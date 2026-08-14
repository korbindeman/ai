//! Client, registry, and generation handle.

use crate::backend::ModelBackend;
use crate::capability::ModelCapabilities;
use crate::error::LlmError;
use crate::event::{CallControl, CallEvent, EventSink, GenerationEvent, WireCapture};
use crate::id::{BackendId, CallId, ModelRef};
use crate::message::{ContentBlock, ToolResultBlock};
use crate::recording::{RecordedCall, RecordedOutcome, SCHEMA_VERSION};
use crate::request::{GenerationRequest, TokenCount, TokenCountRequest};
use crate::response::GenerationResult;
use crate::tool::{OutputFormat, ToolChoice};
use futures::Stream;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const DEFAULT_EVENT_BUFFER_CAPACITY: usize = 64;

/// Owns a map from backend ID to [`ModelBackend`].
///
/// The crate does not maintain a global registry or a global default model.
#[derive(Clone)]
pub struct LlmClient {
    backends: BTreeMap<BackendId, Arc<dyn ModelBackend>>,
    event_buffer_capacity: usize,
    wire_capture: WireCapture,
}

impl LlmClient {
    /// Create a builder.
    pub fn builder() -> LlmClientBuilder {
        LlmClientBuilder::new()
    }

    /// Validate `request` and start a generation stream.
    ///
    /// Validation is synchronous and happens before a runner task is spawned.
    pub fn generate(&self, request: GenerationRequest) -> Result<Generation, LlmError> {
        let backend = self.backend(&request.model.backend)?;
        let capabilities = backend.capabilities(&request.model.model);
        validate_generation_request(&request, &capabilities)?;

        let call_id = CallId::new();
        let started = Instant::now();
        let cancel = CancellationToken::new();
        let deadline = request.options.timeout.map(|timeout| started + timeout);
        let sequence = Arc::new(AtomicU64::new(0));
        let (tx, rx) = mpsc::channel(self.event_buffer_capacity);
        let events = EventSink::new(
            call_id,
            started,
            Arc::clone(&sequence),
            tx.clone(),
            cancel.clone(),
        );
        let control = CallControl::new(call_id, cancel.clone(), deadline, self.wire_capture);

        tracing::info!(
            call_id = %call_id,
            backend = %request.model.backend,
            model = %request.model.model,
            "generation started"
        );

        let runner_backend = Arc::clone(backend);
        let runner_events = events.clone();
        let runner_control = control.clone();
        let runner_cancel = cancel.clone();
        let runner_request = request;
        let handle = tokio::spawn(async move {
            run_generation(
                runner_backend,
                runner_request,
                runner_events,
                runner_control,
                runner_cancel,
                started,
            )
            .await;
        });

        Ok(Generation {
            call_id,
            rx,
            cancel,
            handle: Some(handle),
            terminated: false,
        })
    }

    /// Generate and collect the stream into a final result.
    pub async fn complete(&self, request: GenerationRequest) -> Result<GenerationResult, LlmError> {
        self.generate(request)?.finish().await
    }

    /// Generate and collect a serializable call record.
    ///
    /// A backend error produces a successful record with a failed outcome.
    /// Request validation errors return [`Err`] before a record exists.
    pub async fn record_call(&self, request: GenerationRequest) -> Result<RecordedCall, LlmError> {
        let capabilities = self.capabilities(&request.model)?;
        let started_at_unix_ms = unix_now_ms();
        let mut generation = self.generate(request.clone())?;
        let call_id = generation.call_id();
        let mut events = Vec::new();
        let outcome = loop {
            match generation.next_event().await {
                None => {
                    break RecordedOutcome::Failed(
                        LlmError::internal("generation ended without a terminal event").report(),
                    );
                }
                Some(Ok(event)) => {
                    let finished = matches!(event.event, GenerationEvent::Finished(_));
                    events.push(event.clone());
                    if let GenerationEvent::Finished(result) = event.event {
                        let _ = finished;
                        break RecordedOutcome::Succeeded(result);
                    }
                }
                Some(Err(error)) => {
                    break RecordedOutcome::Failed(error.report());
                }
            }
        };

        Ok(RecordedCall {
            schema_version: SCHEMA_VERSION,
            call_id,
            started_at_unix_ms,
            request,
            capabilities,
            events,
            outcome,
        })
    }

    /// Return capabilities for `model`.
    pub fn capabilities(&self, model: &ModelRef) -> Result<ModelCapabilities, LlmError> {
        Ok(self.backend(&model.backend)?.capabilities(&model.model))
    }

    /// List models for a registered backend.
    pub async fn list_models(
        &self,
        backend: &BackendId,
    ) -> Result<Vec<crate::capability::ModelInfo>, LlmError> {
        self.backend(backend)?.list_models().await
    }

    /// Count tokens using the backend for `request.model`.
    pub async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCount, LlmError> {
        let backend = self.backend(&request.model.backend)?;
        let call_id = CallId::new();
        let cancel = CancellationToken::new();
        let control = CallControl::new(call_id, cancel, None, self.wire_capture);
        backend.count_tokens(request, control).await
    }

    /// Complete a call with native structured output and deserialize it.
    ///
    /// This helper uses [`OutputFormat::JsonSchema`]. It does not fall back to
    /// tool calling. If the model lacks structured output, the client returns
    /// an unsupported-capability error.
    pub async fn complete_structured<T: DeserializeOwned>(
        &self,
        mut request: GenerationRequest,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Result<T, LlmError> {
        request.options.output_format = OutputFormat::JsonSchema {
            name: name.into(),
            schema,
            strict,
        };
        let result = self.complete(request).await?;
        let text = result
            .text()
            .ok_or_else(|| LlmError::invalid_response("structured output produced no text"))?;
        serde_json::from_str(&text).map_err(|error| {
            LlmError::invalid_response("structured output is not valid JSON").with_source(error)
        })
    }

    fn backend(&self, id: &BackendId) -> Result<&Arc<dyn ModelBackend>, LlmError> {
        self.backends
            .get(id)
            .ok_or_else(|| LlmError::unknown_backend(id.clone()))
    }
}

/// Builder for [`LlmClient`].
#[derive(Default)]
pub struct LlmClientBuilder {
    backends: BTreeMap<BackendId, Arc<dyn ModelBackend>>,
    event_buffer_capacity: usize,
    wire_capture: WireCapture,
}

impl LlmClientBuilder {
    /// Create a builder with a default event buffer capacity of 64.
    pub fn new() -> Self {
        Self {
            backends: BTreeMap::new(),
            event_buffer_capacity: DEFAULT_EVENT_BUFFER_CAPACITY,
            wire_capture: WireCapture::Off,
        }
    }

    /// Register a backend.
    ///
    /// Duplicate backend IDs are rejected. `id` must not be empty.
    pub fn backend(
        mut self,
        id: impl Into<String>,
        backend: impl ModelBackend,
    ) -> Result<Self, LlmError> {
        let id = BackendId::new(id)?;
        if self.backends.contains_key(&id) {
            return Err(LlmError::invalid_request(format!(
                "duplicate backend id: {id}"
            )));
        }
        self.backends.insert(id, Arc::new(backend));
        Ok(self)
    }

    /// Set the bounded event-buffer capacity.
    ///
    /// The capacity must be greater than zero.
    pub fn event_buffer_capacity(mut self, capacity: usize) -> Result<Self, LlmError> {
        if capacity == 0 {
            return Err(LlmError::invalid_request(
                "event buffer capacity must be greater than zero",
            ));
        }
        self.event_buffer_capacity = capacity;
        Ok(self)
    }

    /// Set the wire-capture level for built-in HTTP adapters.
    pub fn wire_capture(mut self, wire_capture: WireCapture) -> Self {
        self.wire_capture = wire_capture;
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<LlmClient, LlmError> {
        Ok(LlmClient {
            backends: self.backends,
            event_buffer_capacity: self.event_buffer_capacity,
            wire_capture: self.wire_capture,
        })
    }
}

impl fmt::Debug for LlmClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmClientBuilder")
            .field(
                "backends",
                &self.backends.keys().cloned().collect::<Vec<_>>(),
            )
            .field("event_buffer_capacity", &self.event_buffer_capacity)
            .field("wire_capture", &self.wire_capture)
            .finish()
    }
}

/// A live generation stream.
///
/// Dropping this value cancels the call and aborts its runner task. There is
/// no separate completion handle, so a consumer cannot deadlock by waiting
/// without reading events.
#[must_use = "dropping Generation cancels the call"]
pub struct Generation {
    call_id: CallId,
    rx: mpsc::Receiver<Result<CallEvent, LlmError>>,
    cancel: CancellationToken,
    handle: Option<JoinHandle<()>>,
    terminated: bool,
}

impl fmt::Debug for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Generation")
            .field("call_id", &self.call_id)
            .field("terminated", &self.terminated)
            .finish_non_exhaustive()
    }
}

impl Generation {
    /// Identifier for this call.
    pub fn call_id(&self) -> CallId {
        self.call_id
    }

    /// Cancel the call. The stream still yields the terminal cancelled error.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Wait for the next sequenced event.
    pub async fn next_event(&mut self) -> Option<Result<CallEvent, LlmError>> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_next(cx)).await
    }

    /// Drain remaining events and return the final result.
    pub async fn finish(mut self) -> Result<GenerationResult, LlmError> {
        let mut result = None;
        while let Some(item) = self.next_event().await {
            if let CallEvent {
                event: GenerationEvent::Finished(finished),
                ..
            } = item?
            {
                result = Some(finished);
            }
        }
        result.ok_or_else(|| LlmError::internal("generation ended without a result"))
    }
}

impl Stream for Generation {
    type Item = Result<CallEvent, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminated {
            return Poll::Ready(None);
        }
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(item)) => {
                let terminal = match &item {
                    Ok(event) => matches!(event.event, GenerationEvent::Finished(_)),
                    Err(_) => true,
                };
                if terminal {
                    this.terminated = true;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                this.terminated = true;
                let error = if this.cancel.is_cancelled() {
                    LlmError::cancelled()
                } else {
                    LlmError::internal("generation ended without a terminal event")
                };
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn run_generation(
    backend: Arc<dyn ModelBackend>,
    request: GenerationRequest,
    events: EventSink,
    control: CallControl,
    cancel: CancellationToken,
    started: Instant,
) {
    let backend_id = request.model.backend.clone();
    let model_id = request.model.model.clone();
    let call_id = control.call_id();
    let generate = backend.generate(request, events.clone(), control.clone());
    let result = if let Some(remaining) = control.remaining() {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmError::cancelled()),
            _ = tokio::time::sleep(remaining) => Err(LlmError::timeout("generation timed out")),
            result = generate => result,
        }
    } else {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmError::cancelled()),
            result = generate => result,
        }
    };

    let result = match result {
        Ok(_) if cancel.is_cancelled() => Err(LlmError::cancelled()),
        Ok(_) if deadline_exceeded(control.deadline()) => {
            Err(LlmError::timeout("generation timed out"))
        }
        other => other,
    };

    let duration_ms = started.elapsed().as_millis();
    match result {
        Ok(generation) => {
            tracing::info!(
                call_id = %call_id,
                backend = %backend_id,
                model = %model_id,
                duration_ms = duration_ms,
                input_tokens = generation.usage.input_tokens,
                output_tokens = generation.usage.output_tokens,
                "generation finished"
            );
            let _ = events.emit_finished(generation).await;
        }
        Err(error) => {
            let error = error
                .with_call_id(call_id)
                .with_backend(backend_id.clone())
                .with_model(model_id.clone());
            tracing::info!(
                call_id = %call_id,
                backend = %backend_id,
                model = %model_id,
                duration_ms = duration_ms,
                error.kind = ?error.kind(),
                "generation failed"
            );
            let _ = events.emit_error(error).await;
        }
    }
}

fn deadline_exceeded(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

pub(crate) fn validate_generation_request(
    request: &GenerationRequest,
    capabilities: &ModelCapabilities,
) -> Result<(), LlmError> {
    validate_tools(&request.tools, &request.options.tool_choice)?;
    validate_backend_state(&request.model.backend, &request.backend_state)?;
    validate_capabilities(request, capabilities)?;
    Ok(())
}

fn validate_tools(
    tools: &[crate::tool::ToolDefinition],
    choice: &ToolChoice,
) -> Result<(), LlmError> {
    let mut names = HashSet::new();
    for tool in tools {
        if !names.insert(tool.name.as_str()) {
            return Err(LlmError::invalid_request(format!(
                "duplicate tool name: {}",
                tool.name
            )));
        }
    }
    match choice {
        ToolChoice::Named { name } if !names.contains(name.as_str()) => Err(
            LlmError::invalid_request(format!("named tool is not defined: {name}")),
        ),
        ToolChoice::Required if tools.is_empty() => Err(LlmError::invalid_request(
            "tool choice required needs at least one tool",
        )),
        _ => Ok(()),
    }
}

fn validate_backend_state(
    backend: &BackendId,
    state: &[crate::request::BackendState],
) -> Result<(), LlmError> {
    for item in state {
        if &item.backend != backend {
            return Err(LlmError::invalid_request(format!(
                "backend state belongs to {}, not {backend}",
                item.backend
            )));
        }
    }
    Ok(())
}

fn validate_capabilities(
    request: &GenerationRequest,
    capabilities: &ModelCapabilities,
) -> Result<(), LlmError> {
    if capabilities.image_input.is_unsupported() && request_has_image(request) {
        return Err(LlmError::unsupported_capability("image_input"));
    }
    if capabilities.tools.is_unsupported() && !request.tools.is_empty() {
        return Err(LlmError::unsupported_capability("tools"));
    }
    if capabilities.structured_output.is_unsupported()
        && matches!(
            request.options.output_format,
            OutputFormat::JsonSchema { .. }
        )
    {
        return Err(LlmError::unsupported_capability("structured_output"));
    }
    if capabilities.reasoning.is_unsupported() && request.options.reasoning.is_some() {
        return Err(LlmError::unsupported_capability("reasoning"));
    }
    if capabilities.temperature.is_unsupported() && request.options.temperature.is_some() {
        return Err(LlmError::unsupported_capability("temperature"));
    }
    if capabilities.top_p.is_unsupported() && request.options.top_p.is_some() {
        return Err(LlmError::unsupported_capability("top_p"));
    }
    if capabilities.stop_sequences.is_unsupported() && !request.options.stop_sequences.is_empty() {
        return Err(LlmError::unsupported_capability("stop_sequences"));
    }
    Ok(())
}

fn request_has_image(request: &GenerationRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::Image(_) => true,
            ContentBlock::ToolResult { content, .. } => content
                .iter()
                .any(|block| matches!(block, ToolResultBlock::Image(_))),
            _ => false,
        })
    })
}

pub(crate) fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
