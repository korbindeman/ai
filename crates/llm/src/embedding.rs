//! Embedding backends and client.

use crate::capability::EmbeddingCapabilities;
use crate::error::LlmError;
use crate::event::{CallControl, WireCapture};
use crate::extension::Extension;
use crate::id::{BackendId, CallId, ModelId, ModelRef};
use crate::response::Usage;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// A backend that embeds text.
#[async_trait]
pub trait EmbeddingBackend: Send + Sync + 'static {
    /// Return embedding capabilities for `model`.
    fn capabilities(&self, model: &ModelId) -> EmbeddingCapabilities;

    /// Embed `request.inputs` in input order.
    async fn embed(
        &self,
        request: EmbeddingRequest,
        control: CallControl,
    ) -> Result<EmbeddingResult, LlmError>;
}

/// An embedding request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model to call.
    pub model: ModelRef,
    /// Texts to embed, in order.
    pub inputs: Vec<String>,
    /// Requested vector size, when the backend supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Namespaced request extensions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<Extension>,
    /// Deadline for the call.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::serde_util::serialize_opt_duration_millis",
        deserialize_with = "crate::serde_util::deserialize_opt_duration_millis"
    )]
    pub timeout: Option<Duration>,
}

impl EmbeddingRequest {
    /// Create a request for `model` and `inputs`.
    pub fn new(model: ModelRef, inputs: Vec<String>) -> Self {
        Self {
            model,
            inputs,
            dimensions: None,
            extensions: Vec::new(),
            timeout: None,
        }
    }
}

/// Result of an embedding call.
///
/// `vectors` contains one vector for each input, in input order. Mixed
/// dimensions in one result are rejected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Model that produced the vectors.
    pub model: ModelRef,
    /// Embedding vectors, one per input.
    pub vectors: Vec<Vec<f32>>,
    /// Dimension of each vector.
    pub dimensions: u32,
    /// Token usage. Unknown fields remain `None`.
    pub usage: Usage,
    /// Namespaced result extensions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<Extension>,
}

/// Owns a map from backend ID to [`EmbeddingBackend`].
#[derive(Clone)]
pub struct EmbeddingClient {
    backends: BTreeMap<BackendId, Arc<dyn EmbeddingBackend>>,
    wire_capture: WireCapture,
}

impl EmbeddingClient {
    /// Create a builder.
    pub fn builder() -> EmbeddingClientBuilder {
        EmbeddingClientBuilder::new()
    }

    /// Return capabilities for `model`.
    pub fn capabilities(&self, model: &ModelRef) -> Result<EmbeddingCapabilities, LlmError> {
        Ok(self.backend(&model.backend)?.capabilities(&model.model))
    }

    /// Embed texts using the registered backend for `request.model`.
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResult, LlmError> {
        let backend = Arc::clone(self.backend(&request.model.backend)?);
        let capabilities = backend.capabilities(&request.model.model);
        if capabilities.custom_dimensions.is_unsupported() && request.dimensions.is_some() {
            return Err(LlmError::unsupported_capability("custom_dimensions"));
        }

        let call_id = CallId::new();
        let started = Instant::now();
        let cancel = CancellationToken::new();
        let deadline = request.timeout.map(|timeout| started + timeout);
        let control = CallControl::new(call_id, cancel.clone(), deadline, self.wire_capture);
        let expected_count = request.inputs.len();

        tracing::info!(
            call_id = %call_id,
            backend = %request.model.backend,
            model = %request.model.model,
            "embedding started"
        );

        let embed = backend.embed(request, control.clone());
        let result = if let Some(remaining) = control.remaining() {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Err(LlmError::cancelled()),
                _ = tokio::time::sleep(remaining) => Err(LlmError::timeout("embedding timed out")),
                result = embed => result,
            }
        } else {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Err(LlmError::cancelled()),
                result = embed => result,
            }
        };

        let result = match result {
            Ok(_) if cancel.is_cancelled() => Err(LlmError::cancelled()),
            other => other,
        };

        match result {
            Ok(result) => {
                validate_embedding_result(&result, expected_count)?;
                tracing::info!(
                    call_id = %call_id,
                    duration_ms = started.elapsed().as_millis(),
                    "embedding finished"
                );
                Ok(result)
            }
            Err(error) => {
                tracing::info!(
                    call_id = %call_id,
                    duration_ms = started.elapsed().as_millis(),
                    error.kind = ?error.kind(),
                    "embedding failed"
                );
                Err(error.with_call_id(call_id))
            }
        }
    }

    fn backend(&self, id: &BackendId) -> Result<&Arc<dyn EmbeddingBackend>, LlmError> {
        self.backends
            .get(id)
            .ok_or_else(|| LlmError::unknown_backend(id.clone()))
    }
}

/// Builder for [`EmbeddingClient`].
#[derive(Default)]
pub struct EmbeddingClientBuilder {
    backends: BTreeMap<BackendId, Arc<dyn EmbeddingBackend>>,
    wire_capture: WireCapture,
}

impl EmbeddingClientBuilder {
    /// Create a builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an embedding backend.
    pub fn backend(
        mut self,
        id: impl Into<String>,
        backend: impl EmbeddingBackend,
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

    /// Set the wire-capture level for built-in HTTP adapters.
    pub fn wire_capture(mut self, wire_capture: WireCapture) -> Self {
        self.wire_capture = wire_capture;
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<EmbeddingClient, LlmError> {
        Ok(EmbeddingClient {
            backends: self.backends,
            wire_capture: self.wire_capture,
        })
    }
}

fn validate_embedding_result(
    result: &EmbeddingResult,
    expected_count: usize,
) -> Result<(), LlmError> {
    if result.vectors.len() != expected_count {
        return Err(LlmError::invalid_response(format!(
            "expected {expected_count} embedding vectors, got {}",
            result.vectors.len()
        )));
    }
    for vector in &result.vectors {
        if vector.len() as u32 != result.dimensions {
            return Err(LlmError::invalid_response(
                "embedding result contains mixed vector dimensions",
            ));
        }
    }
    Ok(())
}
