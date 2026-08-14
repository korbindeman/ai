//! Backends for tests.

use crate::backend::ModelBackend;
use crate::capability::ModelCapabilities;
use crate::error::LlmError;
use crate::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use crate::id::ModelId;
use crate::message::ContentBlock;
use crate::request::{GenerationRequest, semantic_request_for_compare};
use crate::response::{GenerationResult, StopReason, Usage};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

/// One scripted generation response.
#[derive(Debug)]
pub enum ScriptedResponse {
    /// Emit events, then return a successful result.
    Success {
        /// Output events to emit, in order.
        events: Vec<OutputEvent>,
        /// Final result returned to the client.
        result: GenerationResult,
    },
    /// Emit events, then return an error.
    Failure {
        /// Output events to emit before the error.
        events: Vec<OutputEvent>,
        /// Terminal error.
        error: LlmError,
    },
    /// Wait until the call is cancelled or times out.
    Hang,
    /// Panic inside the backend future.
    Panic(&'static str),
}

impl ScriptedResponse {
    /// A successful text completion with one text delta.
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::Success {
            events: vec![OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Text { text: text.clone() },
            }],
            result: GenerationResult {
                content: vec![ContentBlock::text(text)],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
                backend_state: Vec::new(),
                extensions: Vec::new(),
            },
        }
    }
}

/// A backend that serves ordered request expectations for unit tests.
///
/// If calls occur in the wrong order, or after the script is exhausted, the
/// backend returns [`LlmError::invalid_request`].
#[derive(Debug)]
pub struct ScriptedBackend {
    capabilities: ModelCapabilities,
    turns: Mutex<VecDeque<ScriptedTurn>>,
}

#[derive(Debug)]
struct ScriptedTurn {
    expected: Option<GenerationRequest>,
    response: ScriptedResponse,
}

impl ScriptedBackend {
    /// Create an empty scripted backend.
    pub fn new() -> Self {
        Self {
            capabilities: ModelCapabilities::unknown(),
            turns: Mutex::new(VecDeque::new()),
        }
    }

    /// Set capabilities returned for every model.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Append a turn that ignores the request contents.
    pub fn enqueue(self, response: ScriptedResponse) -> Self {
        self.push(None, response);
        self
    }

    /// Append a turn that requires a semantic request match.
    ///
    /// Comparison excludes correlation metadata and timeout.
    pub fn expect(self, request: GenerationRequest, response: ScriptedResponse) -> Self {
        self.push(Some(request), response);
        self
    }

    fn push(&self, expected: Option<GenerationRequest>, response: ScriptedResponse) {
        self.turns
            .lock()
            .expect("scripted backend mutex")
            .push_back(ScriptedTurn { expected, response });
    }
}

impl Default for ScriptedBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelBackend for ScriptedBackend {
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities {
        let _ = model;
        self.capabilities.clone()
    }

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError> {
        let turn = self
            .turns
            .lock()
            .expect("scripted backend mutex")
            .pop_front()
            .ok_or_else(|| LlmError::invalid_request("scripted backend has no remaining turns"))?;

        if let Some(expected) = turn.expected
            && semantic_request_for_compare(&expected) != semantic_request_for_compare(&request)
        {
            return Err(LlmError::invalid_request(
                "scripted backend received a request in the wrong order",
            ));
        }

        match turn.response {
            ScriptedResponse::Success {
                events: output,
                result,
            } => {
                for event in output {
                    events.emit(event).await?;
                }
                Ok(result)
            }
            ScriptedResponse::Failure {
                events: output,
                error,
            } => {
                for event in output {
                    events.emit(event).await?;
                }
                Err(error)
            }
            ScriptedResponse::Hang => {
                control.cancelled().await;
                Err(LlmError::cancelled())
            }
            ScriptedResponse::Panic(message) => panic!("{message}"),
        }
    }
}
