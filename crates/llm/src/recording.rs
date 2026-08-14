//! Call records and replay.

use crate::backend::ModelBackend;
use crate::capability::ModelCapabilities;
use crate::error::{ErrorReport, LlmError};
use crate::event::{CallControl, CallEvent, EventSink, GenerationEvent};
use crate::id::{CallId, ModelId};
use crate::request::{GenerationRequest, semantic_request_for_compare};
use crate::response::GenerationResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// First call-record schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// A serializable request, event sequence, and outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedCall {
    /// Schema version. The first version is `1`.
    pub schema_version: u32,
    /// Call identifier from the original run.
    pub call_id: CallId,
    /// Unix time in milliseconds when the original call started.
    pub started_at_unix_ms: u64,
    /// Original request, including unknown extensions and backend state.
    pub request: GenerationRequest,
    /// Capabilities used for the original call.
    pub capabilities: ModelCapabilities,
    /// Sequenced events. A successful call includes the terminal `Finished` event.
    pub events: Vec<CallEvent>,
    /// Terminal outcome.
    pub outcome: RecordedOutcome,
}

/// Terminal outcome of a recorded call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecordedOutcome {
    /// The call succeeded. The result equals the terminal `Finished` event.
    Succeeded(GenerationResult),
    /// The call failed after zero or more successful events.
    Failed(ErrorReport),
}

/// Replays a [`RecordedCall`] through [`ModelBackend`].
///
/// The replay backend compares the semantic request with the recorded request.
/// The comparison excludes call ID, start time, deadline, and correlation
/// metadata. By default events are emitted without recorded delays.
pub struct ReplayBackend {
    record: RecordedCall,
    timed: bool,
}

impl ReplayBackend {
    /// Create a replay backend that emits events without recorded delays.
    pub fn new(record: RecordedCall) -> Self {
        Self {
            record,
            timed: false,
        }
    }

    /// Emit events using the recorded elapsed times.
    pub fn with_recorded_timing(mut self) -> Self {
        self.timed = true;
        self
    }
}

#[async_trait]
impl ModelBackend for ReplayBackend {
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities {
        let _ = model;
        self.record.capabilities.clone()
    }

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError> {
        if let Some(summary) = request_mismatch(&self.record.request, &request) {
            return Err(LlmError::invalid_request(format!(
                "replay request mismatch: {summary}"
            )));
        }

        let mut last_elapsed = 0;
        for event in &self.record.events {
            if control.is_cancelled() {
                return Err(LlmError::cancelled());
            }
            let GenerationEvent::Output(output) = &event.event else {
                continue;
            };
            if self.timed {
                let wait = event.elapsed_micros.saturating_sub(last_elapsed);
                if wait > 0 {
                    tokio::select! {
                        _ = control.cancelled() => return Err(LlmError::cancelled()),
                        _ = tokio::time::sleep(Duration::from_micros(wait)) => {}
                    }
                }
                last_elapsed = event.elapsed_micros;
            }
            events.emit(output.clone()).await?;
        }

        match &self.record.outcome {
            RecordedOutcome::Succeeded(result) => Ok(result.clone()),
            RecordedOutcome::Failed(report) => Err(LlmError::from_report(report.clone())),
        }
    }
}

fn request_mismatch(recorded: &GenerationRequest, actual: &GenerationRequest) -> Option<String> {
    let recorded = semantic_request_for_compare(recorded);
    let actual = semantic_request_for_compare(actual);
    if recorded == actual {
        return None;
    }

    let mut parts = Vec::new();
    if recorded.model != actual.model {
        parts.push(format!(
            "model: recorded {} actual {}",
            recorded.model, actual.model
        ));
    }
    if recorded.instructions != actual.instructions {
        parts.push("instructions differ".into());
    }
    if recorded.messages != actual.messages {
        parts.push(format!(
            "messages: recorded {} actual {}",
            recorded.messages.len(),
            actual.messages.len()
        ));
    }
    if recorded.tools != actual.tools {
        parts.push("tools differ".into());
    }
    if recorded.options != actual.options {
        parts.push("options differ".into());
    }
    if recorded.backend_state != actual.backend_state {
        parts.push("backend_state differs".into());
    }
    if recorded.extensions != actual.extensions {
        parts.push("extensions differ".into());
    }
    if parts.is_empty() {
        parts.push("semantic request differs".into());
    }
    Some(parts.join("; "))
}
