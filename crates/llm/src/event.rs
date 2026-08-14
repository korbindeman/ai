//! Stream events, event sink, and call control.

use crate::error::LlmError;
use crate::extension::Extension;
use crate::id::CallId;
use crate::response::{GenerationResult, Usage};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A nonterminal event from a backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputEvent {
    /// Incremental content.
    ContentDelta {
        /// Output index for parallel outputs. Use `0` for a single output.
        output_index: u32,
        /// Incremental content.
        delta: ContentDelta,
    },
    /// Updated usage.
    Usage(Usage),
    /// A sanitized provider request or response record.
    Wire(WireEvent),
    /// Namespaced event data.
    Extension(Extension),
}

/// Incremental content within an output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Incremental text.
    Text {
        /// Text fragment.
        text: String,
    },
    /// Incremental reasoning text.
    Reasoning {
        /// Reasoning fragment.
        text: String,
    },
    /// Incremental tool-call data.
    ToolCall {
        /// Tool-call index for interleaved tool calls.
        tool_index: u32,
        /// Tool-call identifier, when this fragment includes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool name, when this fragment includes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Incremental JSON arguments.
        arguments_delta: String,
    },
    /// Namespaced incremental data.
    Extension(Extension),
}

/// An output event or the final result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum GenerationEvent {
    /// A nonterminal event.
    Output(OutputEvent),
    /// The terminal successful result. Last event in a successful stream.
    Finished(GenerationResult),
}

/// A sequenced generation event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallEvent {
    /// Call that produced the event.
    pub call_id: CallId,
    /// Sequence number assigned by the client. The first sequence number is zero.
    pub sequence: u64,
    /// Microseconds since the call started, from one monotonic clock.
    pub elapsed_micros: u64,
    /// Event payload.
    pub event: GenerationEvent,
}

/// How much provider wire data to capture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireCapture {
    /// Do not emit wire events.
    #[default]
    Off,
    /// Method, sanitized URL, status, duration, and selected safe headers.
    Metadata,
    /// Metadata plus request bodies, response bodies, and stream frames.
    Bodies,
}

/// Direction of a wire event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireDirection {
    /// Outgoing request.
    Request,
    /// Incoming response.
    Response,
}

/// Whether a wire payload may contain sensitive content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    /// The payload is safe to log.
    Public,
    /// The payload may contain prompts, output, or other sensitive data.
    Sensitive,
}

/// A sanitized provider request or response record.
///
/// Wire events never contain authorization headers, API keys, cookies, or
/// signed URL parameters. Body capture uses [`Sensitivity::Sensitive`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireEvent {
    /// Request or response.
    pub direction: WireDirection,
    /// Adapter-defined kind, such as `http` or `sse_frame`.
    pub kind: String,
    /// Sanitized payload.
    pub payload: serde_json::Value,
    /// Sensitivity classification.
    pub sensitivity: Sensitivity,
}

/// Failure when emitting an output event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitError {
    /// The consumer cancelled or dropped the generation.
    Cancelled,
}

impl From<EmitError> for LlmError {
    fn from(value: EmitError) -> Self {
        match value {
            EmitError::Cancelled => LlmError::cancelled(),
        }
    }
}

/// Cloneable sink that delivers output events on a bounded channel.
#[derive(Clone)]
pub struct EventSink {
    call_id: CallId,
    started: Instant,
    sequence: Arc<AtomicU64>,
    tx: mpsc::Sender<Result<CallEvent, LlmError>>,
    cancel: CancellationToken,
}

impl EventSink {
    pub(crate) fn new(
        call_id: CallId,
        started: Instant,
        sequence: Arc<AtomicU64>,
        tx: mpsc::Sender<Result<CallEvent, LlmError>>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            call_id,
            started,
            sequence,
            tx,
            cancel,
        }
    }

    /// Emit a nonterminal output event.
    ///
    /// The sink assigns the sequence number and elapsed time before delivery.
    /// A full bounded channel applies backpressure. After the consumer drops
    /// the generation, this method returns [`EmitError::Cancelled`].
    pub async fn emit(&self, event: OutputEvent) -> Result<(), EmitError> {
        self.send(GenerationEvent::Output(event)).await
    }

    pub(crate) async fn emit_finished(&self, result: GenerationResult) -> Result<(), EmitError> {
        self.send(GenerationEvent::Finished(result)).await
    }

    pub(crate) async fn emit_error(&self, error: LlmError) -> Result<(), EmitError> {
        self.tx
            .send(Err(error))
            .await
            .map_err(|_| EmitError::Cancelled)
    }

    async fn send(&self, event: GenerationEvent) -> Result<(), EmitError> {
        if self.cancel.is_cancelled() {
            return Err(EmitError::Cancelled);
        }
        let call_event = self.wrap(event);
        tokio::select! {
            result = self.tx.send(Ok(call_event)) => {
                result.map_err(|_| EmitError::Cancelled)
            }
            _ = self.cancel.cancelled() => Err(EmitError::Cancelled),
        }
    }

    fn wrap(&self, event: GenerationEvent) -> CallEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        CallEvent {
            call_id: self.call_id,
            sequence,
            elapsed_micros: u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX),
            event,
        }
    }
}

/// Per-call cancellation, deadline, and wire-capture controls.
#[derive(Clone)]
pub struct CallControl {
    call_id: CallId,
    cancel: CancellationToken,
    deadline: Option<Instant>,
    wire_capture: WireCapture,
}

impl CallControl {
    pub(crate) fn new(
        call_id: CallId,
        cancel: CancellationToken,
        deadline: Option<Instant>,
        wire_capture: WireCapture,
    ) -> Self {
        Self {
            call_id,
            cancel,
            deadline,
            wire_capture,
        }
    }

    /// Identifier for this call.
    pub fn call_id(&self) -> CallId {
        self.call_id
    }

    /// Return true when the consumer cancelled or dropped the generation.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Wait until the call is cancelled.
    pub async fn cancelled(&self) {
        self.cancel.cancelled().await;
    }

    /// Deadline for the call, when one was set.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Remaining time until the deadline, when one was set.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Wire-capture level for this call.
    pub fn wire_capture(&self) -> WireCapture {
        self.wire_capture
    }
}
