//! Generation requests and options.

use crate::extension::Extension;
use crate::id::{BackendId, ModelRef};
use crate::message::Message;
use crate::tool::{OutputFormat, ToolChoice, ToolDefinition};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

/// Opaque continuation data that a backend needs for a later call.
///
/// Backend state is sensitive by default. The payload is redacted in [`Debug`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendState {
    /// Backend that owns this state.
    pub backend: BackendId,
    /// Backend-defined kind, such as `responses_item`.
    pub kind: String,
    /// Opaque payload. Adapters must preserve it without normalization.
    pub payload: serde_json::Value,
}

impl BackendState {
    /// Create backend state.
    pub fn new(backend: BackendId, kind: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            backend,
            kind: kind.into(),
            payload,
        }
    }
}

impl fmt::Debug for BackendState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendState")
            .field("backend", &self.backend)
            .field("kind", &self.kind)
            .field("payload", &"<redacted>")
            .finish()
    }
}

/// Reasoning configuration for a generation request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningOptions {
    /// Qualitative reasoning effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Token budget for reasoning, when the backend uses a budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Qualitative reasoning effort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
}

/// Sampling and output options for a generation request.
///
/// The default selects text output and automatic tool choice. It does not set
/// a timeout, token limit, temperature, or `top_p`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Sequences that stop generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Tool selection strategy.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Requested output shape.
    #[serde(default)]
    pub output_format: OutputFormat,
    /// Reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOptions>,
    /// Deadline for the call.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::serde_util::serialize_opt_duration_millis",
        deserialize_with = "crate::serde_util::deserialize_opt_duration_millis"
    )]
    pub timeout: Option<Duration>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            tool_choice: ToolChoice::Auto,
            output_format: OutputFormat::Text,
            reasoning: None,
            timeout: None,
        }
    }
}

/// A generation request.
///
/// Request metadata is for consumer correlation. An adapter must not send it
/// to a provider. Provider metadata belongs in a namespaced extension.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Model to call.
    pub model: ModelRef,
    /// System instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Sampling and output options.
    #[serde(default)]
    pub options: GenerationOptions,
    /// Opaque continuation data from earlier calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_state: Vec<BackendState>,
    /// Namespaced request extensions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<Extension>,
    /// Consumer correlation metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl GenerationRequest {
    /// Create a request for `model` with default options.
    pub fn new(model: ModelRef) -> Self {
        Self {
            model,
            instructions: None,
            messages: Vec::new(),
            tools: Vec::new(),
            options: GenerationOptions::default(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Set system instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set conversation messages.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Set tools.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set generation options.
    pub fn with_options(mut self, options: GenerationOptions) -> Self {
        self.options = options;
        self
    }
}

/// A token-counting request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenCountRequest {
    /// Model to count against.
    pub model: ModelRef,
    /// System instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Tools included in the count.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
}

/// Token-count result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCount {
    /// Estimated input tokens.
    pub input_tokens: u64,
}

pub(crate) fn semantic_request_for_compare(request: &GenerationRequest) -> GenerationRequest {
    let mut cloned = request.clone();
    cloned.metadata.clear();
    cloned.options.timeout = None;
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::BackendId;

    #[test]
    fn backend_state_debug_redacts_payload() {
        let state = BackendState::new(
            BackendId::new("chatgpt").unwrap(),
            "responses_item",
            serde_json::json!({"encrypted_content": "SECRET_PAYLOAD"}),
        );
        let debug = format!("{state:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("SECRET_PAYLOAD"));
        assert!(!debug.contains("encrypted_content"));
    }

    #[test]
    fn default_options_are_unconstrained_text() {
        let options = GenerationOptions::default();
        assert!(options.max_output_tokens.is_none());
        assert!(options.temperature.is_none());
        assert!(options.top_p.is_none());
        assert!(options.timeout.is_none());
        assert!(matches!(options.tool_choice, ToolChoice::Auto));
        assert!(matches!(options.output_format, OutputFormat::Text));
    }
}
