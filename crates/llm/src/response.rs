//! Generation results and usage.

use crate::extension::Extension;
use crate::message::ContentBlock;
use crate::request::BackendState;
use serde::{Deserialize, Serialize};

/// Token usage reported by a backend.
///
/// Unknown values remain `None`. The crate does not replace an unknown value
/// with zero.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Cached input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// Output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Copy each present field from `other` onto this value.
    pub fn merge(&mut self, other: Self) {
        if other.input_tokens.is_some() {
            self.input_tokens = other.input_tokens;
        }
        if other.cached_input_tokens.is_some() {
            self.cached_input_tokens = other.cached_input_tokens;
        }
        if other.output_tokens.is_some() {
            self.output_tokens = other.output_tokens;
        }
        if other.reasoning_tokens.is_some() {
            self.reasoning_tokens = other.reasoning_tokens;
        }
    }
}

/// Why the model stopped generating.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StopReason {
    /// The model ended its turn.
    EndTurn,
    /// The model issued one or more tool calls.
    ToolCall,
    /// The model reached the output token limit.
    MaxOutputTokens,
    /// A content filter stopped the model.
    ContentFilter,
    /// A backend-specific stop reason.
    Other {
        /// Backend-specific reason.
        reason: String,
    },
}

impl StopReason {
    /// Map a provider stop-reason string onto the common enum.
    pub fn from_provider(reason: &str) -> Self {
        match reason {
            "end_turn" | "stop" | "eos" | "completed" => Self::EndTurn,
            "tool_use" | "tool_calls" | "function_call" => Self::ToolCall,
            "max_tokens" | "length" | "max_output_tokens" => Self::MaxOutputTokens,
            "content_filter" | "content_filtered" => Self::ContentFilter,
            other => Self::Other {
                reason: other.to_string(),
            },
        }
    }
}

/// Final result of a generation call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationResult {
    /// User-visible output.
    pub content: Vec<ContentBlock>,
    /// Why the model stopped.
    pub stop_reason: StopReason,
    /// Token usage. Unknown fields remain `None`.
    pub usage: Usage,
    /// Opaque continuation data for a later call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_state: Vec<BackendState>,
    /// Namespaced result extensions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<Extension>,
}

impl GenerationResult {
    /// Concatenate text blocks in the result.
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }

    /// Tool-call blocks in the result.
    pub fn tool_calls(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolCall { .. }))
            .collect()
    }
}

impl Default for GenerationResult {
    fn default() -> Self {
        Self {
            content: Vec::new(),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_keeps_unknown_usage_unknown() {
        let mut usage = Usage {
            input_tokens: Some(10),
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
        };
        usage.merge(Usage {
            input_tokens: None,
            cached_input_tokens: None,
            output_tokens: Some(4),
            reasoning_tokens: None,
        });
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.output_tokens, Some(4));
        assert_eq!(usage.reasoning_tokens, None);
    }

    #[test]
    fn maps_provider_stop_reasons() {
        assert_eq!(StopReason::from_provider("end_turn"), StopReason::EndTurn);
        assert_eq!(StopReason::from_provider("stop"), StopReason::EndTurn);
        assert_eq!(StopReason::from_provider("tool_use"), StopReason::ToolCall);
        assert_eq!(
            StopReason::from_provider("tool_calls"),
            StopReason::ToolCall
        );
        assert_eq!(
            StopReason::from_provider("max_tokens"),
            StopReason::MaxOutputTokens
        );
        assert_eq!(
            StopReason::from_provider("content_filter"),
            StopReason::ContentFilter
        );
        assert_eq!(
            StopReason::from_provider("weird_stop"),
            StopReason::Other {
                reason: "weird_stop".into()
            }
        );
    }
}
