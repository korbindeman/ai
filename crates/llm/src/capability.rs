//! Model capability descriptions.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::id::ModelId;

/// Whether a model supports a capability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Support {
    /// The model supports the capability.
    Supported,
    /// The model does not support the capability.
    Unsupported,
    /// Support is not known. The client may send the request.
    #[default]
    Unknown,
}

impl Support {
    /// Return true when the client must reject a request that uses this capability.
    pub fn is_unsupported(self) -> bool {
        matches!(self, Self::Unsupported)
    }
}

/// Model-specific capability information.
///
/// Built-in adapters do not invent context limits from a backend-wide default.
/// Configured model metadata can override discovered metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Image input.
    pub image_input: Support,
    /// Tool calling.
    pub tools: Support,
    /// Native structured output.
    pub structured_output: Support,
    /// Reasoning output or configuration.
    pub reasoning: Support,
    /// Token counting.
    pub token_counting: Support,
    /// Temperature sampling.
    pub temperature: Support,
    /// Nucleus sampling.
    pub top_p: Support,
    /// Stop sequences.
    pub stop_sequences: Support,
    /// Context window in tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Maximum output tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Backend-specific capability metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

impl ModelCapabilities {
    /// Capabilities with every field set to [`Support::Unknown`].
    pub fn unknown() -> Self {
        Self::default()
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            image_input: Support::Unknown,
            tools: Support::Unknown,
            structured_output: Support::Unknown,
            reasoning: Support::Unknown,
            token_counting: Support::Unknown,
            temperature: Support::Unknown,
            top_p: Support::Unknown,
            stop_sequences: Support::Unknown,
            context_window: None,
            max_output_tokens: None,
            extensions: BTreeMap::new(),
        }
    }
}

/// A model advertised by a backend.
///
/// The common crate does not attach cost tiers or guessed context limits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: ModelId,
    /// Human-readable name.
    pub display_name: String,
    /// Model-specific capabilities.
    pub capabilities: ModelCapabilities,
    /// Backend-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Embedding-specific capability information.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCapabilities {
    /// Whether the caller may request a custom vector size.
    pub custom_dimensions: Support,
    /// Backend-specific capability metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

impl EmbeddingCapabilities {
    /// Capabilities with unknown support.
    pub fn unknown() -> Self {
        Self::default()
    }
}

impl Default for EmbeddingCapabilities {
    fn default() -> Self {
        Self {
            custom_dimensions: Support::Unknown,
            extensions: BTreeMap::new(),
        }
    }
}
