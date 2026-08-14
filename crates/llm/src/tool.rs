//! Tool definitions and output format.

use serde::{Deserialize, Serialize};

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool arguments.
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Create a tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Strategy for selecting tools.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    #[default]
    Auto,
    /// Do not call tools.
    None,
    /// Require the model to call a tool.
    Required,
    /// Require the model to call a named tool.
    Named {
        /// Tool name.
        name: String,
    },
}

/// Requested shape of model output.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Unstructured text.
    #[default]
    Text,
    /// Native structured output that conforms to a JSON Schema.
    JsonSchema {
        /// Schema name.
        name: String,
        /// JSON Schema.
        schema: serde_json::Value,
        /// Whether the backend should enforce the schema strictly.
        strict: bool,
    },
}
