//! Semantic conversation messages.

use crate::extension::Extension;
use serde::{Deserialize, Serialize};

/// Role of a message in a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Message from the user.
    User,
    /// Message from the assistant.
    Assistant,
}

/// How reasoning text should be treated by a consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningVisibility {
    /// A user-visible summary of reasoning.
    Summary,
    /// A detailed reasoning trace.
    Trace,
}

/// Image bytes or a remote image URL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// MIME type, such as `image/png`.
        media_type: String,
        /// Base64 payload without a `data:` prefix.
        data: String,
    },
    /// Remote image URL.
    Url {
        /// Image URL.
        url: String,
    },
}

/// An image in a message or tool result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    /// Image bytes or URL.
    pub source: ImageSource,
    /// Optional caption shown beside the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Image {
    /// Create a base64 image.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            label: None,
        }
    }

    /// Create an image from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url { url: url.into() },
            label: None,
        }
    }

    /// Set the image label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Content in a tool result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    /// Plain text.
    Text {
        /// Tool result text.
        text: String,
    },
    /// An image produced by the tool.
    Image(Image),
    /// Namespaced data outside the common interface.
    Extension(Extension),
}

/// One block of message content.
///
/// Reasoning text is semantic content. Plain or encrypted continuation data
/// belongs in [`crate::BackendState`], not in a content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// Text.
        text: String,
    },
    /// An image.
    Image(Image),
    /// Model reasoning.
    Reasoning {
        /// Reasoning text.
        text: String,
        /// Whether the text is a summary or a trace.
        visibility: ReasoningVisibility,
    },
    /// A tool call from the assistant.
    ToolCall {
        /// Tool-call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Tool arguments.
        arguments: serde_json::Value,
    },
    /// A tool result from the user.
    ToolResult {
        /// Identifier of the tool call this result answers.
        tool_call_id: String,
        /// Result content.
        content: Vec<ToolResultBlock>,
        /// Whether the tool failed.
        is_error: bool,
    },
    /// Namespaced data outside the common interface.
    Extension(Extension),
}

impl ContentBlock {
    /// Create a text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool-call block.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Create a tool-result block with text content.
    pub fn tool_result_text(
        tool_call_id: impl Into<String>,
        text: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: vec![ToolResultBlock::Text { text: text.into() }],
            is_error,
        }
    }
}

/// A message in a conversation.
///
/// The crate does not define a tool role. Each adapter maps tool results to
/// its wire format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create a user message that contains a tool result.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        text: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::tool_result_text(tool_call_id, text, is_error)],
        }
    }

    /// Concatenate text blocks in this message.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Return true when this is a user message.
    pub fn is_user(&self) -> bool {
        self.role == Role::User
    }

    /// Return true when this is an assistant message.
    pub fn is_assistant(&self) -> bool {
        self.role == Role::Assistant
    }
}
