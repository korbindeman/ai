//! Anthropic Messages API adapter.

use crate::assemble::OutputAssembler;
use crate::backend::ModelBackend;
use crate::capability::ModelCapabilities;
use crate::error::LlmError;
use crate::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use crate::extension::reject_unknown_extensions;
use crate::id::ModelId;
use crate::message::{ContentBlock, ImageSource, Message, Role, ToolResultBlock};
use crate::providers::http_util::{
    emit_http_request, emit_http_response, emit_sse_frame, error_from_response, expose,
    secret_from_string, send,
};
use crate::request::{GenerationRequest, TokenCount, TokenCountRequest};
use crate::response::{GenerationResult, StopReason, Usage};
use crate::tool::{OutputFormat, ToolChoice, ToolDefinition};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

/// Anthropic Messages API backend.
///
/// Model IDs are strings. This adapter does not contain a Claude model enum.
pub struct Anthropic {
    api_key: SecretString,
    http: reqwest::Client,
    base_url: String,
    capabilities: BTreeMap<String, ModelCapabilities>,
    default_capabilities: ModelCapabilities,
}

impl Anthropic {
    /// Create an adapter with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: secret_from_string(api_key),
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.into(),
            capabilities: BTreeMap::new(),
            default_capabilities: ModelCapabilities::unknown(),
        }
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Override capabilities for one model.
    pub fn with_model_capabilities(
        mut self,
        model: impl Into<String>,
        capabilities: ModelCapabilities,
    ) -> Self {
        self.capabilities.insert(model.into(), capabilities);
        self
    }

    /// Override default capabilities for models without a specific entry.
    pub fn with_default_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.default_capabilities = capabilities;
        self
    }
}

impl fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Anthropic")
            .field("api_key", &self.api_key)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ModelBackend for Anthropic {
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities {
        self.capabilities
            .get(model.as_str())
            .cloned()
            .unwrap_or_else(|| self.default_capabilities.clone())
    }

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError> {
        reject_unknown_extensions(&request.extensions, &[])?;
        let body = translate_request(&request)?;
        let url = format!("{}/v1/messages", self.base_url);
        let started = Instant::now();
        emit_http_request(
            &events,
            &control,
            "POST",
            &url,
            &[
                ("anthropic-version", API_VERSION),
                ("content-type", "application/json"),
            ],
            Some(&body),
        )
        .await?;

        let builder = self
            .http
            .post(&url)
            .header("x-api-key", expose(&self.api_key))
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body);
        let response = send(builder, &control).await?;
        if !response.status().is_success() {
            return Err(error_from_response(&events, &control, started, response).await);
        }

        emit_http_response(
            &events,
            &control,
            response.status().as_u16(),
            started.elapsed(),
            None,
            None,
        )
        .await?;
        consume_stream(response, events, control).await
    }

    async fn count_tokens(
        &self,
        request: TokenCountRequest,
        control: CallControl,
    ) -> Result<TokenCount, LlmError> {
        let generation = GenerationRequest {
            model: request.model.clone(),
            instructions: request.instructions.clone(),
            messages: request.messages.clone(),
            tools: request.tools.clone(),
            options: crate::request::GenerationOptions::default(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
            metadata: BTreeMap::new(),
        };
        let mut body = translate_request(&generation)?;
        if let Value::Object(map) = &mut body {
            map.remove("stream");
            map.remove("max_tokens");
        }
        let url = format!("{}/v1/messages/count_tokens", self.base_url);
        let builder = self
            .http
            .post(&url)
            .header("x-api-key", expose(&self.api_key))
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body);
        let response = send(builder, &control).await?;
        if !response.status().is_success() {
            let status = response.status();
            let retry_after = crate::providers::http_util::retry_after_ms(&response);
            let text = response.text().await.unwrap_or_default();
            return Err(crate::providers::http_util::classify_http_error(
                status,
                &text,
                retry_after,
            ));
        }
        let parsed: CountTokensResponse = response.json().await.map_err(|error| {
            LlmError::invalid_response("token count response is not valid JSON").with_source(error)
        })?;
        Ok(TokenCount {
            input_tokens: u64::from(parsed.input_tokens),
        })
    }
}

#[derive(Deserialize)]
struct CountTokensResponse {
    input_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<Value>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    config_type: String,
    budget_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Vec<AnthropicToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicToolResultContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
}

#[derive(Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Serialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    content_block: Option<StreamContentBlock>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    message: Option<StreamMessage>,
}

#[derive(Deserialize)]
struct StreamMessage {
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize, Default)]
struct StreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(rename = "type", default)]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Clone, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

fn translate_request(request: &GenerationRequest) -> Result<Value, LlmError> {
    let thinking = request.options.reasoning.as_ref().and_then(|options| {
        options.budget_tokens.map(|budget_tokens| ThinkingConfig {
            config_type: "enabled".into(),
            budget_tokens,
        })
    });
    if request.options.reasoning.is_some() && thinking.is_none() {
        return Err(LlmError::invalid_request(
            "Anthropic reasoning requires budget_tokens",
        ));
    }

    let output_format = match &request.options.output_format {
        OutputFormat::Text => None,
        OutputFormat::JsonSchema { schema, .. } => Some(serde_json::json!({
            "type": "json_schema",
            "schema": schema,
        })),
    };

    let body = AnthropicRequest {
        model: request.model.model.as_str().to_string(),
        messages: translate_messages(&request.messages)?,
        max_tokens: request.options.max_output_tokens,
        system: request.instructions.clone(),
        stream: true,
        tools: if request.tools.is_empty() {
            None
        } else {
            Some(translate_tools(&request.tools))
        },
        tool_choice: if request.tools.is_empty() {
            None
        } else {
            Some(translate_tool_choice(&request.options.tool_choice))
        },
        thinking,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        stop_sequences: request.options.stop_sequences.clone(),
        output_format,
    };
    serde_json::to_value(body).map_err(|error| {
        LlmError::internal("failed to serialize Anthropic request").with_source(error)
    })
}

fn translate_messages(messages: &[Message]) -> Result<Vec<AnthropicMessage>, LlmError> {
    messages
        .iter()
        .map(|message| {
            Ok(AnthropicMessage {
                role: match message.role {
                    Role::User => "user".into(),
                    Role::Assistant => "assistant".into(),
                },
                content: translate_blocks(&message.content)?,
            })
        })
        .collect()
}

fn translate_blocks(blocks: &[ContentBlock]) -> Result<Vec<AnthropicContentBlock>, LlmError> {
    let mut out = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                out.push(AnthropicContentBlock::Text { text: text.clone() })
            }
            ContentBlock::Image(image) => out.push(AnthropicContentBlock::Image {
                source: image_source(&image.source)?,
            }),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => out.push(AnthropicContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            }),
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
            } => out.push(AnthropicContentBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: translate_tool_result(content)?,
                is_error: is_error.then_some(true),
            }),
            ContentBlock::Reasoning { .. } => {}
            ContentBlock::Extension(extension) => {
                return Err(LlmError::unsupported_extension(
                    &extension.namespace,
                    &extension.name,
                ));
            }
        }
    }
    Ok(out)
}

fn translate_tool_result(
    content: &[ToolResultBlock],
) -> Result<Vec<AnthropicToolResultContent>, LlmError> {
    let mut out = Vec::new();
    for block in content {
        match block {
            ToolResultBlock::Text { text } => {
                out.push(AnthropicToolResultContent::Text { text: text.clone() })
            }
            ToolResultBlock::Image(image) => out.push(AnthropicToolResultContent::Image {
                source: image_source(&image.source)?,
            }),
            ToolResultBlock::Extension(extension) => {
                return Err(LlmError::unsupported_extension(
                    &extension.namespace,
                    &extension.name,
                ));
            }
        }
    }
    Ok(out)
}

fn image_source(source: &ImageSource) -> Result<AnthropicImageSource, LlmError> {
    match source {
        ImageSource::Base64 { media_type, data } => Ok(AnthropicImageSource {
            source_type: "base64".into(),
            media_type: Some(media_type.clone()),
            data: Some(data.clone()),
            url: None,
        }),
        ImageSource::Url { url } => Ok(AnthropicImageSource {
            source_type: "url".into(),
            media_type: None,
            data: None,
            url: Some(url.clone()),
        }),
    }
}

fn translate_tools(tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|tool| AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect()
}

fn translate_tool_choice(choice: &ToolChoice) -> AnthropicToolChoice {
    match choice {
        ToolChoice::Auto => AnthropicToolChoice {
            choice_type: "auto".into(),
            name: None,
        },
        ToolChoice::None => AnthropicToolChoice {
            choice_type: "none".into(),
            name: None,
        },
        ToolChoice::Required => AnthropicToolChoice {
            choice_type: "any".into(),
            name: None,
        },
        ToolChoice::Named { name } => AnthropicToolChoice {
            choice_type: "tool".into(),
            name: Some(name.clone()),
        },
    }
}

fn map_usage(usage: &AnthropicUsage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens,
        cached_input_tokens: usage.cache_read_input_tokens,
        output_tokens: usage.output_tokens,
        reasoning_tokens: None,
    }
}

async fn consume_stream(
    response: reqwest::Response,
    events: EventSink,
    control: CallControl,
) -> Result<GenerationResult, LlmError> {
    let mut stream = response.bytes_stream().eventsource();
    let mut assembler = OutputAssembler::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = Usage::default();

    loop {
        let next = tokio::select! {
            item = stream.next() => item,
            _ = control.cancelled() => return Err(LlmError::cancelled()),
        };
        let Some(event_result) = next else {
            break;
        };
        let event =
            event_result.map_err(|error| LlmError::transport("stream error").with_source(error))?;
        emit_sse_frame(&events, &control, &event.data).await?;
        let stream_event: StreamEvent = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => {
                return Err(
                    LlmError::invalid_response("malformed Anthropic stream event")
                        .with_source(error),
                );
            }
        };

        match stream_event.event_type.as_str() {
            "message_start" => {
                if let Some(message) = stream_event.message
                    && let Some(parsed) = message.usage
                {
                    usage.merge(map_usage(&parsed));
                    let event = OutputEvent::Usage(usage.clone());
                    assembler.apply(&event);
                    events.emit(event).await?;
                }
            }
            "content_block_start" => {
                if let (Some(index), Some(block)) = (stream_event.index, stream_event.content_block)
                    && block.block_type == "tool_use"
                {
                    let event = OutputEvent::ContentDelta {
                        output_index: 0,
                        delta: ContentDelta::ToolCall {
                            tool_index: index,
                            id: block.id,
                            name: block.name,
                            arguments_delta: String::new(),
                        },
                    };
                    assembler.apply(&event);
                    events.emit(event).await?;
                }
            }
            "content_block_delta" => {
                if let (Some(index), Some(delta)) = (stream_event.index, stream_event.delta) {
                    let event = match delta.delta_type.as_str() {
                        "text_delta" => delta.text.map(|text| OutputEvent::ContentDelta {
                            output_index: index,
                            delta: ContentDelta::Text { text },
                        }),
                        "thinking_delta" => delta.thinking.map(|text| OutputEvent::ContentDelta {
                            output_index: index,
                            delta: ContentDelta::Reasoning { text },
                        }),
                        "input_json_delta" => {
                            delta
                                .partial_json
                                .map(|arguments_delta| OutputEvent::ContentDelta {
                                    output_index: 0,
                                    delta: ContentDelta::ToolCall {
                                        tool_index: index,
                                        id: None,
                                        name: None,
                                        arguments_delta,
                                    },
                                })
                        }
                        _ => None,
                    };
                    if let Some(event) = event {
                        assembler.apply(&event);
                        events.emit(event).await?;
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = stream_event.delta
                    && let Some(reason) = delta.stop_reason
                {
                    stop_reason = StopReason::from_provider(&reason);
                }
                if let Some(parsed) = stream_event.usage {
                    usage.merge(map_usage(&parsed));
                    let event = OutputEvent::Usage(usage.clone());
                    assembler.apply(&event);
                    events.emit(event).await?;
                }
            }
            "error" => {
                return Err(LlmError::backend("provider request failed"));
            }
            _ => {}
        }
    }

    let content = assembler.into_content()?;
    if content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolCall { .. }))
    {
        stop_reason = StopReason::ToolCall;
    }
    Ok(GenerationResult {
        content,
        stop_reason,
        usage,
        backend_state: Vec::new(),
        extensions: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ModelRef;
    use crate::message::Message;
    use crate::request::GenerationRequest;

    fn request() -> GenerationRequest {
        GenerationRequest::new(ModelRef::new("anthropic", "claude-sonnet-4-6").unwrap())
            .with_messages(vec![Message::user("hello")])
            .with_instructions("be brief")
    }

    #[test]
    fn translates_text_and_system() {
        let body = translate_request(&request()).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn translates_base64_image() {
        let mut request = request();
        request.messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Image(crate::message::Image::base64(
                "image/png",
                "AAAA",
            ))],
        }];
        let body = translate_request(&request).unwrap();
        let image = &body["messages"][0]["content"][0];
        assert_eq!(image["type"], "image");
        assert_eq!(image["source"]["type"], "base64");
        assert_eq!(image["source"]["data"], "AAAA");
    }

    #[test]
    fn translates_tools_and_choice() {
        let mut request = request();
        request.tools = vec![ToolDefinition::new(
            "lookup",
            "look up a thing",
            serde_json::json!({"type": "object"}),
        )];
        request.options.tool_choice = ToolChoice::Named {
            name: "lookup".into(),
        };
        let body = translate_request(&request).unwrap();
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "lookup");
    }

    #[test]
    fn debug_redacts_api_key() {
        let backend = Anthropic::new("sk-sentinel-secret-key-do-not-leak");
        let debug = format!("{backend:?}");
        assert!(!debug.contains("sk-sentinel-secret-key-do-not-leak"));
    }
}
