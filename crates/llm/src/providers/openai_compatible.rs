//! OpenAI-compatible chat-completions adapter.

use crate::assemble::OutputAssembler;
use crate::backend::ModelBackend;
use crate::capability::{EmbeddingCapabilities, ModelCapabilities};
use crate::embedding::{EmbeddingBackend, EmbeddingRequest, EmbeddingResult};
use crate::error::LlmError;
use crate::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use crate::extension::reject_unknown_extensions;
use crate::id::ModelId;
use crate::message::{ContentBlock, Image, ImageSource, Message, Role, ToolResultBlock};
use crate::providers::http_util::{
    emit_http_request, emit_http_response, emit_sse_frame, error_from_response, expose,
    secret_from_string, send,
};
use crate::request::GenerationRequest;
use crate::response::{GenerationResult, StopReason, Usage};
use crate::tool::{OutputFormat, ToolChoice, ToolDefinition};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

/// Configuration for an OpenAI-compatible chat-completions endpoint.
pub struct OpenAiCompatibleConfig {
    /// API base URL, such as `https://api.openai.com/v1`.
    pub base_url: String,
    /// Optional bearer token.
    pub api_key: Option<String>,
    /// Extra headers that are safe to log.
    pub headers: Vec<(String, String)>,
    /// Per-model generation capabilities.
    pub capabilities: BTreeMap<String, ModelCapabilities>,
    /// Capabilities used when a model has no specific entry.
    pub default_capabilities: ModelCapabilities,
    /// Per-model embedding capabilities.
    pub embedding_capabilities: BTreeMap<String, EmbeddingCapabilities>,
    /// Embedding capabilities used when a model has no specific entry.
    pub default_embedding_capabilities: EmbeddingCapabilities,
}

impl OpenAiCompatibleConfig {
    /// Create a config for `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            headers: Vec::new(),
            capabilities: BTreeMap::new(),
            default_capabilities: ModelCapabilities::unknown(),
            embedding_capabilities: BTreeMap::new(),
            default_embedding_capabilities: EmbeddingCapabilities::unknown(),
        }
    }

    /// Set the optional API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Add a safe header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Configurable OpenAI-compatible backend.
///
/// Supports hosted endpoints and local compatible servers. Chat-completions
/// streaming is the primary generation path.
pub struct OpenAiCompatible {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: Option<SecretString>,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) capabilities: BTreeMap<String, ModelCapabilities>,
    pub(crate) default_capabilities: ModelCapabilities,
    pub(crate) embedding_capabilities: BTreeMap<String, EmbeddingCapabilities>,
    pub(crate) default_embedding_capabilities: EmbeddingCapabilities,
    pub(crate) include_usage: bool,
    pub(crate) allowed_extensions: Vec<(String, String)>,
}

impl OpenAiCompatible {
    /// Create an adapter from `config`.
    pub fn new(config: OpenAiCompatibleConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key.map(secret_from_string),
            headers: config.headers,
            capabilities: config.capabilities,
            default_capabilities: config.default_capabilities,
            embedding_capabilities: config.embedding_capabilities,
            default_embedding_capabilities: config.default_embedding_capabilities,
            include_usage: false,
            allowed_extensions: Vec::new(),
        }
    }

    pub(crate) fn with_include_usage(mut self) -> Self {
        self.include_usage = true;
        self
    }

    pub(crate) fn allow_extension(
        mut self,
        namespace: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        self.allowed_extensions
            .push((namespace.into(), name.into()));
        self
    }

    fn apply_headers(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", expose(api_key)));
        }
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        builder
    }

    fn safe_headers(&self) -> Vec<(&str, &str)> {
        self.headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect()
    }

    pub(crate) fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }
}

impl fmt::Debug for OpenAiCompatible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatible")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ModelBackend for OpenAiCompatible {
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
        let allowed: Vec<(&str, &str)> = self
            .allowed_extensions
            .iter()
            .map(|(namespace, name)| (namespace.as_str(), name.as_str()))
            .collect();
        reject_unknown_extensions(&request.extensions, &allowed)?;
        let body = translate_chat_request(&request, self.include_usage)?;
        let url = self.chat_url();
        let started = Instant::now();
        emit_http_request(
            &events,
            &control,
            "POST",
            &url,
            &self.safe_headers(),
            Some(&body),
        )
        .await?;

        let builder = self
            .apply_headers(
                self.http
                    .post(&url)
                    .header("content-type", "application/json"),
            )
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
        consume_chat_stream(response, events, control).await
    }
}

#[async_trait]
impl EmbeddingBackend for OpenAiCompatible {
    fn capabilities(&self, model: &ModelId) -> EmbeddingCapabilities {
        self.embedding_capabilities
            .get(model.as_str())
            .cloned()
            .unwrap_or_else(|| self.default_embedding_capabilities.clone())
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        control: CallControl,
    ) -> Result<EmbeddingResult, LlmError> {
        reject_unknown_extensions(&request.extensions, &[])?;
        let mut body = json!({
            "model": request.model.model.as_str(),
            "input": request.inputs,
        });
        if let Some(dimensions) = request.dimensions {
            body["dimensions"] = json!(dimensions);
        }
        let url = self.embeddings_url();
        let builder = self
            .apply_headers(
                self.http
                    .post(&url)
                    .header("content-type", "application/json"),
            )
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
        let parsed: OpenAiEmbeddingResponse = response.json().await.map_err(|error| {
            LlmError::invalid_response("embedding response is not valid JSON").with_source(error)
        })?;
        let mut data = parsed.data;
        data.sort_by_key(|item| item.index);
        let vectors: Vec<Vec<f32>> = data.into_iter().map(|item| item.embedding).collect();
        let dimensions = vectors
            .first()
            .map(|vector| vector.len() as u32)
            .unwrap_or(0);
        Ok(EmbeddingResult {
            model: request.model,
            vectors,
            dimensions,
            usage: parsed.usage.map_or_else(Usage::default, |usage| Usage {
                input_tokens: usage.prompt_tokens,
                cached_input_tokens: None,
                output_tokens: None,
                reasoning_tokens: None,
            }),
            extensions: Vec::new(),
        })
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<Value>,
}

#[derive(Serialize)]
pub(crate) struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<WireContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum WireContent {
    Text(String),
    Parts(Vec<WireContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub(crate) enum WireContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: WireImageUrl },
}

#[derive(Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct WireToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionCall>,
}

#[derive(Deserialize)]
struct StreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    cost: Option<f64>,
}

#[derive(Deserialize)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
    index: u32,
}

pub(crate) fn translate_chat_request(
    request: &GenerationRequest,
    include_usage: bool,
) -> Result<Value, LlmError> {
    let extra_tools = openrouter_plugin_tools(request)?;
    let mut tools = translate_tools(&request.tools);
    tools.extend(extra_tools);

    let body = ChatRequest {
        model: request.model.model.as_str().to_string(),
        messages: translate_messages(request.instructions.as_deref(), &request.messages)?,
        stream: true,
        max_tokens: request.options.max_output_tokens,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        stop: request.options.stop_sequences.clone(),
        tools: (!tools.is_empty()).then_some(tools),
        tool_choice: if request.tools.is_empty() {
            None
        } else {
            Some(translate_tool_choice(&request.options.tool_choice))
        },
        response_format: match &request.options.output_format {
            OutputFormat::Text => None,
            OutputFormat::JsonSchema {
                name,
                schema,
                strict,
            } => Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": strict,
                    "schema": schema,
                }
            })),
        },
        reasoning_effort: request.options.reasoning.as_ref().and_then(|options| {
            options.effort.map(|effort| match effort {
                crate::request::ReasoningEffort::Low => "low".into(),
                crate::request::ReasoningEffort::Medium => "medium".into(),
                crate::request::ReasoningEffort::High => "high".into(),
            })
        }),
        stream_options: include_usage.then_some(json!({"include_usage": true})),
    };
    serde_json::to_value(body).map_err(|error| {
        LlmError::internal("failed to serialize OpenAI-compatible request").with_source(error)
    })
}

fn openrouter_plugin_tools(request: &GenerationRequest) -> Result<Vec<Value>, LlmError> {
    let wants_web_search = request
        .extensions
        .iter()
        .any(|extension| extension.is("openrouter", "web_search"));
    if !wants_web_search {
        return Ok(Vec::new());
    }
    if request.tools.iter().any(|tool| tool.name == "web_search") {
        return Err(LlmError::invalid_request(
            "openrouter.web_search cannot be combined with a function tool named web_search",
        ));
    }
    Ok(vec![json!({ "type": "openrouter:web_search" })])
}

pub(crate) fn translate_messages(
    system: Option<&str>,
    messages: &[Message],
) -> Result<Vec<WireMessage>, LlmError> {
    let mut result = Vec::new();
    if let Some(system) = system.filter(|value| !value.is_empty()) {
        result.push(WireMessage {
            role: "system".into(),
            content: Some(WireContent::Text(system.to_string())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for message in messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let mut text_parts = Vec::new();
        let mut image_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        let mut result_images = Vec::new();

        for block in &message.content {
            match block {
                ContentBlock::Text { text } => text_parts.push(text.clone()),
                ContentBlock::Image(image) => image_parts.push(image.clone()),
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    tool_calls.push(WireToolCall {
                        id: id.clone(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(arguments)
                                .unwrap_or_else(|_| "{}".into()),
                        },
                    });
                }
                ContentBlock::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    let mut text = String::new();
                    for block in content {
                        match block {
                            ToolResultBlock::Text { text: value } => text.push_str(value),
                            ToolResultBlock::Image(image) => result_images.push(image.clone()),
                            ToolResultBlock::Extension(extension) => {
                                return Err(LlmError::unsupported_extension(
                                    &extension.namespace,
                                    &extension.name,
                                ));
                            }
                        }
                    }
                    tool_results.push((tool_call_id.clone(), text));
                }
                ContentBlock::Reasoning { .. } => {}
                ContentBlock::Extension(extension) => {
                    return Err(LlmError::unsupported_extension(
                        &extension.namespace,
                        &extension.name,
                    ));
                }
            }
        }

        for (tool_call_id, content) in tool_results {
            result.push(WireMessage {
                role: "tool".into(),
                content: Some(WireContent::Text(content)),
                tool_calls: None,
                tool_call_id: Some(tool_call_id),
            });
        }

        if !result_images.is_empty() {
            let mut parts = vec![WireContentPart::Text {
                text: "Images produced by the tool result(s) above:".into(),
            }];
            for image in result_images {
                if let Some(label) = &image.label {
                    parts.push(WireContentPart::Text {
                        text: label.clone(),
                    });
                }
                parts.push(WireContentPart::ImageUrl {
                    image_url: WireImageUrl {
                        url: image_url(&image)?,
                    },
                });
            }
            result.push(WireMessage {
                role: "user".into(),
                content: Some(WireContent::Parts(parts)),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        if !text_parts.is_empty() || !image_parts.is_empty() || !tool_calls.is_empty() {
            let content = if image_parts.is_empty() {
                if text_parts.is_empty() {
                    None
                } else {
                    Some(WireContent::Text(text_parts.join("")))
                }
            } else {
                let mut parts: Vec<WireContentPart> = text_parts
                    .into_iter()
                    .map(|text| WireContentPart::Text { text })
                    .collect();
                for image in image_parts {
                    parts.push(WireContentPart::ImageUrl {
                        image_url: WireImageUrl {
                            url: image_url(&image)?,
                        },
                    });
                }
                Some(WireContent::Parts(parts))
            };
            result.push(WireMessage {
                role: role.into(),
                content,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
            });
        }
    }
    Ok(result)
}

fn image_url(image: &Image) -> Result<String, LlmError> {
    match &image.source {
        ImageSource::Base64 { media_type, data } => Ok(format!("data:{media_type};base64,{data}")),
        ImageSource::Url { url } => Ok(url.clone()),
    }
}

fn translate_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

fn translate_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".into()),
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Required => Value::String("required".into()),
        ToolChoice::Named { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

fn map_usage(usage: &WireUsage) -> Usage {
    Usage {
        input_tokens: usage.prompt_tokens,
        cached_input_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        output_tokens: usage.completion_tokens,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
    }
}

fn cost_extension(usage: &WireUsage) -> Option<crate::extension::Extension> {
    usage
        .cost
        .map(|cost| crate::extension::Extension::new("openrouter", "cost", json!({ "usd": cost })))
}

async fn consume_chat_stream(
    response: reqwest::Response,
    events: EventSink,
    control: CallControl,
) -> Result<GenerationResult, LlmError> {
    let mut stream = response.bytes_stream().eventsource();
    let mut assembler = OutputAssembler::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = Usage::default();
    let mut extensions = Vec::new();

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
        if event.data == "[DONE]" {
            break;
        }
        emit_sse_frame(&events, &control, &event.data).await?;
        let chunk: StreamChunk = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => {
                return Err(
                    LlmError::invalid_response("malformed OpenAI-compatible stream event")
                        .with_source(error),
                );
            }
        };

        if let Some(parsed) = chunk.usage {
            usage.merge(map_usage(&parsed));
            if let Some(extension) = cost_extension(&parsed) {
                extensions.push(extension.clone());
                events.emit(OutputEvent::Extension(extension)).await?;
            }
            let event = OutputEvent::Usage(usage.clone());
            assembler.apply(&event);
            events.emit(event).await?;
        }

        let Some(choice) = chunk.choices.first() else {
            continue;
        };
        if let Some(text) = &choice.delta.content
            && !text.is_empty()
        {
            let event = OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Text { text: text.clone() },
            };
            assembler.apply(&event);
            events.emit(event).await?;
        }
        if let Some(text) = &choice.delta.reasoning
            && !text.is_empty()
        {
            let event = OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Reasoning { text: text.clone() },
            };
            assembler.apply(&event);
            events.emit(event).await?;
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tool_call in tool_calls {
                let event = OutputEvent::ContentDelta {
                    output_index: 0,
                    delta: ContentDelta::ToolCall {
                        tool_index: tool_call.index,
                        id: tool_call.id.clone(),
                        name: tool_call
                            .function
                            .as_ref()
                            .and_then(|function| function.name.clone()),
                        arguments_delta: tool_call
                            .function
                            .as_ref()
                            .and_then(|function| function.arguments.clone())
                            .unwrap_or_default(),
                    },
                };
                assembler.apply(&event);
                events.emit(event).await?;
            }
        }
        if let Some(reason) = &choice.finish_reason {
            stop_reason = StopReason::from_provider(reason);
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
        extensions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ModelRef;
    use crate::message::Message;

    fn request() -> GenerationRequest {
        GenerationRequest::new(ModelRef::new("openai", "gpt-4.1").unwrap())
            .with_messages(vec![Message::user("hi")])
    }

    #[test]
    fn tool_result_images_ride_as_user_image_parts() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: "t1".into(),
                content: vec![
                    ToolResultBlock::Text {
                        text: "{\"ok\":true}".into(),
                    },
                    ToolResultBlock::Image(
                        Image::base64("image/png", "AAAA").with_label("frame at t=0s"),
                    ),
                ],
                is_error: false,
            }],
        }];
        let wire = translate_messages(None, &messages).unwrap();
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0].role, "tool");
        assert_eq!(wire[1].role, "user");
        let json = serde_json::to_string(&wire[1]).unwrap();
        assert!(json.contains("image_url"));
        assert!(json.contains("data:image/png;base64,AAAA"));
        assert!(json.contains("frame at t=0s"));
    }

    #[test]
    fn imageless_tool_results_stay_text_only() {
        let messages = vec![Message::tool_result("t1", "done", false)];
        let wire = translate_messages(None, &messages).unwrap();
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "tool");
    }

    #[test]
    fn rejects_web_search_plugin_with_function_tool() {
        let mut request = request();
        request.tools = vec![ToolDefinition::new(
            "web_search",
            "search",
            json!({"type": "object"}),
        )];
        request.extensions = vec![crate::extension::Extension::new(
            "openrouter",
            "web_search",
            json!(true),
        )];
        let error = translate_chat_request(&request, false).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::InvalidRequest);
    }

    #[test]
    fn attaches_web_search_plugin() {
        let mut request = request();
        request.extensions = vec![crate::extension::Extension::new(
            "openrouter",
            "web_search",
            json!(true),
        )];
        let body = translate_chat_request(&request, true).unwrap();
        assert_eq!(body["tools"][0]["type"], "openrouter:web_search");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn debug_redacts_api_key() {
        let backend = OpenAiCompatible::new(
            OpenAiCompatibleConfig::new("https://example.test/v1")
                .with_api_key("sk-sentinel-secret-key-do-not-leak"),
        );
        let debug = format!("{backend:?}");
        assert!(!debug.contains("sk-sentinel-secret-key-do-not-leak"));
    }
}
