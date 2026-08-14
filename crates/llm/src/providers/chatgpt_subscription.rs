//! ChatGPT subscription adapter for the OpenAI Responses API.
//!
//! Authentication is supplied through [`AccessTokenSource`]. This adapter does
//! not contain keychain access, configuration UI, or product-specific behavior.
//! Encrypted reasoning and provider items are preserved in [`crate::BackendState`].

use crate::backend::ModelBackend;
use crate::capability::ModelCapabilities;
use crate::error::LlmError;
use crate::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use crate::extension::reject_unknown_extensions;
use crate::id::{BackendId, ModelId};
use crate::message::{ContentBlock, ImageSource, Message, Role, ToolResultBlock};
use crate::providers::http_util::{
    emit_http_request, emit_http_response, emit_sse_frame, error_from_response, expose,
    secret_from_string, send,
};
use crate::request::{BackendState, GenerationRequest};
use crate::response::{GenerationResult, StopReason, Usage};
use crate::tool::{ToolChoice, ToolDefinition};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use secrecy::SecretString;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const RESPONSES_PATH: &str = "/responses";
const STATE_KIND: &str = "responses_item";

/// Short-lived access token for one Responses API request.
pub struct AccessToken {
    token: SecretString,
    /// ChatGPT account ID, when known.
    pub account_id: Option<String>,
}

impl AccessToken {
    /// Create an access token.
    pub fn new(token: impl Into<String>, account_id: Option<String>) -> Self {
        Self {
            token: secret_from_string(token),
            account_id,
        }
    }
}

impl fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessToken")
            .field("token", &self.token)
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// Supplies access tokens for the ChatGPT subscription adapter.
///
/// The source owns refresh policy. The adapter requests a token before each
/// generation and may request a second token after a 401 if no semantic output
/// event has been emitted.
#[async_trait]
pub trait AccessTokenSource: Send + Sync + 'static {
    /// Return a usable access token.
    async fn access_token(&self) -> Result<AccessToken, LlmError>;
}

/// ChatGPT subscription backend using the Responses API.
pub struct ChatGptSubscription {
    http: reqwest::Client,
    tokens: Arc<dyn AccessTokenSource>,
    base_url: String,
    capabilities: ModelCapabilities,
}

impl ChatGptSubscription {
    /// Create an adapter that requests tokens from `tokens`.
    pub fn new(tokens: Arc<dyn AccessTokenSource>) -> Self {
        Self {
            http: reqwest::Client::new(),
            tokens,
            base_url: DEFAULT_BASE_URL.into(),
            capabilities: ModelCapabilities::unknown(),
        }
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Override model capabilities.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

impl fmt::Debug for ChatGptSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatGptSubscription")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ModelBackend for ChatGptSubscription {
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
        reject_unknown_extensions(&request.extensions, &[])?;
        match self.run_once(&request, &events, &control).await? {
            Attempt::Ok(result) => Ok(result),
            Attempt::Unauthorized => {
                if control.is_cancelled() {
                    return Err(LlmError::cancelled());
                }
                match self.run_once(&request, &events, &control).await? {
                    Attempt::Ok(result) => Ok(result),
                    Attempt::Unauthorized => Err(LlmError::authentication("authentication failed")),
                }
            }
        }
    }
}

enum Attempt {
    Ok(GenerationResult),
    Unauthorized,
}

impl ChatGptSubscription {
    async fn run_once(
        &self,
        request: &GenerationRequest,
        events: &EventSink,
        control: &CallControl,
    ) -> Result<Attempt, LlmError> {
        let token = self.tokens.access_token().await?;
        let body = build_body(request)?;
        let url = format!("{}{RESPONSES_PATH}", self.base_url);
        let started = Instant::now();
        emit_http_request(
            events,
            control,
            "POST",
            &url,
            &[
                ("content-type", "application/json"),
                ("openai-beta", "responses=experimental"),
            ],
            Some(&body),
        )
        .await?;

        let mut builder = self
            .http
            .post(&url)
            .header("authorization", format!("Bearer {}", expose(&token.token)))
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("openai-beta", "responses=experimental")
            .json(&body);
        if let Some(account_id) = token.account_id.as_deref()
            && !account_id.is_empty()
        {
            builder = builder.header("chatgpt-account-id", account_id);
        }
        let response = send(builder, control).await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            let _ = response.text().await;
            return Ok(Attempt::Unauthorized);
        }
        if !response.status().is_success() {
            return Err(error_from_response(events, control, started, response).await);
        }
        emit_http_response(
            events,
            control,
            response.status().as_u16(),
            started.elapsed(),
            None,
            None,
        )
        .await?;
        Ok(Attempt::Ok(
            consume_stream(response, events, control, &request.model.backend).await?,
        ))
    }
}

async fn consume_stream(
    response: reqwest::Response,
    events: &EventSink,
    control: &CallControl,
    backend: &BackendId,
) -> Result<GenerationResult, LlmError> {
    let mut stream = response.bytes_stream().eventsource();
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut backend_state = Vec::new();
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
        emit_sse_frame(events, control, &event.data).await?;
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => {
                return Err(
                    LlmError::invalid_response("malformed Responses API stream event")
                        .with_source(error),
                );
            }
        };
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    text.push_str(delta);
                    events
                        .emit(OutputEvent::ContentDelta {
                            output_index: 0,
                            delta: ContentDelta::Text {
                                text: delta.to_string(),
                            },
                        })
                        .await?;
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item") {
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let block = function_call_block(item)?;
                        if let ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } = &block
                        {
                            events
                                .emit(OutputEvent::ContentDelta {
                                    output_index: 0,
                                    delta: ContentDelta::ToolCall {
                                        tool_index: tool_calls.len() as u32,
                                        id: Some(id.clone()),
                                        name: Some(name.clone()),
                                        arguments_delta: serde_json::to_string(arguments)
                                            .unwrap_or_else(|_| "{}".into()),
                                    },
                                })
                                .await?;
                        }
                        tool_calls.push(block);
                    }
                    backend_state.push(BackendState::new(
                        backend.clone(),
                        STATE_KIND,
                        item.clone(),
                    ));
                }
            }
            Some("response.completed") => {
                if let Some(parsed) = value
                    .get("response")
                    .and_then(|response| response.get("usage"))
                {
                    usage = parse_usage(parsed);
                    events.emit(OutputEvent::Usage(usage.clone())).await?;
                }
            }
            Some("error") => {
                return Err(LlmError::backend("provider request failed"));
            }
            _ => {}
        }
    }

    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::text(text));
    }
    let has_tools = !tool_calls.is_empty();
    content.extend(tool_calls);
    Ok(GenerationResult {
        content,
        stop_reason: if has_tools {
            StopReason::ToolCall
        } else {
            StopReason::EndTurn
        },
        usage,
        backend_state,
        extensions: Vec::new(),
    })
}

fn function_call_block(item: &Value) -> Result<ContentBlock, LlmError> {
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let arguments = match item.get("arguments") {
        Some(Value::String(args)) => serde_json::from_str(args).map_err(|error| {
            LlmError::invalid_response("tool-call arguments are not valid JSON").with_source(error)
        })?,
        Some(other) => other.clone(),
        None => Value::Object(Map::new()),
    };
    Ok(ContentBlock::tool_call(id, name, arguments))
}

fn parse_usage(usage: &Value) -> Usage {
    let field = |key: &str| usage.get(key).and_then(Value::as_u64);
    Usage {
        input_tokens: field("input_tokens"),
        cached_input_tokens: field("cached_input_tokens"),
        output_tokens: field("output_tokens"),
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64),
    }
}

fn build_body(request: &GenerationRequest) -> Result<Value, LlmError> {
    let mut body = Map::new();
    body.insert(
        "model".into(),
        json!(strip_provider_prefix(request.model.model.as_str())),
    );
    body.insert("input".into(), json!(to_input_items(request)?));
    body.insert("stream".into(), json!(true));
    body.insert("store".into(), json!(false));
    body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    if let Some(instructions) = request
        .instructions
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        body.insert("instructions".into(), json!(instructions));
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".into(),
            json!(request.tools.iter().map(format_tool).collect::<Vec<_>>()),
        );
        body.insert(
            "tool_choice".into(),
            format_tool_choice(&request.options.tool_choice),
        );
    }
    Ok(Value::Object(body))
}

fn strip_provider_prefix(model: &str) -> &str {
    model.strip_prefix("openai/").unwrap_or(model)
}

fn to_input_items(request: &GenerationRequest) -> Result<Vec<Value>, LlmError> {
    let native_items: Vec<Value> = request
        .backend_state
        .iter()
        .filter(|state| state.kind == STATE_KIND)
        .map(|state| state.payload.clone())
        .collect();
    let native_call_ids: HashSet<String> = native_items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .filter_map(|item| {
            item.get("call_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect();

    let mut items = Vec::new();
    let mut inserted_native = false;
    for message in &request.messages {
        match message.role {
            Role::User => append_user_message(&mut items, message, &native_call_ids)?,
            Role::Assistant => {
                if !inserted_native && !native_items.is_empty() {
                    items.extend(native_items.iter().cloned());
                    inserted_native = true;
                }
                append_assistant_fallback(&mut items, message, &native_items, &native_call_ids)?;
            }
        }
    }
    Ok(items)
}

fn append_user_message(
    items: &mut Vec<Value>,
    message: &Message,
    native_call_ids: &HashSet<String>,
) -> Result<(), LlmError> {
    let mut content_parts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } => {
                content_parts.push(json!({"type": "input_text", "text": text}));
            }
            ContentBlock::Image(image) => {
                content_parts.push(json!({
                    "type": "input_image",
                    "image_url": image_url(image)?,
                }));
            }
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                let text = tool_result_text(content)?;
                if native_call_ids.contains(tool_call_id) {
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": text,
                    }));
                } else {
                    content_parts.push(json!({
                        "type": "input_text",
                        "text": format!("[Historical tool result for {tool_call_id}]\n{text}"),
                    }));
                }
                append_tool_result_images(items, content)?;
            }
            ContentBlock::ToolCall { .. } | ContentBlock::Reasoning { .. } => {}
            ContentBlock::Extension(extension) => {
                return Err(LlmError::unsupported_extension(
                    &extension.namespace,
                    &extension.name,
                ));
            }
        }
    }
    if !content_parts.is_empty() {
        items.push(json!({
            "type": "message",
            "role": "user",
            "content": content_parts,
        }));
    }
    Ok(())
}

fn append_assistant_fallback(
    items: &mut Vec<Value>,
    message: &Message,
    native_items: &[Value],
    native_call_ids: &HashSet<String>,
) -> Result<(), LlmError> {
    let has_native_message = native_items
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("message"));
    let mut text: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    for block in &message.content {
        if let ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = block
            && !native_call_ids.contains(id)
        {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&format!(
                "[Historical call to tool `{name}` with arguments: {arguments}]"
            ));
        }
        if let ContentBlock::Extension(extension) = block {
            return Err(LlmError::unsupported_extension(
                &extension.namespace,
                &extension.name,
            ));
        }
    }
    if !text.is_empty() && !has_native_message {
        items.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }));
    }
    Ok(())
}

fn append_tool_result_images(
    items: &mut Vec<Value>,
    content: &[ToolResultBlock],
) -> Result<(), LlmError> {
    let images: Vec<&crate::message::Image> = content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Image(image) => Some(image),
            _ => None,
        })
        .collect();
    if images.is_empty() {
        return Ok(());
    }
    let mut parts = vec![json!({
        "type": "input_text",
        "text": "Images produced by the tool result(s) above:",
    })];
    for image in images {
        if let Some(label) = &image.label {
            parts.push(json!({"type": "input_text", "text": label}));
        }
        parts.push(json!({
            "type": "input_image",
            "image_url": image_url(image)?,
        }));
    }
    items.push(json!({
        "type": "message",
        "role": "user",
        "content": parts,
    }));
    Ok(())
}

fn tool_result_text(content: &[ToolResultBlock]) -> Result<String, LlmError> {
    let mut text = String::new();
    for block in content {
        match block {
            ToolResultBlock::Text { text: value } => text.push_str(value),
            ToolResultBlock::Image(_) => {}
            ToolResultBlock::Extension(extension) => {
                return Err(LlmError::unsupported_extension(
                    &extension.namespace,
                    &extension.name,
                ));
            }
        }
    }
    Ok(text)
}

fn image_url(image: &crate::message::Image) -> Result<String, LlmError> {
    match &image.source {
        ImageSource::Base64 { media_type, data } => Ok(format!("data:{media_type};base64,{data}")),
        ImageSource::Url { url } => Ok(url.clone()),
    }
}

fn format_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": sanitize_schema(&tool.input_schema),
    })
}

fn format_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named { name } => json!({"type": "function", "name": name}),
    }
}

fn sanitize_schema(node: &Value) -> Value {
    match node {
        Value::Object(map) => {
            let mut out = Map::new();
            let strip_enum = matches!(map.get("enum"), Some(Value::Array(_)))
                && map
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind != "string");
            for (key, value) in map {
                if strip_enum && key == "enum" {
                    continue;
                }
                out.insert(key.clone(), sanitize_schema(value));
            }
            if strip_enum && let Some(Value::Array(values)) = map.get("enum") {
                let hint = format!(
                    "(allowed values: {})",
                    values
                        .iter()
                        .map(value_to_plain_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                match out.get("description").and_then(Value::as_str) {
                    Some(existing) => {
                        out.insert("description".into(), json!(format!("{existing} {hint}")));
                    }
                    None => {
                        out.insert("description".into(), json!(hint));
                    }
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sanitize_schema).collect()),
        other => other.clone(),
    }
}

fn value_to_plain_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ModelRef;
    use crate::message::Message;

    fn request() -> GenerationRequest {
        GenerationRequest::new(ModelRef::new("chatgpt", "openai/gpt-5.6-sol").unwrap())
            .with_messages(vec![Message::user("hi")])
            .with_instructions("be terse")
    }

    #[test]
    fn strips_openai_prefix() {
        assert_eq!(strip_provider_prefix("openai/gpt-5.6-sol"), "gpt-5.6-sol");
        assert_eq!(strip_provider_prefix("gpt-5.6-luna"), "gpt-5.6-luna");
    }

    #[test]
    fn build_body_requests_encrypted_reasoning() {
        let body = build_body(&request()).unwrap();
        assert_eq!(body["model"], json!("gpt-5.6-sol"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["store"], json!(false));
        assert_eq!(body["instructions"], json!("be terse"));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn native_response_items_replay_from_backend_state() {
        let reasoning = json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque"
        });
        let message = json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "checking"}]
        });
        let call = json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "search",
            "arguments": "{\"q\":\"rust\"}"
        });
        let mut request = request();
        request.messages = vec![
            Message::user("question"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::text("checking"),
                    ContentBlock::tool_call("call_1", "search", json!({"q": "rust"})),
                ],
            },
            Message::tool_result("call_1", "results", false),
        ];
        request.backend_state = vec![
            BackendState::new(request.model.backend.clone(), STATE_KIND, reasoning.clone()),
            BackendState::new(request.model.backend.clone(), STATE_KIND, message.clone()),
            BackendState::new(request.model.backend.clone(), STATE_KIND, call.clone()),
        ];
        let items = to_input_items(&request).unwrap();
        assert_eq!(items[0]["role"], json!("user"));
        assert_eq!(items[1], reasoning);
        assert_eq!(items[2], message);
        assert_eq!(items[3], call);
        assert_eq!(items[4]["type"], json!("function_call_output"));
        assert_eq!(items[4]["call_id"], json!("call_1"));
    }

    #[test]
    fn access_token_debug_redacts_secret() {
        let token = AccessToken::new("sk-sentinel-secret-key-do-not-leak", None);
        let debug = format!("{token:?}");
        assert!(!debug.contains("sk-sentinel-secret-key-do-not-leak"));
    }
}
