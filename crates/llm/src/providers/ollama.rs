//! Ollama generation and embedding adapter.

use crate::assemble::OutputAssembler;
use crate::backend::ModelBackend;
use crate::capability::{EmbeddingCapabilities, ModelCapabilities, ModelInfo};
use crate::embedding::{EmbeddingBackend, EmbeddingRequest, EmbeddingResult};
use crate::error::LlmError;
use crate::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use crate::extension::reject_unknown_extensions;
use crate::id::ModelId;
use crate::message::{ContentBlock, ImageSource, Message, Role, ToolResultBlock};
use crate::providers::http_util::{
    emit_http_request, emit_http_response, emit_sse_frame, error_from_response, send,
};
use crate::request::GenerationRequest;
use crate::response::{GenerationResult, StopReason, Usage};
use crate::tool::{OutputFormat, ToolChoice, ToolDefinition};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::time::Instant;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Ollama local-server backend.
///
/// The adapter does not assume that every model supports the same context size
/// or feature set.
pub struct Ollama {
    http: reqwest::Client,
    base_url: String,
    capabilities: ModelCapabilities,
    embedding_capabilities: EmbeddingCapabilities,
}

impl Ollama {
    /// Create an adapter for `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            capabilities: ModelCapabilities::unknown(),
            embedding_capabilities: EmbeddingCapabilities::unknown(),
        }
    }

    /// Create an adapter for `http://localhost:11434`.
    pub fn localhost() -> Self {
        Self::new(DEFAULT_BASE_URL)
    }

    /// Override generation capabilities.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Override embedding capabilities.
    pub fn with_embedding_capabilities(mut self, capabilities: EmbeddingCapabilities) -> Self {
        self.embedding_capabilities = capabilities;
        self
    }
}

impl fmt::Debug for Ollama {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ollama")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ModelBackend for Ollama {
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
        if matches!(
            request.options.output_format,
            OutputFormat::JsonSchema { .. }
        ) {
            return Err(LlmError::unsupported_capability("structured_output"));
        }
        let body = translate_request(&request)?;
        let url = format!("{}/api/chat", self.base_url);
        let started = Instant::now();
        emit_http_request(
            &events,
            &control,
            "POST",
            &url,
            &[("content-type", "application/json")],
            Some(&body),
        )
        .await?;
        let builder = self
            .http
            .post(&url)
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
        consume_ndjson(response, events, control).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<TagModel>,
        }
        #[derive(Deserialize)]
        struct TagModel {
            name: String,
        }

        let url = format!("{}/api/tags", self.base_url);
        let cancel = tokio_util::sync::CancellationToken::new();
        let control = CallControl::new(
            crate::id::CallId::new(),
            cancel,
            None,
            crate::event::WireCapture::Off,
        );
        let response = send(self.http.get(&url), &control).await?;
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
        let tags: TagsResponse = response.json().await.map_err(|error| {
            LlmError::invalid_response("model list is not valid JSON").with_source(error)
        })?;
        tags.models
            .into_iter()
            .map(|model| {
                Ok(ModelInfo {
                    id: ModelId::new(model.name.clone())?,
                    display_name: model.name,
                    capabilities: ModelCapabilities::unknown(),
                    metadata: Default::default(),
                })
            })
            .collect()
    }
}

#[async_trait]
impl EmbeddingBackend for Ollama {
    fn capabilities(&self, model: &ModelId) -> EmbeddingCapabilities {
        let _ = model;
        self.embedding_capabilities.clone()
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        control: CallControl,
    ) -> Result<EmbeddingResult, LlmError> {
        reject_unknown_extensions(&request.extensions, &[])?;
        let body = json!({
            "model": request.model.model.as_str(),
            "input": request.inputs,
        });
        let url = format!("{}/api/embed", self.base_url);
        let builder = self
            .http
            .post(&url)
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
        let parsed: OllamaEmbedResponse = response.json().await.map_err(|error| {
            LlmError::invalid_response("embedding response is not valid JSON").with_source(error)
        })?;
        let dimensions = parsed
            .embeddings
            .first()
            .map(|vector| vector.len() as u32)
            .unwrap_or(0);
        Ok(EmbeddingResult {
            model: request.model,
            vectors: parsed.embeddings,
            dimensions,
            usage: Usage::default(),
            extensions: Vec::new(),
        })
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaToolDef,
}

#[derive(Serialize)]
struct OllamaToolDef {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    message: OllamaResponseMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

#[derive(Deserialize, Default)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

fn translate_request(request: &GenerationRequest) -> Result<Value, LlmError> {
    let tools = if request.tools.is_empty() {
        None
    } else {
        let mut tools = translate_tools(&request.tools);
        if let ToolChoice::Named { name } = &request.options.tool_choice {
            tools.retain(|tool| tool.function.name == *name);
        }
        Some(tools)
    };
    let body = OllamaRequest {
        model: request.model.model.as_str().to_string(),
        messages: translate_messages(request.instructions.as_deref(), &request.messages)?,
        stream: true,
        options: Some(OllamaOptions {
            num_predict: request.options.max_output_tokens,
            temperature: request.options.temperature,
            top_p: request.options.top_p,
            stop: (!request.options.stop_sequences.is_empty())
                .then(|| request.options.stop_sequences.clone()),
        }),
        tools,
    };
    serde_json::to_value(body).map_err(|error| {
        LlmError::internal("failed to serialize Ollama request").with_source(error)
    })
}

fn translate_messages(
    system: Option<&str>,
    messages: &[Message],
) -> Result<Vec<OllamaMessage>, LlmError> {
    let mut result = Vec::new();
    if let Some(system) = system.filter(|value| !value.is_empty()) {
        result.push(OllamaMessage {
            role: "system".into(),
            content: system.to_string(),
            images: None,
            tool_calls: None,
        });
    }
    for message in messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let mut text_parts = Vec::new();
        let mut images = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();

        for block in &message.content {
            match block {
                ContentBlock::Text { text } => text_parts.push(text.clone()),
                ContentBlock::Image(image) => match &image.source {
                    ImageSource::Base64 { data, .. } => images.push(data.clone()),
                    ImageSource::Url { .. } => {
                        return Err(LlmError::unsupported_capability("image_url"));
                    }
                },
                ContentBlock::ToolCall {
                    name, arguments, ..
                } => {
                    tool_calls.push(OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    });
                }
                ContentBlock::ToolResult { content, .. } => {
                    let mut text = String::new();
                    for block in content {
                        match block {
                            ToolResultBlock::Text { text: value } => text.push_str(value),
                            ToolResultBlock::Image(_) => {
                                return Err(LlmError::unsupported_capability("tool_result_image"));
                            }
                            ToolResultBlock::Extension(extension) => {
                                return Err(LlmError::unsupported_extension(
                                    &extension.namespace,
                                    &extension.name,
                                ));
                            }
                        }
                    }
                    tool_results.push(text);
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

        for content in tool_results {
            result.push(OllamaMessage {
                role: "tool".into(),
                content,
                images: None,
                tool_calls: None,
            });
        }

        let content = text_parts.join("");
        if !content.is_empty() || !tool_calls.is_empty() || !images.is_empty() {
            result.push(OllamaMessage {
                role: role.into(),
                content,
                images: (!images.is_empty()).then_some(images),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            });
        }
    }
    Ok(result)
}

fn translate_tools(tools: &[ToolDefinition]) -> Vec<OllamaTool> {
    tools
        .iter()
        .map(|tool| OllamaTool {
            tool_type: "function".into(),
            function: OllamaToolDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        })
        .collect()
}

async fn consume_ndjson(
    response: reqwest::Response,
    events: EventSink,
    control: CallControl,
) -> Result<GenerationResult, LlmError> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut assembler = OutputAssembler::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = Usage::default();
    let mut tool_index = 0;

    loop {
        let next = tokio::select! {
            item = stream.next() => item,
            _ = control.cancelled() => return Err(LlmError::cancelled()),
        };
        let Some(chunk) = next else {
            break;
        };
        let chunk =
            chunk.map_err(|error| LlmError::transport("stream error").with_source(error))?;
        buffer.extend_from_slice(&chunk);
        while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer[..pos].to_vec();
            buffer.drain(..=pos);
            if line.is_empty() {
                continue;
            }
            emit_sse_frame(&events, &control, &String::from_utf8_lossy(&line)).await?;
            apply_line(
                &line,
                &events,
                &mut assembler,
                &mut stop_reason,
                &mut usage,
                &mut tool_index,
            )
            .await?;
        }
    }
    if !buffer.is_empty() {
        apply_line(
            &buffer,
            &events,
            &mut assembler,
            &mut stop_reason,
            &mut usage,
            &mut tool_index,
        )
        .await?;
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

async fn apply_line(
    line: &[u8],
    events: &EventSink,
    assembler: &mut OutputAssembler,
    stop_reason: &mut StopReason,
    usage: &mut Usage,
    tool_index: &mut u32,
) -> Result<(), LlmError> {
    let event: OllamaResponse = serde_json::from_slice(line).map_err(|error| {
        LlmError::invalid_response("malformed Ollama stream event").with_source(error)
    })?;
    if !event.message.content.is_empty() {
        let output = OutputEvent::ContentDelta {
            output_index: 0,
            delta: ContentDelta::Text {
                text: event.message.content,
            },
        };
        assembler.apply(&output);
        events.emit(output).await?;
    }
    if let Some(tool_calls) = event.message.tool_calls {
        for tool_call in tool_calls {
            let arguments = serde_json::to_string(&tool_call.function.arguments)
                .unwrap_or_else(|_| "{}".into());
            let output = OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::ToolCall {
                    tool_index: *tool_index,
                    id: Some(format!("call_{tool_index}")),
                    name: Some(tool_call.function.name),
                    arguments_delta: arguments,
                },
            };
            *tool_index += 1;
            assembler.apply(&output);
            events.emit(output).await?;
        }
    }
    if event.done {
        *usage = Usage {
            input_tokens: event.prompt_eval_count,
            cached_input_tokens: None,
            output_tokens: event.eval_count,
            reasoning_tokens: None,
        };
        let output = OutputEvent::Usage(usage.clone());
        assembler.apply(&output);
        events.emit(output).await?;
        *stop_reason = StopReason::EndTurn;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ModelRef;
    use crate::message::Message;
    use crate::request::GenerationRequest;

    #[test]
    fn translates_system_and_text() {
        let request = GenerationRequest::new(ModelRef::new("ollama", "llama3").unwrap())
            .with_instructions("be brief")
            .with_messages(vec![Message::user("hi")]);
        let body = translate_request(&request).unwrap();
        assert_eq!(body["model"], "llama3");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(body.get("options").unwrap().get("num_predict").is_none());
    }

    #[test]
    fn tool_results_become_tool_messages() {
        let messages = vec![Message::tool_result("t1", "done", false)];
        let wire = translate_messages(None, &messages).unwrap();
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "tool");
        assert_eq!(wire[0].content, "done");
    }
}
