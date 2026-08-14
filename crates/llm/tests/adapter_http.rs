//! HTTP fixture tests for built-in adapters.

#![cfg(all(
    feature = "anthropic",
    feature = "openai-compatible",
    feature = "openrouter",
    feature = "ollama",
    feature = "chatgpt-subscription"
))]

use llm::async_trait;
use llm::event::{Sensitivity, WireCapture, WireDirection};
use llm::{
    AccessToken, AccessTokenSource, Anthropic, ChatGptSubscription, EmbeddingClient,
    EmbeddingRequest, GenerationOptions, GenerationRequest, LlmClient, LlmError, Message, ModelRef,
    Ollama, OpenAiCompatible, OpenAiCompatibleConfig, OpenRouter,
};
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SENTINEL: &str = "sk-sentinel-secret-key-do-not-leak";

fn sse(events: &[&str]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(event);
        body.push_str("\n\n");
    }
    body
}

fn text_request(backend: &str, model: &str) -> GenerationRequest {
    GenerationRequest::new(ModelRef::new(backend, model).unwrap())
        .with_messages(vec![Message::user("hello")])
}

#[tokio::test]
async fn anthropic_text_stream_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(&[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":0}}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            ]),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let client = LlmClient::builder()
        .backend(
            "anthropic",
            Anthropic::new(SENTINEL).with_base_url(server.uri()),
        )
        .unwrap()
        .wire_capture(WireCapture::Bodies)
        .build()
        .unwrap();
    let mut generation = client
        .generate(text_request("anthropic", "claude-sonnet-4-6"))
        .unwrap();
    let mut saw_sensitive_body = false;
    let mut result = None;
    while let Some(item) = generation.next_event().await {
        let event = item.unwrap();
        let rendered = format!("{event:?}");
        assert!(!rendered.contains(SENTINEL), "{rendered}");
        match event.event {
            llm::GenerationEvent::Output(llm::OutputEvent::Wire(wire)) => {
                if matches!(wire.kind.as_str(), "http" | "sse_frame")
                    && (wire.payload.get("body").is_some() || wire.kind == "sse_frame")
                {
                    assert_eq!(wire.sensitivity, Sensitivity::Sensitive);
                    saw_sensitive_body = true;
                }
                if wire.direction == WireDirection::Request {
                    let headers = wire.payload.get("headers").cloned().unwrap_or_default();
                    let header_json = headers.to_string().to_ascii_lowercase();
                    assert!(!header_json.contains("x-api-key"));
                    assert!(!header_json.contains("authorization"));
                }
            }
            llm::GenerationEvent::Finished(finished) => result = Some(finished),
            _ => {}
        }
    }
    let result = result.expect("finished event");
    assert_eq!(result.text().as_deref(), Some("Hi"));
    assert_eq!(result.usage.input_tokens, Some(3));
    assert!(saw_sensitive_body);
    let debug = format!("{:?}", Anthropic::new(SENTINEL));
    assert!(!debug.contains(SENTINEL));
    assert!(!format!("{result:?}").contains(SENTINEL));
}

#[tokio::test]
async fn anthropic_tool_and_reasoning_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(&[
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"lookup"}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            ]),
            "text/event-stream",
        ))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "anthropic",
            Anthropic::new(SENTINEL).with_base_url(server.uri()),
        )
        .unwrap()
        .build()
        .unwrap();
    let mut request = text_request("anthropic", "claude-sonnet-4-6");
    request.tools = vec![llm::ToolDefinition::new(
        "lookup",
        "look",
        serde_json::json!({"type": "object"}),
    )];
    let result = client.complete(request).await.unwrap();
    assert_eq!(result.tool_calls().len(), 1);
    assert!(matches!(result.stop_reason, llm::StopReason::ToolCall));
}

#[tokio::test]
async fn anthropic_auth_and_rate_limit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error":{"message":"invalid api key"}}"#),
        )
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "anthropic",
            Anthropic::new(SENTINEL).with_base_url(server.uri()),
        )
        .unwrap()
        .build()
        .unwrap();
    let error = client
        .complete(text_request("anthropic", "claude-sonnet-4-6"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Authentication);
    assert!(!error.to_string().contains(SENTINEL));
    assert!(!error.to_string().contains('{'));
}

#[tokio::test]
async fn openai_interleaved_tool_calls_and_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(&[
                r#"{"choices":[{"delta":{"content":"go"}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"a","arguments":"{\"x\":"}}]}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"b","arguments":"{\"y\":1}"}}]}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}"#,
                r#"{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":6,"cost":0.01}}"#,
                "[DONE]",
            ]),
            "text/event-stream",
        ))
        .mount(&server)
        .await;
    let backend =
        OpenAiCompatible::new(OpenAiCompatibleConfig::new(server.uri()).with_api_key(SENTINEL));
    let client = LlmClient::builder()
        .backend("openai", backend)
        .unwrap()
        .wire_capture(WireCapture::Bodies)
        .build()
        .unwrap();
    let result = client
        .complete(text_request("openai", "gpt-4.1"))
        .await
        .unwrap();
    assert_eq!(result.tool_calls().len(), 2);
    assert_eq!(result.usage.input_tokens, Some(4));
    assert!(!format!("{result:?}").contains(SENTINEL));
}

#[tokio::test]
async fn openai_rate_limit_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "2")
                .set_body_string("rate limit"),
        )
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "openai",
            OpenAiCompatible::new(OpenAiCompatibleConfig::new(server.uri()).with_api_key(SENTINEL)),
        )
        .unwrap()
        .build()
        .unwrap();
    let error = client
        .complete(text_request("openai", "gpt-4.1"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::RateLimited);
    assert_eq!(error.report().retry_after_ms, Some(2000));
}

#[tokio::test]
async fn openai_malformed_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(&[r#"{"choices":[{"delta":{"content":"ok"}}]"#, "[DONE]"]),
            "text/event-stream",
        ))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "openai",
            OpenAiCompatible::new(OpenAiCompatibleConfig::new(server.uri()).with_api_key(SENTINEL)),
        )
        .unwrap()
        .build()
        .unwrap();
    let error = client
        .complete(text_request("openai", "gpt-4.1"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidResponse);
}

#[tokio::test]
async fn openai_embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2], "index": 1},
                {"embedding": [0.3, 0.4], "index": 0}
            ],
            "usage": {"prompt_tokens": 2}
        })))
        .mount(&server)
        .await;
    let embeddings = EmbeddingClient::builder()
        .backend(
            "openai",
            OpenAiCompatible::new(OpenAiCompatibleConfig::new(server.uri()).with_api_key(SENTINEL)),
        )
        .unwrap()
        .build()
        .unwrap();
    let result = embeddings
        .embed(EmbeddingRequest::new(
            ModelRef::new("openai", "text-embedding-3-small").unwrap(),
            vec!["a".into(), "b".into()],
        ))
        .await
        .unwrap();
    assert_eq!(result.vectors[0], vec![0.3, 0.4]);
    assert_eq!(result.vectors[1], vec![0.1, 0.2]);
}

#[tokio::test]
async fn openrouter_rejects_duplicate_web_search() {
    let backend = OpenRouter::new(SENTINEL);
    let client = LlmClient::builder()
        .backend("openrouter", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut request = text_request("openrouter", "openai/gpt-4.1");
    request.tools = vec![llm::ToolDefinition::new(
        "web_search",
        "search",
        serde_json::json!({"type": "object"}),
    )];
    request.extensions = vec![llm::Extension::new(
        "openrouter",
        "web_search",
        serde_json::json!(true),
    )];
    let error = client.complete(request).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}

#[tokio::test]
async fn ollama_text_and_embed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "{\"message\":{\"content\":\"hi\"},\"done\":false}\n{\"message\":{\"content\":\"!\"},\"done\":true,\"prompt_eval_count\":2,\"eval_count\":1}\n",
            "application/x-ndjson",
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [[1.0, 2.0], [3.0, 4.0]]
        })))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend("ollama", Ollama::new(server.uri()))
        .unwrap()
        .build()
        .unwrap();
    let result = client
        .complete(text_request("ollama", "llama3"))
        .await
        .unwrap();
    assert_eq!(result.text().as_deref(), Some("hi!"));
    assert_eq!(result.usage.input_tokens, Some(2));

    let embeddings = EmbeddingClient::builder()
        .backend("ollama", Ollama::new(server.uri()))
        .unwrap()
        .build()
        .unwrap();
    let embedded = embeddings
        .embed(EmbeddingRequest::new(
            ModelRef::new("ollama", "nomic").unwrap(),
            vec!["a".into(), "b".into()],
        ))
        .await
        .unwrap();
    assert_eq!(embedded.vectors.len(), 2);
}

#[tokio::test]
async fn chatgpt_preserves_responses_items_in_backend_state() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", format!("Bearer {SENTINEL}")))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(&[
                r#"{"type":"response.output_text.delta","delta":"ok"}"#,
                r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}}"#,
                r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}}"#,
                r#"{"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2}}}"#,
            ]),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    struct Tokens;
    #[async_trait]
    impl AccessTokenSource for Tokens {
        async fn access_token(&self) -> Result<AccessToken, LlmError> {
            Ok(AccessToken::new(SENTINEL, None))
        }
    }

    let client = LlmClient::builder()
        .backend(
            "chatgpt",
            ChatGptSubscription::new(Arc::new(Tokens)).with_base_url(server.uri()),
        )
        .unwrap()
        .build()
        .unwrap();
    let result = client
        .complete(text_request("chatgpt", "openai/gpt-5.6-sol"))
        .await
        .unwrap();
    assert_eq!(result.text().as_deref(), Some("ok"));
    assert_eq!(result.backend_state.len(), 2);
    assert_eq!(result.backend_state[0].kind, "responses_item");
    assert_eq!(
        result.backend_state[0].payload["encrypted_content"],
        "opaque"
    );
    assert_eq!(result.tool_calls().len(), 1);
    let debug = format!("{:?}", AccessToken::new(SENTINEL, None));
    assert!(!debug.contains(SENTINEL));
}

#[tokio::test]
async fn cancellation_during_anthropic_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_raw(sse(&[r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"slow"}}"#]), "text/event-stream"),
        )
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "anthropic",
            Anthropic::new(SENTINEL).with_base_url(server.uri()),
        )
        .unwrap()
        .build()
        .unwrap();
    let mut request = text_request("anthropic", "claude-sonnet-4-6");
    request.options = GenerationOptions {
        timeout: Some(Duration::from_millis(30)),
        ..GenerationOptions::default()
    };
    let error = client.complete(request).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Timeout);
}

#[tokio::test]
async fn unknown_extension_is_rejected_by_anthropic() {
    let client = LlmClient::builder()
        .backend(
            "anthropic",
            Anthropic::new(SENTINEL).with_base_url("http://127.0.0.1:1"),
        )
        .unwrap()
        .build()
        .unwrap();
    let mut request = text_request("anthropic", "claude-sonnet-4-6");
    request.extensions = vec![llm::Extension::new("other", "x", serde_json::json!(true))];
    let error = client.complete(request).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::UnsupportedExtension);
}

#[tokio::test]
async fn context_limit_from_nested_json() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string(
            r#"{"error":{"message":"Provider returned error","metadata":{"raw":"{\"error\":\"This model's maximum context length is 128000 tokens\"}"}}}"#,
        ))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .backend(
            "openai",
            OpenAiCompatible::new(OpenAiCompatibleConfig::new(server.uri()).with_api_key(SENTINEL)),
        )
        .unwrap()
        .build()
        .unwrap();
    let error = client
        .complete(text_request("openai", "gpt-4.1"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::ContextLimit);
    assert_eq!(error.to_string(), "context limit exceeded");
}
