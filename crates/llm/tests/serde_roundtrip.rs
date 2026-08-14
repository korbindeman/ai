//! Serialization round-trips for public semantic types.

use llm::{
    BackendId, BackendState, CallEvent, CallId, ContentBlock, ContentDelta, EmbeddingCapabilities,
    EmbeddingRequest, EmbeddingResult, ErrorKind, ErrorReport, Extension, GenerationEvent,
    GenerationOptions, GenerationRequest, GenerationResult, Image, Message, ModelCapabilities,
    ModelId, ModelInfo, ModelRef, OutputEvent, OutputFormat, ReasoningEffort, ReasoningOptions,
    ReasoningVisibility, RecordedCall, RecordedOutcome, Role, SCHEMA_VERSION, Sensitivity,
    StopReason, Support, TokenCount, TokenCountRequest, ToolChoice, ToolDefinition, Usage,
    WireCapture, WireDirection, WireEvent,
};
use std::time::Duration;

fn round_trip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value).unwrap();
    let restored: T = serde_json::from_str(&json).unwrap();
    assert_eq!(&restored, value, "{json}");
}

#[test]
fn round_trips_public_semantic_types() {
    round_trip(&BackendId::new("anthropic").unwrap());
    round_trip(&ModelId::new("claude").unwrap());
    round_trip(&ModelRef::new("anthropic", "claude").unwrap());
    round_trip(&CallId::new());
    round_trip(&Message::user("hi"));
    round_trip(&ContentBlock::text("hi"));
    round_trip(&ContentBlock::Image(
        Image::base64("image/png", "AAAA").with_label("frame"),
    ));
    round_trip(&ContentBlock::tool_call(
        "c1",
        "lookup",
        serde_json::json!({"q": 1}),
    ));
    round_trip(&ContentBlock::Reasoning {
        text: "think".into(),
        visibility: ReasoningVisibility::Trace,
    });
    round_trip(&ContentBlock::tool_result_text("c1", "ok", false));
    round_trip(&ContentBlock::Extension(Extension::new(
        "ns",
        "x",
        serde_json::json!({"keep": true}),
    )));
    round_trip(&Role::User);
    round_trip(&Support::Unsupported);
    round_trip(&ToolDefinition::new(
        "lookup",
        "look",
        serde_json::json!({"type": "object"}),
    ));
    round_trip(&ToolChoice::Auto);
    round_trip(&ToolChoice::None);
    round_trip(&ToolChoice::Required);
    round_trip(&ToolChoice::Named {
        name: "lookup".into(),
    });
    round_trip(&OutputFormat::Text);
    round_trip(&OutputFormat::JsonSchema {
        name: "out".into(),
        schema: serde_json::json!({"type": "object"}),
        strict: true,
    });
    round_trip(&ReasoningOptions {
        effort: Some(ReasoningEffort::High),
        budget_tokens: Some(128),
    });
    round_trip(&StopReason::ToolCall);
    round_trip(&StopReason::MaxOutputTokens);
    round_trip(&StopReason::ContentFilter);
    round_trip(&StopReason::Other {
        reason: "provider_stop".into(),
    });
    round_trip(&TokenCountRequest {
        model: ModelRef::new("lab", "echo").unwrap(),
        instructions: Some("sys".into()),
        messages: vec![Message::user("hi")],
        tools: Vec::new(),
    });
    round_trip(&TokenCount { input_tokens: 12 });
    round_trip(&ModelInfo {
        id: ModelId::new("echo").unwrap(),
        display_name: "Echo".into(),
        capabilities: ModelCapabilities::unknown(),
        metadata: Default::default(),
    });
    round_trip(&EmbeddingCapabilities::unknown());
    round_trip(&WireCapture::Bodies);
    round_trip(&WireEvent {
        direction: WireDirection::Request,
        kind: "http".into(),
        payload: serde_json::json!({"method": "POST"}),
        sensitivity: Sensitivity::Sensitive,
    });
    round_trip(&CallEvent {
        call_id: CallId::new(),
        sequence: 0,
        elapsed_micros: 1,
        event: GenerationEvent::Output(OutputEvent::Usage(Usage {
            input_tokens: Some(3),
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
        })),
    });
    round_trip(&GenerationOptions {
        timeout: Some(Duration::from_millis(1500)),
        ..GenerationOptions::default()
    });
    round_trip(
        &GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
            .with_messages(vec![Message::user("hi")]),
    );
    round_trip(&GenerationResult {
        content: vec![ContentBlock::text("ok")],
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            input_tokens: Some(3),
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
        },
        backend_state: Vec::new(),
        extensions: Vec::new(),
    });
    round_trip(&OutputEvent::ContentDelta {
        output_index: 0,
        delta: ContentDelta::Text { text: "x".into() },
    });
    round_trip(&ErrorReport {
        kind: ErrorKind::RateLimited,
        message: "rate limited".into(),
        call_id: None,
        backend: None,
        model: None,
        status: Some(429),
        code: None,
        retryable: true,
        retry_after_ms: Some(1000),
    });
    round_trip(&EmbeddingRequest::new(
        ModelRef::new("lab", "ones").unwrap(),
        vec!["a".into()],
    ));
    round_trip(&EmbeddingResult {
        model: ModelRef::new("lab", "ones").unwrap(),
        vectors: vec![vec![1.0, 0.0]],
        dimensions: 2,
        usage: Usage::default(),
        extensions: Vec::new(),
    });
    round_trip(&ModelCapabilities::unknown());
}

#[test]
fn round_trips_unknown_extensions_and_backend_state() {
    let mut request = GenerationRequest::new(ModelRef::new("lab", "echo").unwrap());
    request.extensions.push(Extension::new(
        "future.vendor",
        "feature",
        serde_json::json!({"nested": {"keep": true}, "extra": [1, 2, 3]}),
    ));
    request.backend_state.push(BackendState::new(
        BackendId::new("lab").unwrap(),
        "opaque",
        serde_json::json!({"encrypted_content": "PAYLOAD_EXACT", "version": 2}),
    ));
    let json = serde_json::to_string(&request).unwrap();
    let restored: GenerationRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.extensions[0].payload["nested"]["keep"], true);
    assert_eq!(
        restored.backend_state[0].payload["encrypted_content"],
        "PAYLOAD_EXACT"
    );
    assert_eq!(restored, request);

    let record = RecordedCall {
        schema_version: SCHEMA_VERSION,
        call_id: CallId::new(),
        started_at_unix_ms: 1,
        request,
        capabilities: ModelCapabilities::unknown(),
        events: vec![CallEvent {
            call_id: CallId::new(),
            sequence: 0,
            elapsed_micros: 10,
            event: GenerationEvent::Output(OutputEvent::Extension(Extension::new(
                "future.vendor",
                "delta",
                serde_json::json!({"keep": "me"}),
            ))),
        }],
        outcome: RecordedOutcome::Failed(ErrorReport {
            kind: ErrorKind::Backend,
            message: "backend error".into(),
            call_id: None,
            backend: None,
            model: None,
            status: None,
            code: None,
            retryable: false,
            retry_after_ms: None,
        }),
    };
    round_trip(&record);
}

#[test]
fn backend_state_debug_redacts_payload() {
    let state = BackendState::new(
        BackendId::new("lab").unwrap(),
        "item",
        serde_json::json!({"secret": "VISIBLE_IF_LEAKED"}),
    );
    let debug = format!("{state:?}");
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("VISIBLE_IF_LEAKED"));
}

#[test]
fn role_user_and_assistant_only() {
    assert_eq!(Message::user("a").role, Role::User);
    assert_eq!(Message::assistant("b").role, Role::Assistant);
}
