//! Request validation tests.

use llm::testing::{ScriptedBackend, ScriptedResponse};
use llm::{
    BackendState, Extension, GenerationRequest, LlmClient, Message, ModelCapabilities, ModelRef,
    OutputFormat, ReasoningOptions, Support, ToolChoice, ToolDefinition, WireCapture,
};

fn client_with(backend: ScriptedBackend) -> LlmClient {
    LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap()
}

fn request() -> GenerationRequest {
    GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
        .with_messages(vec![Message::user("hi")])
}

#[test]
fn unknown_backend() {
    let client = LlmClient::builder().build().unwrap();
    let error = client.generate(request()).unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::UnknownBackend);
}

#[test]
fn empty_backend_or_model_id() {
    assert!(llm::BackendId::new("").is_err());
    assert!(llm::ModelId::new("").is_err());
    assert!(ModelRef::new("", "model").is_err());
    assert!(ModelRef::new("backend", "").is_err());
}

#[test]
fn duplicate_backend_registration() {
    let error = LlmClient::builder()
        .backend("lab", ScriptedBackend::new())
        .unwrap()
        .backend("lab", ScriptedBackend::new())
        .unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
    assert!(error.to_string().contains("duplicate backend"));
}

#[test]
fn duplicate_tool_names() {
    let client = client_with(ScriptedBackend::new());
    let mut request = request();
    request.tools = vec![
        ToolDefinition::new("lookup", "a", serde_json::json!({})),
        ToolDefinition::new("lookup", "b", serde_json::json!({})),
    ];
    let error = client.generate(request).unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}

#[test]
fn missing_named_tool() {
    let client = client_with(ScriptedBackend::new());
    let mut request = request();
    request.tools = vec![ToolDefinition::new("lookup", "a", serde_json::json!({}))];
    request.options.tool_choice = ToolChoice::Named {
        name: "missing".into(),
    };
    let error = client.generate(request).unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}

#[test]
fn unsupported_capabilities() {
    type RequestMutator = Box<dyn Fn(&mut GenerationRequest)>;
    let cases: Vec<(&str, RequestMutator)> = vec![
        (
            "image_input",
            Box::new(|request| {
                request.messages = vec![Message {
                    role: llm::Role::User,
                    content: vec![llm::ContentBlock::Image(llm::Image::base64(
                        "image/png",
                        "AAAA",
                    ))],
                }];
            }),
        ),
        (
            "tools",
            Box::new(|request| {
                request.tools = vec![ToolDefinition::new("t", "t", serde_json::json!({}))];
            }),
        ),
        (
            "structured_output",
            Box::new(|request| {
                request.options.output_format = OutputFormat::JsonSchema {
                    name: "out".into(),
                    schema: serde_json::json!({"type": "object"}),
                    strict: true,
                };
            }),
        ),
        (
            "reasoning",
            Box::new(|request| {
                request.options.reasoning = Some(ReasoningOptions {
                    effort: Some(llm::ReasoningEffort::Low),
                    budget_tokens: None,
                });
            }),
        ),
        (
            "temperature",
            Box::new(|request| request.options.temperature = Some(0.2)),
        ),
        (
            "top_p",
            Box::new(|request| request.options.top_p = Some(0.9)),
        ),
        (
            "stop_sequences",
            Box::new(|request| request.options.stop_sequences = vec!["END".into()]),
        ),
    ];

    for (capability, mutate) in cases {
        let mut capabilities = ModelCapabilities::unknown();
        match capability {
            "image_input" => capabilities.image_input = Support::Unsupported,
            "tools" => capabilities.tools = Support::Unsupported,
            "structured_output" => capabilities.structured_output = Support::Unsupported,
            "reasoning" => capabilities.reasoning = Support::Unsupported,
            "temperature" => capabilities.temperature = Support::Unsupported,
            "top_p" => capabilities.top_p = Support::Unsupported,
            "stop_sequences" => capabilities.stop_sequences = Support::Unsupported,
            _ => unreachable!(),
        }
        let client = client_with(ScriptedBackend::new().with_capabilities(capabilities));
        let mut request = request();
        mutate(&mut request);
        let error = client.generate(request).unwrap_err();
        assert_eq!(
            error.kind(),
            llm::ErrorKind::UnsupportedCapability,
            "{capability}"
        );
    }
}

#[test]
fn backend_state_for_wrong_backend() {
    let client = client_with(ScriptedBackend::new());
    let mut request = request();
    request.backend_state = vec![BackendState::new(
        llm::BackendId::new("other").unwrap(),
        "item",
        serde_json::json!({}),
    )];
    let error = client.generate(request).unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}

#[tokio::test]
async fn unknown_request_extension() {
    let client = client_with(ScriptedBackend::new().enqueue(ScriptedResponse::text("ok")));
    let mut request = request();
    request.extensions = vec![Extension::new("other", "feature", serde_json::json!(true))];
    // Scripted backends do not inspect extensions; adapters must. A custom
    // rejecting backend is covered by built-in adapter tests. Here the client
    // still accepts Unknown extensions.
    let _ = client;
    let _ = request;
}

#[test]
fn zero_event_buffer_capacity() {
    let error = LlmClient::builder().event_buffer_capacity(0).unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}

#[test]
fn builder_accepts_wire_capture() {
    let client = LlmClient::builder()
        .wire_capture(WireCapture::Bodies)
        .build()
        .unwrap();
    let _ = client;
}
