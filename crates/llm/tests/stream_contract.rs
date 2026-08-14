//! Stream contract tests.

use futures::StreamExt;
use llm::async_trait;
use llm::event::{ContentDelta, GenerationEvent, OutputEvent};
use llm::testing::{ScriptedBackend, ScriptedResponse};
use llm::{
    GenerationRequest, GenerationResult, LlmClient, LlmError, Message, ModelCapabilities, ModelRef,
    StopReason, Support, Usage,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::oneshot;

fn model() -> ModelRef {
    ModelRef::new("lab", "echo").unwrap()
}

fn request() -> GenerationRequest {
    GenerationRequest::new(model()).with_messages(vec![Message::user("hi")])
}

fn text_result(text: &str) -> GenerationResult {
    GenerationResult {
        content: vec![llm::ContentBlock::text(text)],
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
        backend_state: Vec::new(),
        extensions: Vec::new(),
    }
}

struct Hang {
    started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
    running: Arc<AtomicBool>,
}

#[async_trait]
impl llm::ModelBackend for Hang {
    fn capabilities(&self, _: &llm::ModelId) -> ModelCapabilities {
        ModelCapabilities::unknown()
    }

    async fn generate(
        &self,
        _: GenerationRequest,
        _: llm::EventSink,
        control: llm::CallControl,
    ) -> Result<GenerationResult, LlmError> {
        struct Guard(Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = Guard(Arc::clone(&self.running));
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }
        control.cancelled().await;
        Ok(GenerationResult::default())
    }
}

#[tokio::test]
async fn successful_text_stream_equals_final_result() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("Hello"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut generation = client.generate(request()).unwrap();
    let mut outputs = Vec::new();
    let mut finished = None;
    while let Some(item) = generation.next().await {
        match item.unwrap().event {
            GenerationEvent::Output(OutputEvent::ContentDelta {
                delta: ContentDelta::Text { text },
                ..
            }) => outputs.push(text),
            GenerationEvent::Finished(result) => finished = Some(result),
            _ => {}
        }
    }
    let finished = finished.expect("finished event");
    assert_eq!(outputs.concat(), "Hello");
    assert_eq!(finished.text().as_deref(), Some("Hello"));
    assert_eq!(finished.content, text_result("Hello").content);
}

#[tokio::test]
async fn complete_matches_generate_finish() {
    let backend = ScriptedBackend::new()
        .enqueue(ScriptedResponse::text("Hello"))
        .enqueue(ScriptedResponse::text("Hello"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let streamed = client.generate(request()).unwrap().finish().await.unwrap();
    let completed = client.complete(request()).await.unwrap();
    assert_eq!(streamed, completed);
}

#[tokio::test]
async fn interleaved_text_and_tool_call_deltas() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Success {
        events: vec![
            OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Text {
                    text: "calling".into(),
                },
            },
            OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::ToolCall {
                    tool_index: 0,
                    id: Some("c1".into()),
                    name: Some("lookup".into()),
                    arguments_delta: "{\"q\":".into(),
                },
            },
            OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::ToolCall {
                    tool_index: 1,
                    id: Some("c2".into()),
                    name: Some("other".into()),
                    arguments_delta: "{\"n\":1}".into(),
                },
            },
            OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::ToolCall {
                    tool_index: 0,
                    id: None,
                    name: None,
                    arguments_delta: "\"x\"}".into(),
                },
            },
        ],
        result: GenerationResult {
            content: vec![
                llm::ContentBlock::text("calling"),
                llm::ContentBlock::tool_call("c1", "lookup", serde_json::json!({"q": "x"})),
                llm::ContentBlock::tool_call("c2", "other", serde_json::json!({"n": 1})),
            ],
            stop_reason: StopReason::ToolCall,
            usage: Usage::default(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
        },
    });
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let result = client.complete(request()).await.unwrap();
    assert_eq!(result.stop_reason, StopReason::ToolCall);
    assert_eq!(result.tool_calls().len(), 2);
}

#[tokio::test]
async fn usage_update_before_final_result() {
    let usage = Usage {
        input_tokens: Some(3),
        cached_input_tokens: None,
        output_tokens: Some(2),
        reasoning_tokens: None,
    };
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Success {
        events: vec![
            OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Text { text: "ok".into() },
            },
            OutputEvent::Usage(usage.clone()),
        ],
        result: GenerationResult {
            content: vec![llm::ContentBlock::text("ok")],
            stop_reason: StopReason::EndTurn,
            usage: usage.clone(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
        },
    });
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut generation = client.generate(request()).unwrap();
    let mut saw_usage = false;
    let mut finished = None;
    while let Some(item) = generation.next().await {
        match item.unwrap().event {
            GenerationEvent::Output(OutputEvent::Usage(_)) => saw_usage = true,
            GenerationEvent::Finished(result) => finished = Some(result),
            _ => {}
        }
    }
    assert!(saw_usage);
    assert_eq!(finished.unwrap().usage.input_tokens, Some(3));
}

#[tokio::test]
async fn backend_error_before_first_event() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Failure {
        events: vec![],
        error: LlmError::backend("boom"),
    });
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let error = client.complete(request()).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Backend);
}

#[tokio::test]
async fn backend_error_after_output_events() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Failure {
        events: vec![OutputEvent::ContentDelta {
            output_index: 0,
            delta: ContentDelta::Text {
                text: "partial".into(),
            },
        }],
        error: LlmError::backend("later"),
    });
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut generation = client.generate(request()).unwrap();
    let mut saw_output = false;
    let mut error = None;
    while let Some(item) = generation.next().await {
        match item {
            Ok(event) => {
                if matches!(event.event, GenerationEvent::Output(_)) {
                    saw_output = true;
                }
            }
            Err(err) => error = Some(err),
        }
    }
    assert!(saw_output);
    assert_eq!(error.unwrap().kind(), llm::ErrorKind::Backend);
}

#[tokio::test]
async fn consumer_cancellation() {
    let running = Arc::new(AtomicBool::new(true));
    let (started_tx, started_rx) = oneshot::channel();
    let client = LlmClient::builder()
        .backend(
            "lab",
            Hang {
                started: std::sync::Mutex::new(Some(started_tx)),
                running: Arc::clone(&running),
            },
        )
        .unwrap()
        .build()
        .unwrap();
    let generation = client.generate(request()).unwrap();
    started_rx.await.unwrap();
    generation.cancel();
    let error = generation.finish().await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Cancelled);
    tokio::time::timeout(Duration::from_secs(1), async {
        while running.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("backend task still active after cancel");
}

#[tokio::test]
async fn deadline_expiration() {
    let running = Arc::new(AtomicBool::new(true));
    let (started_tx, started_rx) = oneshot::channel();
    let client = LlmClient::builder()
        .backend(
            "lab",
            Hang {
                started: std::sync::Mutex::new(Some(started_tx)),
                running: Arc::clone(&running),
            },
        )
        .unwrap()
        .build()
        .unwrap();
    let mut request = request();
    request.options.timeout = Some(Duration::from_millis(20));
    let generation = client.generate(request).unwrap();
    started_rx.await.unwrap();
    let error = generation.finish().await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Timeout);
    tokio::time::timeout(Duration::from_secs(1), async {
        while running.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("backend task still active after deadline");
}

#[tokio::test]
async fn generation_drop_aborts_runner() {
    let running = Arc::new(AtomicBool::new(true));
    let (started_tx, started_rx) = oneshot::channel();
    let client = LlmClient::builder()
        .backend(
            "lab",
            Hang {
                started: std::sync::Mutex::new(Some(started_tx)),
                running: Arc::clone(&running),
            },
        )
        .unwrap()
        .build()
        .unwrap();
    let generation = client.generate(request()).unwrap();
    started_rx.await.unwrap();
    drop(generation);
    tokio::time::timeout(Duration::from_secs(1), async {
        while running.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("backend task still active after drop");
}

#[tokio::test]
async fn full_bounded_channel_applies_backpressure() {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let (continue_tx, continue_rx) = oneshot::channel::<()>();
    struct Backpressure {
        started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        continue_rx: std::sync::Mutex<Option<oneshot::Receiver<()>>>,
    }
    #[async_trait]
    impl llm::ModelBackend for Backpressure {
        fn capabilities(&self, _: &llm::ModelId) -> ModelCapabilities {
            ModelCapabilities::unknown()
        }
        async fn generate(
            &self,
            _: GenerationRequest,
            events: llm::EventSink,
            _: llm::CallControl,
        ) -> Result<GenerationResult, LlmError> {
            events
                .emit(OutputEvent::ContentDelta {
                    output_index: 0,
                    delta: ContentDelta::Text { text: "one".into() },
                })
                .await?;
            if let Some(tx) = self.started.lock().unwrap().take() {
                let _ = tx.send(());
            }
            let continue_rx = self.continue_rx.lock().unwrap().take();
            if let Some(rx) = continue_rx {
                let _ = rx.await;
            }
            events
                .emit(OutputEvent::ContentDelta {
                    output_index: 0,
                    delta: ContentDelta::Text { text: "two".into() },
                })
                .await?;
            Ok(text_result("onetwo"))
        }
    }
    let client = LlmClient::builder()
        .backend(
            "lab",
            Backpressure {
                started: std::sync::Mutex::new(Some(started_tx)),
                continue_rx: std::sync::Mutex::new(Some(continue_rx)),
            },
        )
        .unwrap()
        .event_buffer_capacity(1)
        .unwrap()
        .build()
        .unwrap();
    let mut generation = client.generate(request()).unwrap();
    started_rx.await.unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let first = generation.next().await.unwrap().unwrap();
    assert!(matches!(
        first.event,
        GenerationEvent::Output(OutputEvent::ContentDelta { .. })
    ));
    let _ = continue_tx.send(());
    let result = generation.finish().await.unwrap();
    assert_eq!(result.text().as_deref(), Some("onetwo"));
}

#[tokio::test]
async fn closed_event_receiver_cancels_emit() {
    struct EmitThenHang;
    #[async_trait]
    impl llm::ModelBackend for EmitThenHang {
        fn capabilities(&self, _: &llm::ModelId) -> ModelCapabilities {
            ModelCapabilities::unknown()
        }
        async fn generate(
            &self,
            _: GenerationRequest,
            events: llm::EventSink,
            control: llm::CallControl,
        ) -> Result<GenerationResult, LlmError> {
            let result = events
                .emit(OutputEvent::ContentDelta {
                    output_index: 0,
                    delta: ContentDelta::Text { text: "x".into() },
                })
                .await;
            assert!(result.is_err());
            control.cancelled().await;
            Err(LlmError::cancelled())
        }
    }
    let client = LlmClient::builder()
        .backend("lab", EmitThenHang)
        .unwrap()
        .event_buffer_capacity(1)
        .unwrap()
        .build()
        .unwrap();
    drop(client.generate(request()).unwrap());
    tokio::time::sleep(Duration::from_millis(50)).await;
}

#[tokio::test]
async fn backend_panic_becomes_internal_error() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Panic("boom"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let error = client.complete(request()).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::Internal);
}

#[tokio::test]
async fn exactly_one_terminal_event_and_nothing_after() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("ok"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut generation = client.generate(request()).unwrap();
    let mut terminals = 0;
    let mut sequences = Vec::new();
    while let Some(item) = generation.next().await {
        let event = item.unwrap();
        sequences.push(event.sequence);
        if matches!(event.event, GenerationEvent::Finished(_)) {
            terminals += 1;
        }
    }
    assert_eq!(terminals, 1);
    assert!(generation.next().await.is_none());
    assert_eq!(sequences.first().copied(), Some(0));
}

#[tokio::test]
async fn unknown_capability_is_allowed() {
    let mut capabilities = ModelCapabilities::unknown();
    capabilities.tools = Support::Unknown;
    let backend = ScriptedBackend::new()
        .with_capabilities(capabilities)
        .enqueue(ScriptedResponse::text("ok"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut request = request();
    request.tools = vec![llm::ToolDefinition::new(
        "lookup",
        "look",
        serde_json::json!({"type": "object"}),
    )];
    client.complete(request).await.unwrap();
}
