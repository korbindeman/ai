//! Recording and replay tests.

use llm::testing::{ScriptedBackend, ScriptedResponse};
use llm::{GenerationRequest, LlmClient, Message, ModelRef, RecordedOutcome, ReplayBackend};

fn request() -> GenerationRequest {
    GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
        .with_messages(vec![Message::user("Hi")])
}

#[tokio::test]
async fn record_and_replay_preserves_events_and_result() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("Hello"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let mut recorded = client.record_call(request()).await.unwrap();
    assert_eq!(recorded.schema_version, 1);
    match &recorded.outcome {
        RecordedOutcome::Succeeded(result) => {
            assert_eq!(result.text().as_deref(), Some("Hello"));
            let last = recorded.events.last().unwrap();
            assert!(matches!(last.event, llm::GenerationEvent::Finished(_)));
        }
        other => panic!("expected success, got {other:?}"),
    }
    for event in &mut recorded.events {
        event.elapsed_micros = 10_000_000;
    }
    let original_outputs: Vec<_> = recorded
        .events
        .iter()
        .filter_map(|event| match &event.event {
            llm::GenerationEvent::Output(output) => Some(output.clone()),
            llm::GenerationEvent::Finished(_) => None,
        })
        .collect();

    let replay = LlmClient::builder()
        .backend("lab", ReplayBackend::new(recorded.clone()))
        .unwrap()
        .build()
        .unwrap();
    let mut replay_request = request();
    replay_request
        .metadata
        .insert("correlation".into(), "abc".into());
    replay_request.options.timeout = Some(std::time::Duration::from_secs(9));
    let started = std::time::Instant::now();
    let mut replay_generation = replay.generate(replay_request).unwrap();
    let mut replay_outputs = Vec::new();
    let mut replay_result = None;
    while let Some(item) = replay_generation.next_event().await {
        match item.unwrap().event {
            llm::GenerationEvent::Output(output) => replay_outputs.push(output),
            llm::GenerationEvent::Finished(result) => replay_result = Some(result),
        }
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "replay waited for recorded elapsed time"
    );
    assert_eq!(replay_outputs, original_outputs);
    assert_eq!(
        replay_result.expect("replay result").text().as_deref(),
        Some("Hello")
    );
}

#[tokio::test]
async fn strict_replay_rejects_different_semantic_request() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("Hello"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let recorded = client.record_call(request()).await.unwrap();
    let replay = LlmClient::builder()
        .backend("lab", ReplayBackend::new(recorded))
        .unwrap()
        .build()
        .unwrap();
    let different = GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
        .with_messages(vec![Message::user("other")]);
    let error = replay.complete(different).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
    assert!(error.to_string().contains("mismatch"));
}

#[tokio::test]
async fn failed_backend_still_records() {
    let backend = ScriptedBackend::new().enqueue(ScriptedResponse::Failure {
        events: vec![],
        error: llm::LlmError::backend("nope"),
    });
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let recorded = client.record_call(request()).await.unwrap();
    match recorded.outcome {
        RecordedOutcome::Failed(report) => {
            assert_eq!(report.kind, llm::ErrorKind::Backend);
        }
        RecordedOutcome::Succeeded(_) => panic!("expected failed outcome"),
    }
}

#[tokio::test]
async fn scripted_backend_rejects_out_of_order_calls() {
    let first = request();
    let second = GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
        .with_messages(vec![Message::user("second")]);
    let backend = ScriptedBackend::new()
        .expect(first.clone(), ScriptedResponse::text("one"))
        .expect(second.clone(), ScriptedResponse::text("two"));
    let client = LlmClient::builder()
        .backend("lab", backend)
        .unwrap()
        .build()
        .unwrap();
    let error = client.complete(second).await.unwrap_err();
    assert_eq!(error.kind(), llm::ErrorKind::InvalidRequest);
}
