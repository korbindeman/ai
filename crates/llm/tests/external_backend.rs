//! External `ModelBackend` and `EmbeddingBackend` implementations.

use llm::async_trait;
use llm::event::{CallControl, ContentDelta, EventSink, OutputEvent};
use llm::{
    EmbeddingBackend, EmbeddingCapabilities, EmbeddingClient, EmbeddingRequest, EmbeddingResult,
    GenerationRequest, GenerationResult, LlmClient, LlmError, Message, ModelBackend,
    ModelCapabilities, ModelId, ModelRef, StopReason, Usage,
};

struct Echo;

#[async_trait]
impl ModelBackend for Echo {
    fn capabilities(&self, _model: &ModelId) -> ModelCapabilities {
        ModelCapabilities::unknown()
    }

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        _control: CallControl,
    ) -> Result<GenerationResult, LlmError> {
        let text = request
            .messages
            .last()
            .map(|message| message.text())
            .unwrap_or_default();
        events
            .emit(OutputEvent::ContentDelta {
                output_index: 0,
                delta: ContentDelta::Text { text: text.clone() },
            })
            .await?;
        Ok(GenerationResult {
            content: vec![llm::ContentBlock::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            backend_state: Vec::new(),
            extensions: Vec::new(),
        })
    }
}

struct Ones;

#[async_trait]
impl EmbeddingBackend for Ones {
    fn capabilities(&self, _model: &ModelId) -> EmbeddingCapabilities {
        EmbeddingCapabilities::unknown()
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        _control: CallControl,
    ) -> Result<EmbeddingResult, LlmError> {
        Ok(EmbeddingResult {
            model: request.model,
            vectors: request.inputs.iter().map(|_| vec![1.0, 0.0, 0.0]).collect(),
            dimensions: 3,
            usage: Usage::default(),
            extensions: Vec::new(),
        })
    }
}

#[tokio::test]
async fn external_model_backend_can_register_and_stream() {
    let client = LlmClient::builder()
        .backend("lab", Echo)
        .unwrap()
        .build()
        .unwrap();
    let request = GenerationRequest::new(ModelRef::new("lab", "echo").unwrap())
        .with_messages(vec![Message::user("ping")]);
    let streamed = client
        .generate(request.clone())
        .unwrap()
        .finish()
        .await
        .unwrap();
    let completed = client.complete(request).await.unwrap();
    assert_eq!(streamed.text().as_deref(), Some("ping"));
    assert_eq!(completed, streamed);
}

#[tokio::test]
async fn external_embedding_backend_can_register_and_embed() {
    let client = EmbeddingClient::builder()
        .backend("lab", Ones)
        .unwrap()
        .build()
        .unwrap();
    let result = client
        .embed(EmbeddingRequest::new(
            ModelRef::new("lab", "ones").unwrap(),
            vec!["a".into(), "b".into()],
        ))
        .await
        .unwrap();
    assert_eq!(result.vectors.len(), 2);
    assert_eq!(result.dimensions, 3);
}
