//! One model interface for Rust applications.
//!
//! The crate supports hosted APIs, local servers, in-process models, recorded
//! models, and consumer-defined models. A consumer can add a backend without a
//! change to this crate.
//!
//! The crate provides transport and model-call behavior. It does not provide
//! agent behavior, tool execution, or prompt policy.
//!
//! # Examples
//!
//! ## Register an Anthropic backend and complete a call
//!
//! ```no_run
//! # #[cfg(feature = "anthropic")]
//! # {
//! use llm::{GenerationRequest, LlmClient, Message, ModelRef};
//! use llm::providers::anthropic::Anthropic;
//!
//! # async fn example() -> Result<(), llm::LlmError> {
//! let client = LlmClient::builder()
//!     .backend("anthropic", Anthropic::new("your-api-key"))?
//!     .build()?;
//!
//! let request = GenerationRequest::new(ModelRef::new("anthropic", "claude-sonnet-4-6")?)
//!     .with_messages(vec![Message::user("Hello")]);
//! let result = client.complete(request).await?;
//! let _ = result.text();
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Read a typed generation stream
//!
//! ```
//! use llm::event::{ContentDelta, GenerationEvent, OutputEvent};
//! use llm::testing::{ScriptedBackend, ScriptedResponse};
//! use llm::{GenerationRequest, LlmClient, Message, ModelRef};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), llm::LlmError> {
//! let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("Hello"));
//! let client = LlmClient::builder().backend("lab", backend)?.build()?;
//! let request = GenerationRequest::new(ModelRef::new("lab", "echo")?)
//!     .with_messages(vec![Message::user("Hi")]);
//!
//! let mut generation = client.generate(request)?;
//! while let Some(event) = generation.next_event().await {
//!     if let GenerationEvent::Output(OutputEvent::ContentDelta {
//!         delta: ContentDelta::Text { text },
//!         ..
//!     }) = event?.event
//!     {
//!         let _ = text;
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Implement and register a consumer-defined local backend
//!
//! ```
//! use async_trait::async_trait;
//! use llm::event::{CallControl, ContentDelta, EventSink, OutputEvent};
//! use llm::{
//!     GenerationRequest, GenerationResult, LlmClient, LlmError, Message, ModelBackend,
//!     ModelCapabilities, ModelId, ModelRef, StopReason, Usage,
//! };
//!
//! struct Echo;
//!
//! #[async_trait]
//! impl ModelBackend for Echo {
//!     fn capabilities(&self, _model: &ModelId) -> ModelCapabilities {
//!         ModelCapabilities::unknown()
//!     }
//!
//!     async fn generate(
//!         &self,
//!         request: GenerationRequest,
//!         events: EventSink,
//!         _control: CallControl,
//!     ) -> Result<GenerationResult, LlmError> {
//!         let text = request.messages.last().map(|message| message.text()).unwrap_or_default();
//!         events
//!             .emit(OutputEvent::ContentDelta {
//!                 output_index: 0,
//!                 delta: ContentDelta::Text { text: text.clone() },
//!             })
//!             .await?;
//!         Ok(GenerationResult {
//!             content: vec![llm::ContentBlock::text(text)],
//!             stop_reason: StopReason::EndTurn,
//!             usage: Usage::default(),
//!             backend_state: Vec::new(),
//!             extensions: Vec::new(),
//!         })
//!     }
//! }
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), llm::LlmError> {
//! let client = LlmClient::builder().backend("local", Echo)?.build()?;
//! let request = GenerationRequest::new(ModelRef::new("local", "echo")?)
//!     .with_messages(vec![Message::user("ping")]);
//! let result = client.complete(request).await?;
//! assert_eq!(result.text().as_deref(), Some("ping"));
//! # Ok(())
//! # }
//! ```
//!
//! ## Register and call an embedding backend
//!
//! ```
//! use async_trait::async_trait;
//! use llm::event::CallControl;
//! use llm::{
//!     EmbeddingBackend, EmbeddingCapabilities, EmbeddingClient, EmbeddingRequest,
//!     EmbeddingResult, LlmError, ModelId, ModelRef, Usage,
//! };
//!
//! struct Ones;
//!
//! #[async_trait]
//! impl EmbeddingBackend for Ones {
//!     fn capabilities(&self, _model: &ModelId) -> EmbeddingCapabilities {
//!         EmbeddingCapabilities::unknown()
//!     }
//!
//!     async fn embed(
//!         &self,
//!         request: EmbeddingRequest,
//!         _control: CallControl,
//!     ) -> Result<EmbeddingResult, LlmError> {
//!         Ok(EmbeddingResult {
//!             model: request.model,
//!             vectors: request.inputs.iter().map(|_| vec![1.0, 0.0]).collect(),
//!             dimensions: 2,
//!             usage: Usage::default(),
//!             extensions: Vec::new(),
//!         })
//!     }
//! }
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), llm::LlmError> {
//! let client = EmbeddingClient::builder().backend("lab", Ones)?.build()?;
//! let result = client
//!     .embed(EmbeddingRequest::new(
//!         ModelRef::new("lab", "ones")?,
//!         vec!["a".into(), "b".into()],
//!     ))
//!     .await?;
//! assert_eq!(result.vectors.len(), 2);
//! # Ok(())
//! # }
//! ```
//!
//! ## Record and replay a call
//!
//! ```
//! use llm::testing::{ScriptedBackend, ScriptedResponse};
//! use llm::{GenerationRequest, LlmClient, Message, ModelRef, ReplayBackend};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), llm::LlmError> {
//! let backend = ScriptedBackend::new().enqueue(ScriptedResponse::text("Hello"));
//! let client = LlmClient::builder().backend("lab", backend)?.build()?;
//! let request = GenerationRequest::new(ModelRef::new("lab", "echo")?)
//!     .with_messages(vec![Message::user("Hi")]);
//! let recorded = client.record_call(request.clone()).await?;
//!
//! let replay = LlmClient::builder()
//!     .backend("lab", ReplayBackend::new(recorded))?
//!     .build()?;
//! let result = replay.complete(request).await?;
//! assert_eq!(result.text().as_deref(), Some("Hello"));
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

#[cfg(feature = "http")]
mod assemble;
mod backend;
mod capability;
mod client;
mod embedding;
mod error;
pub mod event;
mod extension;
mod id;
mod message;
mod recording;
mod request;
mod response;
mod serde_util;
pub mod testing;
mod tool;

pub mod providers;

pub use async_trait::async_trait;
pub use backend::ModelBackend;
pub use capability::{EmbeddingCapabilities, ModelCapabilities, ModelInfo, Support};
pub use client::{Generation, LlmClient, LlmClientBuilder};
pub use embedding::{
    EmbeddingBackend, EmbeddingClient, EmbeddingClientBuilder, EmbeddingRequest, EmbeddingResult,
};
pub use error::{ErrorKind, ErrorReport, LlmError};
pub use event::{
    CallControl, CallEvent, ContentDelta, EmitError, EventSink, GenerationEvent, OutputEvent,
    Sensitivity, WireCapture, WireDirection, WireEvent,
};
pub use extension::{Extension, reject_unknown_extensions};
pub use id::{BackendId, CallId, EmptyIdentifierError, ModelId, ModelRef};
pub use message::{
    ContentBlock, Image, ImageSource, Message, ReasoningVisibility, Role, ToolResultBlock,
};
pub use recording::{RecordedCall, RecordedOutcome, ReplayBackend, SCHEMA_VERSION};
pub use request::{
    BackendState, GenerationOptions, GenerationRequest, ReasoningEffort, ReasoningOptions,
    TokenCount, TokenCountRequest,
};
pub use response::{GenerationResult, StopReason, Usage};
pub use tool::{OutputFormat, ToolChoice, ToolDefinition};

#[cfg(feature = "anthropic")]
pub use providers::anthropic::Anthropic;
#[cfg(feature = "chatgpt-subscription")]
pub use providers::chatgpt_subscription::{AccessToken, AccessTokenSource, ChatGptSubscription};
#[cfg(feature = "ollama")]
pub use providers::ollama::Ollama;
#[cfg(feature = "openai-compatible")]
pub use providers::openai_compatible::{OpenAiCompatible, OpenAiCompatibleConfig};
#[cfg(feature = "openrouter")]
pub use providers::openrouter::OpenRouter;
