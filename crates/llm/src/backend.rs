//! Public backend trait.

use crate::capability::{ModelCapabilities, ModelInfo};
use crate::error::LlmError;
use crate::event::{CallControl, EventSink};
use crate::id::ModelId;
use crate::request::{GenerationRequest, TokenCount, TokenCountRequest};
use crate::response::GenerationResult;
use async_trait::async_trait;

/// A backend that generates from one family of models.
///
/// The trait is public and unsealed. Optional methods have default
/// implementations so external backends do not break when new operations are
/// added.
#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    /// Return capabilities for `model`.
    ///
    /// Use [`ModelCapabilities::unknown`] when the backend does not know.
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities;

    /// Run a generation call.
    ///
    /// Emit nonterminal events through `events`. Return the final result. The
    /// client creates the terminal stream event from that result.
    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError>;

    /// List models known to the backend.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let _ = self;
        Err(LlmError::unsupported_operation("list_models"))
    }

    /// Count tokens for a request.
    async fn count_tokens(
        &self,
        request: TokenCountRequest,
        control: CallControl,
    ) -> Result<TokenCount, LlmError> {
        let _ = (self, request, control);
        Err(LlmError::unsupported_operation("count_tokens"))
    }
}
