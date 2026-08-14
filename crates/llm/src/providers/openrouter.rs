//! OpenRouter adapter built on the OpenAI-compatible backend.

use crate::backend::ModelBackend;
use crate::capability::{ModelCapabilities, ModelInfo, Support};
use crate::error::LlmError;
use crate::event::{CallControl, EventSink};
use crate::id::ModelId;
use crate::providers::http_util::{expose, send};
use crate::providers::openai_compatible::{OpenAiCompatible, OpenAiCompatibleConfig};
use crate::request::{GenerationRequest, TokenCount, TokenCountRequest};
use crate::response::GenerationResult;
use async_trait::async_trait;
use serde::Deserialize;
use std::fmt;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter chat-completions backend.
///
/// Wraps [`OpenAiCompatible`] and adds OpenRouter headers, usage fields, model
/// discovery, and the `openrouter.web_search` request extension.
pub struct OpenRouter {
    inner: OpenAiCompatible,
}

impl OpenRouter {
    /// Create an adapter with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAiCompatible::new(
            OpenAiCompatibleConfig::new(DEFAULT_BASE_URL)
                .with_api_key(api_key)
                .with_header("HTTP-Referer", "https://github.com")
                .with_header("X-Title", "llm"),
        )
        .with_include_usage()
        .allow_extension("openrouter", "web_search");
        Self { inner }
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
}

impl fmt::Debug for OpenRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenRouter")
            .field("inner", &self.inner)
            .finish()
    }
}

#[async_trait]
impl ModelBackend for OpenRouter {
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities {
        self.inner.capabilities(model)
    }

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError> {
        self.inner.generate(request, events, control).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelEntry>,
        }
        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
            name: String,
            #[serde(default)]
            context_length: Option<u64>,
        }

        let url = format!("{}/models", self.inner.base_url);
        let mut builder = self.inner.http.get(&url);
        if let Some(api_key) = &self.inner.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", expose(api_key)));
        }
        let cancel = tokio_util::sync::CancellationToken::new();
        let control = CallControl::new(
            crate::id::CallId::new(),
            cancel,
            None,
            crate::event::WireCapture::Off,
        );
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
        let models: ModelsResponse = response.json().await.map_err(|error| {
            LlmError::invalid_response("model list is not valid JSON").with_source(error)
        })?;
        models
            .data
            .into_iter()
            .map(|entry| {
                Ok(ModelInfo {
                    id: ModelId::new(entry.id)?,
                    display_name: entry.name,
                    capabilities: ModelCapabilities {
                        context_window: entry.context_length,
                        token_counting: Support::Unknown,
                        ..ModelCapabilities::unknown()
                    },
                    metadata: Default::default(),
                })
            })
            .collect()
    }

    async fn count_tokens(
        &self,
        request: TokenCountRequest,
        control: CallControl,
    ) -> Result<TokenCount, LlmError> {
        self.inner.count_tokens(request, control).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_api_key() {
        let backend = OpenRouter::new("sk-or-sentinel-secret-key-do-not-leak");
        let debug = format!("{backend:?}");
        assert!(!debug.contains("sk-or-sentinel-secret-key-do-not-leak"));
    }
}
