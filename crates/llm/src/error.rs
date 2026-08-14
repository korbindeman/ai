//! Errors for model transport operations.

use crate::id::{BackendId, CallId, EmptyIdentifierError, ModelId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Classification of an [`LlmError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// No backend is registered for the requested ID.
    UnknownBackend,
    /// The backend does not implement the requested operation.
    UnsupportedOperation,
    /// The model does not support a requested capability.
    UnsupportedCapability,
    /// The backend does not understand a request extension.
    UnsupportedExtension,
    /// The request failed local validation.
    InvalidRequest,
    /// Authentication failed.
    Authentication,
    /// The caller is not allowed to use the model or resource.
    Permission,
    /// The backend rejected the call because of rate limits.
    RateLimited,
    /// The request exceeded the model context window.
    ContextLimit,
    /// The requested model is not available.
    ModelUnavailable,
    /// A network or protocol transport failure.
    Transport,
    /// The backend returned a response that could not be interpreted.
    InvalidResponse,
    /// The call exceeded its deadline.
    Timeout,
    /// The call was cancelled.
    Cancelled,
    /// An unclassified backend failure.
    Backend,
    /// An internal crate failure.
    Internal,
}

/// Serializable and redacted form of [`LlmError`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Error classification.
    pub kind: ErrorKind,
    /// Short application-neutral message.
    pub message: String,
    /// Call that produced the error, when known.
    pub call_id: Option<CallId>,
    /// Backend that produced the error, when known.
    pub backend: Option<BackendId>,
    /// Model that produced the error, when known.
    pub model: Option<ModelId>,
    /// HTTP status, when the error came from an HTTP adapter.
    pub status: Option<u16>,
    /// Provider error code, when known.
    pub code: Option<String>,
    /// Whether a later retry might succeed.
    pub retryable: bool,
    /// Suggested wait before retry, in milliseconds.
    pub retry_after_ms: Option<u64>,
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Error from a model or embedding call.
///
/// [`std::fmt::Display`] and [`ErrorReport`] contain a short safe message. Diagnostic
/// detail may live in the source error and must not include credentials.
#[derive(thiserror::Error)]
#[error("{inner}")]
pub struct LlmError {
    inner: Box<LlmErrorInner>,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

struct LlmErrorInner {
    kind: ErrorKind,
    message: String,
    call_id: Option<CallId>,
    backend: Option<BackendId>,
    model: Option<ModelId>,
    status: Option<u16>,
    code: Option<String>,
    retryable: bool,
    retry_after_ms: Option<u64>,
}

impl fmt::Display for LlmErrorInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmError")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("call_id", &self.inner.call_id)
            .field("backend", &self.inner.backend)
            .field("model", &self.inner.model)
            .field("status", &self.inner.status)
            .field("code", &self.inner.code)
            .field("retryable", &self.inner.retryable)
            .field("retry_after_ms", &self.inner.retry_after_ms)
            .finish_non_exhaustive()
    }
}

impl LlmError {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let retryable = matches!(
            kind,
            ErrorKind::RateLimited | ErrorKind::Timeout | ErrorKind::Transport
        );
        Self {
            inner: Box::new(LlmErrorInner {
                kind,
                message: message.into(),
                call_id: None,
                backend: None,
                model: None,
                status: None,
                code: None,
                retryable,
                retry_after_ms: None,
            }),
            source: None,
        }
    }

    /// The backend ID is not registered on the client.
    pub fn unknown_backend(backend: BackendId) -> Self {
        Self::new(
            ErrorKind::UnknownBackend,
            format!("unknown backend: {backend}"),
        )
        .with_backend(backend)
    }

    /// The backend does not implement `operation`.
    pub fn unsupported_operation(operation: impl Into<String>) -> Self {
        let operation = operation.into();
        Self::new(
            ErrorKind::UnsupportedOperation,
            format!("unsupported operation: {operation}"),
        )
        .with_code(operation)
    }

    /// The model does not support `capability`.
    pub fn unsupported_capability(capability: impl Into<String>) -> Self {
        let capability = capability.into();
        Self::new(
            ErrorKind::UnsupportedCapability,
            format!("unsupported capability: {capability}"),
        )
        .with_code(capability)
    }

    /// The backend does not understand a request extension.
    pub fn unsupported_extension(namespace: &str, name: &str) -> Self {
        Self::new(
            ErrorKind::UnsupportedExtension,
            format!("unsupported extension: {namespace}.{name}"),
        )
        .with_code(format!("{namespace}.{name}"))
    }

    /// The request failed local validation.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, message)
    }

    /// Authentication failed.
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Authentication, message)
    }

    /// The caller is not allowed to use the model or resource.
    pub fn permission(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Permission, message)
    }

    /// The backend rejected the call because of rate limits.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::RateLimited, message)
    }

    /// The request exceeded the model context window.
    pub fn context_limit(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ContextLimit, message)
    }

    /// The requested model is not available.
    pub fn model_unavailable(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ModelUnavailable, message)
    }

    /// A network or protocol transport failure.
    pub fn transport(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Transport, message)
    }

    /// The backend returned a response that could not be interpreted.
    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidResponse, message)
    }

    /// The call exceeded its deadline.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }

    /// The call was cancelled.
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled, "generation cancelled")
    }

    /// An unclassified backend failure.
    pub fn backend(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Backend, message)
    }

    /// An internal crate failure.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    /// Reconstruct an error from a serializable report.
    ///
    /// The source error is not preserved.
    pub fn from_report(report: ErrorReport) -> Self {
        Self {
            inner: Box::new(LlmErrorInner {
                kind: report.kind,
                message: report.message,
                call_id: report.call_id,
                backend: report.backend,
                model: report.model,
                status: report.status,
                code: report.code,
                retryable: report.retryable,
                retry_after_ms: report.retry_after_ms,
            }),
            source: None,
        }
    }

    /// Attach a source error that is omitted from [`std::fmt::Display`] and [`ErrorReport`].
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Attach the call ID.
    pub fn with_call_id(mut self, call_id: CallId) -> Self {
        self.inner.call_id = Some(call_id);
        self
    }

    /// Attach the backend ID.
    pub fn with_backend(mut self, backend: BackendId) -> Self {
        self.inner.backend = Some(backend);
        self
    }

    /// Attach the model ID.
    pub fn with_model(mut self, model: ModelId) -> Self {
        self.inner.model = Some(model);
        self
    }

    /// Attach an HTTP status.
    pub fn with_status(mut self, status: u16) -> Self {
        self.inner.status = Some(status);
        self
    }

    /// Attach a provider error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.inner.code = Some(code.into());
        self
    }

    /// Override whether a later retry might succeed.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.inner.retryable = retryable;
        self
    }

    /// Attach a suggested wait before retry, in milliseconds.
    pub fn with_retry_after_ms(mut self, retry_after_ms: u64) -> Self {
        self.inner.retry_after_ms = Some(retry_after_ms);
        self.inner.retryable = true;
        self
    }

    /// Error classification.
    pub fn kind(&self) -> ErrorKind {
        self.inner.kind
    }

    /// Short application-neutral message.
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    /// Call that produced the error, when known.
    pub fn call_id(&self) -> Option<CallId> {
        self.inner.call_id
    }

    /// HTTP status, when the error came from an HTTP adapter.
    pub fn status(&self) -> Option<u16> {
        self.inner.status
    }

    /// Serializable and redacted form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport {
            kind: self.inner.kind,
            message: self.inner.message.clone(),
            call_id: self.inner.call_id,
            backend: self.inner.backend.clone(),
            model: self.inner.model.clone(),
            status: self.inner.status,
            code: self.inner.code.clone(),
            retryable: self.inner.retryable,
            retry_after_ms: self.inner.retry_after_ms,
        }
    }
}

impl From<EmptyIdentifierError> for LlmError {
    fn from(value: EmptyIdentifierError) -> Self {
        Self::invalid_request(value.to_string())
    }
}

impl From<ErrorReport> for LlmError {
    fn from(value: ErrorReport) -> Self {
        Self::from_report(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_safe_message_only() {
        let err = LlmError::transport("transport error")
            .with_source(std::io::Error::other("secret=sk-sentinel-in-source"));
        assert_eq!(err.to_string(), "transport error");
        assert!(!err.to_string().contains("sk-sentinel"));
        let report = err.report();
        assert_eq!(report.message, "transport error");
        assert!(!format!("{report:?}").contains("sk-sentinel"));
        assert!(!format!("{err:?}").contains("sk-sentinel"));
    }

    #[test]
    fn report_round_trips_through_json() {
        let err = LlmError::rate_limited("rate limited")
            .with_status(429)
            .with_retry_after_ms(1500);
        let report = err.report();
        let json = serde_json::to_string(&report).unwrap();
        let restored: ErrorReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, report);
    }
}
