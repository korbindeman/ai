//! Public identifiers for backends, models, and calls.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// A backend or model identifier was empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmptyIdentifierError {
    /// Which identifier was empty (`"backend"` or `"model"`).
    pub what: &'static str,
}

impl fmt::Display for EmptyIdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} id must not be empty", self.what)
    }
}

impl std::error::Error for EmptyIdentifierError {}

/// Identifier for a registered model backend.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct BackendId(String);

impl BackendId {
    /// Create a backend ID.
    ///
    /// Returns [`EmptyIdentifierError`] when `value` is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, EmptyIdentifierError> {
        Ok(Self(parse_id(value.into(), "backend")?))
    }

    /// Borrow the ID as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for BackendId {
    type Error = EmptyIdentifierError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for BackendId {
    type Error = EmptyIdentifierError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<BackendId> for String {
    fn from(value: BackendId) -> Self {
        value.0
    }
}

impl AsRef<str> for BackendId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BackendId").field(&self.0).finish()
    }
}

/// Identifier for a model within a backend.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelId(String);

impl ModelId {
    /// Create a model ID.
    ///
    /// Returns [`EmptyIdentifierError`] when `value` is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, EmptyIdentifierError> {
        Ok(Self(parse_id(value.into(), "model")?))
    }

    /// Borrow the ID as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ModelId {
    type Error = EmptyIdentifierError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for ModelId {
    type Error = EmptyIdentifierError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ModelId> for String {
    fn from(value: ModelId) -> Self {
        value.0
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ModelId").field(&self.0).finish()
    }
}

/// A backend ID and a model ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// Backend that serves the model.
    pub backend: BackendId,
    /// Model identifier understood by that backend.
    pub model: ModelId,
}

impl ModelRef {
    /// Create a model reference from backend and model ID strings.
    pub fn new(
        backend: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, EmptyIdentifierError> {
        Ok(Self {
            backend: BackendId::new(backend)?,
            model: ModelId::new(model)?,
        })
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.backend, self.model)
    }
}

/// Identifier for one generation or embedding call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallId(Uuid);

impl CallId {
    /// Create a UUID version 7 call ID.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Create a call ID from an existing UUID.
    ///
    /// Use this for deterministic fixtures and imported call records.
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Return the inner UUID.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for CallId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for CallId {
    fn from(value: Uuid) -> Self {
        Self::from_uuid(value)
    }
}

fn parse_id(value: String, what: &'static str) -> Result<String, EmptyIdentifierError> {
    if value.is_empty() {
        Err(EmptyIdentifierError { what })
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_backend_id() {
        assert!(BackendId::new("").is_err());
        assert!(BackendId::new("anthropic").is_ok());
    }

    #[test]
    fn rejects_empty_model_id() {
        assert!(ModelId::new("").is_err());
        assert!(ModelId::new("claude-sonnet-4-6").is_ok());
    }

    #[test]
    fn call_id_is_uuid_v7() {
        let id = CallId::new();
        assert_eq!(id.as_uuid().get_version(), Some(uuid::Version::SortRand));
    }

    #[test]
    fn call_id_from_existing_uuid() {
        let uuid = Uuid::nil();
        assert_eq!(CallId::from_uuid(uuid).as_uuid(), uuid);
    }

    #[test]
    fn serde_rejects_empty_backend_id() {
        let err = serde_json::from_str::<BackendId>("\"\"").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }
}
