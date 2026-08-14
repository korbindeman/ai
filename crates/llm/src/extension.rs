//! Namespaced extensions for behavior outside the common interface.

use crate::error::LlmError;
use serde::{Deserialize, Serialize};

/// Namespaced JSON data for a feature that is not part of the common interface.
///
/// Use a stable owner name as the namespace, such as `openrouter` or
/// `my_company.runtime`. The common crate namespace is `llm`.
///
/// An adapter must return an error for each request extension that it does not
/// understand. An extension must not duplicate a field that exists in the
/// common interface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Extension {
    /// Stable owner name, such as `openrouter`.
    pub namespace: String,
    /// Feature name within the namespace.
    pub name: String,
    /// JSON payload.
    pub payload: serde_json::Value,
}

impl Extension {
    /// Create an extension.
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            payload,
        }
    }

    /// Return `{namespace}.{name}`.
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.namespace, self.name)
    }

    /// Return true when this extension has `namespace` and `name`.
    pub fn is(&self, namespace: &str, name: &str) -> bool {
        self.namespace == namespace && self.name == name
    }
}

/// Return an unsupported-extension error for any extension not in `allowed`.
pub fn reject_unknown_extensions(
    extensions: &[Extension],
    allowed: &[(&str, &str)],
) -> Result<(), LlmError> {
    for extension in extensions {
        if !allowed
            .iter()
            .any(|(namespace, name)| extension.is(namespace, name))
        {
            return Err(LlmError::unsupported_extension(
                &extension.namespace,
                &extension.name,
            ));
        }
    }
    Ok(())
}
