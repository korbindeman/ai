//! Built-in HTTP adapters.

#[cfg(feature = "http")]
pub(crate) mod http_util;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "chatgpt-subscription")]
pub mod chatgpt_subscription;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

#[cfg(feature = "openrouter")]
pub mod openrouter;
