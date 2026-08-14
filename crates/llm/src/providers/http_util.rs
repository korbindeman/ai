//! Shared HTTP helpers for built-in adapters.

use crate::error::LlmError;
use crate::event::{
    CallControl, EventSink, OutputEvent, Sensitivity, WireCapture, WireDirection, WireEvent,
};
use reqwest::{Response, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;
use std::time::{Duration, Instant};

const SENSITIVE_QUERY_KEYS: &[&str] = &[
    "signature",
    "sig",
    "x-amz-signature",
    "token",
    "access_token",
    "api_key",
    "apikey",
    "key",
    "expires",
    "expires_at",
    "hmac",
    "credential",
    "signed",
];

const SENSITIVE_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
    "openai-api-key",
];

pub(crate) fn secret_from_string(value: impl Into<String>) -> SecretString {
    SecretString::from(value.into())
}

pub(crate) fn expose(secret: &SecretString) -> &str {
    secret.expose_secret()
}

pub(crate) fn sanitize_url(raw: &str) -> String {
    let Some((base, query)) = raw.split_once('?') else {
        return raw.to_string();
    };
    let redacted: Vec<String> = query
        .split('&')
        .map(|pair| {
            let key = pair.split('=').next().unwrap_or(pair);
            if is_sensitive_query(key) {
                format!("{key}=REDACTED")
            } else {
                pair.to_string()
            }
        })
        .collect();
    format!("{base}?{}", redacted.join("&"))
}

fn is_sensitive_query(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    SENSITIVE_QUERY_KEYS
        .iter()
        .any(|candidate| key == *candidate || key.ends_with(&format!("-{candidate}")))
}

pub(crate) fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SENSITIVE_HEADERS.contains(&name.as_str())
}

pub(crate) async fn send(
    builder: reqwest::RequestBuilder,
    control: &CallControl,
) -> Result<Response, LlmError> {
    tokio::select! {
        result = builder.send() => {
            result.map_err(|error| LlmError::transport("transport error").with_source(error))
        }
        _ = control.cancelled() => Err(LlmError::cancelled()),
    }
}

pub(crate) async fn emit_http_request(
    events: &EventSink,
    control: &CallControl,
    method: &str,
    url: &str,
    safe_headers: &[(&str, &str)],
    body: Option<&Value>,
) -> Result<(), LlmError> {
    let capture = control.wire_capture();
    if matches!(capture, WireCapture::Off) {
        return Ok(());
    }
    let mut payload = serde_json::json!({
        "method": method,
        "url": sanitize_url(url),
        "headers": safe_header_map(safe_headers),
    });
    let sensitivity = if matches!(capture, WireCapture::Bodies) {
        if let Some(body) = body {
            payload["body"] = body.clone();
        }
        Sensitivity::Sensitive
    } else {
        Sensitivity::Public
    };
    events
        .emit(OutputEvent::Wire(WireEvent {
            direction: WireDirection::Request,
            kind: "http".into(),
            payload,
            sensitivity,
        }))
        .await
        .map_err(LlmError::from)
}

pub(crate) async fn emit_http_response(
    events: &EventSink,
    control: &CallControl,
    status: u16,
    duration: Duration,
    body: Option<&Value>,
    raw_body: Option<&str>,
) -> Result<(), LlmError> {
    let capture = control.wire_capture();
    if matches!(capture, WireCapture::Off) {
        return Ok(());
    }
    let mut payload = serde_json::json!({
        "status": status,
        "duration_ms": duration.as_millis(),
    });
    let sensitivity = if matches!(capture, WireCapture::Bodies) {
        if let Some(body) = body {
            payload["body"] = body.clone();
        } else if let Some(raw_body) = raw_body {
            payload["body"] = Value::String(raw_body.to_string());
        }
        Sensitivity::Sensitive
    } else {
        Sensitivity::Public
    };
    events
        .emit(OutputEvent::Wire(WireEvent {
            direction: WireDirection::Response,
            kind: "http".into(),
            payload,
            sensitivity,
        }))
        .await
        .map_err(LlmError::from)
}

pub(crate) async fn emit_sse_frame(
    events: &EventSink,
    control: &CallControl,
    data: &str,
) -> Result<(), LlmError> {
    if !matches!(control.wire_capture(), WireCapture::Bodies) {
        return Ok(());
    }
    events
        .emit(OutputEvent::Wire(WireEvent {
            direction: WireDirection::Response,
            kind: "sse_frame".into(),
            payload: Value::String(data.to_string()),
            sensitivity: Sensitivity::Sensitive,
        }))
        .await
        .map_err(LlmError::from)
}

pub(crate) fn classify_http_error(
    status: StatusCode,
    body: &str,
    retry_after: Option<u64>,
) -> LlmError {
    let status_code = status.as_u16();
    let lower = body.to_ascii_lowercase();
    let detail = extract_provider_detail(body);
    let detail_lower = detail.as_deref().map(str::to_ascii_lowercase);

    if looks_like_context_limit(&lower, detail_lower.as_deref()) {
        return LlmError::context_limit("context limit exceeded").with_status(status_code);
    }
    if looks_like_model_unavailable(&lower, detail_lower.as_deref()) {
        return LlmError::model_unavailable("model unavailable").with_status(status_code);
    }

    let mut error = match status_code {
        401 => LlmError::authentication("authentication failed"),
        403 => LlmError::permission("permission denied"),
        404 => LlmError::model_unavailable("model unavailable"),
        408 | 504 => LlmError::timeout("request timed out"),
        429 => LlmError::rate_limited("rate limited"),
        code if (500..600).contains(&code) => LlmError::transport("provider unavailable"),
        _ => LlmError::backend("provider request failed"),
    }
    .with_status(status_code);

    if let Some(retry_after) = retry_after {
        error = error.with_retry_after_ms(retry_after);
    } else if status_code == 429 {
        error = error.with_retryable(true);
    }
    error
}

pub(crate) fn retry_after_ms(response: &Response) -> Option<u64> {
    let header = response.headers().get("retry-after")?.to_str().ok()?;
    parse_retry_after(header)
}

pub(crate) fn parse_retry_after(header: &str) -> Option<u64> {
    if let Ok(seconds) = header.trim().parse::<u64>() {
        return Some(seconds.saturating_mul(1000));
    }
    None
}

pub(crate) async fn error_from_response(
    events: &EventSink,
    control: &CallControl,
    started: Instant,
    response: Response,
) -> LlmError {
    let status = response.status();
    let retry_after = retry_after_ms(&response);
    let body = response.text().await.unwrap_or_default();
    let _ = emit_http_response(
        events,
        control,
        status.as_u16(),
        started.elapsed(),
        serde_json::from_str(&body).ok().as_ref(),
        Some(&body),
    )
    .await;
    classify_http_error(status, &body, retry_after)
}

fn looks_like_context_limit(lower: &str, detail_lower: Option<&str>) -> bool {
    let haystack = detail_lower.unwrap_or(lower);
    haystack.contains("context length")
        || haystack.contains("maximum context")
        || haystack.contains("too many tokens")
        || haystack.contains("context_length")
        || haystack.contains("context window")
}

fn looks_like_model_unavailable(lower: &str, detail_lower: Option<&str>) -> bool {
    let haystack = detail_lower.unwrap_or(lower);
    haystack.contains("model")
        && (haystack.contains("not found")
            || haystack.contains("does not exist")
            || haystack.contains("unavailable"))
}

fn extract_provider_detail(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let err = value.get("error").unwrap_or(&value);
    if let Some(raw) = err
        .pointer("/metadata/raw")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        && let Ok(inner) = serde_json::from_str::<Value>(raw)
        && let Some(detail) = first_string_field(&inner, &["error", "message", "msg"])
    {
        return Some(detail);
    }
    first_string_field(err, &["message", "error", "msg"])
}

fn first_string_field(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(text) = value.get(*key).and_then(Value::as_str)
            && let trimmed = text.trim()
            && !trimmed.is_empty()
        {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn safe_header_map(headers: &[(&str, &str)]) -> Value {
    let mut map = serde_json::Map::new();
    for (name, value) in headers {
        if !is_sensitive_header(name) {
            map.insert((*name).to_string(), Value::String((*value).to_string()));
        }
    }
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_signed_query_values() {
        let url = "https://files.example/object?X-Amz-Signature=abc&expires=1&path=foo";
        let sanitized = sanitize_url(url);
        assert!(sanitized.contains("X-Amz-Signature=REDACTED"));
        assert!(sanitized.contains("expires=REDACTED"));
        assert!(sanitized.contains("path=foo"));
        assert!(!sanitized.contains("abc"));
    }

    #[test]
    fn classifies_nested_context_limit_without_dumping_json() {
        let body = r#"{
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": "{\"error\":\"This model's maximum context length is 128000 tokens\"}"
                }
            }
        }"#;
        let error = classify_http_error(StatusCode::BAD_REQUEST, body, None);
        assert_eq!(error.kind(), crate::error::ErrorKind::ContextLimit);
        assert_eq!(error.to_string(), "context limit exceeded");
        assert!(!error.to_string().contains('{'));
    }

    #[test]
    fn classifies_rate_limit_and_retry_after() {
        let error = classify_http_error(StatusCode::TOO_MANY_REQUESTS, "rate limit", Some(2000));
        assert_eq!(error.kind(), crate::error::ErrorKind::RateLimited);
        assert_eq!(error.report().retry_after_ms, Some(2000));
    }
}
