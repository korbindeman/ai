# LLM crate redesign

Status: approved for implementation

Audience: the agent that implements the redesign

Crate: `/Users/korbin/dev/ai/crates/llm`

## Purpose

The `llm` crate provides one model interface for Rust applications.

The interface supports hosted APIs, local servers, in-process models, recorded models, and consumer-defined models.

A consumer can add a backend without a change to the `llm` crate.

The crate provides transport and model-call behavior. It does not provide agent behavior.

This redesign replaces the current provider and model enums with public backend traits.

## Required outcomes

The implementation must provide these outcomes:

1. A consumer can implement a model backend in another crate.
2. A consumer can register that backend under a new backend ID.
3. A consumer can use any model ID without a central enum change.
4. One typed stream represents all generation calls.
5. A helper can collect that stream into a final result.
6. Generation and embedding use separate backend traits.
7. Requests, events, results, and error reports support Serde.
8. Cancellation, deadlines, and bounded backpressure have defined behavior.
9. Recorded calls can run again through a replay backend.
10. Built-in adapters support Anthropic, OpenAI-compatible APIs, OpenRouter, and Ollama.
11. Provider continuation data remains separate from semantic message content.
12. Logs and error messages do not contain credentials.

## Non-goals

The crate must not contain these systems:

- An agent loop.
- Tool execution.
- Prompt policy.
- Conversation storage.
- Memory or retrieval.
- Model-selection policy.
- Automatic generation retries.
- Application-specific authentication user interfaces.
- Application-specific error messages.

The crate describes tool calls and tool results. A consumer executes the tools.

The crate exposes model metadata. A consumer selects the model.

## Terminology

Use one term for each concept.

| Term | Meaning |
| --- | --- |
| Backend | An implementation that calls or runs one family of models. |
| Model reference | A backend ID and a model ID. |
| Generation | One model call that produces an event stream. |
| Output event | A nonterminal event from a backend. |
| Generation event | An output event or the final result. |
| Backend state | Opaque data that a backend needs for a later call. |
| Extension | Namespaced JSON data for behavior outside the common interface. |
| Wire event | A sanitized provider request or response record. |
| Call record | A serializable request, event sequence, and outcome. |

Do not use `provider` as the common abstraction name. A local model has no external provider.

Provider-specific modules can use `provider` for wire-level concepts.

## Design rules

The implementation must obey these rules:

- Keep the required backend trait small.
- Give optional operations default implementations.
- Do not seal public backend traits.
- Do not use central enums for backend IDs or model IDs.
- Do not hard-code model catalogs in common code.
- Do not silently ignore an unsupported request field.
- Do not put provider-native data in semantic content blocks.
- Do not expose raw credentials through `Debug`, `Display`, events, or tracing.
- Do not spawn detached tasks that survive a dropped generation.
- Do not retry a generation after an output event.
- Do not require HTTP dependencies for a consumer-defined local backend.

## Public identifiers

Use owned newtypes for public identifiers.

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BackendId(String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub backend: BackendId,
    pub model: ModelId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallId(Uuid);
```

Each string identifier constructor must reject an empty value.

`CallId::new()` must create a UUID version 7 value.

The public API must also support construction from an existing UUID.

This support permits deterministic fixtures and imported call records.

## Semantic message types

All semantic message types must derive `Clone`, `Debug`, `Serialize`, `Deserialize`, and `PartialEq`.

```rust
pub enum Role {
    User,
    Assistant,
}

pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

pub enum ContentBlock {
    Text { text: String },
    Image(Image),
    Reasoning {
        text: String,
        visibility: ReasoningVisibility,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        content: Vec<ToolResultBlock>,
        is_error: bool,
    },
    Extension(Extension),
}

pub enum ToolResultBlock {
    Text { text: String },
    Image(Image),
    Extension(Extension),
}

pub enum ReasoningVisibility {
    Summary,
    Trace,
}

pub struct Image {
    pub source: ImageSource,
    pub label: Option<String>,
}

pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
}
```

The crate must not treat reasoning content as provider continuation data.

Plain or encrypted continuation data belongs in `BackendState`.

The crate must not create a `Tool` role. Each adapter maps tool results to its wire format.

## Tools and output format

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named { name: String },
}

pub enum OutputFormat {
    Text,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}
```

The client must reject duplicate tool names before it calls a backend.

If the named tool does not exist, the client must reject `ToolChoice::Named`.

The structured-output helper must use `OutputFormat::JsonSchema`.

The helper must not hide a tool-call fallback.

If a model lacks structured output, the client must return an unsupported-capability error.

## Extensions

Extensions provide a controlled escape hatch for future model features.

```rust
pub struct Extension {
    pub namespace: String,
    pub name: String,
    pub payload: serde_json::Value,
}
```

The namespace must use a stable owner name, such as `openrouter` or `my_company.runtime`.

The common crate namespace is `llm`.

An adapter must return an error for each request extension that it does not understand.

An adapter must not ignore an extension.

An extension must not duplicate a field that exists in the common interface.

Extensions must remain serializable and visible in call records.

## Backend state

Backend state preserves opaque continuation data across stateless calls.

```rust
pub struct BackendState {
    pub backend: BackendId,
    pub kind: String,
    pub payload: serde_json::Value,
}
```

The client must reject backend state for a different backend.

An adapter must preserve the payload without normalization.

Backend state must not appear in `Message` or `ContentBlock`.

Backend state is sensitive by default. Its `Debug` implementation must redact the payload.

The ChatGPT subscription adapter uses backend state for Responses API items.

## Generation request

```rust
pub struct GenerationRequest {
    pub model: ModelRef,
    pub instructions: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub options: GenerationOptions,
    pub backend_state: Vec<BackendState>,
    pub extensions: Vec<Extension>,
    pub metadata: BTreeMap<String, String>,
}

pub struct GenerationOptions {
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub tool_choice: ToolChoice,
    pub output_format: OutputFormat,
    pub reasoning: Option<ReasoningOptions>,
    pub timeout: Option<Duration>,
}

pub struct ReasoningOptions {
    pub effort: Option<ReasoningEffort>,
    pub budget_tokens: Option<u32>,
}

pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}
```

`GenerationOptions::default()` must select text output and automatic tool choice.

The default value must not set a timeout, token limit, temperature, or `top_p`.

Request metadata is for consumer correlation. An adapter must not send it to a provider.

Provider metadata belongs in a namespaced extension.

## Model capabilities

Capability information must be model-specific.

```rust
pub enum Support {
    Supported,
    Unsupported,
    Unknown,
}

pub struct ModelCapabilities {
    pub image_input: Support,
    pub tools: Support,
    pub structured_output: Support,
    pub reasoning: Support,
    pub token_counting: Support,
    pub temperature: Support,
    pub top_p: Support,
    pub stop_sequences: Support,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub extensions: BTreeMap<String, serde_json::Value>,
}
```

If the applicable value is `Unsupported`, the client must reject the request.

If the value is `Unknown`, the client can send the request.

If the provider rejects that request, the backend remains responsible for an accurate error.

Built-in adapters must not invent context limits from a backend-wide default.

Configured model metadata can override discovered metadata.

## Usage and final result

```rust
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

pub enum StopReason {
    EndTurn,
    ToolCall,
    MaxOutputTokens,
    ContentFilter,
    Other(String),
}

pub struct GenerationResult {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub backend_state: Vec<BackendState>,
    pub extensions: Vec<Extension>,
}
```

Unknown usage values must remain `None`.

The crate must not replace an unknown value with zero.

Provider-reported cost belongs in a provider extension until the crate defines exact money semantics.

## Stream events

A backend emits nonterminal output events through an `EventSink`.

```rust
pub enum OutputEvent {
    ContentDelta {
        output_index: u32,
        delta: ContentDelta,
    },
    Usage(Usage),
    Wire(WireEvent),
    Extension(Extension),
}

pub enum ContentDelta {
    Text { text: String },
    Reasoning { text: String },
    ToolCall {
        tool_index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    Extension(Extension),
}

pub enum GenerationEvent {
    Output(OutputEvent),
    Finished(GenerationResult),
}

pub struct CallEvent {
    pub call_id: CallId,
    pub sequence: u64,
    pub elapsed_micros: u64,
    pub event: GenerationEvent,
}
```

The client assigns sequence numbers. The first sequence number is zero.

The client uses one monotonic clock for all elapsed values in a call.

Each successful stream contains exactly one `Finished` event.

The `Finished` event is the last event in that stream.

Each failed stream contains one terminal `Err` item and no `Finished` event.

The final result must contain all user-visible output from the output events.

## Backend trait

The required backend trait is public and unsealed.

```rust
#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn capabilities(&self, model: &ModelId) -> ModelCapabilities;

    async fn generate(
        &self,
        request: GenerationRequest,
        events: EventSink,
        control: CallControl,
    ) -> Result<GenerationResult, LlmError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Err(LlmError::unsupported_operation("list_models"))
    }

    async fn count_tokens(
        &self,
        request: TokenCountRequest,
        control: CallControl,
    ) -> Result<TokenCount, LlmError> {
        Err(LlmError::unsupported_operation("count_tokens"))
    }
}
```

The optional methods need default implementations to protect external backend implementations from unnecessary changes.

`ModelInfo` contains a model ID, a display name, capabilities, and backend metadata.

The common crate must not attach cost tiers or guessed context limits to `ModelInfo`.

## Event sink and call control

`EventSink` is a cloneable crate type. It hides the internal bounded channel.

```rust
impl EventSink {
    pub async fn emit(&self, event: OutputEvent) -> Result<(), EmitError>;
}
```

The sink assigns the sequence number and elapsed time before delivery.

A full bounded channel causes the sink to apply backpressure.

The sink returns `EmitError::Cancelled` after the consumer drops the generation.

`CallControl` exposes these operations:

```rust
impl CallControl {
    pub fn call_id(&self) -> CallId;
    pub fn is_cancelled(&self) -> bool;
    pub async fn cancelled(&self);
    pub fn deadline(&self) -> Option<Instant>;
    pub fn remaining(&self) -> Option<Duration>;
    pub fn wire_capture(&self) -> WireCapture;
}
```

A consumer-defined backend can use the same cancellation contract as a built-in backend.

## Client and registry

`LlmClient` owns a map from `BackendId` to `Arc<dyn ModelBackend>`.

The builder must accept built-in and consumer-defined backend values.

```rust
let client = LlmClient::builder()
    .backend("anthropic", Anthropic::new(api_key))?
    .backend("lab", MyLocalBackend::new())?
    .event_buffer_capacity(64)?
    .wire_capture(WireCapture::Metadata)
    .build()?;
```

The builder must reject duplicate backend IDs.

The event buffer capacity must be greater than zero.

`LlmClient` must provide these primary methods:

```rust
impl LlmClient {
    pub fn generate(&self, request: GenerationRequest) -> Result<Generation, LlmError>;

    pub async fn complete(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, LlmError>;

    pub async fn record_call(
        &self,
        request: GenerationRequest,
    ) -> Result<RecordedCall, LlmError>;

    pub fn capabilities(&self, model: &ModelRef)
        -> Result<ModelCapabilities, LlmError>;
}
```

`generate` must perform synchronous request validation before it spawns a task.

`complete` must use `generate` and drain the same stream.

The crate must not maintain a global registry or a global default model.

## Generation handle

`Generation` implements `Stream<Item = Result<CallEvent, LlmError>>`.

It also provides these operations:

```rust
impl Generation {
    pub fn call_id(&self) -> CallId;
    pub fn cancel(&self);
    pub async fn finish(self) -> Result<GenerationResult, LlmError>;
}
```

`finish` drains all remaining events and returns the final result.

Dropping `Generation` must cancel the call and abort its runner task.

The implementation must not expose a separate completion handle.

This rule prevents a bounded-channel deadlock for a consumer that waits without reading events.

## Cancellation and deadlines

The client creates one cancellation token for each call.

The client passes that token through `CallControl`.

The runner must select between the backend future, cancellation, and the deadline.

Cancellation returns an error with the `Cancelled` kind.

A deadline returns an error with the `Timeout` kind.

The runner must abort the backend future after either condition.

If cancellation occurs, a backend must stop network reads and local inference.

No successful final result can occur after cancellation or timeout.

## Retry behavior

The crate must not retry a generation automatically.

Retries are consumer policy because a repeated call can create a different output or external effect.

An adapter can refresh an expired access token before it sends a generation request.

An adapter can reconnect before it receives a semantic output event.

Each reconnect must remain visible through wire events and tracing.

## Error model

`LlmError` must use `thiserror` and preserve an optional source error.

`ErrorReport` is the serializable and redacted form of `LlmError`.

```rust
pub enum ErrorKind {
    UnknownBackend,
    UnsupportedOperation,
    UnsupportedCapability,
    UnsupportedExtension,
    InvalidRequest,
    Authentication,
    Permission,
    RateLimited,
    ContextLimit,
    ModelUnavailable,
    Transport,
    InvalidResponse,
    Timeout,
    Cancelled,
    Backend,
    Internal,
}

pub struct ErrorReport {
    pub kind: ErrorKind,
    pub message: String,
    pub call_id: Option<CallId>,
    pub backend: Option<BackendId>,
    pub model: Option<ModelId>,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub retryable: bool,
    pub retry_after_ms: Option<u64>,
}
```

The safe message must be short and application-neutral.

The error source can contain diagnostic detail. It must not appear in `Display` or `ErrorReport`.

Provider bodies belong in protected wire events, not in public error messages.

The crate must provide public constructors for errors from consumer-defined backends.

## Wire observation

Built-in HTTP adapters must support three wire-capture levels:

```rust
pub enum WireCapture {
    Off,
    Metadata,
    Bodies,
}

pub enum WireDirection {
    Request,
    Response,
}

pub enum Sensitivity {
    Public,
    Sensitive,
}

pub struct WireEvent {
    pub direction: WireDirection,
    pub kind: String,
    pub payload: serde_json::Value,
    pub sensitivity: Sensitivity,
}
```

Metadata capture includes the method, sanitized URL, status, duration, and selected safe headers.

Body capture also includes request bodies, response bodies, and provider stream frames.

Wire events must never contain authorization headers, API keys, cookies, or signed URL parameters.

Wire events with bodies must carry a `Sensitive` classification.

Tracing must include the call ID, backend ID, model ID, status, duration, usage, and error kind.

Tracing must not include prompts, output text, image data, backend state, or credentials.

## Recording and replay

The crate must define a stable call-record format.

```rust
pub struct RecordedCall {
    pub schema_version: u32,
    pub call_id: CallId,
    pub started_at_unix_ms: u64,
    pub request: GenerationRequest,
    pub capabilities: ModelCapabilities,
    pub events: Vec<CallEvent>,
    pub outcome: RecordedOutcome,
}

pub enum RecordedOutcome {
    Succeeded(GenerationResult),
    Failed(ErrorReport),
}
```

The first schema version is `1`.

The format must preserve unknown extensions and backend state exactly.

For a successful call, `events` includes the terminal `Finished` event.

The successful outcome must equal the result in that terminal event.

For a failed call, `events` contains the successful events that occurred before the error.

The failed outcome contains the terminal error report.

`LlmClient::record_call` must collect a generation into a `RecordedCall`.

A backend error must produce a successful call-record operation with a failed outcome.

Request validation errors can stop `record_call` before a call record exists.

`ReplayBackend` must implement `ModelBackend`.

The replay backend must compare the semantic request with the recorded request.

The comparison must exclude the call ID, start time, deadline, and correlation metadata.

The default replay mode emits events without recorded delays.

An optional replay mode can use the recorded elapsed times.

The replay backend emits only recorded `Output` events through its event sink.

The replay backend returns the recorded result. The client creates a new terminal event.

A request mismatch must return `InvalidRequest` with a structural difference summary.

`ScriptedBackend` must support ordered request expectations for unit tests.

If calls occur in the wrong order, the scripted backend must fail.

## Embeddings

Generation and embedding use separate traits.

```rust
#[async_trait]
pub trait EmbeddingBackend: Send + Sync + 'static {
    fn capabilities(&self, model: &ModelId) -> EmbeddingCapabilities;

    async fn embed(
        &self,
        request: EmbeddingRequest,
        control: CallControl,
    ) -> Result<EmbeddingResult, LlmError>;
}

pub struct EmbeddingRequest {
    pub model: ModelRef,
    pub inputs: Vec<String>,
    pub dimensions: Option<u32>,
    pub extensions: Vec<Extension>,
    pub timeout: Option<Duration>,
}

pub struct EmbeddingResult {
    pub model: ModelRef,
    pub vectors: Vec<Vec<f32>>,
    pub dimensions: u32,
    pub usage: Usage,
    pub extensions: Vec<Extension>,
}
```

The embedding client uses a separate backend registry.

The result must contain one vector for each input, in input order.

The client must reject mixed vector dimensions in one result.

Built-in embedding adapters must support OpenAI-compatible endpoints and Ollama.

## Built-in adapters

Each adapter must keep wire types private to its module.

Each adapter must translate between public semantic types and its wire types.

### Anthropic

The Anthropic adapter must support these features:

- Text input and output.
- Base64 image input.
- Tool calls and tool results.
- Streaming text, reasoning, and tool-call deltas.
- Structured output for models that support it.
- Token counting.
- Provider usage and stop reasons.
- Reasoning budget configuration.

The adapter must accept model IDs as strings.

The adapter must not contain a fixed Claude model enum.

### OpenAI-compatible

The OpenAI-compatible adapter is a configurable backend.

Its configuration includes a base URL, an optional API key, safe headers, and declared model capabilities.

It must support chat-completions streaming first.

It must support interleaved tool-call deltas by tool index.

The adapter must work with hosted endpoints and local compatible servers.

The adapter must not claim support for an extension that the configured endpoint lacks.

Use `/Users/korbin/dev/puddle/crates/llm/src/openai_compat.rs` as translation and parser source material.

### OpenRouter

The OpenRouter adapter wraps the OpenAI-compatible adapter.

It adds OpenRouter headers, usage fields, model discovery, and OpenRouter extensions.

Server-side web search uses the `openrouter.web_search` request extension.

The adapter must reject a request that also defines a function tool named `web_search`.

This error prevents duplicate tool registration.

### Ollama

The Ollama adapter must support a configurable base URL.

It must support text generation, tools, streaming, model discovery, and embeddings.

It must not assume that every Ollama model supports the same context size or feature set.

Use the existing AI crate adapter as the primary source material.

### ChatGPT subscription

The optional ChatGPT subscription adapter uses the Responses API.

It must accept a public asynchronous access-token source trait.

The adapter must not contain keychain, configuration UI, or Puddle-specific behavior.

It must preserve encrypted reasoning and provider items in `BackendState`.

Use `/Users/korbin/dev/puddle/crates/llm/src/chatgpt_subscription.rs` as source material.

## Authentication and secrets

Built-in adapters must store credentials in `secrecy::SecretString` or an equivalent redacted type.

Credential types must not implement serialization.

Custom `Debug` implementations must show a redacted value.

HTTP adapter constructors must accept credentials directly.

The crate must not read application environment variables without an explicit constructor call.

The ChatGPT subscription adapter must request tokens through its token-source trait.

## Cargo features

The core crate must compile without an HTTP client.

Use this feature structure:

```toml
[features]
default = []
http = ["dep:reqwest", "dep:eventsource-stream"]
anthropic = ["http"]
openai-compatible = ["http"]
openrouter = ["openai-compatible"]
ollama = ["http"]
chatgpt-subscription = ["http"]
```

Core dependencies can include these crates:

- `async-trait`
- `futures`
- `serde`
- `serde_json`
- `thiserror`
- `tokio` with the required runtime and synchronization features
- `tokio-util` for cancellation
- `tracing`
- `uuid` with version 7 and Serde support

HTTP features can add these crates:

- `reqwest` with Rustls, JSON, and streaming support
- `eventsource-stream`
- `secrecy`

Do not enable all Tokio features in the reusable crate.

Select only the required Tokio features.

## Module layout

Use this initial module layout:

```text
src/
  lib.rs
  backend.rs
  capability.rs
  client.rs
  embedding.rs
  error.rs
  event.rs
  extension.rs
  id.rs
  message.rs
  recording.rs
  request.rs
  response.rs
  tool.rs
  testing.rs
  providers/
    mod.rs
    anthropic.rs
    chatgpt_subscription.rs
    ollama.rs
    openai_compatible.rs
    openrouter.rs
```

Keep provider wire types in the applicable provider module.

Do not create a second crate during this redesign.

## Current API migration

This redesign is a breaking change. The crate is at version `0.1.0`.

Apply these replacements:

| Current API | Replacement |
| --- | --- |
| `Provider` enum | `BackendId` plus a registered `ModelBackend` |
| `Model` enum | `ModelRef` |
| `ProviderConfig` enum | Backend constructors and backend-owned configuration |
| `CompletionOptions` | `GenerationOptions` |
| `Response` | `GenerationResult` |
| `StreamCompletion` | `GenerationResult` in the terminal stream event |
| `StreamHandle` | `Generation` |
| `EmbeddingClient` with a closed provider enum | `EmbeddingBackend` plus an embedding registry |
| `ContentBlock::Thinking` | `ContentBlock::Reasoning` |
| `ContentBlock::ToolUse` | `ContentBlock::ToolCall` |
| `complete_structured` tool forcing | Native `OutputFormat::JsonSchema` helper |

Remove the `prompt!` macro from the crate.

Prompt-file loading is a consumer concern and does not belong in a model transport crate.

Do not keep deprecated wrappers for the old provider and model enums.

The clean break prevents two competing public interfaces.

## Implementation sequence

Perform the work in this order:

1. Add fixture tests for the useful current wire translations.
2. Add the new identifier, message, request, result, event, and error types.
3. Add `ModelBackend`, `EventSink`, `CallControl`, `Generation`, and `LlmClient`.
4. Add the external-backend integration test.
5. Add recording, replay, and scripted backends.
6. Port the Anthropic adapter.
7. Port the OpenAI-compatible adapter from Puddle.
8. Implement OpenRouter as an OpenAI-compatible specialization.
9. Port the Ollama generation and embedding adapters.
10. Add the generic embedding registry.
11. Port the ChatGPT subscription adapter behind its feature.
12. Remove the old API and unused modules.
13. Update the crate documentation and examples.
14. Run all required quality commands.

Do not keep both APIs during the final quality pass.

## Test requirements

The default test suite must not call an external network service.

Use local HTTP fixtures for built-in adapter tests.

### Public API tests

Add an integration test that implements `ModelBackend` outside the library crate.

This test must register the backend and complete a streamed call.

Compile the test with no built-in provider features.

Add the same type of test for `EmbeddingBackend`.

### Stream contract tests

Cover these cases:

- A successful text stream.
- Interleaved text and tool-call deltas.
- A usage update before the final result.
- A backend error before the first event.
- A backend error after output events.
- Consumer cancellation.
- Deadline expiration.
- Generation drop.
- A full bounded channel.
- A closed event receiver.
- A backend panic or unexpected runner exit.
- Exactly one terminal event.
- No event after the terminal event.
- Equality between streamed output and final semantic output.

### Request validation tests

Cover these cases:

- An unknown backend.
- An empty backend ID or model ID.
- Duplicate backend registration.
- Duplicate tool names.
- A missing named tool.
- Unsupported images, tools, output format, reasoning, and sampling values.
- Backend state for the wrong backend.
- An unknown request extension.
- A zero event-buffer capacity.

### Serialization tests

Round-trip every public semantic type through JSON.

Round-trip call records that contain unknown extensions.

Round-trip backend state without a payload change.

Make sure that redacted `Debug` output does not contain backend-state payloads.

### Recording tests

Record a scripted call and replay it.

Compare the replay event order and final result with the original values.

Make sure that strict replay rejects a different semantic request.

Make sure that replay ignores correlation metadata and recorded elapsed time.

### Adapter fixture tests

Each built-in adapter needs fixtures for these applicable cases:

- Request translation.
- Text streaming.
- Multiple content blocks.
- Tool-call streaming with fragmented arguments.
- Multiple interleaved tool calls.
- Tool results.
- Image input.
- Reasoning output.
- Usage values.
- Stop-reason mapping.
- Provider continuation state.
- Malformed stream data.
- Authentication errors.
- Rate limits and `Retry-After`.
- Context-limit errors.
- Provider errors with nested JSON.
- Cancellation during a response stream.

Do not use `unwrap` on provider response data.

### Security tests

Put sentinel credentials in each built-in adapter.

Make sure that these values do not occur in `Debug`, errors, traces, or wire events.

Make sure that sanitized URLs remove signed query values.

Make sure that body capture has the sensitive classification.

## Public documentation

The crate-level documentation must include these examples:

1. Register an Anthropic backend and complete a call.
2. Read a typed generation stream.
3. Implement and register a consumer-defined local backend.
4. Register and call an embedding backend.
5. Record and replay a call.

Each public type and public method needs a useful Rustdoc comment.

The custom-backend example must compile as a doctest or integration test.

## Required quality commands

Run these commands from `/Users/korbin/dev/ai`:

```text
cargo fmt --all --check
cargo test -p llm --no-default-features
cargo test -p llm --all-features
cargo clippy -p llm --all-features --all-targets -- -D warnings
cargo doc -p llm --all-features --no-deps
```

All commands must succeed before handoff.

## Acceptance criteria

The redesign is complete after all statements below are true:

- No public provider or model enum exists.
- A downstream crate can implement both backend traits.
- A downstream crate can register arbitrary backend and model IDs.
- The core crate compiles without `reqwest`.
- `complete` and manual stream collection return the same final result.
- Cancellation and deadline tests prove that no task remains active.
- The stream has bounded backpressure and one terminal event.
- Every public semantic value supports Serde.
- Unknown usage remains unknown.
- Backend state stays outside semantic messages.
- Unknown request extensions fail instead of disappearing.
- Recording and replay preserve typed events and backend state.
- Anthropic, OpenAI-compatible, OpenRouter, and Ollama fixture tests pass.
- OpenAI-compatible and Ollama embedding fixture tests pass.
- The ChatGPT subscription adapter preserves Responses API continuation state.
- No built-in adapter hard-codes a closed model list.
- No credential appears in logs, errors, events, or `Debug` output.
- All required quality commands succeed.

## Source material

Use the current AI crate for these implementations:

- Anthropic generation and token counting.
- Ollama generation, tools, streaming, and embeddings.
- OpenRouter request and response behavior.

Use the Puddle crate for these implementations:

- The open backend trait direction.
- OpenAI-compatible translation and streaming tool assembly.
- Responses API continuation state.
- ChatGPT subscription authentication boundaries.
- Provider error parsing tests.

Port behavior, not application coupling.

Do not copy Puddle configuration UI text, UI errors, keychain access, or product-specific names.

This document contains all approved design decisions for the redesign.
