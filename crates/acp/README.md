# acp

Rust client library for the [Agent Client Protocol](https://github.com/anthropics/agent-client-protocol). Manages ACP agent processes, sessions, and tool execution (terminal, filesystem, permissions).

Built-in support for auto-installing and launching Claude Code, Codex, and OpenCode agents.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
acp = { path = "crates/acp" }
tokio = { version = "1", features = ["full"] }
```

### Resolve and launch an agent

```rust
use std::path::PathBuf;
use acp::manager::{AgentManager, ClientInfo};
use acp::resolve;

#[tokio::main]
async fn main() {
    // Resolve the agent binary (auto-installs npm packages on first use)
    let agent = resolve::resolve("claude").expect("failed to resolve agent");

    // Create the manager
    let mut mgr = AgentManager::new(ClientInfo {
        name: "my-app".into(),
        version: "0.1.0".into(),
        title: Some("My App".into()),
    });

    // Must run on a LocalSet because ACP futures are !Send
    let local = tokio::task::LocalSet::new();
    local.run_until(async {
        let args: Vec<&str> = agent.args.iter().map(|s| s.as_str()).collect();
        let session_id = mgr
            .start_session(
                agent.program.to_str().unwrap(),
                &args,
                PathBuf::from("."),
                "my-session",
            )
            .await
            .expect("failed to start session");

        // Send a prompt
        let stop_reason = mgr
            .prompt(
                &session_id,
                vec![acp::schema::ContentBlock::Text(
                    acp::schema::TextContent::new("Hello, what files are in this directory?"),
                )],
            )
            .await
            .expect("prompt failed");

        println!("stop reason: {stop_reason}");

        mgr.kill_session(&session_id);
    }).await;
}
```

### Streaming updates

```rust
mgr.on_update(|session_id, update| {
    match update {
        acp::SessionUpdate::Text(text) => print!("{text}"),
        acp::SessionUpdate::ToolCallStarted { title, .. } => {
            println!("[tool] {title}");
        }
        acp::SessionUpdate::ToolCallDone { .. } => {}
        acp::SessionUpdate::Plan(_) => {}
    }
});
```

### Agent resolution

Built-in agents are resolved by short name. npm-based agents are auto-installed to `~/.acp/agents/` on first use.

| Name | Type | Package |
|------|------|---------|
| `claude` | npm | `@zed-industries/claude-agent-acp` |
| `codex` | npm | `@zed-industries/codex-acp` |
| `opencode` | binary | `opencode` (from PATH) |

```rust
// Resolve default agent (claude)
let agent = resolve::resolve_default().unwrap();

// Resolve by name
let agent = resolve::resolve("codex").unwrap();

// With extra args and env vars
let agent = resolve::resolve_with_args("claude", &[], &env_vars).unwrap();

// With custom cache directory
let agent = resolve::resolve_with_cache("claude", &cache_dir, &[], &env).unwrap();

// Any command — falls back to PATH lookup
let agent = resolve::resolve("my-custom-agent").unwrap();
```

### Manager configuration

```rust
let mut mgr = AgentManager::new(client_info);

// Auto-approve tool calls (default: true)
mgr.set_auto_approve(true);

// Inject env vars into all spawned processes
mgr.set_env(HashMap::from([
    ("MY_VAR".into(), "value".into()),
]));

// Enable per-session JSON-RPC message logging
mgr.set_log_dir(PathBuf::from("/tmp/acp-logs"));
```

### Sessions with MCP servers

```rust
let session_id = mgr
    .start_session_with_mcp(
        agent_cmd,
        &agent_args,
        cwd,
        vec![mcp_server],
        "label",
        &agent_env,
    )
    .await?;
```

### Session lifecycle

```rust
// Send a prompt (blocks until agent finishes)
let stop_reason = mgr.prompt(&session_id, content).await?;

// Soft cancel (session stays alive for follow-up prompts)
mgr.cancel(&session_id).await?;

// Kill session and agent process
mgr.kill_session(&session_id);

// List active sessions
let ids = mgr.session_ids();
```
