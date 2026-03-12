use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use agent_client_protocol as acp;
use acp::{StreamMessageContent, StreamMessageDirection};
use tokio::process::Command;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::client::Client;
use crate::{Error, Result, UpdateCallback};

/// A handle to an active ACP session and its owning agent process.
struct SessionEntry {
    conn: Rc<acp::ClientSideConnection>,
    _child_ref: Rc<RefCell<Option<tokio::process::Child>>>,
}

/// Client identity sent during the ACP handshake.
pub struct ClientInfo {
    pub name: String,
    pub version: String,
    pub title: Option<String>,
}

/// Manages ACP agent processes and sessions.
///
/// All methods must be called from within a `tokio::task::LocalSet` because
/// ACP futures are `!Send`.
#[derive(Clone)]
pub struct AgentManager {
    sessions: Rc<RefCell<HashMap<String, SessionEntry>>>,
    update_callback: Rc<RefCell<Option<UpdateCallback>>>,
    auto_approve_permissions: bool,
    extra_env: Rc<HashMap<String, String>>,
    client_info: Rc<ClientInfo>,
    /// Optional directory for per-session JSON-RPC logs.
    log_dir: Option<PathBuf>,
}

impl AgentManager {
    pub fn new(client_info: ClientInfo) -> Self {
        Self {
            sessions: Rc::new(RefCell::new(HashMap::new())),
            update_callback: Rc::new(RefCell::new(None)),
            auto_approve_permissions: true,
            extra_env: Rc::new(HashMap::new()),
            client_info: Rc::new(client_info),
            log_dir: None,
        }
    }

    /// Set whether tool calls are auto-approved. Defaults to `true`.
    pub fn set_auto_approve(&mut self, auto_approve: bool) {
        self.auto_approve_permissions = auto_approve;
    }

    /// Set environment variables injected into every spawned subprocess
    /// (both agent processes and terminal commands).
    pub fn set_env(&mut self, env: HashMap<String, String>) {
        self.extra_env = Rc::new(env);
    }

    /// Set a directory for per-session JSON-RPC message logs.
    /// Each session writes to `<log_dir>/<label>.log`.
    pub fn set_log_dir(&mut self, dir: PathBuf) {
        self.log_dir = Some(dir);
    }

    /// Set a callback to receive session updates from all agents.
    /// The callback receives `(session_id, update)`.
    pub fn on_update(&self, callback: impl Fn(&str, crate::SessionUpdate) + 'static) {
        *self.update_callback.borrow_mut() = Some(Box::new(callback));
    }

    /// Spawn a new agent process, create a session, and return the session ID.
    ///
    /// `agent_cmd` is the command to run (e.g., "claude" or "npx").
    /// `agent_args` are additional arguments (e.g., ["--acp"]).
    /// `cwd` is the working directory for the session.
    /// `label` is used for log file naming.
    pub async fn start_session(
        &self,
        agent_cmd: &str,
        agent_args: &[&str],
        cwd: PathBuf,
        label: &str,
    ) -> Result<String> {
        self.start_session_inner(agent_cmd, agent_args, cwd, vec![], label, &HashMap::new())
            .await
    }

    /// Spawn a new agent process with MCP servers configured.
    ///
    /// `agent_env` contains additional environment variables layered on top of
    /// the base env (from `set_env()`), so these values take precedence.
    pub async fn start_session_with_mcp(
        &self,
        agent_cmd: &str,
        agent_args: &[&str],
        cwd: PathBuf,
        mcp_servers: Vec<acp::McpServer>,
        label: &str,
        agent_env: &HashMap<String, String>,
    ) -> Result<String> {
        self.start_session_inner(agent_cmd, agent_args, cwd, mcp_servers, label, agent_env)
            .await
    }

    async fn start_session_inner(
        &self,
        agent_cmd: &str,
        agent_args: &[&str],
        cwd: PathBuf,
        mcp_servers: Vec<acp::McpServer>,
        label: &str,
        agent_env: &HashMap<String, String>,
    ) -> Result<String> {
        tracing::debug!(cmd = agent_cmd, args = ?agent_args, cwd = %cwd.display(), "spawning agent process");

        let mut cmd = Command::new(agent_cmd);
        cmd.args(agent_args)
            .current_dir(&cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        for (k, v) in self.extra_env.iter() {
            cmd.env(k, v);
        }
        for (k, v) in agent_env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn()?;

        let stdin = child.stdin.take().unwrap().compat_write();
        let stdout = child.stdout.take().unwrap().compat();

        // Capture agent stderr and log line-by-line.
        if let Some(stderr) = child.stderr.take() {
            let agent_cmd_owned = agent_cmd.to_string();
            let cwd_owned = cwd.display().to_string();
            tokio::task::spawn_local(async move {
                use tokio::io::AsyncBufReadExt;
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if !line.is_empty() {
                        tracing::debug!(agent = agent_cmd_owned, cwd = cwd_owned, stderr_line = %line, "agent stderr");
                    }
                }
            });
        }

        // Create ACP client
        let client = Client::new(
            self.update_callback.clone(),
            self.auto_approve_permissions,
            self.extra_env.clone(),
        );

        let (conn, handle_io) =
            acp::ClientSideConnection::new(client, stdin, stdout, |fut| {
                tokio::task::spawn_local(fut);
            });
        let conn = Rc::new(conn);

        // Drive the I/O loop in background
        tokio::task::spawn_local(async move {
            if let Err(e) = handle_io.await {
                tracing::error!(error = %e, "ACP I/O error");
            }
        });

        // Optional per-session JSON-RPC logging.
        if let Some(log_dir) = &self.log_dir {
            let mut stream_rx = conn.subscribe();
            let log_dir = log_dir.clone();
            let log_label = label.to_string();
            tokio::task::spawn_local(async move {
                if let Err(e) = tokio::fs::create_dir_all(&log_dir).await {
                    tracing::warn!(error = %e, "failed to create session log dir");
                    return;
                }
                let log_path = log_dir.join(format!("{log_label}.log"));
                let file = match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .await
                {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!(path = %log_path.display(), error = %e, "failed to open session log");
                        return;
                    }
                };
                use tokio::io::AsyncWriteExt;
                let mut writer = tokio::io::BufWriter::new(file);

                while let Ok(msg) = stream_rx.recv().await {
                    let arrow = match msg.direction {
                        StreamMessageDirection::Outgoing => "->",
                        StreamMessageDirection::Incoming => "<-",
                    };
                    let now = chrono::Local::now().format("%H:%M:%S");
                    let line = match &msg.message {
                        StreamMessageContent::Request { id, method, params } => {
                            let params_str = truncate_json(params.as_ref());
                            format!("[{now}] {arrow} {method} id={id} params={params_str}\n")
                        }
                        StreamMessageContent::Response { id, result } => {
                            let result_str = match result {
                                Ok(val) => truncate_json(val.as_ref()),
                                Err(e) => format!("error: {e}"),
                            };
                            format!("[{now}] {arrow} response id={id} result={result_str}\n")
                        }
                        StreamMessageContent::Notification { method, params } => {
                            let params_str = truncate_json(params.as_ref());
                            format!("[{now}] {arrow} {method} params={params_str}\n")
                        }
                    };
                    if let Err(e) = writer.write_all(line.as_bytes()).await {
                        tracing::warn!(error = %e, "session log write error");
                        break;
                    }
                    let _ = writer.flush().await;
                }
            });
        }

        // Initialize ACP handshake
        let mut impl_info =
            acp::Implementation::new(&self.client_info.name, &self.client_info.version);
        if let Some(title) = &self.client_info.title {
            impl_info = impl_info.title(title);
        }

        acp::Agent::initialize(
            conn.as_ref(),
            acp::InitializeRequest::new(acp::ProtocolVersion::V1)
                .client_capabilities(
                    acp::ClientCapabilities::new()
                        .fs(
                            acp::FileSystemCapabilities::new()
                                .read_text_file(true)
                                .write_text_file(true),
                        )
                        .terminal(true),
                )
                .client_info(impl_info),
        )
        .await?;

        // Create session
        let session_resp = acp::Agent::new_session(
            conn.as_ref(),
            acp::NewSessionRequest::new(cwd).mcp_servers(mcp_servers),
        )
        .await?;

        let session_id = session_resp.session_id.to_string();
        tracing::info!(session_id, "ACP session created");

        // Store session
        let child_ref = Rc::new(RefCell::new(Some(child)));
        self.sessions.borrow_mut().insert(
            session_id.clone(),
            SessionEntry {
                conn: conn.clone(),
                _child_ref: child_ref,
            },
        );

        Ok(session_id)
    }

    /// Send a prompt to an existing session and wait for completion.
    /// Returns the stop reason as a string.
    pub async fn prompt(
        &self,
        session_id: &str,
        content: Vec<acp::ContentBlock>,
    ) -> Result<String> {
        tracing::debug!(session_id, blocks = content.len(), "sending prompt to agent");
        let conn = {
            let sessions = self.sessions.borrow();
            let entry = sessions
                .get(session_id)
                .ok_or_else(|| Error::SessionNotFound(session_id.to_string()))?;
            entry.conn.clone()
        };

        let resp = acp::Agent::prompt(
            conn.as_ref(),
            acp::PromptRequest::new(
                acp::SessionId::from(session_id.to_string()),
                content,
            ),
        )
        .await?;

        Ok(format!("{:?}", resp.stop_reason))
    }

    /// Cancel a running prompt on a session (soft cancel — session stays alive).
    pub async fn cancel(&self, session_id: &str) -> Result<()> {
        let conn = {
            let sessions = self.sessions.borrow();
            let entry = sessions
                .get(session_id)
                .ok_or_else(|| Error::SessionNotFound(session_id.to_string()))?;
            entry.conn.clone()
        };

        acp::Agent::cancel(
            conn.as_ref(),
            acp::CancelNotification::new(acp::SessionId::from(
                session_id.to_string(),
            )),
        )
        .await?;

        Ok(())
    }

    /// Kill a session and its agent process.
    /// Dropping the session entry kills the child process via `kill_on_drop`.
    pub fn kill_session(&self, session_id: &str) {
        tracing::debug!(session_id, "killing ACP session");
        self.sessions.borrow_mut().remove(session_id);
    }

    /// List all active session IDs.
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions.borrow().keys().cloned().collect()
    }
}

/// Truncate a JSON value to a reasonable size for logging.
fn truncate_json(value: Option<&serde_json::Value>) -> String {
    const MAX_LEN: usize = 500;
    let Some(value) = value else {
        return "null".to_string();
    };
    let s = value.to_string();
    if s.len() <= MAX_LEN {
        s
    } else {
        let end = s.floor_char_boundary(MAX_LEN);
        let truncated = s.len() - end;
        format!("{}[...truncated {truncated} bytes]", &s[..end])
    }
}
