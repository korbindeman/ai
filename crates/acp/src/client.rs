use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use agent_client_protocol as acp;

use crate::{SessionUpdate, UpdateCallback};

/// State for a spawned terminal process.
/// Starts with a running child, transitions to completed with output.
enum TerminalState {
    Running {
        child: tokio::process::Child,
        output_limit: usize,
    },
    Done {
        output: String,
        truncated: bool,
        exit_status: acp::TerminalExitStatus,
    },
}

/// ACP Client implementation.
pub struct Client {
    update_callback: Rc<RefCell<Option<UpdateCallback>>>,
    auto_approve: bool,
    terminals: Rc<RefCell<HashMap<String, TerminalState>>>,
    next_terminal_id: Rc<std::cell::Cell<u64>>,
    extra_env: Rc<HashMap<String, String>>,
    /// When set, only tools whose title matches one of these names are allowed.
    allowed_tools: Option<Rc<HashSet<String>>>,
    /// Names of MCP servers for this session — tools from these are always allowed.
    mcp_server_names: HashSet<String>,
}

impl Client {
    pub(crate) fn new(
        update_callback: Rc<RefCell<Option<UpdateCallback>>>,
        auto_approve: bool,
        extra_env: Rc<HashMap<String, String>>,
        allowed_tools: Option<Rc<HashSet<String>>>,
        mcp_server_names: HashSet<String>,
    ) -> Self {
        Self {
            update_callback,
            auto_approve,
            terminals: Rc::new(RefCell::new(HashMap::new())),
            next_terminal_id: Rc::new(std::cell::Cell::new(1)),
            extra_env,
            allowed_tools,
            mcp_server_names,
        }
    }

    /// Check whether a tool call is allowed by the filter.
    /// Returns `Ok(())` if allowed, or an ACP error if blocked.
    fn check_tool_allowed(&self, title: &str) -> acp::Result<()> {
        let Some(ref allowed) = self.allowed_tools else {
            return Ok(());
        };

        // Exact match on title.
        if allowed.contains(title) {
            return Ok(());
        }

        // Prefix match: title may be "Read /path/to/file" while allowed has "Read".
        let first_word = title.split_whitespace().next().unwrap_or(title);
        if allowed.contains(first_word) {
            return Ok(());
        }

        // MCP tools are always allowed. Claude Code names them as
        // "mcp__<server>__<tool>"; check if the title references a known
        // MCP server name.
        for server_name in &self.mcp_server_names {
            if title.contains(server_name.as_str()) {
                return Ok(());
            }
        }

        Err(acp::Error::into_internal_error(std::io::Error::other(
            format!(
                "Tool '{}' is not available in this session. Use the document MCP tools instead.",
                first_word
            ),
        )))
    }
}

/// Collect output from a completed child process.
/// Combines stdout and stderr, truncates to the limit (keeping the tail),
/// and builds the exit status.
async fn collect_output(
    child: tokio::process::Child,
    output_limit: usize,
) -> (String, bool, acp::TerminalExitStatus) {
    match child.wait_with_output().await {
        Ok(output) => {
            let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));

            let truncated = combined.len() > output_limit;
            if truncated {
                let start = combined.len() - output_limit;
                let start = combined.ceil_char_boundary(start);
                combined = combined[start..].to_string();
            }

            let exit_status = acp::TerminalExitStatus::new()
                .exit_code(output.status.code().map(|c| c as u32));
            (combined, truncated, exit_status)
        }
        Err(e) => {
            let exit_status =
                acp::TerminalExitStatus::new().signal(format!("io_error: {e}"));
            (String::new(), false, exit_status)
        }
    }
}

/// Take a running child out of the terminal state map, leaving a placeholder Done.
/// Returns the child and output_limit if the terminal was Running.
fn take_running_child(
    terminals: &RefCell<HashMap<String, TerminalState>>,
    terminal_id: &str,
) -> Option<(tokio::process::Child, usize)> {
    let mut terminals = terminals.borrow_mut();
    let state = terminals.get_mut(terminal_id)?;
    if !matches!(state, TerminalState::Running { .. }) {
        return None;
    }
    let taken = std::mem::replace(
        state,
        TerminalState::Done {
            output: String::new(),
            truncated: false,
            exit_status: acp::TerminalExitStatus::new(),
        },
    );
    if let TerminalState::Running {
        child,
        output_limit,
    } = taken
    {
        Some((child, output_limit))
    } else {
        None
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Client for Client {
    async fn request_permission(
        &self,
        args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        let tool_title = args
            .tool_call
            .fields
            .title
            .as_deref()
            .unwrap_or("unknown");

        // Check the tool filter before proceeding.
        self.check_tool_allowed(tool_title)?;

        if self.auto_approve {
            tracing::debug!(tool = tool_title, "auto-approving permission");
            let option_id = args
                .options
                .iter()
                .find(|o| o.kind == acp::PermissionOptionKind::AllowOnce)
                .or_else(|| {
                    args.options
                        .iter()
                        .find(|o| o.kind == acp::PermissionOptionKind::AllowAlways)
                })
                .map(|o| o.option_id.clone())
                .unwrap_or_else(|| args.options[0].option_id.clone());

            Ok(acp::RequestPermissionResponse::new(
                acp::RequestPermissionOutcome::Selected(
                    acp::SelectedPermissionOutcome::new(option_id),
                ),
            ))
        } else {
            tracing::warn!(tool = tool_title, "permission denied (non-interactive mode)");
            Ok(acp::RequestPermissionResponse::new(
                acp::RequestPermissionOutcome::Cancelled,
            ))
        }
    }

    async fn session_notification(
        &self,
        args: acp::SessionNotification,
    ) -> acp::Result<()> {
        let cb = self.update_callback.borrow();
        let Some(callback) = cb.as_ref() else {
            return Ok(());
        };

        let session_id = args.session_id.to_string();

        match args.update {
            acp::SessionUpdate::AgentMessageChunk(chunk) => {
                if let acp::ContentBlock::Text(text) = chunk.content {
                    callback(&session_id, SessionUpdate::Text(text.text));
                }
            }
            acp::SessionUpdate::ToolCall(tc) => {
                callback(
                    &session_id,
                    SessionUpdate::ToolCallStarted {
                        id: tc.tool_call_id.to_string(),
                        title: tc.title.clone(),
                    },
                );
            }
            acp::SessionUpdate::ToolCallUpdate(tcu) => {
                callback(
                    &session_id,
                    SessionUpdate::ToolCallDone {
                        id: tcu.tool_call_id.to_string(),
                    },
                );
            }
            acp::SessionUpdate::Plan(plan) => {
                if let Ok(value) = serde_json::to_value(&plan) {
                    callback(&session_id, SessionUpdate::Plan(value));
                }
            }
            _ => {}
        }

        Ok(())
    }

    async fn read_text_file(
        &self,
        args: acp::ReadTextFileRequest,
    ) -> acp::Result<acp::ReadTextFileResponse> {
        let content = tokio::fs::read_to_string(&args.path)
            .await
            .map_err(acp::Error::into_internal_error)?;
        Ok(acp::ReadTextFileResponse::new(content))
    }

    async fn write_text_file(
        &self,
        args: acp::WriteTextFileRequest,
    ) -> acp::Result<acp::WriteTextFileResponse> {
        let path = std::path::Path::new(&args.path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(acp::Error::into_internal_error)?;
        }

        tokio::fs::write(&args.path, &args.content)
            .await
            .map_err(acp::Error::into_internal_error)?;
        Ok(acp::WriteTextFileResponse::default())
    }

    async fn create_terminal(
        &self,
        args: acp::CreateTerminalRequest,
    ) -> acp::Result<acp::CreateTerminalResponse> {
        let mut cmd = tokio::process::Command::new(&args.command);
        cmd.args(&args.args);

        if let Some(cwd) = &args.cwd {
            cmd.current_dir(cwd);
        }

        for env_var in &args.env {
            cmd.env(&env_var.name, &env_var.value);
        }
        for (k, v) in self.extra_env.iter() {
            cmd.env(k, v);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.stdin(std::process::Stdio::null());
        cmd.kill_on_drop(true);

        let child = cmd.spawn().map_err(|e| {
            acp::Error::into_internal_error(std::io::Error::other(format!(
                "failed to spawn terminal: {e}"
            )))
        })?;

        let id_num = self.next_terminal_id.get();
        self.next_terminal_id.set(id_num + 1);
        let terminal_id = format!("term-{id_num}");

        let output_limit = args.output_byte_limit.unwrap_or(100_000) as usize;

        self.terminals.borrow_mut().insert(
            terminal_id.clone(),
            TerminalState::Running {
                child,
                output_limit,
            },
        );

        Ok(acp::CreateTerminalResponse::new(acp::TerminalId::from(
            terminal_id,
        )))
    }

    async fn terminal_output(
        &self,
        args: acp::TerminalOutputRequest,
    ) -> acp::Result<acp::TerminalOutputResponse> {
        let tid = args.terminal_id.to_string();
        self.finish_if_exited(&tid).await;

        let terminals = self.terminals.borrow();
        let state = terminals.get(&tid).ok_or_else(|| {
            acp::Error::into_internal_error(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "terminal not found",
            ))
        })?;

        match state {
            TerminalState::Running { .. } => {
                Ok(acp::TerminalOutputResponse::new(String::new(), false))
            }
            TerminalState::Done {
                output,
                truncated,
                exit_status,
                ..
            } => {
                let mut resp =
                    acp::TerminalOutputResponse::new(output.clone(), *truncated);
                resp.exit_status = Some(exit_status.clone());
                Ok(resp)
            }
        }
    }

    async fn wait_for_terminal_exit(
        &self,
        args: acp::WaitForTerminalExitRequest,
    ) -> acp::Result<acp::WaitForTerminalExitResponse> {
        let tid = args.terminal_id.to_string();

        {
            let terminals = self.terminals.borrow();
            if let Some(TerminalState::Done { exit_status, .. }) = terminals.get(&tid) {
                return Ok(acp::WaitForTerminalExitResponse::new(exit_status.clone()));
            }
        }

        let Some((child, output_limit)) = take_running_child(&self.terminals, &tid)
        else {
            return Err(acp::Error::into_internal_error(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "terminal not found",
            )));
        };

        let (output, truncated, exit_status) = collect_output(child, output_limit).await;

        self.terminals.borrow_mut().insert(
            tid,
            TerminalState::Done {
                output,
                truncated,
                exit_status: exit_status.clone(),
            },
        );

        Ok(acp::WaitForTerminalExitResponse::new(exit_status))
    }

    async fn kill_terminal(
        &self,
        args: acp::KillTerminalRequest,
    ) -> acp::Result<acp::KillTerminalResponse> {
        let tid = args.terminal_id.to_string();
        let child = {
            let mut terminals = self.terminals.borrow_mut();
            let state = terminals.get_mut(&tid).ok_or_else(|| {
                acp::Error::into_internal_error(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "terminal not found",
                ))
            })?;
            if let TerminalState::Running { child, .. } = state {
                Some(child.id())
            } else {
                None
            }
        };

        // Kill outside the borrow by taking the running child.
        if child.is_some() {
            if let Some((mut child, output_limit)) =
                take_running_child(&self.terminals, &tid)
            {
                let _ = child.kill().await;
                let (output, truncated, exit_status) =
                    collect_output(child, output_limit).await;
                self.terminals.borrow_mut().insert(
                    tid,
                    TerminalState::Done {
                        output,
                        truncated,
                        exit_status,
                    },
                );
            }
        }

        Ok(acp::KillTerminalResponse::default())
    }

    async fn release_terminal(
        &self,
        args: acp::ReleaseTerminalRequest,
    ) -> acp::Result<acp::ReleaseTerminalResponse> {
        let tid = args.terminal_id.to_string();
        let entry = self.terminals.borrow_mut().remove(&tid);

        if let Some(TerminalState::Running { mut child, .. }) = entry {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }

        Ok(acp::ReleaseTerminalResponse::default())
    }
}

impl Client {
    /// If the terminal's child has exited, transition to Done state.
    async fn finish_if_exited(&self, terminal_id: &str) {
        let exited = {
            let mut terminals = self.terminals.borrow_mut();
            if let Some(TerminalState::Running { child, .. }) =
                terminals.get_mut(terminal_id)
            {
                matches!(child.try_wait(), Ok(Some(_)))
            } else {
                false
            }
        };

        if !exited {
            return;
        }

        let Some((child, output_limit)) =
            take_running_child(&self.terminals, terminal_id)
        else {
            return;
        };

        let (output, truncated, exit_status) = collect_output(child, output_limit).await;

        self.terminals.borrow_mut().insert(
            terminal_id.to_string(),
            TerminalState::Done {
                output,
                truncated,
                exit_status,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client(
        allowed: Option<Vec<&str>>,
        mcp_names: Vec<&str>,
    ) -> Client {
        Client::new(
            Rc::new(RefCell::new(None)),
            true,
            Rc::new(HashMap::new()),
            allowed.map(|v| Rc::new(v.into_iter().map(String::from).collect())),
            mcp_names.into_iter().map(String::from).collect(),
        )
    }

    #[test]
    fn no_filter_allows_everything() {
        let client = make_client(None, vec![]);
        assert!(client.check_tool_allowed("Read").is_ok());
        assert!(client.check_tool_allowed("Write").is_ok());
        assert!(client.check_tool_allowed("Bash").is_ok());
    }

    #[test]
    fn filter_allows_listed_tools() {
        let client = make_client(Some(vec!["Read", "WebSearch"]), vec![]);
        assert!(client.check_tool_allowed("Read").is_ok());
        assert!(client.check_tool_allowed("WebSearch").is_ok());
    }

    #[test]
    fn filter_blocks_unlisted_tools() {
        let client = make_client(Some(vec!["Read", "WebSearch"]), vec![]);
        let err = client.check_tool_allowed("Write").unwrap_err();
        assert!(err.to_string().contains("not available"));

        let err = client.check_tool_allowed("Bash").unwrap_err();
        assert!(err.to_string().contains("not available"));

        let err = client.check_tool_allowed("Edit").unwrap_err();
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn filter_matches_title_prefix() {
        let client = make_client(Some(vec!["Read"]), vec![]);
        // Title like "Read /path/to/file.txt" should match "Read"
        assert!(client.check_tool_allowed("Read /path/to/file.txt").is_ok());
        // But "ReadFile" should NOT match
        assert!(client.check_tool_allowed("ReadFile").is_err());
    }

    #[test]
    fn mcp_tools_bypass_filter() {
        let client = make_client(Some(vec!["Read"]), vec!["enki"]);
        // MCP tool referencing a known server name should be allowed
        assert!(client.check_tool_allowed("mcp__enki__enki_edit_file").is_ok());
        // Built-in tool not in the list should still be blocked
        assert!(client.check_tool_allowed("Write").is_err());
    }
}
