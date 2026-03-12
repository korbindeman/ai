//! Resolve agent binaries for ACP sessions.
//!
//! Built-in agents are either npm packages (installed into a cache directory
//! on first use) or standalone binaries resolved from PATH.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A built-in agent that can be resolved by short name.
struct BuiltinAgent {
    name: &'static str,
    kind: BuiltinKind,
}

enum BuiltinKind {
    /// npm package installed into `<cache_dir>/<cache_name>/`.
    Npm {
        package: &'static str,
        entry_point: &'static str,
        cache_name: &'static str,
    },
    /// Standalone binary resolved from PATH.
    Binary {
        binary: &'static str,
        default_args: &'static [&'static str],
    },
}

const BUILTINS: &[BuiltinAgent] = &[
    BuiltinAgent {
        name: "claude",
        kind: BuiltinKind::Npm {
            package: "@zed-industries/claude-agent-acp",
            entry_point: "node_modules/@zed-industries/claude-agent-acp/dist/index.js",
            cache_name: "claude-agent-acp",
        },
    },
    BuiltinAgent {
        name: "codex",
        kind: BuiltinKind::Npm {
            package: "@zed-industries/codex-acp",
            entry_point: "node_modules/@zed-industries/codex-acp/bin/codex-acp.js",
            cache_name: "codex-acp",
        },
    },
    BuiltinAgent {
        name: "opencode",
        kind: BuiltinKind::Binary {
            binary: "opencode",
            default_args: &["acp"],
        },
    },
];

pub const DEFAULT_AGENT: &str = "claude";

/// Return the short names of all built-in agents.
pub fn builtin_names() -> Vec<&'static str> {
    BUILTINS.iter().map(|b| b.name).collect()
}

fn find_builtin(name: &str) -> Option<&'static BuiltinAgent> {
    BUILTINS.iter().find(|b| b.name == name)
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("node not found on PATH — install Node.js first")]
    NodeNotFound,
    #[error("npm install failed: {0}")]
    NpmInstallFailed(String),
    #[error("home directory not found")]
    NoHomeDir,
    #[error("agent not found: {0}")]
    AgentNotFound(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Resolved agent command ready to spawn.
#[derive(Debug, Clone)]
pub struct AgentCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

fn find_node() -> Result<PathBuf, ResolveError> {
    which::which("node").map_err(|_| ResolveError::NodeNotFound)
}

/// Default cache directory: `~/.acp/agents/`.
fn default_cache_root() -> Result<PathBuf, ResolveError> {
    let home = dirs::home_dir().ok_or(ResolveError::NoHomeDir)?;
    Ok(home.join(".acp").join("agents"))
}

fn npm_install(cache: &Path, package: &str) -> Result<(), ResolveError> {
    std::fs::create_dir_all(cache)?;

    let prefix = cache
        .to_str()
        .ok_or_else(|| ResolveError::NpmInstallFailed("cache path is not valid UTF-8".into()))?;

    let output = Command::new("npm")
        .args(["install", "--prefix", prefix, package])
        .output()
        .map_err(|e| ResolveError::NpmInstallFailed(e.to_string()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ResolveError::NpmInstallFailed(stderr.to_string()));
    }

    Ok(())
}

fn resolve_builtin(
    builtin: &BuiltinAgent,
    cache_root: &Path,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<AgentCommand, ResolveError> {
    match &builtin.kind {
        BuiltinKind::Npm {
            package,
            entry_point,
            cache_name,
        } => {
            let node = find_node()?;
            let cache = cache_root.join(cache_name);
            let entry = cache.join(entry_point);

            if !entry.exists() {
                tracing::info!(package, path = %cache.display(), "installing agent package");
                npm_install(&cache, package)?;

                if !entry.exists() {
                    return Err(ResolveError::NpmInstallFailed(format!(
                        "entry point not found after install: {}",
                        entry.display()
                    )));
                }
            }

            let mut args = vec![entry.to_string_lossy().into_owned()];
            args.extend_from_slice(extra_args);

            Ok(AgentCommand {
                program: node,
                args,
                env: env.clone(),
            })
        }
        BuiltinKind::Binary {
            binary,
            default_args,
        } => {
            let program = which::which(binary)
                .map_err(|_| ResolveError::AgentNotFound(binary.to_string()))?;

            let mut args: Vec<String> =
                default_args.iter().map(|s| s.to_string()).collect();
            args.extend_from_slice(extra_args);

            Ok(AgentCommand {
                program,
                args,
                env: env.clone(),
            })
        }
    }
}

/// Resolve an agent by its short name ("claude", "codex", "opencode").
///
/// Uses the default cache directory (`~/.acp/agents/`). For npm-based agents,
/// installs the package on first use.
pub fn resolve(name: &str) -> Result<AgentCommand, ResolveError> {
    resolve_with_args(name, &[], &HashMap::new())
}

/// Resolve an agent by short name with extra arguments and environment variables.
pub fn resolve_with_args(
    name: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<AgentCommand, ResolveError> {
    let cache_root = default_cache_root()?;
    resolve_with_cache(name, &cache_root, extra_args, env)
}

/// Resolve an agent by short name with a custom cache directory.
pub fn resolve_with_cache(
    name: &str,
    cache_root: &Path,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<AgentCommand, ResolveError> {
    if let Some(builtin) = find_builtin(name) {
        return resolve_builtin(builtin, cache_root, extra_args, env);
    }

    // Not a builtin — resolve from PATH or as a path.
    let program = if name.contains('/') {
        PathBuf::from(name)
    } else {
        which::which(name).map_err(|_| ResolveError::AgentNotFound(name.to_string()))?
    };

    let args = extra_args.to_vec();

    Ok(AgentCommand {
        program,
        args,
        env: env.clone(),
    })
}

/// Resolve the default agent ("claude") with no extra args.
pub fn resolve_default() -> Result<AgentCommand, ResolveError> {
    resolve(DEFAULT_AGENT)
}
