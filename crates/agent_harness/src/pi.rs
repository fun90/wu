//! Drives `pi --mode rpc` over JSONL on stdio (https://github.com/earendil-works/pi).
//!
//! Pi has no permission prompts, so every run is effectively full access.
// ponytail: no session listing/history or model cache yet; add when needed.

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use futures::stream::{BoxStream, FuturesUnordered};
use futures::{Future, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use crate::{
    AgentEvent, DoneStatus, HarnessError, HarnessId, Model, PermissionMode, ReasoningLevel, RunControls,
    RunRequest, Skill, SkillRef, SlashCommand, SteeringMode, ToolCall, UserInputAnswer, UserInputQuestion,
};
use crate::{Harness, StderrTail};

const INSTALL_HINT: &str = "npm install -g @earendil-works/pi-coding-agent";
const INTERRUPT_GRACE: Duration = Duration::from_secs(5);
const KILL_GRACE: Duration = Duration::from_secs(2);
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
const REASONING_LEVELS: &[ReasoningLevel] = &[
    ReasoningLevel::Minimal,
    ReasoningLevel::Low,
    ReasoningLevel::Medium,
    ReasoningLevel::High,
    ReasoningLevel::XHigh,
    ReasoningLevel::Max,
];

fn resolve_pi_executable() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PI_EXECUTABLE").filter(|path| !path.is_empty()) {
        return crate::executable::validate_native_override(&PathBuf::from(path)).ok();
    }
    let mut extra = Vec::new();
    if let Some(home) = crate::executable::home_dir() {
        extra.push(home.join(".local").join("bin").join("pi"));
        extra.push(home.join(".bun").join("bin").join("pi"));
        extra.push(home.join(".npm-global").join("bin").join("pi"));
    }
    extra.push(PathBuf::from("/opt/homebrew/bin/pi"));
    extra.push(PathBuf::from("/usr/local/bin/pi"));
    crate::executable::find_on_paths("pi", extra)
}

#[derive(Default)]
pub struct PiHarness {
    executable: Option<PathBuf>,
}

impl PiHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_executable(mut self, path: impl Into<PathBuf>) -> Self {
        self.executable = Some(path.into());
        self
    }

    fn resolve_executable(&self) -> Result<PathBuf, HarnessError> {
        if let Some(path) = &self.executable {
            return crate::executable::validate_native_override(path);
        }
        resolve_pi_executable().ok_or_else(|| HarnessError::NotInstalled(INSTALL_HINT.into()))
    }

    fn command(&self, cwd: &str) -> Result<Command, HarnessError> {
        let exe = self.resolve_executable()?;
        let mut cmd = Command::new(&exe);
        crate::compose_child_path(&mut cmd, &exe);
        cmd.args(["--mode", "rpc"]);
        if cwd.is_empty() {
            cmd.current_dir(crate::executable::scratch_dir());
        } else {
            cmd.current_dir(cwd);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        Ok(cmd)
    }
}

#[async_trait]
impl Harness for PiHarness {
    fn id(&self) -> HarnessId {
        HarnessId::Pi
    }
    fn display_name(&self) -> &str {
        "Pi"
    }
    fn supports_steering(&self) -> bool {
        true
    }
    fn steering_mode(&self) -> SteeringMode {
        SteeringMode::StepBoundary
    }
    fn reasoning_levels(&self) -> &[ReasoningLevel] {
        REASONING_LEVELS
    }
    fn installed(&self) -> bool {
        self.resolve_executable().is_ok()
    }
    fn executable_path(&self) -> Option<PathBuf> {
        self.resolve_executable().ok()
    }
    fn update_args(&self) -> &'static [&'static str] {
        &["update", "self"]
    }

    async fn models(&self) -> Result<Vec<Model>, HarnessError> {
        let mut cmd = self.command("")?;
        cmd.arg("--no-session");
        let mut rpc = Rpc::start(cmd)?;
        let result = tokio::time::timeout(PROBE_TIMEOUT, async {
            rpc.send(json!({"id": "models", "type": "get_available_models"}))
                .await?;
            let data = rpc.response("models").await?;
            Ok::<_, HarnessError>(models_from_wire(&data))
        })
        .await
        .unwrap_or_else(|_| Err(HarnessError::Protocol("pi did not answer in time".into())));
        rpc.shutdown().await;
        result
    }

    async fn commands(&self, cwd: &Path) -> Result<Vec<SlashCommand>, HarnessError> {
        let mut commands = vec![
            SlashCommand {
                name: "compact".into(),
                description: "Compact the conversation context".into(),
                input_hint: Some("[instructions]".into()),
                is_skill: false,
            },
            SlashCommand {
                name: "reload".into(),
                description: "Reload skills and commands".into(),
                input_hint: None,
                is_skill: false,
            },
        ];
        let data = self.get_commands(cwd).await?;
        commands.extend(commands_from_wire(&data));
        // Pi expands a leading `/skill:<name>` itself, so skills are also valid slash commands.
        commands.extend(wire_commands(&data, true).filter_map(|command| {
            Some(SlashCommand {
                name: command["name"].as_str()?.to_owned(),
                description: command["description"].as_str().unwrap_or_default().to_owned(),
                input_hint: None,
                is_skill: true,
            })
        }));
        Ok(commands)
    }

    async fn skills(&self, cwd: &Path) -> Result<Vec<Skill>, HarnessError> {
        Ok(skills_from_wire(&self.get_commands(cwd).await?))
    }

    async fn run(
        &self,
        request: RunRequest,
        controls: RunControls,
    ) -> Result<BoxStream<'static, Result<AgentEvent, HarnessError>>, HarnessError> {
        let mut cmd = self.command(&request.cwd)?;
        if let Some(resume) = &request.resume {
            cmd.args(["--session", resume]);
        } else if let Some(fork) = &request.fork {
            cmd.args(["--fork", fork]);
        }
        if request.permission == PermissionMode::ReadOnly {
            // Pi has no approval prompts, so read-only means withholding the writing tools.
            cmd.args(["--tools", "read,grep,find,ls", "--no-mcp"]);
        }
        if let Some(model) = &request.model {
            cmd.args(["--model", model]);
        }
        if let Some(level) = request.reasoning.and_then(thinking_arg) {
            cmd.args(["--thinking", level]);
        }
        let rpc = Rpc::start(cmd)?;
        let (event_tx, event_rx) = mpsc::channel::<Result<AgentEvent, HarnessError>>(256);
        tokio::spawn(run_session(rpc, request, controls, event_tx));
        Ok(futures::stream::unfold(event_rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        })
        .boxed())
    }
}

impl PiHarness {
    async fn get_commands(&self, cwd: &Path) -> Result<Value, HarnessError> {
        let mut cmd = self.command(&cwd.display().to_string())?;
        cmd.arg("--no-session");
        let mut rpc = Rpc::start(cmd)?;
        let result = tokio::time::timeout(PROBE_TIMEOUT, async {
            rpc.send(json!({"id": "commands", "type": "get_commands"}))
                .await?;
            rpc.response("commands").await
        })
        .await
        .unwrap_or_else(|_| Err(HarnessError::Protocol("pi did not answer in time".into())));
        rpc.shutdown().await;
        result
    }
}

fn wire_commands<'a>(data: &'a Value, skills: bool) -> impl Iterator<Item = &'a Value> {
    data["commands"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(move |command| (command["source"] == "skill") == skills)
}

fn commands_from_wire(data: &Value) -> Vec<SlashCommand> {
    wire_commands(data, false)
        .filter_map(|command| {
            Some(SlashCommand {
                name: command["name"].as_str()?.to_owned(),
                description: command["description"].as_str().unwrap_or_default().to_owned(),
                input_hint: None,
                is_skill: false,
            })
        })
        .collect()
}

/// Pi names skills `skill:<name>`; the UI wants the bare name.
fn skills_from_wire(data: &Value) -> Vec<Skill> {
    wire_commands(data, true)
        .filter_map(|command| {
            let name = command["name"].as_str()?;
            Some(Skill {
                name: name.strip_prefix("skill:").unwrap_or(name).to_owned(),
                description: command["description"].as_str().unwrap_or_default().to_owned(),
                path: command["sourceInfo"]["path"].as_str().map(str::to_owned),
                enabled: true,
            })
        })
        .collect()
}

/// Pi expands only a leading `/skill:<name>`, so other selections are read from their files.
fn skill_prompt(prompt: &str, skills: &[SkillRef]) -> String {
    let mut paths: Vec<&str> = Vec::new();
    for skill in skills {
        if !paths.contains(&skill.path.as_str()) {
            paths.push(&skill.path);
        }
    }
    if let [only] = skills
        && let Some(rest) = prompt.trim_start().strip_prefix('$')
        && let Some(rest) = rest.strip_prefix(only.name.as_str())
        && rest.chars().next().is_none_or(char::is_whitespace)
    {
        return format!("/skill:{}{rest}", only.name);
    }
    if paths.is_empty() {
        return prompt.to_owned();
    }
    format!("{prompt}\n\nBefore you start, read and follow these skill files: {}.", paths.join(", "))
}

fn thinking_arg(level: ReasoningLevel) -> Option<&'static str> {
    Some(match level {
        ReasoningLevel::Minimal => "minimal",
        ReasoningLevel::Low => "low",
        ReasoningLevel::Medium => "medium",
        ReasoningLevel::High => "high",
        ReasoningLevel::XHigh => "xhigh",
        ReasoningLevel::Max | ReasoningLevel::Ultra | ReasoningLevel::Ultracode => "max",
        ReasoningLevel::Ultrathink => return None,
    })
}

/// Ids are `provider/model`, the form `--model` accepts.
fn models_from_wire(data: &Value) -> Vec<Model> {
    data["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|model| {
            let id = model["id"].as_str()?;
            let provider = model["provider"].as_str()?;
            let reasoning = model["reasoning"].as_bool().unwrap_or(false);
            Some(Model {
                id: format!("{provider}/{id}"),
                label: model["name"].as_str().unwrap_or(id).to_owned(),
                description: Some(provider.to_owned()),
                reasoning_levels: if reasoning {
                    REASONING_LEVELS.to_vec()
                } else {
                    Vec::new()
                },
                options: Vec::new(),
            })
        })
        .collect()
}

/// A running `pi --mode rpc` child with LF-framed JSON on both pipes.
struct Rpc {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr: StderrTail,
}

impl Rpc {
    fn start(mut cmd: Command) -> Result<Self, HarnessError> {
        let mut child = cmd.spawn().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                HarnessError::NotInstalled(INSTALL_HINT.into())
            } else {
                HarnessError::Io(error)
            }
        })?;
        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| HarnessError::Protocol("pi has no stdout".into()))?;
        let stderr = StderrTail::default();
        if let Some(mut pipe) = child.stderr.take() {
            let tail = stderr.clone();
            tokio::spawn(async move {
                let mut text = String::new();
                let mut buf = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut buf).await {
                    if n == 0 {
                        break;
                    }
                    text.push_str(&String::from_utf8_lossy(&buf[..n]));
                    while let Some(at) = text.find('\n') {
                        tail.push(&text[..at]);
                        text.drain(..=at);
                    }
                }
                tail.push(&text);
            });
        }
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr,
        })
    }

    async fn send(&mut self, command: Value) -> Result<(), HarnessError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| HarnessError::Protocol("pi stdin is closed".into()))?;
        let mut line = command.to_string();
        line.push('\n');
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    /// `None` once pi closed stdout. Splits only on LF: JSON strings may hold U+2028.
    async fn next_record(&mut self) -> Option<Value> {
        loop {
            let mut line = Vec::new();
            match self.stdout.read_until(b'\n', &mut line).await {
                Ok(0) | Err(_) => return None,
                Ok(_) => {}
            }
            if let Ok(record) = serde_json::from_slice::<Value>(line.trim_ascii()) {
                return Some(record);
            }
        }
    }

    /// Waits for the response to `id` and returns its `data`; other records are dropped.
    async fn response(&mut self, id: &str) -> Result<Value, HarnessError> {
        while let Some(record) = self.next_record().await {
            if record["type"] == "response" && record["id"] == id {
                return if record["success"].as_bool().unwrap_or(false) {
                    Ok(record["data"].clone())
                } else {
                    Err(HarnessError::Protocol(
                        record["error"]
                            .as_str()
                            .unwrap_or("pi command failed")
                            .to_owned(),
                    ))
                };
            }
        }
        Err(self.exited().await)
    }

    async fn exited(&mut self) -> HarnessError {
        let status = self.child.wait().await.ok();
        HarnessError::Protocol(crate::crash_message("pi", status, &self.stderr))
    }

    async fn shutdown(&mut self) {
        self.stdin = None;
        crate::shutdown_child(&mut self.child, KILL_GRACE).await;
    }
}

type AnswerFuture =
    Pin<Box<dyn Future<Output = (String, &'static str, Option<Vec<UserInputAnswer>>)> + Send>>;

async fn run_session(
    mut rpc: Rpc,
    request: RunRequest,
    mut controls: RunControls,
    event_tx: mpsc::Sender<Result<AgentEvent, HarnessError>>,
) {
    let (status, error, session_id) = drive(&mut rpc, &request, &mut controls, &event_tx).await;
    rpc.shutdown().await;
    let _ = event_tx
        .send(Ok(AgentEvent::Done {
            status,
            result: None,
            error,
            session_id,
        }))
        .await;
}

async fn drive(
    rpc: &mut Rpc,
    request: &RunRequest,
    controls: &mut RunControls,
    event_tx: &mpsc::Sender<Result<AgentEvent, HarnessError>>,
) -> (DoneStatus, Option<String>, Option<String>) {
    macro_rules! emit {
        ($event:expr) => {
            if event_tx.send(Ok($event)).await.is_err() {
                return (DoneStatus::Interrupted, None, None);
            }
        };
    }
    macro_rules! fail {
        ($error:expr, $session:expr) => {
            return (DoneStatus::Errored, Some($error.to_string()), $session)
        };
    }

    // The state response comes before any event, so SessionStarted always leads.
    if let Err(error) = rpc.send(json!({"id": "state", "type": "get_state"})).await {
        fail!(error, None);
    }
    let state = match rpc.response("state").await {
        Ok(state) => state,
        Err(error) => fail!(error, None),
    };
    let session_id = state["sessionId"].as_str().map(str::to_owned);
    let mut context_window = state["model"]["contextWindow"].as_u64();
    let model = match (
        state["model"]["provider"].as_str(),
        state["model"]["id"].as_str(),
    ) {
        (Some(provider), Some(id)) => format!("{provider}/{id}"),
        _ => request.model.clone().unwrap_or_default(),
    };
    let mut message_id = uuid::Uuid::new_v4().to_string();
    emit!(AgentEvent::SessionStarted {
        harness: HarnessId::Pi,
        model,
        tools: Vec::new(),
        cwd: request.cwd.clone(),
        session_id: session_id.clone().unwrap_or_default(),
        assistant_message_id: message_id.clone(),
    });

    if let Some(("compact", instructions)) = crate::leading_command(&request.prompt) {
        // Not in get_commands: pi's RPC exposes compaction as its own command.
        let mut compact = json!({"id": "compact", "type": "compact"});
        if !instructions.is_empty() {
            compact["customInstructions"] = json!(instructions);
        }
        if let Err(error) = rpc.send(compact).await {
            fail!(error, session_id);
        }
        let (status, error, text) = tokio::select! {
            result = rpc.response("compact") => match result {
                Ok(data) => {
                    let before = data["tokensBefore"].as_u64().unwrap_or_default();
                    (DoneStatus::Completed, None, format!("Context compacted (was {before} tokens)."))
                }
                Err(error) => (DoneStatus::Errored, Some(error.to_string()), String::new()),
            },
            _ = controls.interrupt.cancelled() => {
                if let Err(error) = rpc.send(json!({"type": "abort_compaction"})).await {
                    log::warn!("pi abort_compaction failed: {error}");
                }
                (DoneStatus::Interrupted, None, String::new())
            }
        };
        if !text.is_empty() {
            emit!(AgentEvent::TextDelta { text });
        }
        return (status, error, session_id);
    }

    let prompt = json!({
        "id": "prompt",
        "type": "prompt",
        "message": skill_prompt(&request.prompt, &request.skills),
        "images": images(&request.attachments),
    });
    if let Err(error) = rpc.send(prompt).await {
        fail!(error, session_id);
    }

    let mut steering_open = true;
    let mut users_seen = 0u32;
    let mut last_error: Option<String> = None;
    let mut interrupted = false;
    let mut deadline: Option<tokio::time::Instant> = None;
    let mut inputs: FuturesUnordered<AnswerFuture> = FuturesUnordered::new();

    loop {
        tokio::select! {
            record = rpc.next_record() => {
                let Some(record) = record else {
                    let error = rpc.exited().await;
                    fail!(error, session_id);
                };
                match record["type"].as_str().unwrap_or_default() {
                    "response" if record["id"] == "prompt" && record["success"] == false => {
                        fail!(record["error"].as_str().unwrap_or("pi rejected the prompt"), session_id);
                    }
                    "agent_settled" => {
                        let status = if record["aborted"].as_bool().unwrap_or(false) || interrupted {
                            DoneStatus::Interrupted
                        } else if last_error.is_some() {
                            DoneStatus::Errored
                        } else {
                            DoneStatus::Completed
                        };
                        return (status, last_error, session_id);
                    }
                    "extension_ui_request" => {
                        if let Some((id, method, questions)) = ui_questions(&record) {
                            let answer = (controls.request_input)(questions.clone());
                            emit!(AgentEvent::InputRequested { request_id: id.clone(), questions });
                            inputs.push(Box::pin(async move { (id, method, answer.await.ok()) }));
                        }
                    }
                    _ => {
                        let before = users_seen;
                        let events = translate(&record, &mut users_seen, &mut context_window, &mut last_error);
                        if users_seen > before && users_seen > 1 {
                            let previous = std::mem::replace(&mut message_id, uuid::Uuid::new_v4().to_string());
                            emit!(AgentEvent::Steered {
                                assistant_message_id: Some(previous),
                                next_assistant_message_id: Some(message_id.clone()),
                            });
                        }
                        for event in events {
                            emit!(event);
                        }
                    }
                }
            }
            steer = controls.steering.recv(), if steering_open && !interrupted => match steer {
                Some(steer) => {
                    let command = json!({
                        "type": "steer",
                        "message": skill_prompt(&steer.prompt, &steer.skills),
                        "images": images(&steer.attachments),
                    });
                    if let Err(error) = rpc.send(command).await {
                        fail!(error, session_id);
                    }
                }
                None => steering_open = false,
            },
            Some((id, method, answer)) = inputs.next(), if !inputs.is_empty() => {
                let reply = ui_reply(&id, method, answer.as_deref());
                emit!(AgentEvent::InputResolved { request_id: id });
                if let Err(error) = rpc.send(reply).await {
                    fail!(error, session_id);
                }
            }
            _ = controls.interrupt.cancelled(), if !interrupted => {
                interrupted = true;
                deadline = Some(tokio::time::Instant::now() + INTERRUPT_GRACE);
                // Cancel pending dialogs so extensions don't hang the abort.
                let _ = rpc.send(json!({"type": "abort"})).await;
            }
            _ = tokio::time::sleep_until(deadline.unwrap_or_else(tokio::time::Instant::now)), if deadline.is_some() => {
                return (DoneStatus::Interrupted, None, session_id);
            }
        }
    }
}

fn images(paths: &[String]) -> Vec<Value> {
    paths
        .iter()
        .filter_map(|path| {
            let bytes = std::fs::read(path).ok()?;
            let mime = match path.rsplit('.').next()?.to_ascii_lowercase().as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "webp" => "image/webp",
                _ => return None,
            };
            Some(json!({
                "type": "image",
                "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                "mimeType": mime,
            }))
        })
        .collect()
}

/// Dialog requests become questions; fire-and-forget methods return `None`.
fn ui_questions(record: &Value) -> Option<(String, &'static str, Vec<UserInputQuestion>)> {
    let id = record["id"].as_str()?.to_owned();
    let title = record["title"].as_str().unwrap_or("Pi").to_owned();
    let (method, question) = match record["method"].as_str()? {
        "select" => {
            let options = record["options"]
                .as_array()?
                .iter()
                .filter_map(|option| option.as_str().map(str::to_owned))
                .collect();
            ("select", (title.clone(), options, None, false))
        }
        "confirm" => {
            let text = record["message"].as_str().unwrap_or(&title).to_owned();
            (
                "confirm",
                (text, vec!["Yes".into(), "No".into()], None, false),
            )
        }
        "input" => ("input", (title.clone(), Vec::new(), None, false)),
        "editor" => (
            "editor",
            (
                title.clone(),
                Vec::new(),
                record["prefill"].as_str().map(str::to_owned),
                true,
            ),
        ),
        _ => return None,
    };
    let (text, options, prefill, multiline) = question;
    Some((
        id.clone(),
        method,
        vec![UserInputQuestion {
            id,
            header: title,
            question: text,
            options,
            multi_select: false,
            prefill,
            multiline,
        }],
    ))
}

fn ui_reply(id: &str, method: &str, answer: Option<&[UserInputAnswer]>) -> Value {
    let label = answer
        .and_then(|a| a.first())
        .and_then(|a| a.labels.first());
    match (label, method) {
        (None, _) => json!({"type": "extension_ui_response", "id": id, "cancelled": true}),
        (Some(label), "confirm") => {
            json!({"type": "extension_ui_response", "id": id, "confirmed": label == "Yes"})
        }
        (Some(label), _) => json!({"type": "extension_ui_response", "id": id, "value": label}),
    }
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|part| part["type"] == "text")
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn tool_call(name: &str, args: &Value) -> ToolCall {
    let text = |key: &str| args[key].as_str().map(str::to_owned);
    let path = || text("path").unwrap_or_default();
    match name {
        "bash" => ToolCall::Exec {
            command: text("command").unwrap_or_default(),
        },
        "read" => ToolCall::ReadFile { path: path() },
        "write" => ToolCall::WriteFile {
            path: path(),
            content: text("content"),
        },
        "edit" => {
            let first = &args["edits"][0];
            ToolCall::EditFile {
                path: path(),
                old_string: first["oldText"].as_str().map(str::to_owned),
                new_string: first["newText"].as_str().map(str::to_owned),
            }
        }
        "grep" => ToolCall::Search {
            pattern: text("pattern").unwrap_or_default(),
            path: text("path"),
        },
        "find" => ToolCall::Glob {
            pattern: text("pattern").unwrap_or_default(),
        },
        _ => ToolCall::Unknown {
            name: name.to_owned(),
            input: Some(args.clone()),
        },
    }
}

/// Maps one pi session event; `settled`/dialog records are handled by the caller.
fn translate(
    record: &Value,
    users_seen: &mut u32,
    context_window: &mut Option<u64>,
    last_error: &mut Option<String>,
) -> Vec<AgentEvent> {
    match record["type"].as_str().unwrap_or_default() {
        "message_start" => {
            match record["message"]["role"].as_str() {
                Some("user") => *users_seen += 1,
                Some("assistant") => *last_error = None,
                _ => {}
            }
            Vec::new()
        }
        "message_update" => {
            let update = &record["assistantMessageEvent"];
            let delta = update["delta"].as_str().unwrap_or_default().to_owned();
            match update["type"].as_str() {
                Some("text_delta") => vec![AgentEvent::TextDelta { text: delta }],
                Some("thinking_delta") => vec![AgentEvent::ReasoningDelta { text: delta }],
                _ => Vec::new(),
            }
        }
        "message_end" if record["message"]["role"] == "assistant" => {
            let message = &record["message"];
            if message["stopReason"] == "error" {
                *last_error = Some(
                    message["errorMessage"]
                        .as_str()
                        .unwrap_or("pi request failed")
                        .to_owned(),
                );
            }
            let usage = &message["usage"];
            let mut events = Vec::new();
            if let (Some(input), Some(output)) = (usage["input"].as_u64(), usage["output"].as_u64())
            {
                events.push(AgentEvent::Usage {
                    input_tokens: input,
                    output_tokens: output,
                });
            }
            if let Some(total) = usage["totalTokens"].as_u64().filter(|total| *total > 0) {
                events.push(AgentEvent::ContextUsage {
                    tokens: Some(total),
                    window: *context_window,
                });
            }
            events
        }
        "tool_execution_start" => vec![AgentEvent::ToolCall {
            id: record["toolCallId"].as_str().unwrap_or_default().to_owned(),
            call: tool_call(
                record["toolName"].as_str().unwrap_or_default(),
                &record["args"],
            ),
        }],
        "tool_execution_end" => vec![AgentEvent::ToolResult {
            id: record["toolCallId"].as_str().unwrap_or_default().to_owned(),
            is_error: record["isError"].as_bool().unwrap_or(false),
            output: Some(text_of(&record["result"]["content"])).filter(|text| !text.is_empty()),
            diff: None,
        }],
        "compaction_start" => vec![AgentEvent::Compacting { active: true }],
        "compaction_end" => {
            let mut events = vec![AgentEvent::Compacting { active: false }];
            if record["result"].is_object() {
                events.push(AgentEvent::Compacted {
                    tokens: record["result"]["estimatedTokensAfter"].as_u64(),
                    manual: record["reason"] == "manual",
                });
            }
            events
        }
        "auto_retry_end" if record["success"] == false => {
            *last_error = record["finalError"].as_str().map(str::to_owned);
            Vec::new()
        }
        "session_info_changed" => record["name"]
            .as_str()
            .map(|title| AgentEvent::TitleChanged {
                title: title.to_owned(),
            })
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(records: &[Value]) -> (Vec<AgentEvent>, u32, Option<String>) {
        let (mut users, mut window, mut error) = (0, Some(1000), None);
        let events = records
            .iter()
            .flat_map(|record| translate(record, &mut users, &mut window, &mut error))
            .collect();
        (events, users, error)
    }

    #[test]
    fn skills_use_source_path_and_prompt_is_expanded() {
        let data = json!({"commands": [
            {"name": "skill:demo", "description": "d", "source": "skill", "sourceInfo": {"path": "/s/demo/SKILL.md"}},
            {"name": "mcp", "description": "m", "source": "extension", "sourceInfo": {"path": "/e.ts"}},
        ]});
        let skills = skills_from_wire(&data);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "demo");
        assert_eq!(skills[0].path.as_deref(), Some("/s/demo/SKILL.md"));
        assert_eq!(commands_from_wire(&data).len(), 1);

        let reference = SkillRef { name: "demo".into(), path: "/s/demo/SKILL.md".into() };
        assert_eq!(skill_prompt("$demo fix it", std::slice::from_ref(&reference)), "/skill:demo fix it");
        assert!(skill_prompt("use $demo", &[reference]).contains("/s/demo/SKILL.md"));
    }

    #[test]
    fn translates_text_tools_and_errors() {
        let (events, users, error) = run(&[
            json!({"type": "message_start", "message": {"role": "user"}}),
            json!({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hi"}}),
            json!({"type": "tool_execution_start", "toolCallId": "t", "toolName": "bash", "args": {"command": "ls"}}),
            json!({"type": "tool_execution_end", "toolCallId": "t", "isError": false,
                   "result": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}),
            json!({"type": "message_end", "message": {"role": "assistant", "stopReason": "error",
                   "errorMessage": "boom", "usage": {"input": 3, "output": 4, "totalTokens": 7}}}),
        ]);
        assert_eq!(users, 1);
        assert_eq!(error.as_deref(), Some("boom"));
        assert_eq!(events[0], AgentEvent::TextDelta { text: "hi".into() });
        assert_eq!(
            events[1],
            AgentEvent::ToolCall {
                id: "t".into(),
                call: ToolCall::Exec {
                    command: "ls".into()
                }
            }
        );
        assert_eq!(
            events[2],
            AgentEvent::ToolResult {
                id: "t".into(),
                is_error: false,
                output: Some("a\nb".into()),
                diff: None
            }
        );
        assert_eq!(
            events[3],
            AgentEvent::Usage {
                input_tokens: 3,
                output_tokens: 4
            }
        );
        assert_eq!(
            events[4],
            AgentEvent::ContextUsage {
                tokens: Some(7),
                window: Some(1000)
            }
        );
    }

    #[test]
    fn models_are_provider_qualified_and_dialogs_round_trip() {
        let data =
            json!({"models": [{"id": "m", "name": "M", "provider": "p", "reasoning": true}]});
        let models = models_from_wire(&data);
        assert_eq!(models[0].id, "p/m");
        assert!(!models[0].reasoning_levels.is_empty());

        let request = json!({"type": "extension_ui_request", "id": "u", "method": "confirm", "title": "T", "message": "Sure?"});
        let (id, method, questions) = ui_questions(&request).unwrap();
        assert_eq!((id.as_str(), method), ("u", "confirm"));
        assert_eq!(questions[0].options, ["Yes", "No"]);
        let yes = [UserInputAnswer {
            question_id: "u".into(),
            labels: vec!["Yes".into()],
        }];
        assert_eq!(ui_reply("u", method, Some(&yes))["confirmed"], true);
        assert_eq!(ui_reply("u", method, None)["cancelled"], true);
        assert!(ui_questions(&json!({"id": "n", "method": "notify"})).is_none());
    }

    /// Needs a logged-in `pi`: `cargo test -p agent_harness pi:: -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn live_pi_lists_models_and_answers() {
        let harness = PiHarness::new();
        let models = harness.models().await.unwrap();
        println!(
            "{} models, first: {:?}",
            models.len(),
            models.first().map(|m| &m.id)
        );
        let (_steer_tx, steering) = mpsc::channel(1);
        let (_perm_tx, permission) = tokio::sync::watch::channel(crate::PermissionMode::Auto);
        let controls = RunControls {
            request_input: Box::new(|_| tokio::sync::oneshot::channel().1),
            steering,
            interrupt: crate::CancellationToken::new(),
            permission,
        };
        let request = RunRequest {
            prompt: "Reply with the single word: pong".into(),
            model: models.first().map(|m| m.id.clone()),
            reasoning: None,
            model_options: Default::default(),
            cwd: std::env::temp_dir().display().to_string(),
            permission: crate::PermissionMode::Auto,
            resume: None,
            fork: None,
            attachments: Vec::new(),
            skills: Vec::new(),
            auto_compact_tokens: None,
            chrome: false,
            drop_prompt: None,
        };
        let mut stream = harness.run(request, controls).await.unwrap();
        let mut text = String::new();
        let mut done = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                AgentEvent::TextDelta { text: delta } => text.push_str(&delta),
                AgentEvent::Done { status, error, .. } => done = Some((status, error)),
                other => println!("{other:?}"),
            }
        }
        println!("text: {text:?} done: {done:?}");
        assert_eq!(done.map(|d| d.0), Some(DoneStatus::Completed));
        assert!(text.to_lowercase().contains("pong"));
    }
}
