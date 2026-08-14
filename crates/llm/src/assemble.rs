//! Assemble semantic content from output events.

use crate::event::{ContentDelta, OutputEvent};
use crate::message::{ContentBlock, ReasoningVisibility};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct OutputAssembler {
    texts: BTreeMap<u32, String>,
    reasoning: BTreeMap<u32, String>,
    tools: BTreeMap<u32, AssemblingTool>,
    order: Vec<Slot>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Text(u32),
    Reasoning(u32),
    Tool(u32),
}

#[derive(Default)]
struct AssemblingTool {
    id: String,
    name: String,
    arguments: String,
}

impl OutputAssembler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn apply(&mut self, event: &OutputEvent) {
        match event {
            OutputEvent::ContentDelta {
                output_index,
                delta,
            } => match delta {
                ContentDelta::Text { text } => {
                    self.note(Slot::Text(*output_index));
                    self.texts.entry(*output_index).or_default().push_str(text);
                }
                ContentDelta::Reasoning { text } => {
                    self.note(Slot::Reasoning(*output_index));
                    self.reasoning
                        .entry(*output_index)
                        .or_default()
                        .push_str(text);
                }
                ContentDelta::ToolCall {
                    tool_index,
                    id,
                    name,
                    arguments_delta,
                } => {
                    self.note(Slot::Tool(*tool_index));
                    let tool = self.tools.entry(*tool_index).or_default();
                    if let Some(id) = id {
                        tool.id = id.clone();
                    }
                    if let Some(name) = name {
                        tool.name = name.clone();
                    }
                    tool.arguments.push_str(arguments_delta);
                }
                ContentDelta::Extension(_) => {}
            },
            OutputEvent::Usage(_) | OutputEvent::Wire(_) | OutputEvent::Extension(_) => {}
        }
    }

    pub(crate) fn into_content(self) -> Result<Vec<ContentBlock>, crate::error::LlmError> {
        let mut content = Vec::new();
        for slot in self.order {
            match slot {
                Slot::Text(index) => {
                    if let Some(text) = self.texts.get(&index)
                        && !text.is_empty()
                    {
                        content.push(ContentBlock::text(text));
                    }
                }
                Slot::Reasoning(index) => {
                    if let Some(text) = self.reasoning.get(&index)
                        && !text.is_empty()
                    {
                        content.push(ContentBlock::Reasoning {
                            text: text.clone(),
                            visibility: ReasoningVisibility::Trace,
                        });
                    }
                }
                Slot::Tool(index) => {
                    if let Some(tool) = self.tools.get(&index) {
                        let arguments = parse_tool_arguments(&tool.arguments)?;
                        content.push(ContentBlock::tool_call(
                            tool.id.clone(),
                            tool.name.clone(),
                            arguments,
                        ));
                    }
                }
            }
        }
        Ok(content)
    }

    fn note(&mut self, slot: Slot) {
        if !self.order.contains(&slot) {
            self.order.push(slot);
        }
    }
}

fn parse_tool_arguments(raw: &str) -> Result<serde_json::Value, crate::error::LlmError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(trimmed).map_err(|error| {
        crate::error::LlmError::invalid_response("tool-call arguments are not valid JSON")
            .with_source(error)
    })
}
