"""Types for StreamingReact strategy."""

from typing import Any

from typing_extensions import TypedDict


class ToolUseContent(TypedDict):
    """Tool use block in scratchpad."""
    type: str  # "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(TypedDict):
    """Tool result block in scratchpad."""
    type: str  # "tool_result"
    tool_use_id: str
    content: str
    is_error: bool


class AssistantScratchpadEntry(TypedDict):
    """Assistant's tool use decision."""
    role: str  # "assistant"
    content: list[ToolUseContent]


class ToolScratchpadEntry(TypedDict):
    """Tool execution results."""
    role: str  # "tool"
    content: list[ToolResultContent]


class AssistantThoughtScratchpadEntry(TypedDict):
    """Assistant's thinking content."""
    role: str  # "assistant_thought"
    content: str


# Union type for all scratchpad entries
ScratchpadEntry = AssistantScratchpadEntry | ToolScratchpadEntry | AssistantThoughtScratchpadEntry


__all__ = [
    "ToolUseContent",
    "ToolResultContent",
    "AssistantScratchpadEntry",
    "ToolScratchpadEntry",
    "AssistantThoughtScratchpadEntry",
    "ScratchpadEntry",
]
