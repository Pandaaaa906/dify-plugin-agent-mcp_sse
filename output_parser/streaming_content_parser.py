"""
Streaming Content Blocks Parser for StreamingReact Agent Strategy.

This parser implements real-time streaming of content blocks inspired by
Anthropic Claude API's message format. Key features:
- Early detection of text blocks for immediate streaming to user
- Support for parallel tool_use blocks
- Stable JSON parsing using json_repair with stream_stable option
- Stop reason detection

Reference: https://docs.anthropic.com/en/api/messages-streaming
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import json_repair
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities.model.llm import LLMResultChunk

logger = logging.getLogger(__name__)
logger.addHandler(plugin_logger_handler)


class StopReason(Enum):
    """Reason why the model stopped generating."""
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"


@dataclass
class ThinkingBlock:
    """Represents a thinking/reasoning block from the model."""
    thinking: str


@dataclass
class ThinkingDelta:
    """Incremental update for thinking block."""
    thinking: str


@dataclass
class ToolUseBlock:
    """Represents a tool invocation block."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class TextDelta:
    """Represents an incremental text update for streaming final answer."""
    text: str


class StreamingContentParser:
    """
    Parser for streaming content blocks in Claude API style format.

    Memory-efficient design:
    - No storage of parsed blocks, only tracks completion state
    - Streaming blocks (text/thinking) yield deltas immediately
    - Non-streaming blocks (tool_use) yield only when complete

    Usage:
        parser = StreamingContentParser()
        for chunk in llm_response:
            for item in parser.feed(chunk.delta.message.content):
                if isinstance(item, (TextDelta, ThinkingDelta)):
                    # Stream to user immediately
                    yield create_text_message(item.text or item.thinking)
                elif isinstance(item, ToolUseBlock):
                    # Collect for execution
                    tool_calls.append(item)
                elif isinstance(item, StopReason):
                    # Handle stop reason
                    handle_stop(item)
    """

    def __init__(self):
        self.buffer: str = ""
        self.stop_reason: Optional[StopReason] = None
        self.start_idx: int = 0
        # self._completed_indices: set[int] = set()  # Blocks fully yielded
        self._text_progress: dict[int, int] = {}  # text block index -> yielded length
        self._thinking_progress: dict[int, int] = {}  # thinking block index -> yielded length
        self._auto_tool_id_counter: int = 0
        self.all_contents_processed = False

    def _generate_tool_id(self) -> str:
        """Generate an auto-incrementing tool ID when model doesn't provide one."""
        self._auto_tool_id_counter += 1
        return f"tool_{self._auto_tool_id_counter}"

    def feed(self, text_chunk: str) -> Generator[Union[ThinkingBlock, ToolUseBlock, TextDelta, StopReason, None], None, None]:
        """
        Feed a text chunk into the parser and yield parsed items.

        Args:
            text_chunk: A chunk of text from the LLM stream.

        Yields:
            ThinkingBlock: When a thinking block is newly parsed (complete).
            ThinkingDelta: Incremental thinking updates.
            ToolUseBlock: When a tool_use block is complete.
            TextDelta: Incremental text updates for streaming final answer.
            StopReason: When the stop_reason is detected.
        """
        self.buffer += text_chunk

        if self.all_contents_processed:
            return

        # Attempt to parse with json_repair for stable results
        try:
            result = json_repair.loads(
                self.buffer,
                stream_stable=True,
                skip_json_loads=True,
            )
        except Exception:
            # Partial JSON not yet parseable
            return

        # skip malformed json
        if (
            not isinstance(result, dict)
            or "content" not in result
            or not isinstance((content := result["content"]), list)
            or not content
        ):
            return

        # TODO Check for stop_reason
        if "stop_reason" in result and self.stop_reason is None:
            reason_str = result["stop_reason"]
            try:
                self.stop_reason = StopReason(reason_str)
            except ValueError:
                return

        # Process each content block
        while self.start_idx < len(content):
            block = content[self.start_idx]
            if not isinstance(block, dict):
                return
            block_type = block.get("type")
            # 如果存在下一个block，则现在这个block是完整的
            block_is_complete = (self.start_idx + 1 < len(content)) or self.stop_reason is not None

            if block_is_complete and block_type == "thinking":
                if b := self._handle_thinking_block(self.start_idx, block):
                    yield b
            elif block_type == "text":
                yield from self._handle_text_block(self.start_idx, block)
                if not block_is_complete:
                    break
            elif block_is_complete and block_type == "tool_use":
                yield from self._handle_tool_use_block(block)
            else:
                logger.debug(f"Unknown content block type: {block_type}")
                break

            if block_is_complete:
                self.start_idx += 1

        yield self.stop_reason

    def _handle_thinking_block(self, idx: int, block: dict) -> ThinkingBlock | None:
        """Handle thinking block with incremental streaming support."""
        thinking_text = block.get("thinking", "")
        # prev_len = self._thinking_progress.get(idx, 0)

        # if len(thinking_text) > prev_len:
        #     delta = thinking_text[prev_len:]
        #     self._thinking_progress[idx] = len(thinking_text)
        #     return ThinkingDelta(thinking=delta)
        if thinking_text:
            return ThinkingBlock(thinking=thinking_text)
        return None

    def _handle_text_block(self, idx: int, block: dict) -> Generator[TextDelta, None, None]:
        """Handle text block with incremental streaming."""
        text_content = block.get("text", "")
        prev_len = self._text_progress.get(idx, 0)

        if len(text_content) > prev_len:
            delta = text_content[prev_len:]
            self._text_progress[idx] = len(text_content)
            yield TextDelta(text=delta)

    def _handle_tool_use_block(self, block: dict) -> Generator[ToolUseBlock, None, None]:
        """Handle tool_use block - only yield when complete."""
        # Check if block has required fields
        tool_name = block.get("name")
        tool_input = block.get("input")

        # tool_use is complete when name and input are present
        if not tool_name or tool_input is None:
            return

        tool_id = block.get("id") or self._generate_tool_id()

        # Ensure tool_input is a dict
        if isinstance(tool_input, str):
            try:
                tool_input = json_repair.loads(tool_input, skip_json_loads=True)
            except Exception:
                tool_input = {"raw_input": tool_input}
        elif not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}

        yield ToolUseBlock(
            id=tool_id,
            name=tool_name,
            input=tool_input
        )

    def feed_stream(
        self,
        llm_response: Generator[LLMResultChunk, None, None],
        usage_dict: dict | None = None
    ) -> Generator[Union[ThinkingDelta, ToolUseBlock, TextDelta, StopReason], None, None]:
        """
        Process an LLM response stream and yield parsed items.

        Args:
            llm_response: Generator of LLM result chunks.
            usage_dict: Optional dict to collect usage statistics.

        Yields:
            Parsed content blocks and stop reason.
        """
        for chunk in llm_response:
            if not isinstance(chunk, LLMResultChunk):
                continue

            delta = chunk.delta
            if not delta or not delta.message:
                continue

            if usage_dict is not None and delta.usage:
                usage_dict["usage"] = delta.usage

            content = delta.message.content
            if not isinstance(content, str):
                continue

            yield from self.feed(content)
