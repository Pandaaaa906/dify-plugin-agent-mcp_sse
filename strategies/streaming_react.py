"""
StreamingReact Agent Strategy with Content Blocks.

This strategy implements a ReAct-style agent with the following improvements:
1. Content Blocks format inspired by Anthropic Claude API
2. Real-time streaming of final answers (text blocks are yielded immediately)
3. Support for parallel tool execution
4. Robust JSON parsing using json_repair with stream_stable

Key features:
- thinking blocks: Model's reasoning process (visible in logs)
- tool_use blocks: Tool invocations (parallel execution supported)
- text blocks: Final answer streamed to user in real-time
- stop_reason: Indicates why generation stopped
"""
import logging
import time
from typing import Any, Optional, cast, Generator, Mapping, List

import orjson
import pydantic
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.invoke_message import InvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.provider_config import LogMetadata
from dify_plugin.entities.tool import (
    ToolInvokeMessage,
    ToolProviderType,
)
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentStrategy,
    ToolEntity,
)
from pydantic import BaseModel

from output_parser.streaming_content_parser import (
    StopReason,
    StreamingContentParser,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
)
from prompt.streaming_content_template import STREAMING_CONTENT_PROMPT_TEMPLATES
from strategies.base import FilterHistoryMessageByModelFeaturesMixin
from strategies.types import ScratchpadEntry
from utils.mcp_client import McpClients


class StreamingReactParams(BaseModel):
    """Parameters for StreamingReact strategy."""
    query: str
    instruction: str
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    mcp_servers_config: str | None
    mcp_resources_as_tools: bool = False
    mcp_prompts_as_tools: bool = False
    maximum_iterations: int = 5

logger = logging.getLogger(__name__)
logger.addHandler(plugin_logger_handler)
logger.setLevel(logging.DEBUG)


class StreamingReactAgentStrategy(FilterHistoryMessageByModelFeaturesMixin, AgentStrategy):
    """
    StreamingReact Agent Strategy with Content Blocks.

    This strategy uses a structured JSON format for model responses:
    {
        "content": [
            {"type": "thinking", "thinking": "..."},
            {"type": "tool_use", "id": "tool_1", "name": "...", "input": {...}},
            {"type": "text", "text": "..."}
        ],
        "stop_reason": "tool_use|end_turn|max_tokens"
    }

    The text block is streamed to the user immediately as it's generated,
    providing a responsive user experience.
    """

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage | InvokeMessage, None, None]:
        """Run StreamingReact agent application."""

        # Validate parameters
        try:
            params = StreamingReactParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e
        # Initialize state
        query = params.query
        instruction = params.instruction or ""
        agent_scratchpad: List[ScratchpadEntry] = []
        iteration_step = 1
        max_iterations = params.maximum_iterations
        run_agent_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer_delivered = False

        # Prepare model
        model = params.model
        stop_sequences = []
        if model.completion_params:
            stop_sequences = model.completion_params.get("stop", [])

        # Convert tools
        tools = params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}

        # Fetch MCP tools
        mcp_clients = None
        mcp_tools = []
        mcp_tool_instances = {}
        if params.mcp_servers_config:
            try:
                servers_config = orjson.loads(params.mcp_servers_config.strip('"'))
            except orjson.JSONDecodeError as e:
                raise ValueError(f"mcp_servers_config must be valid JSON: {e}")

            mcp_clients = McpClients(
                servers_config,
                params.mcp_resources_as_tools,
                params.mcp_prompts_as_tools
            )
            mcp_tools = mcp_clients.fetch_tools()
            mcp_tool_instances = {
                tool.get("name"): tool
                for tool in mcp_tools
                if tool.get("name")
            }

        # Combine Dify and MCP tools for prompts
        prompt_messages_tools = self._init_prompt_tools(tools)
        prompt_messages_tools.extend(self._init_prompt_mcp_tools(mcp_tools))

        # Main agent loop
        while run_agent_state and iteration_step <= max_iterations:
            run_agent_state = False
            round_started_at = time.perf_counter()

            # Create round log
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={LogMetadata.STARTED_AT: round_started_at},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log
            if iteration_step == max_iterations:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            # Prepare prompt messages
            prompt_messages = self._organize_prompt_messages(
                query=query,
                instruction=instruction,
                tools=prompt_messages_tools,
                scratchpad=agent_scratchpad,
                model=model,
            )

            # Calculate max tokens
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(model.entity, prompt_messages, model.completion_params)

            # Invoke model
            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Generation",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            # Stream and parse response
            parser = StreamingContentParser()
            pending_tool_calls: list[ToolUseBlock] = []
            current_usage: Optional[LLMUsage] = None
            text_started = False
            stop_reason = None

            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop_sequences,
            )

            # Parse content using feed_stream with usage tracking
            usage_dict: dict = {}
            current_thinking = ""
            for parsed in parser.feed_stream(chunks, usage_dict):
                if isinstance(parsed, ThinkingBlock):
                    # Log final thinking when complete
                    current_thinking = parsed.thinking
                    yield self.create_log_message(
                        label="Thinking",
                        data={"thought": parsed.thinking},
                        parent=round_log,
                        status=ToolInvokeMessage.LogMessage.LogStatus.SUCCESS,
                    )

                elif isinstance(parsed, ToolUseBlock):
                    pending_tool_calls.append(parsed)

                elif isinstance(parsed, TextDelta):
                    if pending_tool_calls and not current_thinking:
                        current_thinking = parsed.full
                    elif not pending_tool_calls:
                        if not text_started:
                            text_started = True
                            final_answer_delivered = True
                        # Stream text to user immediately
                        yield self.create_text_message(parsed.text)

                elif isinstance(parsed, StopReason):
                    # Stop reason detected
                    stop_reason = parsed

            # Get usage from usage_dict
            if "usage" in usage_dict and usage_dict["usage"] is not None:
                current_usage = usage_dict["usage"]
                self.increase_usage(llm_usage, current_usage)

            # Finish model log
            yield self.finish_log_message(
                log=model_log,
                data={
                    "thought": current_thinking,
                    "stop_reason": stop_reason,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in pending_tool_calls
                    ],
                    "text_delivered": text_started,
                },
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: current_usage.total_price if current_usage else 0,
                    LogMetadata.CURRENCY: current_usage.currency if current_usage else "",
                    LogMetadata.TOTAL_TOKENS: current_usage.total_tokens if current_usage else 0,
                },
            )

            # Execute tools if any
            tool_results = []
            if pending_tool_calls:
                run_agent_state = True
                for tool_call in pending_tool_calls:
                    # Execute each tool with logging
                    yield from self._execute_single_tool(
                        tool_call=tool_call,
                        tool_instances=tool_instances,
                        mcp_clients=mcp_clients,
                        mcp_tool_instances=mcp_tool_instances,
                        parent_log=round_log,
                        tool_results=tool_results,
                    )

                # Add to scratchpad for next iteration
                agent_scratchpad.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in pending_tool_calls
                    ]
                })
                agent_scratchpad.append({
                    "role": "tool",
                    "content": tool_results
                })
            else:
                if current_thinking:
                    agent_scratchpad.append({
                        "role": "assistant_thought",
                        "content": current_thinking,
                    })
                if not final_answer_delivered:
                    # retry if llm didn't response final answer or tool use
                    run_agent_state = True

            yield self.finish_log_message(
                log=round_log,
                data={
                    "stop_reason": stop_reason,
                    "thought": current_thinking,
                    "observation": self._format_scratchpad(agent_scratchpad),
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            iteration_step += 1

        # Cleanup
        if mcp_clients:
            mcp_clients.close()

        # Final metadata
        if not final_answer_delivered:
            yield self.create_text_message("I apologize, but I was unable to complete the task.")

        yield self.create_json_message({
            "execution_metadata": {
                LogMetadata.TOTAL_PRICE: llm_usage["usage"].total_price if llm_usage["usage"] else 0,
                LogMetadata.CURRENCY: llm_usage["usage"].currency if llm_usage["usage"] else "",
                LogMetadata.TOTAL_TOKENS: llm_usage["usage"].total_tokens if llm_usage["usage"] else 0,
            }
        })

    def _organize_prompt_messages(
        self,
        query: str,
        instruction: str,
        tools: list[Any],
        scratchpad: List[ScratchpadEntry],
        model: AgentModelConfig,
    ) -> list[PromptMessage]:
        """Organize prompt messages for the model."""

        # Build tool descriptions
        tool_descriptions = []
        for tool in tools:
            if hasattr(tool, 'model_dump'):
                tool_desc = tool.model_dump(mode="json")
            else:
                tool_desc = {
                    "name": getattr(tool, 'name', 'unknown'),
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'parameters', {}),
                }
            tool_descriptions.append(tool_desc)

        # Format scratchpad as string
        scratchpad_str = self._format_scratchpad(scratchpad)
        history_messages = self._iter_cleanup_history_prompt_messages(model)

        # Build system prompt
        system_prompt = STREAMING_CONTENT_PROMPT_TEMPLATES["english"]["chat"]["prompt"]
        system_prompt = system_prompt.replace("{{instruction}}", instruction)
        system_prompt = system_prompt.replace(
            "{{tools}}",
            orjson.dumps(tool_descriptions).decode('utf-8') if tool_descriptions else "No tools available."
        )

        # Add scratchpad if exists
        if scratchpad_str:
            scratchpad_template = STREAMING_CONTENT_PROMPT_TEMPLATES["english"]["chat"]["agent_scratchpad"]
            system_prompt += scratchpad_template.replace("{{tool_results}}", scratchpad_str)

        messages: list[PromptMessage] = [
            SystemPromptMessage(content=system_prompt),
            *history_messages,
            UserPromptMessage(content=query),
        ]

        return messages

    @staticmethod
    def _format_scratchpad(scratchpad: List[ScratchpadEntry]) -> str:
        """Format scratchpad entries as a string."""
        if not scratchpad:
            return ""

        parts = []
        for entry in scratchpad:
            role = entry.get("role", "")
            content = entry.get("content", [])

            if role == "assistant_thought":
                parts.append(f"Assistant Thoughts: {content}")
            elif role == "assistant":
                parts.append("Assistant decided to use tools:")
                for block in content:
                    if block.get("type") == "tool_use":
                        parts.append(f"  - {block.get('name')} (id: {block.get('id')})")

            elif role == "tool":
                parts.append("Tool execution results:")
                for block in content:
                    if block.get("type") == "tool_result":
                        parts.append(f"  - {block.get('tool_use_id')}: {block.get('content', '')}")

        return "\n\n\n".join(parts)

    def _execute_single_tool(
        self,
        tool_call: ToolUseBlock,
        tool_instances: Mapping[str, ToolEntity],
        mcp_clients: Optional[McpClients],
        mcp_tool_instances: Mapping[str, dict],
        parent_log: Any,
        tool_results: list,
    ) -> Generator[InvokeMessage,  None, None]:
        """
        Execute a single tool call with logging.

        Yields log messages and returns the tool result.
        """
        tool_started_at = time.perf_counter()
        tool_name = tool_call.name

        # Get provider name safely
        provider_name = ""
        if tool_name in tool_instances:
            provider_name = tool_instances[tool_name].identity.provider
        else:
            provider_name = "mcp"

        tool_log = self.create_log_message(
            label=f"CALL {tool_name}",
            data={"input": tool_call.input},
            metadata={
                LogMetadata.STARTED_AT: tool_started_at,
                LogMetadata.PROVIDER: provider_name,
            },
            parent=parent_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield tool_log

        # Execute tool
        try:
            if tool_name in mcp_tool_instances and mcp_clients:
                # MCP tool
                result = mcp_clients.execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_call.input,
                )
                output = self._format_mcp_result(result)
                is_error = False
            elif tool_name in tool_instances:
                # Dify tool
                tool_instance = tool_instances[tool_name]
                tool_params = {**tool_instance.runtime_parameters, **tool_call.input}

                responses = self.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters=tool_params,
                )
                output = self._format_dify_tool_result(responses)
                is_error = False
            else:
                output = f"Error: Tool '{tool_name}' not found"
                is_error = True

        except Exception as e:
            output = f"Tool execution error: {e!s}"
            is_error = True

        # Log completion
        yield self.finish_log_message(
            log=tool_log,
            data={
                "tool_call_args": tool_call.input,
                "tool_name": tool_name,
                "output": output,
                "is_error": is_error,
            },
            metadata={
                LogMetadata.STARTED_AT: tool_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - tool_started_at,
            },
        )

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": output,
                "is_error": is_error,
            }
        )

    def _format_mcp_result(self, result: list[dict]) -> str:
        """Format MCP tool result as string."""
        if not result:
            return ""

        if len(result) == 1:
            item = result[0]
            if item.get("type") == "text":
                return item.get("text", "")
            elif item.get("type") in ("image", "video"):
                return orjson.dumps(item).decode('utf-8')
            elif item.get("type") == "resource":
                return orjson.dumps(item.get('resource', {})).decode('utf-8')

        return orjson.dumps(result).decode('utf-8')

    def _format_dify_tool_result(self, responses: list[ToolInvokeMessage]) -> str:
        """Format Dify tool responses as string."""
        result_parts = []

        for response in responses:
            if response.type == ToolInvokeMessage.MessageType.TEXT:
                text = cast(ToolInvokeMessage.TextMessage, response.message).text
                result_parts.append(text)
            elif response.type == ToolInvokeMessage.MessageType.LINK:
                link = cast(ToolInvokeMessage.TextMessage, response.message).text
                result_parts.append(f"Link: {link}")
            elif response.type in {
                ToolInvokeMessage.MessageType.IMAGE_LINK,
                ToolInvokeMessage.MessageType.IMAGE,
            }:
                result_parts.append("[Image generated and sent to user]")
            elif response.type == ToolInvokeMessage.MessageType.JSON:
                json_obj = cast(ToolInvokeMessage.JsonMessage, response.message).json_object
                result_parts.append(orjson.dumps(json_obj).decode('utf-8'))
            elif response.type == ToolInvokeMessage.MessageType.VARIABLE:
                variable = cast(ToolInvokeMessage.VariableMessage, response.message)
                result_parts.append(f"{variable.variable_name} = {variable.variable_value}")
            else:
                result_parts.append(str(response.message))

        return "\n\n".join(result_parts)

    @staticmethod
    def _init_prompt_mcp_tools(mcp_tools: list[dict]) -> list[Any]:
        """Initialize prompt message tools from MCP tools."""
        from dify_plugin.entities.model.message import PromptMessageTool

        prompt_messages_tools = []
        for tool in mcp_tools:
            prompt_message = PromptMessageTool(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema", {}),
            )
            prompt_messages_tools.append(prompt_message)
        return prompt_messages_tools
