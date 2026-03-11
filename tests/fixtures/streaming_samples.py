"""
Test samples for StreamingContentParser.

Add new test cases to SAMPLES list following the format:
{
    "name": "unique_test_name",
    "llm_response": "...",  # JSON string of LLM response
    "expected": {
        "thinking": N,        # Expected count of ThinkingBlock
        "tool_use": N,        # Expected count of ToolUseBlock
        "text_deltas": N or "multiple",  # Expected count or "multiple" for streaming
        "stop_reason": "..."  # Expected stop reason value
    }
}
"""

SAMPLES = [
    # Test case 1: Simple thinking block only
    {
        "name": "simple_thinking",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "I need to analyze this request carefully."}
  ],
  "stop_reason": "end_turn"
}""",
        "expected": {"thinking": 1, "tool_use": 0, "text_deltas": 0, "stop_reason": "end_turn"}
    },

    # Test case 2: Tool use block
    {
        "name": "tool_use",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "Let me query the product database."},
    {"type": "tool_use", "id": "tool_1", "name": "query_products", "input": {"cat_no": "C4X-10745"}}
  ],
  "stop_reason": "tool_use"
}""",
        "expected": {"thinking": 1, "tool_use": 1, "text_deltas": 0, "stop_reason": "tool_use"}
    },

    # Test case 3: Text response (final answer)
    {
        "name": "text_answer",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "The user is asking about the weather."},
    {"type": "text", "text": "The weather today is sunny with a high of 25°C."}
  ],
  "stop_reason": "end_turn"
}""",
        "expected": {"thinking": 1, "tool_use": 0, "text_deltas": "multiple", "stop_reason": "end_turn"}
    },

    # Test case 4: Multiple tool calls (parallel)
    {
        "name": "parallel_tools",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "I need to fetch data from multiple sources."},
    {"type": "tool_use", "id": "tool_1", "name": "get_user_profile", "input": {"user_id": "123"}},
    {"type": "tool_use", "id": "tool_2", "name": "get_user_orders", "input": {"user_id": "123"}}
  ],
  "stop_reason": "tool_use"
}""",
        "expected": {"thinking": 1, "tool_use": 2, "text_deltas": 0, "stop_reason": "tool_use"}
    },

    # Test case 5: Empty text block (for tool_use scenario)
    {
        "name": "empty_text_with_tool",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "Querying database."},
    {"type": "tool_use", "id": "tool_1", "name": "search_db", "input": {"query": "test"}},
    {"type": "text", "text": ""}
  ],
  "stop_reason": "tool_use"
}""",
        "expected": {"thinking": 1, "tool_use": 1, "text_deltas": 0, "stop_reason": "tool_use"}
    },

    # Test case 6: Long text content (for testing streaming)
    {
        "name": "long_text",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "Let me provide a detailed explanation."},
    {"type": "text", "text": "This is a very long text that should be streamed in multiple chunks. It contains detailed information about the topic at hand and should demonstrate how the parser handles incremental text updates."}
  ],
  "stop_reason": "end_turn"
}""",
        "expected": {"thinking": 1, "tool_use": 0, "text_deltas": "multiple", "stop_reason": "end_turn"}
    },

    # Test case 7: Tool without ID (auto-generated ID)
    {
        "name": "tool_no_id",
        "llm_response": """{
  "content": [
    {"type": "tool_use", "name": "simple_tool", "input": {"param": "value"}}
  ],
  "stop_reason": "tool_use"
}""",
        "expected": {"thinking": 0, "tool_use": 1, "text_deltas": 0, "stop_reason": "tool_use"}
    },

    # Test case 8: Tool with string input (should be converted to dict)
    {
        "name": "tool_string_input",
        "llm_response": """{
  "content": [
    {"type": "tool_use", "id": "tool_1", "name": "process_text", "input": "raw string input"}
  ],
  "stop_reason": "tool_use"
}""",
        "expected": {"thinking": 0, "tool_use": 1, "text_deltas": 0, "stop_reason": "tool_use"}
    },

    # Test case 9: Only text, no thinking
    {
        "name": "text_only",
        "llm_response": """{
  "content": [
    {"type": "text", "text": "Here's the answer to your question."}
  ],
  "stop_reason": "end_turn"
}""",
        "expected": {"thinking": 0, "tool_use": 0, "text_deltas": "multiple", "stop_reason": "end_turn"}
    },

    # Test case 10: Multiple thinking and tool blocks
    {
        "name": "complex_multi_block",
        "llm_response": """{
  "content": [
    {"type": "thinking", "thinking": "First, let me check the weather."},
    {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "Beijing"}},
    {"type": "thinking", "thinking": "Now let me check the traffic."},
    {"type": "tool_use", "id": "tool_2", "name": "get_traffic", "input": {"city": "Beijing"}},
    {"type": "text", "text": "The weather is good but traffic is heavy."}
  ],
  "stop_reason": "end_turn"
}""",
        "expected": {"thinking": 2, "tool_use": 2, "text_deltas": "multiple", "stop_reason": "end_turn"}
    },
]
