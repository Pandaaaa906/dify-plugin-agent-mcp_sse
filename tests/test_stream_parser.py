"""
Unit tests for StreamingContentParser.

Tests the parser's ability to correctly parse various content block types
from streaming LLM responses.
"""

import pytest

from output_parser.streaming_content_parser import (
    StopReason,
    StreamingContentParser,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
)
from tests.utils import mock_llm_stream


# =============================================================================
# Test Data: Import from fixtures
# =============================================================================

from tests.fixtures.streaming_samples import SAMPLES


# =============================================================================
# Test Functions
# =============================================================================

def count_results(results: list) -> dict:
    """Count the number of each block type in results."""
    return {
        "thinking": len([r for r in results if isinstance(r, ThinkingBlock)]),
        "tool_use": len([r for r in results if isinstance(r, ToolUseBlock)]),
        "text_deltas": len([r for r in results if isinstance(r, TextDelta)]),
        "stop_reason": next((r.value for r in results if isinstance(r, StopReason)), None),
    }


class TestStreamingContentParser:
    """Test suite for StreamingContentParser."""

    def _test_sample(self, sample: dict):
        """Generic test runner for a sample."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(sample["llm_response"])

        results = list(parser.feed_stream(stream))
        actual = count_results(results)
        expected = sample["expected"]

        # Handle "multiple" text_deltas expectation
        if expected.get("text_deltas") == "multiple":
            assert actual["text_deltas"] >= 1, f"Expected multiple text_deltas, got {actual['text_deltas']}"
        else:
            assert actual["text_deltas"] == expected["text_deltas"], \
                f"text_deltas mismatch: expected {expected['text_deltas']}, got {actual['text_deltas']}"

        assert actual["thinking"] == expected["thinking"], \
            f"thinking mismatch: expected {expected['thinking']}, got {actual['thinking']}"

        assert actual["tool_use"] == expected["tool_use"], \
            f"tool_use mismatch: expected {expected['tool_use']}, got {actual['tool_use']}"

        assert actual["stop_reason"] == expected["stop_reason"], \
            f"stop_reason mismatch: expected {expected['stop_reason']}, got {actual['stop_reason']}"

    def test_all_samples(self):
        """Run all samples from SAMPLES list."""
        for sample in SAMPLES:
            try:
                self._test_sample(sample)
                print(f"✓ {sample['name']}")
            except AssertionError as e:
                print(f"✗ {sample['name']}: {e}")
                raise

    def test_tool_details(self):
        """Test specific tool attributes."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(SAMPLES[1]["llm_response"])  # tool_use sample

        results = list(parser.feed_stream(stream))
        tool_blocks = [r for r in results if isinstance(r, ToolUseBlock)]

        assert len(tool_blocks) == 1
        assert tool_blocks[0].id == "tool_1"
        assert tool_blocks[0].name == "query_products"
        assert tool_blocks[0].input == {"cat_no": "C4X-10745"}

    def test_text_content(self):
        """Test that text content is correctly reconstructed."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(SAMPLES[2]["llm_response"])  # text_answer sample

        results = list(parser.feed_stream(stream))
        text_deltas = [r for r in results if isinstance(r, TextDelta)]

        full_text = "".join(d.text for d in text_deltas)
        assert "weather today is sunny" in full_text

    def test_tool_auto_id(self):
        """Test that tool without ID gets auto-generated ID."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(SAMPLES[6]["llm_response"])  # tool_no_id sample

        results = list(parser.feed_stream(stream))
        tool_blocks = [r for r in results if isinstance(r, ToolUseBlock)]

        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "simple_tool"
        assert tool_blocks[0].id.startswith("tool_")

    def test_tool_string_input(self):
        """Test that tool string input is converted to dict."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(SAMPLES[7]["llm_response"])  # tool_string_input sample

        results = list(parser.feed_stream(stream))
        tool_blocks = [r for r in results if isinstance(r, ToolUseBlock)]

        assert len(tool_blocks) == 1
        assert tool_blocks[0].input == {"raw_input": "raw string input"}

    def test_parallel_tool_names(self):
        """Test that parallel tools are all parsed."""
        parser = StreamingContentParser()
        stream = mock_llm_stream(SAMPLES[3]["llm_response"])  # parallel_tools sample

        results = list(parser.feed_stream(stream))
        tool_blocks = [r for r in results if isinstance(r, ToolUseBlock)]

        tool_names = {t.name for t in tool_blocks}
        assert tool_names == {"get_user_profile", "get_user_orders"}

    def test_parser_state_across_calls(self):
        """Test that parser maintains state across multiple feed calls."""
        parser = StreamingContentParser()

        # First chunk - incomplete JSON
        first_chunk = '{"content": [{"type": "thinking", "thinking": "Hel'
        results1 = list(parser.feed_stream(mock_llm_stream(first_chunk)))

        # Should not yield anything yet (incomplete)
        assert len([r for r in results1 if isinstance(r, ThinkingBlock)]) == 0

        # Complete the JSON
        second_chunk = 'lo"}], "stop_reason": "end_turn"}'
        results2 = list(parser.feed_stream(mock_llm_stream(second_chunk)))

        thinking_blocks = [r for r in results2 if isinstance(r, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "Hello"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise use basic execution
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        # Fallback: run tests manually
        import sys

        test_class = TestStreamingContentParser()

        # Run sample-based tests
        print("\nRunning sample tests...")
        print("-" * 50)
        try:
            test_class.test_all_samples()
            print("-" * 50)
            print("All sample tests passed!")
        except AssertionError:
            print("-" * 50)
            print("Some tests failed!")
            sys.exit(1)

        # Run additional specific tests
        print("\nRunning additional tests...")
        print("-" * 50)
        additional_tests = [
            "test_tool_details",
            "test_text_content",
            "test_tool_auto_id",
            "test_tool_string_input",
            "test_parallel_tool_names",
            "test_parser_state_across_calls",
        ]

        passed = 0
        failed = 0
        for method_name in additional_tests:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                failed += 1

        print("-" * 50)
        print(f"\nAdditional: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
