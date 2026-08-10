"""
Tests for tool errors in adapters.
"""

import pytest

from haystack.dataclasses import ToolCall, ToolCallResult


class TestHaystackToolErrorDetection:
    """Tests for Haystack tool error detection."""

    def test_tool_call_result_error_flag_detection(self):
        """Verify ToolCallResult.error flag detection logic matches what the adapter checks."""
        success = ToolCallResult(
            result="Found 3 documents",
            origin=ToolCall(tool_name="search", arguments={}),
            error=False,
        )
        assert success.error is False

        failure = ToolCallResult(
            result="API rate limit exceeded",
            origin=ToolCall(tool_name="search", arguments={}),
            error=True,
        )
        assert failure.error is True

    def test_parse_tool_output_includes_error_flag(self):
        """Verify parse_tool_output preserves the error flag in dict output."""
        from django_ai_sdk.adapters.base import parse_tool_output

        failure = ToolCallResult(
            result="Connection refused",
            origin=ToolCall(tool_name="fetch", arguments={"url": "http://..."}),
            error=True,
        )
        parsed = parse_tool_output(failure.to_dict())
        assert parsed["error"] is True
        assert parsed["result"] == "Connection refused"
        assert parsed["origin"]["tool_name"] == "fetch"
