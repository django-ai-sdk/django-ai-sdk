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


class TestReturnedErrors:
    """A tool that returns {"error": ...} failed, as one that raises does."""

    @pytest.mark.parametrize(
        ("result", "error", "failed"),
        [
            ('{"error": "No client \'C-032\'."}', False, True),
            ("{'error': 'written as a Python repr'}", False, True),
            ('{"error": ""}', False, False),
            ('{"code": "C-104"}', False, False),
            ("No such file", False, False),
            ("Connection refused", True, True),
        ],
    )
    def test_tool_failed(self, result: str, error: bool, failed: bool) -> None:
        from django_ai_sdk.agents.tool_agent import tool_failed

        call = ToolCallResult(
            result=result, origin=ToolCall(tool_name="t", arguments={}), error=error
        )
        assert tool_failed(call) is failed

    def test_a_json_result_with_booleans_is_parsed(self) -> None:
        from django_ai_sdk.adapters.base import parse_tool_output

        assert parse_tool_output('{"readable_whole": true, "about": null}') == {
            "readable_whole": True,
            "about": None,
        }
        assert parse_tool_output("{'kept': 'repr'}") == {"kept": "repr"}
        assert parse_tool_output("plain text") == "plain text"

    def test_a_stored_result_says_it_failed(self) -> None:
        from django_ai_sdk.adapters.base import stored_tool_output

        returned = ToolCallResult(
            result='{"error": "No client"}', origin=ToolCall(tool_name="t", arguments={}), error=False
        )
        assert stored_tool_output(returned) == {
            "result": {"error": "No client"},
            "origin": {"tool_name": "t", "arguments": {}, "id": None, "extra": None},
            "error": True,
        }
        ok = ToolCallResult(result='{"code": "C-104"}', origin=returned.origin, error=False)
        assert stored_tool_output(ok)["error"] is False

    def test_a_failed_call_is_no_grounding(self) -> None:
        from django_ai_sdk.oversight.types import ToolCall as OversightCall
        from django_ai_sdk.oversight.types import TurnContext

        context = TurnContext(
            tool_calls=[
                OversightCall(name="lookup", result={"error": "No client"}, error=True),
                OversightCall(name="lookup", result={"code": "C-104"}),
            ]
        )
        assert [source.content for source in context.grounding()] == ['{"code": "C-104"}']
