"""Scoring for tool-calling cases: right tool, valid arguments, expected values."""

from __future__ import annotations

from typing import Final

from quantdiff.errors import SuiteError
from quantdiff.metrics.jsonschema import validate
from quantdiff.types import CaseOutcome, ChatResult, JSONValue, TaskCase, ToolCall

_PREVIEW_CHARS: Final = 60


def check_tool_case(case: TaskCase, result: ChatResult) -> CaseOutcome:
    """Pass when the first tool call matches the case's expectations.

    A case with no `expected_tool` passes only when the model answers without calling
    any tool. Raises SuiteError if the expected tool is not among the case's tools.
    """
    reason = _failure_reason(case, result)
    return CaseOutcome(case_id=case.id, kind=case.kind, passed=reason is None, reason=reason or "")


def arguments_match(expected: JSONValue, actual: JSONValue) -> bool:
    """Lenient equality for tool arguments.

    Numbers compare numerically (1 equals 1.0, but never equals true), strings compare
    after strip() and casefold(), lists compare element by element, and objects match
    when every expected key is present with a matching value.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and expected == actual
    if isinstance(expected, str):
        return isinstance(actual, str) and _fold(expected) == _fold(actual)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(map(arguments_match, expected, actual))
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and arguments_match(value, actual[key]) for key, value in expected.items()
        )
    return expected is None and actual is None


def _failure_reason(case: TaskCase, result: ChatResult) -> str | None:
    expected_tool = case.expected_tool
    if expected_tool is None:
        return _direct_answer_failure(result)
    if not result.tool_calls:
        return f"made no tool call, expected {expected_tool}"

    call = result.tool_calls[0]
    if call.name != expected_tool:
        return f"called {call.name}, expected {expected_tool}"
    if call.arguments is None:
        return f"{call.name} arguments are not a JSON object: {_preview(call.raw_arguments)}"
    schema_errors = validate(call.arguments, _parameters_schema(case, expected_tool))
    if schema_errors:
        extra = f" (and {len(schema_errors) - 1} more)" if len(schema_errors) > 1 else ""
        return f"{call.name} arguments violate the schema: {schema_errors[0]}{extra}"
    return _argument_mismatch(call, case.expected_arguments or {})


def _direct_answer_failure(result: ChatResult) -> str | None:
    if result.tool_calls:
        return f"called {result.tool_calls[0].name}, expected a direct answer"
    return None


def _parameters_schema(case: TaskCase, tool_name: str) -> dict[str, JSONValue]:
    for tool in case.tools:
        if tool.name == tool_name:
            return tool.parameters
    raise SuiteError(f"case {case.id!r} expects tool {tool_name!r} but does not define it")


def _argument_mismatch(call: ToolCall, expected: dict[str, JSONValue]) -> str | None:
    arguments = call.arguments or {}
    for key, value in expected.items():
        if key not in arguments:
            return f"{call.name} is missing argument {key!r}"
        if not arguments_match(value, arguments[key]):
            actual = _preview(repr(arguments[key]))
            return f"{call.name} argument {key!r} is {actual}, expected {_preview(repr(value))}"
    return None


def _fold(text: str) -> str:
    return text.strip().casefold()


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _PREVIEW_CHARS else flat[: _PREVIEW_CHARS - 3] + "..."
