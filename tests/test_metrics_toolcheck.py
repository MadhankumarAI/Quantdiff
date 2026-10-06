from __future__ import annotations

import json
from typing import Any

import pytest

from quantdiff.errors import SuiteError
from quantdiff.metrics.toolcheck import arguments_match, check_tool_case
from quantdiff.types import ChatResult, Message, TaskCase, ToolCall, ToolSpec

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer", "minimum": 1},
            "units": {"enum": ["c", "f"]},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)
FLIGHTS = ToolSpec(
    name="search_flights",
    description="Find flights.",
    parameters={"type": "object", "properties": {"to": {"type": "string"}}},
)


def _case(
    expected_tool: str | None = "get_weather", expected_arguments: dict[str, Any] | None = None
) -> TaskCase:
    return TaskCase(
        id="weather",
        kind="tools",
        messages=(Message("user", "Weather in Paris for 2 days?"),),
        tools=(WEATHER, FLIGHTS),
        expected_tool=expected_tool,
        expected_arguments=expected_arguments,
    )


def _result(*calls: ToolCall, text: str = "") -> ChatResult:
    return ChatResult(
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls",
        prompt_tokens=10,
        completion_tokens=5,
        seconds=0.1,
    )


def _call(name: str, arguments: dict[str, Any] | None, raw: str | None = None) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, raw_arguments=raw or json.dumps(arguments))


def test_correct_call_passes() -> None:
    case = _case(expected_arguments={"city": "paris", "days": 2})
    outcome = check_tool_case(case, _result(_call("get_weather", {"city": " Paris ", "days": 2.0})))
    assert outcome.passed is True
    assert outcome.reason == ""
    assert outcome.case_id == "weather"
    assert outcome.kind == "tools"


def test_wrong_tool_fails_with_a_clear_reason() -> None:
    outcome = check_tool_case(_case(), _result(_call("search_flights", {"to": "CDG"})))
    assert outcome.passed is False
    assert outcome.reason == "called search_flights, expected get_weather"


def test_only_the_first_call_counts() -> None:
    calls = (_call("search_flights", {}), _call("get_weather", {"city": "Paris"}))
    assert check_tool_case(_case(), _result(*calls)).passed is False


def test_missing_call_fails() -> None:
    outcome = check_tool_case(_case(), _result(text="It is sunny."))
    assert outcome.reason == "made no tool call, expected get_weather"


def test_unparseable_arguments_fail() -> None:
    outcome = check_tool_case(_case(), _result(_call("get_weather", None, raw="{city: Paris")))
    assert outcome.passed is False
    assert outcome.reason == "get_weather arguments are not a JSON object: {city: Paris"


def test_schema_violations_fail() -> None:
    outcome = check_tool_case(_case(), _result(_call("get_weather", {"days": 0, "x": 1})))
    assert outcome.passed is False
    assert outcome.reason.startswith("get_weather arguments violate the schema: ")
    assert outcome.reason.endswith("(and 2 more)")


def test_expected_argument_mismatch() -> None:
    case = _case(expected_arguments={"city": "Paris", "units": "c"})
    missing = check_tool_case(case, _result(_call("get_weather", {"city": "Paris"})))
    assert missing.reason == "get_weather is missing argument 'units'"
    wrong = check_tool_case(case, _result(_call("get_weather", {"city": "Lyon", "units": "c"})))
    assert wrong.reason == "get_weather argument 'city' is 'Lyon', expected 'Paris'"


def test_direct_answer_case() -> None:
    case = _case(expected_tool=None)
    assert check_tool_case(case, _result(text="Hello!")).passed is True
    outcome = check_tool_case(case, _result(_call("get_weather", {"city": "Paris"})))
    assert outcome.passed is False
    assert outcome.reason == "called get_weather, expected a direct answer"


def test_expected_tool_missing_from_case_is_a_suite_error() -> None:
    case = _case(expected_tool="book_hotel")
    with pytest.raises(SuiteError, match="book_hotel"):
        check_tool_case(case, _result(_call("book_hotel", {})))


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        (1, 1.0, True),
        (1, True, False),
        (True, 1, False),
        (True, True, True),
        (2.5, 2.5, True),
        (1, "1", False),
        ("New York", "  new york ", True),
        ("Straße", "STRASSE", True),
        ("a", "b", False),
        ([1, "A"], [1.0, "a"], True),
        ([1, 2], [1], False),
        ({"a": {"b": "X"}}, {"a": {"b": "x", "c": 3}}, True),
        ({"a": 1}, {"b": 1}, False),
        ({"a": 1}, [1], False),
        (None, None, True),
        (None, 0, False),
    ],
)
def test_arguments_match(expected: object, actual: object, matches: bool) -> None:
    assert arguments_match(expected, actual) is matches
