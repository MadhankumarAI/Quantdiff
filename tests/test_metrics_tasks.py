from __future__ import annotations

from dataclasses import replace

import pytest

from quantdiff.errors import SuiteError
from quantdiff.metrics.tasks import (
    CHAT_SCORED_BY_AGREEMENT,
    CODE_EXEC_DISABLED,
    MAX_LISTED_FAILURES,
    evaluate_case,
    perf_metrics,
    summarize_tasks,
)
from quantdiff.types import (
    CaseOutcome,
    ChatResult,
    Message,
    TaskCase,
    TaskKind,
    ToolCall,
    ToolSpec,
)
from tests.fakes import text_result

PROMPT = (Message("user", "go"),)
JSON_CASE = TaskCase(
    id="j",
    kind="json",
    messages=PROMPT,
    json_schema={"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}},
)
CODE_CASE = TaskCase(
    id="c", kind="code", messages=PROMPT, entry_point="one", tests="assert one() == 1"
)


def test_json_case_passes_on_valid_fenced_json() -> None:
    outcome = evaluate_case(JSON_CASE, text_result('```json\n{"n": 3}\n```'), allow_code_exec=False)
    assert outcome == CaseOutcome("j", "json", passed=True)


def test_json_case_reports_parse_and_schema_failures() -> None:
    unparsed = evaluate_case(JSON_CASE, text_result("n is 3"), allow_code_exec=False)
    assert unparsed.passed is False
    assert unparsed.reason.startswith("invalid JSON")

    invalid = evaluate_case(JSON_CASE, text_result('{"n": "3", "m": true}'), allow_code_exec=False)
    assert invalid == CaseOutcome(
        "j", "json", passed=False, reason="/n: expected integer, got string"
    )


def test_json_case_rejects_nan_against_numeric_bounds() -> None:
    case = TaskCase(
        id="n", kind="json", messages=PROMPT, json_schema={"type": "number", "maximum": 1}
    )
    outcome = evaluate_case(case, text_result("NaN"), allow_code_exec=False)
    assert outcome == CaseOutcome(
        "n", "json", passed=False, reason="invalid JSON: NaN is not a JSON value"
    )


def test_json_case_without_schema_is_a_suite_error() -> None:
    case = TaskCase(id="j", kind="json", messages=PROMPT)
    with pytest.raises(SuiteError):
        evaluate_case(case, text_result("{}"), allow_code_exec=False)


def test_tools_case_is_delegated() -> None:
    tool = ToolSpec("ping", "Ping.", {"type": "object"})
    case = TaskCase(id="t", kind="tools", messages=PROMPT, tools=(tool,), expected_tool="ping")
    result = ChatResult(
        text="",
        tool_calls=(ToolCall("ping", {}, "{}"),),
        finish_reason="tool_calls",
        prompt_tokens=None,
        completion_tokens=None,
        seconds=0.2,
    )
    assert evaluate_case(case, result, allow_code_exec=False).passed is True


def test_code_case_is_skipped_unless_allowed() -> None:
    outcome = evaluate_case(CODE_CASE, text_result("def one(): return 1"), allow_code_exec=False)
    assert outcome == CaseOutcome("c", "code", passed=None, reason=CODE_EXEC_DISABLED)


def test_code_case_runs_when_allowed() -> None:
    answer = text_result("def one():\n    return 1\n")
    assert evaluate_case(CODE_CASE, answer, allow_code_exec=True).passed is True


def test_chat_case_is_scored_elsewhere() -> None:
    case = TaskCase(id="h", kind="chat", messages=PROMPT)
    outcome = evaluate_case(case, text_result("hi"), allow_code_exec=True)
    assert outcome == CaseOutcome("h", "chat", passed=None, reason=CHAT_SCORED_BY_AGREEMENT)


def _outcomes(kind: TaskKind, passed: int, failed: int, skipped: int) -> list[CaseOutcome]:
    states = [True] * passed + [False] * failed + [None] * skipped
    return [CaseOutcome(f"{kind}{index}", kind, state) for index, state in enumerate(states)]


def test_summarize_tasks_orders_kinds_and_excludes_chat() -> None:
    outcomes = _outcomes("code", 1, 0, 2) + _outcomes("chat", 0, 0, 3) + _outcomes("json", 2, 1, 0)
    summaries = summarize_tasks(outcomes)
    assert [summary.kind for summary in summaries] == ["json", "code"]
    json_metrics, code = summaries
    assert (json_metrics.total, json_metrics.passed, json_metrics.skipped) == (3, 2, 0)
    assert json_metrics.failures == (CaseOutcome("json2", "json", False),)
    assert json_metrics.rate == pytest.approx(2 / 3)
    assert (code.total, code.passed, code.skipped) == (3, 1, 2)
    assert code.rate == 1.0


def test_summarize_tasks_caps_listed_failures() -> None:
    (summary,) = summarize_tasks(_outcomes("tools", 0, MAX_LISTED_FAILURES + 5, 0))
    assert summary.total == MAX_LISTED_FAILURES + 5
    assert len(summary.failures) == MAX_LISTED_FAILURES


def test_summarize_tasks_empty() -> None:
    assert summarize_tasks([]) == ()


def test_perf_metrics() -> None:
    results = [
        text_result("a", seconds=1.0, completion_tokens=30),
        text_result("b", seconds=3.0, completion_tokens=10),
        ChatResult("c", (), "stop", None, None, seconds=2.0),
        text_result("d", seconds=0.0, completion_tokens=99),
    ]
    perf = perf_metrics(results)
    assert perf.tokens_per_second == pytest.approx(40 / 4)
    assert perf.mean_latency_seconds == pytest.approx(6.0 / 4)
    assert perf.source == "wall_clock"


def _rated(tokens: int, rate: float | None, *, seconds: float = 9.0) -> ChatResult:
    return replace(
        text_result("x", seconds=seconds, completion_tokens=tokens), decode_tokens_per_second=rate
    )


def test_perf_metrics_prefers_server_decode_speed() -> None:
    # 30 tokens at 30/s and 10 tokens at 5/s: 40 tokens in 3 seconds of decoding.
    perf = perf_metrics([_rated(30, 30.0), _rated(10, 5.0), _rated(1, None)])
    assert perf.source == "server"
    assert perf.tokens_per_second == pytest.approx(40 / 3)
    assert perf.mean_latency_seconds == pytest.approx(9.0)


def test_perf_metrics_ignores_one_token_rates() -> None:
    perf = perf_metrics([_rated(20, 40.0), _rated(1, 1e6)])
    assert perf.source == "server"
    assert perf.tokens_per_second == pytest.approx(40.0)


def test_perf_metrics_falls_back_to_wall_clock_when_any_rate_is_missing() -> None:
    perf = perf_metrics([_rated(30, 30.0, seconds=1.0), _rated(10, None, seconds=1.0)])
    assert perf.source == "wall_clock"
    assert perf.tokens_per_second == pytest.approx(20.0)


def test_perf_metrics_without_usable_data() -> None:
    assert perf_metrics([]).tokens_per_second is None
    assert perf_metrics([]).mean_latency_seconds is None
    untimed = perf_metrics([ChatResult("c", (), "stop", None, None, seconds=1.0)])
    assert untimed.tokens_per_second is None
    assert untimed.mean_latency_seconds == 1.0
    assert untimed.source == "wall_clock"
