"""Per-case scoring for task suites and the summaries built from it."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from quantdiff.errors import SuiteError
from quantdiff.metrics.codeexec import DEFAULT_TIMEOUT_SECONDS, run_code_case
from quantdiff.metrics.jsonschema import extract_json, validate
from quantdiff.metrics.toolcheck import check_tool_case
from quantdiff.types import CaseOutcome, ChatResult, PerfMetrics, TaskCase, TaskKind, TaskMetrics

MAX_LISTED_FAILURES: Final = 25
SCORED_KINDS: Final[tuple[TaskKind, ...]] = ("json", "tools", "code")
CODE_EXEC_DISABLED: Final = "code execution disabled (pass --allow-code-exec)"
CHAT_SCORED_BY_AGREEMENT: Final = "scored by agreement"


def evaluate_case(
    case: TaskCase,
    result: ChatResult,
    *,
    allow_code_exec: bool,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CaseOutcome:
    """Score one chat result against its case. Code cases are skipped unless allowed."""
    if case.kind == "json":
        return _check_json_case(case, result)
    if case.kind == "tools":
        return check_tool_case(case, result)
    if case.kind == "code":
        if not allow_code_exec:
            return CaseOutcome(case.id, case.kind, passed=None, reason=CODE_EXEC_DISABLED)
        return run_code_case(case, result.text, timeout_seconds=timeout_seconds)
    return CaseOutcome(case.id, case.kind, passed=None, reason=CHAT_SCORED_BY_AGREEMENT)


def summarize_tasks(outcomes: Sequence[CaseOutcome]) -> tuple[TaskMetrics, ...]:
    """Return one TaskMetrics per scored kind present, in json, tools, code order."""
    summaries = []
    for kind in SCORED_KINDS:
        group = [outcome for outcome in outcomes if outcome.kind == kind]
        if not group:
            continue
        failures = [outcome for outcome in group if outcome.passed is False]
        summaries.append(
            TaskMetrics(
                kind=kind,
                total=len(group),
                passed=sum(outcome.passed is True for outcome in group),
                skipped=sum(outcome.passed is None for outcome in group),
                failures=tuple(failures[:MAX_LISTED_FAILURES]),
            )
        )
    return tuple(summaries)


def perf_metrics(results: Sequence[ChatResult]) -> PerfMetrics:
    """Aggregate generation speed and mean request latency over chat results.

    When the server timed its own decoding for every result that generated more than one
    token, `tokens_per_second` is total completion tokens over total decode time (the
    token-weighted harmonic mean of the per-request rates) with source "server". That
    excludes model load and prompt processing, so it is comparable across models. A
    one-token answer has no decode step to time and is left out of that figure.

    Otherwise it falls back to completion tokens over wall-clock request time with source
    "wall_clock", which includes network overhead and prompt evaluation and is therefore
    only comparable between candidates measured on the same machine with the same cases.
    """
    mean_latency = sum(result.seconds for result in results) / len(results) if results else None
    server_rate = _server_decode_rate(results)
    if server_rate is not None:
        return PerfMetrics(server_rate, mean_latency, source="server")
    timed = [
        result for result in results if result.completion_tokens is not None and result.seconds > 0
    ]
    total_seconds = sum(result.seconds for result in timed)
    tokens_per_second = (
        sum(result.completion_tokens or 0 for result in timed) / total_seconds
        if total_seconds > 0
        else None
    )
    return PerfMetrics(tokens_per_second, mean_latency, source="wall_clock")


def _server_decode_rate(results: Sequence[ChatResult]) -> float | None:
    tokens = 0
    decode_seconds: list[float] = []
    for result in results:
        count = result.completion_tokens
        if count is None or count < 2:
            continue
        rate = result.decode_tokens_per_second
        if not rate:
            return None
        tokens += count
        decode_seconds.append(count / rate)
    return tokens / math.fsum(decode_seconds) if decode_seconds else None


def _check_json_case(case: TaskCase, result: ChatResult) -> CaseOutcome:
    if case.json_schema is None:
        raise SuiteError(f"json case {case.id!r} has no json_schema")
    value, problem = extract_json(result.text)
    if problem is not None:
        return CaseOutcome(case.id, case.kind, passed=False, reason=problem)
    errors = validate(value, case.json_schema)
    if errors:
        extra = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        return CaseOutcome(case.id, case.kind, passed=False, reason=errors[0] + extra)
    return CaseOutcome(case.id, case.kind, passed=True)
