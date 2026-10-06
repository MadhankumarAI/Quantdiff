from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from quantdiff.cache import ReferenceCache
from quantdiff.errors import BackendError
from quantdiff.runner import RunPlan, execute
from quantdiff.types import (
    CandidateSpec,
    ChatResult,
    Message,
    ProgressEvent,
    RunSettings,
    ScoringPrompt,
    TaskCase,
    TokenProb,
    TokenStep,
    ToolCall,
    ToolSpec,
)
from tests.fakes import FakeBackend, make_topk, text_result

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
CASES = (
    TaskCase(
        id="json-001",
        kind="json",
        messages=(Message("user", "Give JSON with name Ada."),),
        json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    TaskCase(
        id="tools-001",
        kind="tools",
        messages=(Message("user", "Weather in Oslo?"),),
        tools=(WEATHER,),
        expected_tool="get_weather",
        expected_arguments={"city": "Oslo"},
    ),
    TaskCase(id="chat-001", kind="chat", messages=(Message("user", "Say hi."),)),
)
SCORING = (ScoringPrompt("score-001", "The capital of France is"),)
_PARIS = make_topk((" Paris", -0.1), (" Lyon", -2.5))
_PERIOD = make_topk((".", -0.2), (",", -1.8))
STEPS = [TokenStep(chosen=_PARIS[0], top=_PARIS), TokenStep(chosen=_PERIOD[0], top=_PERIOD)]


def _good_answers(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
    prompt = messages[-1].content
    if tools:
        call = ToolCall("get_weather", {"city": "Oslo"}, json.dumps({"city": "Oslo"}))
        return ChatResult("", (call,), "tool_calls", 10, 5, 0.1)
    if "JSON" in prompt:
        return text_result('{"name": "Ada"}')
    return text_result("hi")


def _bad_answers(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
    if tools:
        return text_result("I cannot check the weather.")
    return text_result("not json at all")


def _plan() -> RunPlan:
    return RunPlan(
        reference=CandidateSpec("ollama", "http://x", "ref", "ref"),
        candidates=(
            CandidateSpec("ollama", "http://x", "good", "good"),
            CandidateSpec("ollama", "http://x", "bad", "bad"),
        ),
        cases=CASES,
        scoring=SCORING,
        settings=RunSettings(
            suites=("json", "tools", "chat"),
            top_k=5,
            score_tokens=2,
            allow_code_exec=False,
            seed=0,
        ),
        title="test run",
        preflight=False,
    )


def _backends() -> dict[str, FakeBackend]:
    ref = FakeBackend(label="ref", chat_handler=_good_answers, steps=list(STEPS))
    good = FakeBackend(
        label="good",
        chat_handler=_good_answers,
        scores=[STEPS[0].top, STEPS[1].top],
    )
    bad = FakeBackend(
        label="bad",
        chat_handler=_bad_answers,
        scores=[make_topk((" Lyon", -0.3), (" Paris", -1.5)), make_topk((",", -0.1), (".", -2.0))],
    )
    return {"ref": ref, "good": good, "bad": bad}


def test_full_run_scores_good_above_bad() -> None:
    backends = _backends()
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])

    good, bad = report.candidates
    assert good.logit is not None
    assert good.logit.top1_agreement == 1.0
    assert good.logit.kld_mean == pytest.approx(0.0, abs=1e-9)
    assert bad.logit is not None
    assert bad.logit.top1_agreement == 0.0
    assert bad.logit.kld_mean > 0.5

    good_rates = {metrics.kind: metrics.rate for metrics in good.tasks}
    bad_rates = {metrics.kind: metrics.rate for metrics in bad.tasks}
    assert good_rates == {"json": 1.0, "tools": 1.0}
    assert bad_rates == {"json": 0.0, "tools": 0.0}
    assert good.agreement is not None
    assert good.agreement.exact_match_rate == 1.0
    assert report.reference.logit is None
    assert all(backend.closed for backend in backends.values())


def test_unreachable_server_fails_fast_before_any_work() -> None:
    backends = _backends()

    def factory(spec: CandidateSpec) -> FakeBackend:
        if spec.label == "bad":
            raise BackendError("connection refused")
        return backends[spec.label]

    with pytest.raises(BackendError, match="bad: connection refused"):
        execute(_plan(), backend_factory=factory)
    assert backends["ref"].chat_calls == []
    assert backends["ref"].closed
    assert backends["good"].closed


def test_every_connection_problem_is_reported_at_once() -> None:
    def factory(spec: CandidateSpec) -> FakeBackend:
        raise BackendError(f"{spec.label} is down")

    with pytest.raises(BackendError) as exc:
        execute(_plan(), backend_factory=factory)
    assert all(f"{label} is down" in str(exc.value) for label in ("ref", "good", "bad"))


def test_teacher_forcing_failure_keeps_task_results() -> None:
    backends = _backends()
    backends["good"].score_error = BackendError("HTTP 500 during scoring")
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    good = report.candidates[0]
    assert good.logit is None
    assert {metrics.kind: metrics.rate for metrics in good.tasks} == {"json": 1.0, "tools": 1.0}
    assert good.errors == ("logit tier failed: HTTP 500 during scoring",)


def test_one_failed_case_does_not_discard_the_others() -> None:
    backends = _backends()

    def flaky(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
        if tools:
            raise BackendError("timed out")
        return _good_answers(messages, tools)

    backends["good"].chat_handler = flaky
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    good = report.candidates[0]
    assert [metrics.kind for metrics in good.tasks] == ["json"]
    assert good.errors[0].startswith("1 case(s) failed with server errors")


def test_candidate_without_logprobs_still_gets_task_metrics() -> None:
    backends = _backends()
    backends["good"].supports_logprobs = False
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    good = report.candidates[0]
    assert good.logit is None
    assert {metrics.kind for metrics in good.tasks} == {"json", "tools"}


def test_reference_cache_skips_reference_on_second_run(tmp_path: Path) -> None:
    cache = ReferenceCache(tmp_path)
    first = _backends()
    fresh = execute(_plan(), backend_factory=lambda spec: first[spec.label], cache=cache)
    assert len(first["ref"].chat_calls) == 1 + len(CASES)

    second = _backends()
    report = execute(_plan(), backend_factory=lambda spec: second[spec.label], cache=cache)
    assert second["ref"].chat_calls == []
    assert report.candidates[0].logit is not None
    assert fresh.reference.perf is not None
    assert report.reference.perf == replace(fresh.reference.perf, source="cached")
    assert report.reference.outcomes == fresh.reference.outcomes


def test_changed_reference_template_invalidates_the_cache(tmp_path: Path) -> None:
    cache = ReferenceCache(tmp_path)
    first = _backends()
    execute(_plan(), backend_factory=lambda spec: first[spec.label], cache=cache)

    second = _backends()
    second["ref"].chat_template = "{{ fixed template }}"
    execute(_plan(), backend_factory=lambda spec: second[spec.label], cache=cache)
    assert len(second["ref"].chat_calls) == 1 + len(CASES)


def test_progress_events_cover_the_whole_run() -> None:
    backends = _backends()
    events: list[ProgressEvent] = []
    execute(_plan(), backend_factory=lambda spec: backends[spec.label], progress=events.append)

    assert events[0].phase == "connect"
    assert events[-1] == ProgressEvent("done", "", events[-1].total, events[-1].total)
    assert events[-2].completed == events[-2].total
    completed = [event.completed for event in events]
    assert completed == sorted(completed)
    assert {event.total for event in events} == {events[0].total}
    assert {event.model for event in events if event.phase == "cases"} == {"ref", "good", "bad"}


def test_per_item_data_is_kept_for_paired_comparisons() -> None:
    backends = _backends()
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])

    good, bad = report.candidates
    assert [(o.case_id, o.passed) for o in report.reference.outcomes] == [
        ("json-001", True),
        ("tools-001", True),
    ]
    assert [(o.case_id, o.passed) for o in good.outcomes] == [
        ("json-001", True),
        ("tools-001", True),
    ]
    assert [(o.case_id, o.passed) for o in bad.outcomes] == [
        ("json-001", False),
        ("tools-001", False),
    ]
    assert good.agreement is not None
    assert good.agreement.per_case == (("chat-001", 1.0),)
    assert bad.logit is not None
    assert [(p.prompt_id, p.positions, p.top1_matches) for p in bad.logit.per_prompt] == [
        ("score-001", 2, 0)
    ]


def test_skipped_code_cases_are_kept_as_outcomes() -> None:
    code = TaskCase(
        id="code-001",
        kind="code",
        messages=(Message("user", "Write add(a, b)."),),
        entry_point="add",
        tests="assert add(1, 2) == 3",
    )
    backends = _backends()
    plan = replace(_plan(), cases=(*CASES, code))
    report = execute(plan, backend_factory=lambda spec: backends[spec.label])
    assert report.candidates[0].outcomes[-1].case_id == "code-001"
    assert report.candidates[0].outcomes[-1].passed is None


def test_weights_identity_reaches_the_report() -> None:
    backends = _backends()
    backends["ref"].weights_id = backends["good"].weights_id = "sha256-same"
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    assert report.reference.info is not None
    assert report.candidates[0].info is not None
    assert report.reference.info.weights_id == report.candidates[0].info.weights_id


def test_each_model_is_warmed_up_before_its_cases() -> None:
    backends = _backends()
    execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    for backend in backends.values():
        assert backend.chat_calls[0] == (Message("user", "Hi"),)
        assert len(backend.chat_calls) == 1 + len(CASES)


def test_failed_warm_up_is_ignored() -> None:
    backends = _backends()

    def cold(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
        if messages[-1].content == "Hi":
            raise BackendError("model is still loading")
        return _good_answers(messages, tools)

    backends["good"].chat_handler = cold
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    good = report.candidates[0]
    assert good.errors == ()
    assert {metrics.kind: metrics.rate for metrics in good.tasks} == {"json": 1.0, "tools": 1.0}


def test_server_decode_speed_is_used_when_reported() -> None:
    backends = _backends()

    def timed(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
        return replace(_good_answers(messages, tools), decode_tokens_per_second=25.0)

    backends["good"].chat_handler = timed
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    good, bad = report.candidates
    assert good.perf is not None
    assert good.perf.source == "server"
    assert good.perf.tokens_per_second == pytest.approx(25.0)
    assert bad.perf is not None
    assert bad.perf.source == "wall_clock"
    assert report.reference.perf is not None
    assert report.reference.perf.source == "wall_clock"


def test_text_forced_candidate_skips_positions_text_cannot_reproduce() -> None:
    # The exact reference stops inside a two-byte character: the first token holds only its
    # first byte, so no text prompt can end there. A text-forced candidate must skip the
    # position after it, not count it as a top-1 miss.
    half = TokenProb("�", -0.1, 7, token_bytes=b"\xc3")
    rest = TokenProb("�", -0.1, 8, token_bytes=b"\xa9")
    split = [TokenStep(chosen=half, top=(half,)), TokenStep(chosen=rest, top=(rest,))]
    backends = _backends()
    backends["ref"].steps = split
    for name in ("good", "bad"):
        backends[name].exact_token_ids = False
        backends[name].scores = [(half,), ()]
    report = execute(_plan(), backend_factory=lambda spec: backends[spec.label])
    logit = report.candidates[0].logit
    assert logit is not None
    # Only the first position is scorable by text; the second is skipped, not a miss.
    assert logit.positions == 1
    assert logit.top1_agreement == 1.0
