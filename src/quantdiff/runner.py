"""Orchestrates a comparison run: connect to everything, run the reference, then each
candidate in turn.

Every server is contacted before any real work starts, so a typo in a model tag fails in
a second instead of after minutes of reference generation. Candidates then run one at a
time on purpose: they usually share a GPU, and running them in parallel would distort
throughput numbers and the servers' own caching. A server error on one request is
recorded and never discards results that were already measured.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from quantdiff._version import __version__
from quantdiff.backends import open_backend
from quantdiff.backends._common import forced_texts
from quantdiff.backends.base import Backend
from quantdiff.cache import ReferenceCache, ReferenceOutputs, cache_key
from quantdiff.errors import BackendError, CapabilityError
from quantdiff.metrics import (
    agreement_metrics,
    evaluate_case,
    logit_metrics,
    perf_metrics,
    summarize_tasks,
)
from quantdiff.metrics.tasks import SCORED_KINDS
from quantdiff.preflight import run_preflight
from quantdiff.suites import suite_digest
from quantdiff.types import (
    CandidateResult,
    CandidateSpec,
    CaseOutcome,
    ChatResult,
    LogitMetrics,
    Message,
    PreflightFinding,
    ProgressEvent,
    ProgressPhase,
    ReferenceTrace,
    Report,
    RunSettings,
    ScoringPrompt,
    ServerInfo,
    TaskCase,
    TokenStep,
    TopK,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[ProgressEvent], None]
BackendFactory = Callable[[CandidateSpec], Backend]

# Progress units are weighted by measured cost so the ETA is honest. One unit is one
# teacher-forced position (a single-token request on a cached prefix). On a consumer GPU a
# chat case costs about 30 of those, a reference scoring prompt about one per generated
# token, and the pre-flight long-prompt probe about as much as six cases.
_CASE_UNITS = 30
_PREFLIGHT_UNITS = 180
_WARMUP_UNITS = 10
_WARMUP_MESSAGES = (Message(role="user", content="Hi"),)


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Everything a run needs, fully resolved and validated."""

    reference: CandidateSpec
    candidates: tuple[CandidateSpec, ...]
    cases: tuple[TaskCase, ...]
    scoring: tuple[ScoringPrompt, ...]
    settings: RunSettings
    title: str
    hf_repo: str | None = None
    offline: bool = False
    preflight: bool = True
    context_probe_tokens: int = 6000
    code_timeout_seconds: float = 10.0


@dataclass(slots=True)
class _Progress:
    """Counts weighted work units and forwards ProgressEvents."""

    sink: ProgressFn
    total: int
    completed: int = 0
    model: str = ""

    def step(self, phase: ProgressPhase, units: int = 1, detail: str = "") -> None:
        self.completed = min(self.completed + units, self.total)
        self.sink(ProgressEvent(phase, self.model, self.completed, self.total, detail))

    def finish(self) -> None:
        self.completed = self.total
        self.model = ""
        self.sink(ProgressEvent("done", "", self.total, self.total))


@dataclass(slots=True)
class _Answers:
    """Chat answers by case id, plus the cases the server failed to answer."""

    results: dict[str, ChatResult] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def execute(
    plan: RunPlan,
    *,
    backend_factory: BackendFactory = open_backend,
    cache: ReferenceCache | None = None,
    progress: ProgressFn | None = None,
) -> Report:
    """Run the plan and return a Report. Pass cache=None to always rerun the reference.

    Raises BackendError before doing any work if a server cannot be reached or does not
    serve the requested model.
    """
    tracker = _Progress(sink=progress or _log_progress, total=_total_units(plan))
    backends = _connect(plan, backend_factory, tracker)
    try:
        reference, *candidates = backends
        ref_backend, ref_info = reference
        tracker.model = plan.reference.label
        outputs, cached = _reference_outputs(plan, ref_backend, ref_info, cache, tracker)
        ref_result = _reference_result(plan, ref_backend, ref_info, outputs, tracker, cached=cached)
        results = tuple(
            _run_candidate(plan, spec, backend, info, ref_backend, ref_info, outputs, tracker)
            for spec, (backend, info) in zip(plan.candidates, candidates, strict=True)
        )
    finally:
        for backend, _ in backends:
            backend.close()
    tracker.finish()

    return Report(
        quantdiff_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        title=plan.title,
        settings=plan.settings,
        reference=ref_result,
        candidates=results,
    )


def _total_units(plan: RunPlan) -> int:
    preflight = _PREFLIGHT_UNITS if plan.preflight else 0
    models = 1 + len(plan.candidates)
    cases = _WARMUP_UNITS + _CASE_UNITS * len(plan.cases)
    scoring = len(plan.scoring) * _scoring_units(plan)
    return models + (cases + scoring + preflight) * models


def _scoring_units(plan: RunPlan) -> int:
    """Units for one scoring prompt: one per generated or teacher-forced token."""
    return max(plan.settings.score_tokens, 1)


def _connect(
    plan: RunPlan, backend_factory: BackendFactory, tracker: _Progress
) -> list[tuple[Backend, ServerInfo]]:
    """Open every backend and fetch its info, or raise one error naming every failure."""
    connected: list[tuple[Backend, ServerInfo]] = []
    problems: list[str] = []
    for spec in (plan.reference, *plan.candidates):
        tracker.model = spec.label
        try:
            backend = backend_factory(spec)
        except BackendError as exc:
            problems.append(f"{spec.label}: {exc}")
            continue
        try:
            connected.append((backend, backend.info()))
        except BackendError as exc:
            backend.close()
            problems.append(f"{spec.label}: {exc}")
        tracker.step("connect")
    if problems:
        for backend, _ in connected:
            backend.close()
        raise BackendError("cannot start the comparison:\n  " + "\n  ".join(problems))
    return connected


# Reference --------------------------------------------------------------------------------


def _reference_outputs(
    plan: RunPlan,
    reference: Backend,
    info: ServerInfo,
    cache: ReferenceCache | None,
    tracker: _Progress,
) -> tuple[ReferenceOutputs, bool]:
    """Return the reference outputs and whether they came from the cache."""
    key = cache_key(
        info,
        base_url=plan.reference.base_url,
        suite_digest=suite_digest(plan.cases, plan.scoring),
        top_k=plan.settings.top_k,
        score_tokens=plan.settings.score_tokens,
        seed=plan.settings.seed,
    )
    if cache is not None:
        cached = cache.load(key)
        if cached is not None:
            units = (
                _WARMUP_UNITS
                + _CASE_UNITS * len(plan.cases)
                + _scoring_units(plan) * len(plan.scoring)
            )
            tracker.step("reference", units, "loaded from cache")
            return cached, True

    answers = _answer_cases(plan, reference, tracker, "reference")
    if answers.failures:
        logger.warning("reference failed %d case(s); they are left out", len(answers.failures))
    traces = _reference_traces(plan, reference, info, tracker)
    outputs = ReferenceOutputs(answers=answers.results, traces=traces)
    if cache is not None and not answers.failures:
        cache.store(key, outputs)
    return outputs, False


def _reference_traces(
    plan: RunPlan, reference: Backend, info: ServerInfo, tracker: _Progress
) -> tuple[ReferenceTrace, ...]:
    if not plan.scoring or not info.supports_logprobs:
        tracker.step("scoring", _scoring_units(plan) * len(plan.scoring), "skipped: no logprobs")
        return ()
    traces = []
    for index, prompt in enumerate(plan.scoring, start=1):
        try:
            steps = reference.generate_scored(
                prompt.text, max_tokens=plan.settings.score_tokens, top_k=plan.settings.top_k
            )
            ids = reference.tokenize(prompt.text) if info.exact_token_ids else None
        except CapabilityError as exc:
            logger.warning("reference exposes no logprobs, skipping the logit tier: %s", exc)
            remaining = len(plan.scoring) - index + 1
            tracker.step("scoring", _scoring_units(plan) * remaining, "skipped: no logprobs")
            return ()
        except BackendError as exc:
            logger.warning("reference failed scoring prompt %s: %s", prompt.id, exc)
        else:
            traces.append(ReferenceTrace(prompt.id, ids, tuple(steps)))
        tracker.step("scoring", _scoring_units(plan), f"{index}/{len(plan.scoring)}")
    return tuple(traces)


def _reference_result(
    plan: RunPlan,
    reference: Backend,
    info: ServerInfo,
    outputs: ReferenceOutputs,
    tracker: _Progress,
    *,
    cached: bool,
) -> CandidateResult:
    findings = _preflight(plan, reference, None, tracker)
    outcomes = _score_cases(plan, outputs.answers)
    perf = perf_metrics(list(outputs.answers.values()))
    return CandidateResult(
        spec=plan.reference,
        info=info,
        logit=None,
        tasks=summarize_tasks(outcomes),
        agreement=None,
        # Timings from an earlier run stay visible but labelled, since the machine's load
        # may differ from the fresh candidate runs.
        perf=replace(perf, source="cached") if cached else perf,
        preflight=findings,
        outcomes=outcomes,
    )


# Candidates -------------------------------------------------------------------------------


def _run_candidate(
    plan: RunPlan,
    spec: CandidateSpec,
    backend: Backend,
    info: ServerInfo,
    reference: Backend,
    ref_info: ServerInfo,
    outputs: ReferenceOutputs,
    tracker: _Progress,
) -> CandidateResult:
    tracker.model = spec.label
    findings = _preflight(plan, backend, reference, tracker)
    answers = _answer_cases(plan, backend, tracker, spec.label)
    logit, logit_error = _logit_tier(plan, backend, info, ref_info, outputs, findings, tracker)

    errors = []
    if answers.failures:
        errors.append(_case_failure_summary(answers.failures))
    if logit_error is not None:
        errors.append(logit_error)
    chats = [
        (case.id, outputs.answers[case.id].text, answers.results[case.id].text)
        for case in plan.cases
        if case.kind == "chat" and case.id in answers.results and case.id in outputs.answers
    ]
    outcomes = _score_cases(plan, answers.results)
    return CandidateResult(
        spec=spec,
        info=info,
        logit=logit,
        tasks=summarize_tasks(outcomes),
        agreement=agreement_metrics(chats) if chats else None,
        perf=perf_metrics(list(answers.results.values())),
        preflight=findings,
        errors=tuple(errors),
        outcomes=outcomes,
    )


def _logit_tier(
    plan: RunPlan,
    backend: Backend,
    info: ServerInfo,
    ref_info: ServerInfo,
    outputs: ReferenceOutputs,
    findings: Sequence[PreflightFinding],
    tracker: _Progress,
) -> tuple[LogitMetrics | None, str | None]:
    """Return logit metrics, or None plus a reason when the tier cannot run fairly."""
    units = len(plan.scoring) * _scoring_units(plan)
    if not outputs.traces or not info.supports_logprobs:
        tracker.step("scoring", units, "skipped")
        return None, None
    if any(f.check == "tokenizer" and f.severity == "fail" for f in findings):
        tracker.step("scoring", units, "skipped")
        return None, "logit tier skipped: tokenizer differs from the reference"
    return _teacher_force(plan, backend, info, ref_info, outputs, tracker)


def _teacher_force(
    plan: RunPlan,
    backend: Backend,
    info: ServerInfo,
    ref_info: ServerInfo,
    outputs: ReferenceOutputs,
    tracker: _Progress,
) -> tuple[LogitMetrics | None, str | None]:
    exact = ref_info.exact_token_ids and info.exact_token_ids
    prompts = {prompt.id: prompt.text for prompt in plan.scoring}
    traces = (
        outputs.traces
        if exact
        else tuple(_text_scorable(trace, prompts[trace.prompt_id]) for trace in outputs.traces)
    )
    budget = len(plan.scoring) * _scoring_units(plan)
    spent = 0
    scored: list[list[TopK]] = []
    for index, trace in enumerate(traces, start=1):
        try:
            tops = backend.score_continuation(
                prompts[trace.prompt_id],
                trace.steps,
                top_k=plan.settings.top_k,
                prompt_token_ids=trace.prompt_token_ids if exact else None,
            )
        except BackendError as exc:
            tracker.step("scoring", budget - spent, "failed")
            return None, f"logit tier failed: {exc}"
        scored.append(tops)
        spent += _scoring_units(plan)
        tracker.step("scoring", _scoring_units(plan), f"{index}/{len(traces)}")
    tracker.step("scoring", budget - spent)
    return logit_metrics(traces, scored, exact_token_ids=exact), None


def _text_scorable(trace: ReferenceTrace, prompt: str) -> ReferenceTrace:
    """The trace with every position that text cannot reproduce marked unscored.

    A reference scored by token ids can stop inside a multi-byte character, and no text
    prompt ends there. Marking those positions unscored (an empty `top`) makes metrics skip
    them for this candidate instead of counting the empty result as a top-1 miss.
    """
    texts = forced_texts(prompt, trace.steps)
    steps = tuple(
        step if text is not None else TokenStep(step.chosen, ())
        for step, text in zip(trace.steps, texts, strict=True)
    )
    return ReferenceTrace(trace.prompt_id, trace.prompt_token_ids, steps)


# Shared steps -----------------------------------------------------------------------------


def _answer_cases(plan: RunPlan, backend: Backend, tracker: _Progress, label: str) -> _Answers:
    _warm_up(plan, backend, tracker, label)
    answers = _Answers()
    total = len(plan.cases)
    for index, case in enumerate(plan.cases, start=1):
        try:
            answers.results[case.id] = backend.chat(
                case.messages,
                max_tokens=case.max_tokens,
                tools=case.tools,
                json_schema=case.json_schema if case.kind == "json" else None,
                seed=plan.settings.seed,
            )
        except BackendError as exc:
            logger.warning("%s failed case %s: %s", label, case.id, exc)
            answers.failures.append(f"{case.id}: {exc}")
        tracker.step("cases", _CASE_UNITS, f"{index}/{total}")
    return answers


def _warm_up(plan: RunPlan, backend: Backend, tracker: _Progress, label: str) -> None:
    """Send one tiny request so model load time never lands in the first case's timing.

    A failure here is only logged: if the server is really broken, the cases report it.
    """
    try:
        backend.chat(_WARMUP_MESSAGES, max_tokens=1, seed=plan.settings.seed)
    except BackendError as exc:
        logger.debug("%s warm-up request failed: %s", label, exc)
    tracker.step("cases", _WARMUP_UNITS, "warm-up")


def _case_failure_summary(failures: Sequence[str]) -> str:
    first = failures[0]
    more = f" (and {len(failures) - 1} more)" if len(failures) > 1 else ""
    return f"{len(failures)} case(s) failed with server errors and were not scored: {first}{more}"


def _score_cases(plan: RunPlan, answers: dict[str, ChatResult]) -> tuple[CaseOutcome, ...]:
    """Outcomes of the answered json, tools and code cases, in plan order."""
    return tuple(
        evaluate_case(
            case,
            answers[case.id],
            allow_code_exec=plan.settings.allow_code_exec,
            timeout_seconds=plan.code_timeout_seconds,
        )
        for case in plan.cases
        if case.id in answers and case.kind in SCORED_KINDS
    )


def _preflight(
    plan: RunPlan, backend: Backend, reference: Backend | None, tracker: _Progress
) -> tuple[PreflightFinding, ...]:
    if not plan.preflight:
        return ()
    findings = run_preflight(
        backend,
        reference=reference,
        hf_repo=plan.hf_repo,
        offline=plan.offline,
        context_probe_tokens=plan.context_probe_tokens,
    )
    tracker.step("preflight", _PREFLIGHT_UNITS)
    return findings


def _log_progress(event: ProgressEvent) -> None:
    logger.info(
        "%s %s %d/%d %s", event.phase, event.model, event.completed, event.total, event.detail
    )
