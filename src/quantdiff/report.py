"""Report files, candidate ranking, the one-line verdict, and methodology notes.

Reports are plain JSON so they can be diffed, archived, and re-rendered later. Loading
is strict: every field is checked by name and type, unknown fields are rejected, and
every problem raises ReportError with the path of the offending field. Version 1 reports
still load: the per-prompt and per-case fields added in version 2 default to empty, so
they render, but the verdict has nothing to pair and says less.

Ranking and the verdict itself live in quantdiff.verdict; this module adapts them for
callers that want a ranked list or a single line of text.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar, get_args

from quantdiff.errors import ReportError
from quantdiff.types import (
    AgreementMetrics,
    BackendKind,
    CandidateResult,
    CandidateSpec,
    CaseOutcome,
    JSONValue,
    LogitMetrics,
    PerfMetrics,
    PerfSource,
    PreflightFinding,
    PromptLogit,
    Report,
    RunSettings,
    ServerInfo,
    Severity,
    TaskKind,
    TaskMetrics,
    TemplateDialect,
)
from quantdiff.verdict import SCORED_TASK_KINDS, judge, kld_thresholds

__all__ = [
    "MAX_REPORT_BYTES",
    "SCHEMA_VERSION",
    "SCORED_TASK_KINDS",
    "RankedCandidate",
    "format_kld",
    "format_rate",
    "format_top1",
    "load_report",
    "notes_for",
    "rank_candidates",
    "report_from_dict",
    "report_to_dict",
    "save_report",
    "task_for",
    "text_forced_note",
    "verdict",
    "verdict_sentences",
]

SCHEMA_VERSION: Final = 2
READABLE_SCHEMA_VERSIONS: Final = (1, 2)
MAX_REPORT_BYTES: Final = 50 * 1024 * 1024

_BACKEND_KINDS: Final[tuple[BackendKind, ...]] = get_args(BackendKind)
_TASK_KINDS: Final[tuple[TaskKind, ...]] = get_args(TaskKind)
_SEVERITIES: Final[tuple[Severity, ...]] = get_args(Severity)
_DIALECTS: Final[tuple[TemplateDialect, ...]] = get_args(TemplateDialect)
_PERF_SOURCES: Final[tuple[PerfSource, ...]] = get_args(PerfSource)
_JSON_TYPE_NAMES: Final[dict[type, str]] = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}

_T = TypeVar("_T")
_S = TypeVar("_S", bound=str)


# Formatting -------------------------------------------------------------------------------


def format_rate(value: float) -> str:
    """Format a task pass rate as a whole percentage, such as "88%"."""
    return f"{value * 100:.0f}%"


def format_top1(value: float) -> str:
    """Format top-1 agreement with one decimal, such as "97.1%"."""
    return f"{value * 100:.1f}%"


def format_kld(value: float) -> str:
    """Format a KL divergence with three significant digits, such as "0.0123"."""
    return f"{value:.3g}"


# Serialization ----------------------------------------------------------------------------


def report_to_dict(report: Report) -> dict[str, JSONValue]:
    """Convert `report` to JSON-compatible data at SCHEMA_VERSION. Inverse of report_from_dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "quantdiff_version": report.quantdiff_version,
        "created_at": report.created_at,
        "title": report.title,
        "settings": _settings_to_dict(report.settings),
        "reference": _candidate_to_dict(report.reference),
        "candidates": [_candidate_to_dict(candidate) for candidate in report.candidates],
        "notes": list(report.notes),
    }


def _settings_to_dict(settings: RunSettings) -> dict[str, JSONValue]:
    return {
        "suites": list(settings.suites),
        "top_k": settings.top_k,
        "score_tokens": settings.score_tokens,
        "allow_code_exec": settings.allow_code_exec,
        "seed": settings.seed,
        "prompts_file": settings.prompts_file,
        "longest_prompt_tokens": settings.longest_prompt_tokens,
        "max_size_bytes": settings.max_size_bytes,
    }


def _candidate_to_dict(result: CandidateResult) -> dict[str, JSONValue]:
    spec = result.spec
    return {
        "spec": {
            "kind": spec.kind,
            "base_url": spec.base_url,
            "model": spec.model,
            "label": spec.label,
            "api_key_env": spec.api_key_env,
        },
        "info": None if result.info is None else _info_to_dict(result.info),
        "logit": None if result.logit is None else _logit_to_dict(result.logit),
        "tasks": [_task_to_dict(task) for task in result.tasks],
        "agreement": None if result.agreement is None else _agreement_to_dict(result.agreement),
        "perf": None if result.perf is None else _perf_to_dict(result.perf),
        "preflight": [_finding_to_dict(finding) for finding in result.preflight],
        "errors": list(result.errors),
        "outcomes": [_outcome_to_dict(outcome) for outcome in result.outcomes],
    }


def _info_to_dict(info: ServerInfo) -> dict[str, JSONValue]:
    return {
        "backend": info.backend,
        "model": info.model,
        "context_length": info.context_length,
        "chat_template": info.chat_template,
        "template_dialect": info.template_dialect,
        "supports_logprobs": info.supports_logprobs,
        "exact_token_ids": info.exact_token_ids,
        "details": [[key, value] for key, value in info.details],
        "size_bytes": info.size_bytes,
        "weights_id": info.weights_id,
    }


def _logit_to_dict(logit: LogitMetrics) -> dict[str, JSONValue]:
    return {
        "prompts": logit.prompts,
        "positions": logit.positions,
        "top1_agreement": logit.top1_agreement,
        "kld_mean": logit.kld_mean,
        "kld_p99": logit.kld_p99,
        "kld_max": logit.kld_max,
        "exact_token_ids": logit.exact_token_ids,
        "per_prompt": [
            {
                "prompt_id": prompt.prompt_id,
                "positions": prompt.positions,
                "top1_matches": prompt.top1_matches,
                "kld_mean": prompt.kld_mean,
            }
            for prompt in logit.per_prompt
        ],
    }


def _task_to_dict(task: TaskMetrics) -> dict[str, JSONValue]:
    return {
        "kind": task.kind,
        "total": task.total,
        "passed": task.passed,
        "skipped": task.skipped,
        "rate": task.rate,
        "failures": [_outcome_to_dict(outcome) for outcome in task.failures],
    }


def _outcome_to_dict(outcome: CaseOutcome) -> dict[str, JSONValue]:
    return {
        "case_id": outcome.case_id,
        "kind": outcome.kind,
        "passed": outcome.passed,
        "reason": outcome.reason,
    }


def _agreement_to_dict(agreement: AgreementMetrics) -> dict[str, JSONValue]:
    return {
        "cases": agreement.cases,
        "exact_match_rate": agreement.exact_match_rate,
        "mean_similarity": agreement.mean_similarity,
        "per_case": [
            {"case_id": case_id, "similarity": similarity}
            for case_id, similarity in agreement.per_case
        ],
    }


def _perf_to_dict(perf: PerfMetrics) -> dict[str, JSONValue]:
    return {
        "tokens_per_second": perf.tokens_per_second,
        "mean_latency_seconds": perf.mean_latency_seconds,
        "source": perf.source,
    }


def _finding_to_dict(finding: PreflightFinding) -> dict[str, JSONValue]:
    return {
        "check": finding.check,
        "severity": finding.severity,
        "message": finding.message,
        "fix": finding.fix,
    }


def report_from_dict(data: object) -> Report:
    """Build a Report from parsed JSON, raising ReportError on any schema violation.

    Accepts every version in READABLE_SCHEMA_VERSIONS. Each version must have exactly its
    own fields, and the result is always a current-version Report.
    """
    fields = _object(
        data,
        "",
        required=(
            "schema_version",
            "quantdiff_version",
            "created_at",
            "title",
            "settings",
            "reference",
            "candidates",
            "notes",
        ),
    )
    version = fields["schema_version"]
    if isinstance(version, bool) or version not in READABLE_SCHEMA_VERSIONS:
        readable = " and ".join(str(v) for v in READABLE_SCHEMA_VERSIONS)
        raise ReportError(
            f"schema_version: unsupported value {version!r}; this quantdiff reads "
            f"versions {readable}"
        )
    reader = _Reader(version=_int(version, "schema_version"))
    return Report(
        schema_version=SCHEMA_VERSION,
        quantdiff_version=_str(fields["quantdiff_version"], "quantdiff_version"),
        created_at=_str(fields["created_at"], "created_at"),
        title=_str(fields["title"], "title"),
        settings=reader.settings(fields["settings"], "settings"),
        reference=reader.candidate(fields["reference"], "reference"),
        candidates=_tuple_of(fields["candidates"], "candidates", reader.candidate),
        notes=_tuple_of(fields["notes"], "notes", _str),
    )


@dataclass(frozen=True, slots=True)
class _Reader:
    """Readers for the parts of a report whose fields depend on the schema version."""

    version: int

    def _fields(self, v1: Sequence[str], added: Sequence[str] = ()) -> tuple[str, ...]:
        return (*v1, *added) if self.version >= 2 else tuple(v1)

    def settings(self, value: object, path: str) -> RunSettings:
        fields = _object(
            value,
            path,
            required=self._fields(
                ("suites", "top_k", "score_tokens", "allow_code_exec", "seed", "prompts_file"),
                ("longest_prompt_tokens",),
            ),
            # Added within schema version 2; reports written before it simply lack the field.
            optional=self._fields((), ("max_size_bytes",)),
        )
        return RunSettings(
            suites=_tuple_of(fields["suites"], f"{path}.suites", _str),
            top_k=_int(fields["top_k"], f"{path}.top_k"),
            score_tokens=_int(fields["score_tokens"], f"{path}.score_tokens"),
            allow_code_exec=_bool(fields["allow_code_exec"], f"{path}.allow_code_exec"),
            seed=_int(fields["seed"], f"{path}.seed", minimum=None),
            prompts_file=_optional(fields["prompts_file"], f"{path}.prompts_file", _str),
            longest_prompt_tokens=_optional(
                fields.get("longest_prompt_tokens"), f"{path}.longest_prompt_tokens", _int
            ),
            max_size_bytes=_optional(fields.get("max_size_bytes"), f"{path}.max_size_bytes", _int),
        )

    def candidate(self, value: object, path: str) -> CandidateResult:
        fields = _object(
            value,
            path,
            required=self._fields(
                ("spec", "info", "logit", "tasks", "agreement", "perf", "preflight", "errors"),
                ("outcomes",),
            ),
        )
        return CandidateResult(
            spec=_spec_from(fields["spec"], f"{path}.spec"),
            info=_optional(fields["info"], f"{path}.info", self.info),
            logit=_optional(fields["logit"], f"{path}.logit", self.logit),
            tasks=_tuple_of(fields["tasks"], f"{path}.tasks", _task_from),
            agreement=_optional(fields["agreement"], f"{path}.agreement", self.agreement),
            perf=_optional(fields["perf"], f"{path}.perf", self.perf),
            preflight=_tuple_of(fields["preflight"], f"{path}.preflight", _finding_from),
            errors=_tuple_of(fields["errors"], f"{path}.errors", _str),
            outcomes=_tuple_of(fields.get("outcomes", []), f"{path}.outcomes", _outcome_from),
        )

    def info(self, value: object, path: str) -> ServerInfo:
        fields = _object(
            value,
            path,
            required=self._fields(
                (
                    "backend",
                    "model",
                    "context_length",
                    "chat_template",
                    "template_dialect",
                    "supports_logprobs",
                    "exact_token_ids",
                    "details",
                ),
                ("size_bytes", "weights_id"),
            ),
        )
        return ServerInfo(
            backend=_choice(fields["backend"], f"{path}.backend", _BACKEND_KINDS),
            model=_str(fields["model"], f"{path}.model"),
            context_length=_optional(fields["context_length"], f"{path}.context_length", _int),
            chat_template=_optional(fields["chat_template"], f"{path}.chat_template", _str),
            template_dialect=_choice(
                fields["template_dialect"], f"{path}.template_dialect", _DIALECTS
            ),
            supports_logprobs=_bool(fields["supports_logprobs"], f"{path}.supports_logprobs"),
            exact_token_ids=_bool(fields["exact_token_ids"], f"{path}.exact_token_ids"),
            details=_tuple_of(fields["details"], f"{path}.details", _detail_from),
            size_bytes=_optional(fields.get("size_bytes"), f"{path}.size_bytes", _int),
            weights_id=_optional(fields.get("weights_id"), f"{path}.weights_id", _str),
        )

    def logit(self, value: object, path: str) -> LogitMetrics:
        fields = _object(
            value,
            path,
            required=self._fields(
                (
                    "prompts",
                    "positions",
                    "top1_agreement",
                    "kld_mean",
                    "kld_p99",
                    "kld_max",
                    "exact_token_ids",
                ),
                ("per_prompt",),
            ),
        )
        return LogitMetrics(
            prompts=_int(fields["prompts"], f"{path}.prompts"),
            positions=_int(fields["positions"], f"{path}.positions"),
            top1_agreement=_fraction(fields["top1_agreement"], f"{path}.top1_agreement"),
            kld_mean=_float(fields["kld_mean"], f"{path}.kld_mean"),
            kld_p99=_float(fields["kld_p99"], f"{path}.kld_p99"),
            kld_max=_float(fields["kld_max"], f"{path}.kld_max"),
            exact_token_ids=_bool(fields["exact_token_ids"], f"{path}.exact_token_ids"),
            per_prompt=_tuple_of(fields.get("per_prompt", []), f"{path}.per_prompt", _prompt_from),
        )

    def agreement(self, value: object, path: str) -> AgreementMetrics:
        fields = _object(
            value,
            path,
            required=self._fields(("cases", "exact_match_rate", "mean_similarity"), ("per_case",)),
        )
        return AgreementMetrics(
            cases=_int(fields["cases"], f"{path}.cases"),
            exact_match_rate=_fraction(fields["exact_match_rate"], f"{path}.exact_match_rate"),
            mean_similarity=_fraction(fields["mean_similarity"], f"{path}.mean_similarity"),
            per_case=_tuple_of(fields.get("per_case", []), f"{path}.per_case", _similarity_from),
        )

    def perf(self, value: object, path: str) -> PerfMetrics:
        fields = _object(
            value,
            path,
            required=self._fields(("tokens_per_second", "mean_latency_seconds"), ("source",)),
        )
        return PerfMetrics(
            tokens_per_second=_optional(
                fields["tokens_per_second"], f"{path}.tokens_per_second", _float
            ),
            mean_latency_seconds=_optional(
                fields["mean_latency_seconds"], f"{path}.mean_latency_seconds", _float
            ),
            source=_choice(fields.get("source", "wall_clock"), f"{path}.source", _PERF_SOURCES),
        )


def _spec_from(value: object, path: str) -> CandidateSpec:
    fields = _object(value, path, required=("kind", "base_url", "model", "label", "api_key_env"))
    return CandidateSpec(
        kind=_choice(fields["kind"], f"{path}.kind", _BACKEND_KINDS),
        base_url=_str(fields["base_url"], f"{path}.base_url"),
        model=_str(fields["model"], f"{path}.model"),
        label=_str(fields["label"], f"{path}.label"),
        api_key_env=_optional(fields["api_key_env"], f"{path}.api_key_env", _str),
    )


def _detail_from(value: object, path: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ReportError(f"{path}: expected a [key, value] pair of strings")
    return _str(value[0], f"{path}[0]"), _str(value[1], f"{path}[1]")


def _prompt_from(value: object, path: str) -> PromptLogit:
    fields = _object(value, path, required=("prompt_id", "positions", "top1_matches", "kld_mean"))
    prompt = PromptLogit(
        prompt_id=_str(fields["prompt_id"], f"{path}.prompt_id"),
        positions=_int(fields["positions"], f"{path}.positions"),
        top1_matches=_int(fields["top1_matches"], f"{path}.top1_matches"),
        kld_mean=_optional(fields["kld_mean"], f"{path}.kld_mean", _float),
    )
    if prompt.top1_matches > prompt.positions:
        raise ReportError(f"{path}: top1_matches exceeds positions")
    return prompt


def _similarity_from(value: object, path: str) -> tuple[str, float]:
    fields = _object(value, path, required=("case_id", "similarity"))
    return (
        _str(fields["case_id"], f"{path}.case_id"),
        _fraction(fields["similarity"], f"{path}.similarity"),
    )


def _task_from(value: object, path: str) -> TaskMetrics:
    # "rate" is derived from the counts on output; it is accepted and ignored on input.
    fields = _object(
        value,
        path,
        required=("kind", "total", "passed", "skipped", "failures"),
        ignored=("rate",),
    )
    task = TaskMetrics(
        kind=_choice(fields["kind"], f"{path}.kind", _TASK_KINDS),
        total=_int(fields["total"], f"{path}.total"),
        passed=_int(fields["passed"], f"{path}.passed"),
        skipped=_int(fields["skipped"], f"{path}.skipped"),
        failures=_tuple_of(fields["failures"], f"{path}.failures", _outcome_from),
    )
    if task.passed + task.skipped > task.total:
        raise ReportError(f"{path}: passed + skipped exceeds total")
    return task


def _outcome_from(value: object, path: str) -> CaseOutcome:
    fields = _object(value, path, required=("case_id", "kind", "passed", "reason"))
    return CaseOutcome(
        case_id=_str(fields["case_id"], f"{path}.case_id"),
        kind=_choice(fields["kind"], f"{path}.kind", _TASK_KINDS),
        passed=_optional(fields["passed"], f"{path}.passed", _bool),
        reason=_str(fields["reason"], f"{path}.reason"),
    )


def _finding_from(value: object, path: str) -> PreflightFinding:
    fields = _object(value, path, required=("check", "severity", "message", "fix"))
    return PreflightFinding(
        check=_str(fields["check"], f"{path}.check"),
        severity=_choice(fields["severity"], f"{path}.severity", _SEVERITIES),
        message=_str(fields["message"], f"{path}.message"),
        fix=_optional(fields["fix"], f"{path}.fix", _str),
    )


# Strict field readers ---------------------------------------------------------------------


def _where(path: str) -> str:
    return path or "report"


def _object(
    value: object,
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    ignored: Sequence[str] = (),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReportError(f"{_where(path)}: expected an object, got {_type_name(value)}")
    allowed = set(required) | set(optional) | set(ignored)
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ReportError(f"{_where(path)}: unknown field {unknown[0]!r}")
    for key in required:
        if key not in value:
            prefix = f"{path}." if path else ""
            raise ReportError(f"{prefix}{key}: missing required field")
    return value


def _str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ReportError(f"{path}: expected a string, got {_type_name(value)}")
    return value


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ReportError(f"{path}: expected true or false, got {_type_name(value)}")
    return value


def _int(value: object, path: str, *, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{path}: expected an integer, got {_type_name(value)}")
    if minimum is not None and value < minimum:
        raise ReportError(f"{path}: must be at least {minimum}, got {value}")
    return value


def _float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{path}: expected a number, got {_type_name(value)}")
    if not math.isfinite(value):
        raise ReportError(f"{path}: must be a finite number")
    return float(value)


def _fraction(value: object, path: str) -> float:
    number = _float(value, path)
    if not 0.0 <= number <= 1.0:
        raise ReportError(f"{path}: must be between 0 and 1, got {number}")
    return number


def _choice(value: object, path: str, options: tuple[_S, ...]) -> _S:
    for option in options:
        if value == option:
            return option
    allowed = ", ".join(options)
    raise ReportError(f"{path}: expected one of {allowed}, got {value!r}")


def _optional(value: object, path: str, read: Callable[[object, str], _T]) -> _T | None:
    return None if value is None else read(value, path)


def _tuple_of(value: object, path: str, read: Callable[[object, str], _T]) -> tuple[_T, ...]:
    if not isinstance(value, list):
        raise ReportError(f"{path}: expected an array, got {_type_name(value)}")
    return tuple(read(item, f"{path}[{index}]") for index, item in enumerate(value))


def _type_name(value: object) -> str:
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


# Files ------------------------------------------------------------------------------------


def save_report(report: Report, path: str | os.PathLike[str]) -> None:
    """Write `report` as UTF-8 JSON. The write is atomic: readers never see a partial file."""
    target = Path(path)
    try:
        text = json.dumps(
            report_to_dict(report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    except ValueError as exc:
        raise ReportError(f"report contains a value JSON cannot represent: {exc}") from exc
    try:
        _write_atomically(target, text + "\n")
    except OSError as exc:
        raise ReportError(f"cannot write report to {target}: {exc.strerror or exc}") from exc


def _write_atomically(target: Path, text: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


def load_report(path: str | os.PathLike[str], *, max_bytes: int = MAX_REPORT_BYTES) -> Report:
    """Read and validate a report file written by save_report."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ReportError(f"cannot read report {source}: {exc.strerror or exc}") from exc
    if len(raw) > max_bytes:
        raise ReportError(f"report {source} is larger than {max_bytes} bytes")
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise ReportError(f"report {source} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"report {source} is not valid JSON: {exc.msg} at line {exc.lineno}"
        ) from exc
    except RecursionError as exc:
        raise ReportError(f"report {source} is nested too deeply") from exc
    return report_from_dict(data)


def _reject_constant(name: str) -> object:
    raise ReportError(f"report contains {name}, which is not valid JSON")


# Ranking ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A candidate with its position on the scorecard and the numbers that placed it."""

    result: CandidateResult
    rank: int
    task_rate: float | None
    """Mean pass rate over json, tools, and code suites that scored at least one case."""
    top1: float | None
    kld_mean: float | None

    @property
    def has_metrics(self) -> bool:
        return self.task_rate is not None or self.top1 is not None or self.kld_mean is not None


def task_for(result: CandidateResult, kind: TaskKind) -> TaskMetrics | None:
    """Return the metrics for one task kind, or None if that suite did not run."""
    return next((task for task in result.tasks if task.kind == kind), None)


def rank_candidates(report: Report) -> tuple[RankedCandidate, ...]:
    """Order candidates best first, in the order the verdict gives them.

    The verdict ranks by status first (recommended, ok, inconclusive, avoid, failed), then
    by mean KLD, top-1 agreement, mean pass rate, and size; see quantdiff.verdict. Ranks run
    1..n over every candidate, failed ones included, so a list position is always a rank.
    """
    by_label = {result.spec.label: result for result in report.candidates}
    ordered = [by_label[call.label] for call in judge(report).candidates]
    return tuple(
        RankedCandidate(
            result=result,
            rank=rank,
            task_rate=_mean_task_rate(result),
            top1=None if result.logit is None else result.logit.top1_agreement,
            kld_mean=None if result.logit is None else result.logit.kld_mean,
        )
        for rank, result in enumerate(ordered, start=1)
    )


def _mean_task_rate(result: CandidateResult) -> float | None:
    rates = [
        rate
        for task in result.tasks
        if task.kind in SCORED_TASK_KINDS and (rate := task.rate) is not None
    ]
    return sum(rates) / len(rates) if rates else None


# Verdict ----------------------------------------------------------------------------------


def verdict(report: Report) -> str:
    """The verdict on one line: the headline followed by its supporting sentences."""
    return " ".join(verdict_sentences(report))


def verdict_sentences(report: Report) -> tuple[str, ...]:
    """The verdict as separate sentences, headline first, so a card can emphasize it."""
    result = judge(report)
    return (result.headline, *result.details)


# Methodology notes ------------------------------------------------------------------------


def notes_for(report: Report) -> tuple[str, ...]:
    """Footnotes that explain how to read the numbers on this particular scorecard."""
    notes = [
        "T1 metrics (top-1, KLD) compare next-token probabilities and need logprobs.",
        "T2 metrics (pass rates, chat agreement) score real answers on any server.",
    ]
    if any(result.logit is not None for result in report.candidates):
        bars = kld_thresholds(report.settings.top_k)
        notes.append(
            f"KLD bands: under {bars.near_lossless:g} near-lossless, under {bars.close:g} small, "
            f"under {bars.large:g} moderate, above that large. KLD here is a lower bound "
            f"computed from the top {report.settings.top_k} tokens, measured against your "
            "reference."
        )
    retokenized = [
        result
        for result in report.candidates
        if result.logit is not None and not result.logit.exact_token_ids
    ]
    if retokenized:
        ollama = all(result.spec.kind == "ollama" for result in retokenized)
        labels = [result.spec.label for result in retokenized]
        notes.append(text_forced_note(labels, ollama=ollama))
    if report.candidates and _mean_task_rate(report.reference) is not None:
        notes.append(
            "The ref row is the reference on the same cases. Signed numbers are differences "
            "from it in percentage points."
        )
    if any(result.agreement is not None for result in report.candidates):
        notes.append("Agree is how similar chat answers are to the reference's answers.")
    results = (report.reference, *report.candidates)
    if not report.settings.allow_code_exec and any(_skipped_code(r) for r in results):
        notes.append("Code cases were skipped because code execution was not enabled.")
    notes.extend(report.notes)
    return tuple(notes)


def text_forced_note(labels: Sequence[str], *, ollama: bool) -> str:
    """The note for logit metrics measured by forcing text rather than token ids: "Ollama"
    when every such candidate is an Ollama model, otherwise the labels by name."""
    if ollama:
        return (
            "Ollama logit metrics are text-forced; on English text they matched "
            "llama-server's exact token-id forcing (docs/calibration.md). Non-Latin text may "
            "read higher."
        )
    return (
        f"Logit metrics for {', '.join(labels)} are text-forced; on English text this matched "
        "exact token-id forcing (docs/calibration.md). Non-Latin text may read higher."
    )


def _skipped_code(result: CandidateResult) -> bool:
    task = task_for(result, "code")
    return task is not None and task.skipped > 0
