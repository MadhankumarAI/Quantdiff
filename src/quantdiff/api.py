"""The Python entry points: compare() for a full run and write_run() to persist one."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from quantdiff.cache import ReferenceCache
from quantdiff.card import render_html, render_markdown
from quantdiff.errors import RenderError, SpecError, SuiteError
from quantdiff.png import render_png
from quantdiff.preflight import MIN_CONTEXT_PROBE_TOKENS
from quantdiff.report import save_report
from quantdiff.runner import ProgressFn, RunPlan, execute
from quantdiff.spec import parse_spec
from quantdiff.suites import (
    BUILTIN_SUITES,
    load_builtin,
    load_cases_file,
    load_scoring_prompts,
    scoring_prompts_from_cases,
)
from quantdiff.types import CandidateSpec, Report, RunSettings, ScoringPrompt, TaskCase
from quantdiff.verdict import Verdict

DEFAULT_SUITES = ("json", "tools", "chat")
PathLike = str | os.PathLike[str]

_TITLE_SEPARATORS: Final = ":-_/."
_MIN_TITLE_PREFIX: Final = 8
_CHARS_PER_TOKEN: Final = 4
_SIZE_PATTERN: Final = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]*)")
_SIZE_UNITS: Final[dict[str, int]] = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "t": 10**12,
    "tb": 10**12,
}
"""Decimal units, as download pages and file managers quote sizes."""


def build_plan(
    ref: str | CandidateSpec,
    candidates: Sequence[str | CandidateSpec],
    *,
    suites: Sequence[str] | None = None,
    prompts: PathLike | None = None,
    scoring_prompts: PathLike | None = None,
    top_k: int = 10,
    score_tokens: int = 32,
    max_cases: int | None = None,
    allow_code_exec: bool = False,
    seed: int = 0,
    hf_repo: str | None = None,
    offline: bool = False,
    preflight: bool = True,
    context_probe_tokens: int = 6000,
    title: str | None = None,
    max_size: int | str | None = None,
) -> RunPlan:
    """Resolve specs, suites and settings into a validated RunPlan.

    With no `suites` and no `prompts`, the default suites run (plus code when
    `allow_code_exec` is set). With `prompts` and no `suites`, only the user's prompts run.
    With `prompts` and no `scoring_prompts`, the logit tier scores the user's own prompt
    texts. `max_cases` caps each suite and the number of scoring prompts. Without a
    `title`, one is derived from the labels. `max_size` is the size budget for the pick,
    in bytes or as text such as "6GB", "6.5G" or "800MB" (decimal units).
    """
    if not candidates:
        raise SpecError("at least one candidate is required")
    if not 1 <= top_k <= 20:
        raise SpecError("top_k must be between 1 and 20")
    if score_tokens < 0:
        raise SpecError("score_tokens must be zero or positive")
    if max_cases is not None and max_cases < 1:
        raise SpecError("max_cases must be at least 1")
    if preflight and context_probe_tokens != 0 and context_probe_tokens < MIN_CONTEXT_PROBE_TOKENS:
        raise SpecError(
            f"context_probe_tokens must be 0 (off) or at least {MIN_CONTEXT_PROBE_TOKENS}"
        )

    budget = parse_size(max_size) if isinstance(max_size, str) else max_size
    if budget is not None and budget < 1:
        raise SpecError("max_size must be a positive number of bytes")
    specs = _unique_labels([_as_spec(ref), *(_as_spec(c) for c in candidates)])
    suite_names = _resolve_suites(suites, has_prompts=prompts is not None, code=allow_code_exec)
    user_cases = () if prompts is None else load_cases_file(prompts)[:max_cases]
    cases = _combine_cases(suite_names, user_cases, max_cases)
    scoring = _load_scoring(scoring_prompts, user_cases, max_cases) if score_tokens > 0 else ()

    settings = RunSettings(
        suites=tuple(suite_names),
        top_k=top_k,
        score_tokens=score_tokens,
        allow_code_exec=allow_code_exec,
        seed=seed,
        prompts_file=None if prompts is None else Path(prompts).name,
        max_size_bytes=budget,
        longest_prompt_tokens=_longest_prompt_tokens(cases, scoring),
    )
    return RunPlan(
        reference=specs[0],
        candidates=tuple(specs[1:]),
        cases=cases,
        scoring=scoring,
        settings=settings,
        title=title or _default_title([spec.label for spec in specs]),
        hf_repo=hf_repo,
        offline=offline,
        preflight=preflight,
        context_probe_tokens=context_probe_tokens,
    )


def compare(
    ref: str | CandidateSpec,
    candidates: Sequence[str | CandidateSpec],
    *,
    suites: Sequence[str] | None = None,
    prompts: PathLike | None = None,
    scoring_prompts: PathLike | None = None,
    top_k: int = 10,
    score_tokens: int = 32,
    max_cases: int | None = None,
    allow_code_exec: bool = False,
    seed: int = 0,
    hf_repo: str | None = None,
    offline: bool = False,
    preflight: bool = True,
    context_probe_tokens: int = 6000,
    title: str | None = None,
    max_size: int | str | None = None,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> Report:
    """Compare candidate model servers against a reference and return the Report.

    Options match build_plan(). `use_cache` reuses reference outputs from earlier runs with
    the same model, suites and settings.
    """
    plan = build_plan(
        ref,
        candidates,
        suites=suites,
        prompts=prompts,
        scoring_prompts=scoring_prompts,
        top_k=top_k,
        score_tokens=score_tokens,
        max_cases=max_cases,
        allow_code_exec=allow_code_exec,
        seed=seed,
        hf_repo=hf_repo,
        offline=offline,
        preflight=preflight,
        context_probe_tokens=context_probe_tokens,
        title=title,
        max_size=max_size,
    )
    return execute(plan, cache=ReferenceCache() if use_cache else None, progress=progress)


@dataclass(frozen=True, slots=True)
class RunFiles:
    """Where write_run() put each artifact. `png` is None when no browser was available."""

    directory: Path
    report: Path
    html: Path
    markdown: Path
    png: Path | None
    png_error: str | None = None


def write_run(
    report: Report,
    out_dir: PathLike = "runs",
    *,
    png: bool = True,
    verdict: Verdict | None = None,
) -> RunFiles:
    """Write report.json, card.html, card.md and (when a browser is available) card.png
    into a new timestamped directory under `out_dir`. The cards show `verdict`, or
    judge(report) when it is None.

    PNG rendering is best effort: a missing browser is reported in `RunFiles.png_error`
    rather than raised, because the other artifacts are already complete.
    """
    run_dir = _new_run_dir(Path(out_dir))
    html = render_html(report, verdict=verdict)
    files = RunFiles(
        directory=run_dir,
        report=run_dir / "report.json",
        html=run_dir / "card.html",
        markdown=run_dir / "card.md",
        png=None,
    )
    save_report(report, files.report)
    files.html.write_text(html, encoding="utf-8")
    files.markdown.write_text(render_markdown(report, verdict=verdict), encoding="utf-8")
    if not png:
        return files
    try:
        return replace(files, png=render_png(html, run_dir / "card.png"))
    except RenderError as exc:
        return replace(files, png_error=str(exc))


def parse_size(text: str) -> int:
    """Bytes in a size such as "6GB", "6.5G", "800MB", "800M" or "6000000000".

    Units are decimal (1 GB is 10**9 bytes) and case does not matter. Raises SpecError
    for anything else, including binary units such as GiB, so a budget is never misread.
    """
    match = _SIZE_PATTERN.fullmatch(text.strip().lower())
    unit = None if match is None else _SIZE_UNITS.get(match.group(2))
    if match is None or unit is None:
        raise SpecError(
            f"max size {text!r} is not a size; use a value such as 6GB, 6.5G, 800MB or a "
            "byte count (decimal units)"
        )
    number = match.group(1)
    if unit == 1 and "." in number:
        raise SpecError(f"max size {text!r} is a fraction of a byte; add a unit such as GB")
    size = round(float(number) * unit)
    if size < 1:
        raise SpecError(f"max size {text!r} must be more than zero")
    return size


def _new_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def _as_spec(value: str | CandidateSpec) -> CandidateSpec:
    return value if isinstance(value, CandidateSpec) else parse_spec(value)


def _unique_labels(specs: Iterable[CandidateSpec]) -> list[CandidateSpec]:
    seen: dict[str, int] = {}
    unique = []
    for spec in specs:
        count = seen.get(spec.label, 0) + 1
        seen[spec.label] = count
        unique.append(spec if count == 1 else replace(spec, label=f"{spec.label} #{count}"))
    return unique


def _default_title(labels: Sequence[str]) -> str:
    """Name a run after its models, reference first.

    Labels that share a prefix ending in one of `:-_/.` (at least 8 characters) become
    "<prefix>: q8_0 vs q4_K_M"; anything else becomes "<reference> vs <n> candidates".
    """
    reference, *others = labels
    prefix = _shared_prefix(labels)
    if prefix:
        return f"{prefix[:-1]}: " + " vs ".join(label[len(prefix) :] for label in labels)
    if len(others) == 1:
        return f"{reference} vs {others[0]}"
    return f"{reference} vs {len(others)} candidates"


def _shared_prefix(labels: Sequence[str]) -> str | None:
    common = os.path.commonprefix(list(labels))
    end = max(common.rfind(separator) for separator in _TITLE_SEPARATORS) + 1
    prefix = common[:end]
    if len(prefix) < _MIN_TITLE_PREFIX or any(len(label) == end for label in labels):
        return None
    return prefix


def _resolve_suites(suites: Sequence[str] | None, *, has_prompts: bool, code: bool) -> list[str]:
    if suites is None:
        if has_prompts:
            return []
        return [*DEFAULT_SUITES, "code"] if code else list(DEFAULT_SUITES)
    names = list(dict.fromkeys(suites))
    unknown = [name for name in names if name not in BUILTIN_SUITES]
    if unknown:
        known = ", ".join(BUILTIN_SUITES)
        raise SuiteError(f"unknown suite(s): {', '.join(unknown)}; choose from {known}")
    if "code" in names and not code:
        raise SpecError(
            "the code suite runs model-written code; add --allow-code-exec to enable it"
        )
    return names


def _combine_cases(
    suite_names: Sequence[str], user_cases: Sequence[TaskCase], max_cases: int | None
) -> tuple[TaskCase, ...]:
    cases: list[TaskCase] = []
    for name in suite_names:
        cases.extend(load_builtin(name)[:max_cases])
    cases.extend(user_cases)
    counts = Counter(case.id for case in cases)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise SuiteError(f"duplicate case ids across suites: {', '.join(duplicates)}")
    if not cases:
        raise SuiteError("nothing to run: no suites selected and no prompts file given")
    return tuple(cases)


def _load_scoring(
    path: PathLike | None, user_cases: Sequence[TaskCase], max_cases: int | None
) -> tuple[ScoringPrompt, ...]:
    if path is None and user_cases:
        return scoring_prompts_from_cases(user_cases)[:max_cases]
    return load_scoring_prompts(path)[:max_cases]


def _longest_prompt_tokens(
    cases: Sequence[TaskCase], scoring: Sequence[ScoringPrompt]
) -> int | None:
    """Rough token count of the longest prompt, at about four characters per token."""
    lengths = [sum(len(message.content) for message in case.messages) for case in cases]
    lengths += [len(prompt.text) for prompt in scoring]
    return math.ceil(max(lengths) / _CHARS_PER_TOKEN) if lengths else None
