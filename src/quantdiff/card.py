"""Scorecards: one recommendation and one table, rendered for a terminal, markdown, and HTML.

Every card answers "which download should I run?" first: the verdict headline leads, and
each candidate row carries the status the verdict engine gave it (RUN, OK, USABLE, UNSURE,
AVOID, FAILED), in the engine's order, under the reference row, with any caveats the engine
attached. All three renderers share one
column model so a number never differs between formats. Columns with nothing to show are
left out. The HTML card is a single self-contained document with no scripts and no
external assets, so it can be opened offline, screenshotted, and attached anywhere.
"""

from __future__ import annotations

import html
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Final, Literal

from quantdiff.report import format_kld, format_rate, format_top1, task_for, text_forced_note
from quantdiff.suites import BUILTIN_SUITES, load_builtin
from quantdiff.types import BackendKind, CandidateResult, Report, TaskKind
from quantdiff.verdict import (
    CALIBRATED_TOP_K,
    LABEL_SEPARATORS,
    CandidateVerdict,
    KldThresholds,
    ServerFinding,
    Status,
    TaskDelta,
    Verdict,
    display_labels,
    judge,
    kld_band,
    kld_thresholds,
)

MISSING: Final = "n/a"
BASELINE: Final = "-"
"""Shown in the reference row for metrics that compare a model against the reference."""
TERMINAL_WIDTH: Final = 100
P99_MIN_POSITIONS: Final = 1000
"""Fewer scored positions than this make a 99th percentile little more than the maximum."""
SIGNIFICANT_MARK: Final = "*"
SIGNIFICANT_FOOTNOTE: Final = "* significant (paired, 95%)"
REFERENCE_CHIP: Final = "REF"
NOTE_CHIP: Final = "NOTE"
"""Chip for a server finding that does not affect the scores on the card."""
USER_PROMPTS: Final = "your prompts"
STATUS_CHIPS: Final[dict[Status, str]] = {
    "recommended": "RUN",
    "ok": "OK",
    "usable": "USABLE",
    "avoid": "AVOID",
    "inconclusive": "UNSURE",
    "failed": "FAILED",
}
NEAR_BAR: Final = "near bar"
"""Tag next to the chip of a candidate whose KLD sits close enough to a band edge that a
rerun could change its status."""
CAVEAT_PREFIX: Final = "caveat:"
OVER_BUDGET: Final = "over budget"
OVER_BUDGET_SHORT: Final = "(over)"
CACHED: Final = "cached"
"""Marker on a reference speed that comes from an earlier run, not this one."""
_LABEL_MIN_WIDTH: Final = 12
_GAP: Final = "  "
_ELLIPSIS: Final = "..."
_SEVERITY_ORDER: Final = ("fail", "warn", "skip", "ok")

Group = Literal["logit", "task"]
Bar = Literal["rate", "kld"]
Align = Literal["left", "right"]
SpeedMode = Literal["server", "wall"]


# Column model -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Metric:
    short: str
    """Terminal header."""
    name: str
    """Markdown and HTML header."""
    group: Group
    value: Callable[[CandidateResult], float | None]
    text: Callable[[float], str]
    bar: Bar
    vs_reference: bool
    """The metric compares a model with the reference, so the reference row omits it."""
    task: TaskKind | None = None
    """The suite of a pass-rate column, whose candidate cells carry the paired delta."""


def _top1(result: CandidateResult) -> float | None:
    return None if result.logit is None else result.logit.top1_agreement


def _kld_mean(result: CandidateResult) -> float | None:
    return None if result.logit is None else result.logit.kld_mean


def _kld_p99(result: CandidateResult) -> float | None:
    logit = result.logit
    if logit is None or logit.positions < P99_MIN_POSITIONS:
        return None
    return logit.kld_p99


def _task_rate(kind: TaskKind) -> Callable[[CandidateResult], float | None]:
    def read(result: CandidateResult) -> float | None:
        task = task_for(result, kind)
        return None if task is None else task.rate

    return read


def _agreement(result: CandidateResult) -> float | None:
    return None if result.agreement is None else result.agreement.mean_similarity


def _task_metric(kind: TaskKind, name: str) -> _Metric:
    return _Metric(
        name, name, "task", _task_rate(kind), format_rate, "rate", vs_reference=False, task=kind
    )


_METRICS: Final = (
    _Metric("Top-1", "Top-1", "logit", _top1, format_top1, "rate", vs_reference=True),
    _Metric("KLD", "KLD mean", "logit", _kld_mean, format_kld, "kld", vs_reference=True),
    _Metric("KLD99", "KLD p99", "logit", _kld_p99, format_kld, "kld", vs_reference=True),
    _task_metric("json", "JSON"),
    _task_metric("tools", "Tools"),
    _task_metric("code", "Code"),
    _Metric("Agree", "Agree", "task", _agreement, format_rate, "rate", vs_reference=True),
)
_SUITE_NAMES: Final[dict[TaskKind, str]] = {
    "json": "JSON",
    "tools": "Tools",
    "code": "Code",
    "chat": "Chat",
}


def format_size(size_bytes: int) -> str:
    """Decimal megabytes or gigabytes, the way model hubs and Ollama print file sizes."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.1f} GB"
    megabytes = size_bytes / 1e6
    return f"{megabytes:.1f} MB" if megabytes < 10 else f"{megabytes:.0f} MB"


def format_size_change(change: float) -> str:
    """A fractional size change as a signed whole percentage, such as "-25%" or "0%"."""
    percent = round(change * 100)
    return f"{percent:+d}%" if percent else "0%"


def _points(value: float) -> str:
    rounded = round(value)
    return f"{rounded:+d}" if rounded else "0"


def format_delta(delta: TaskDelta) -> str:
    """Compact paired delta in points, such as "-30*" (significant) or "-5".

    A gain or tie that is not significant reads "=", so a lucky case or two never looks like
    a candidate beating the reference.
    """
    if delta.significant:
        return _points(delta.delta.estimate) + SIGNIFICANT_MARK
    return "=" if round(delta.delta.estimate) >= 0 else _points(delta.delta.estimate)


def format_interval(delta: TaskDelta) -> str:
    """The compact delta with its 95% interval, such as "-30* [-48, -12]" or "= [-5, +25]"."""
    interval = delta.delta
    return f"{format_delta(delta)} [{_points(interval.low)}, {_points(interval.high)}]"


def is_regression(delta: TaskDelta) -> bool:
    """A significant loss against a reference that is reliable on this suite."""
    return delta.significant and delta.reference_reliable and delta.delta.estimate < 0


def _format_speed(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


@dataclass(frozen=True, slots=True)
class _Row:
    """One table row: the reference, or a candidate with the verdict engine's call on it."""

    result: CandidateResult
    label: str
    call: CandidateVerdict | None
    """None for the reference row."""

    @property
    def is_reference(self) -> bool:
        return self.call is None

    @property
    def chip(self) -> str:
        return REFERENCE_CHIP if self.call is None else STATUS_CHIPS[self.call.status]

    @property
    def status(self) -> Status | None:
        return None if self.call is None else self.call.status

    def delta(self, kind: TaskKind | None) -> TaskDelta | None:
        if self.call is None or kind is None:
            return None
        return next((d for d in self.call.task_deltas if d.kind == kind), None)

    @property
    def near_bar(self) -> bool:
        return self.call is not None and self.call.near_bar

    @property
    def over_budget(self) -> bool:
        return self.call is not None and self.call.fits_budget is False

    @property
    def size_bytes(self) -> int | None:
        if self.call is not None and self.call.size_bytes is not None:
            return self.call.size_bytes
        return None if self.result.info is None else self.result.info.size_bytes


@dataclass(frozen=True, slots=True)
class _Cell:
    text: str
    value: float | None = None
    """The number behind `text`, for bars; None for placeholders."""
    delta: TaskDelta | None = None
    muted: bool = False
    marker: str = ""
    """A short word shown after the value, such as "cached"."""


@dataclass(frozen=True, slots=True)
class _View:
    """Everything the three renderers share for one report."""

    report: Report
    verdict: Verdict
    rows: tuple[_Row, ...]
    prefix: str
    backend: BackendKind | None
    """The backend every model ran on, or None when they differ."""
    metrics: tuple[_Metric, ...]
    """Metric columns that have at least one value to show."""
    unreliable: frozenset[TaskKind]
    """Suites where the reference passes too few cases to judge a regression."""
    speed: SpeedMode | None
    show_size: bool

    @property
    def reference(self) -> _Row:
        return self.rows[0]

    @property
    def candidates(self) -> tuple[_Row, ...]:
        return self.rows[1:]

    @property
    def thresholds(self) -> KldThresholds:
        return kld_thresholds(self.report.settings.top_k)

    @property
    def model_name(self) -> str:
        return self.prefix.rstrip(LABEL_SEPARATORS)

    @property
    def has_significant(self) -> bool:
        return any(d.significant for row in self.candidates for d in _row_deltas(self, row))

    def context(self) -> list[tuple[str, str, str]]:
        """Header facts as (name, value, backend) triples; backend is "" when not shown."""
        reference = self.reference.result.spec
        items = []
        if self.prefix:
            items.append(("Model", self.model_name, ""))
        items.append(("Reference", reference.label, "" if self.backend else reference.kind))
        if self.backend:
            items.append(("Backend", self.backend, ""))
        budget = budget_text(self.report)
        if budget is not None:
            items.append(("Budget", budget, ""))
        return items

    def cell(self, metric: _Metric, row: _Row) -> _Cell:
        if row.is_reference and metric.vs_reference:
            return _Cell(BASELINE)
        value = metric.value(row.result)
        if value is None:
            return _Cell(MISSING)
        muted = metric.task is not None and metric.task in self.unreliable
        return _Cell(metric.text(value), value, row.delta(metric.task), muted)

    def speed_cell(self, row: _Row) -> _Cell:
        perf = row.result.perf
        if perf is None or perf.tokens_per_second is None:
            return _Cell(MISSING)
        text = _format_speed(perf.tokens_per_second)
        if row.is_reference and perf.source == "cached":
            return _Cell(text, perf.tokens_per_second, muted=True, marker=CACHED)
        return _Cell(text, perf.tokens_per_second, muted=self.speed == "wall")

    def size_cell(self, row: _Row) -> tuple[str, str]:
        """Size text and its change against the reference, such as ("398 MB", "-25%")."""
        size = row.size_bytes
        if size is None:
            return MISSING, ""
        change = None if row.call is None else row.call.size_change
        return format_size(size), "" if change is None else format_size_change(change)

    def covers_everyone(self, finding: ServerFinding) -> bool:
        """True when a finding applies to every model on a card with more than one."""
        everyone = {row.result.spec.label for row in self.rows}
        return len(everyone) > 1 and everyone <= set(finding.labels)

    def who(self, finding: ServerFinding) -> str:
        """The models a finding applies to: "both models" or "all N models" when it is every
        one of them, otherwise their short labels."""
        if self.covers_everyone(finding):
            count = len(self.rows)
            return "both models" if count == 2 else f"all {count} models"
        names = {row.result.spec.label: row.label for row in self.rows}
        return ", ".join(names.get(label, label) for label in dict.fromkeys(finding.labels))


def _row_deltas(view: _View, row: _Row) -> list[TaskDelta]:
    """Deltas the table actually shows for this row."""
    return [delta for metric in view.metrics if (delta := view.cell(metric, row).delta) is not None]


def _view(report: Report, verdict: Verdict) -> _View:
    results = (report.reference, *report.candidates)
    names = display_labels(report)
    reference = report.reference.spec.label
    # display_labels drops the stem every label shares; the header states it once instead.
    prefix = reference[: len(reference) - len(names[reference])]
    by_label = {result.spec.label: result for result in report.candidates}
    rows = [_Row(report.reference, names[reference], None)]
    rows += [
        _Row(by_label[call.label], names[call.label], call)
        for call in verdict.candidates
        if call.label in by_label
    ]
    backends = {result.spec.kind for result in results}
    unreliable = frozenset(
        delta.kind
        for call in verdict.candidates
        for delta in call.task_deltas
        if not delta.reference_reliable
    )
    metrics = tuple(
        metric
        for metric in _METRICS
        if any(
            metric.value(row.result) is not None
            for row in rows
            if not (row.is_reference and metric.vs_reference)
        )
    )
    return _View(
        report=report,
        verdict=verdict,
        rows=tuple(rows),
        prefix=prefix,
        backend=backends.pop() if len(backends) == 1 else None,
        metrics=metrics,
        unreliable=unreliable,
        speed=_speed_mode(rows),
        show_size=any(row.size_bytes is not None for row in rows),
    )


def _speed_mode(rows: Sequence[_Row]) -> SpeedMode | None:
    """ "server" when every shown speed is the server's own decode timing, "wall" when any is
    wall-clock or a cached candidate speed, None when there is nothing to show. A cached
    reference speed is shown muted and marked "cached", and does not set the mode."""
    sources = [
        (row.is_reference, perf.source)
        for row in rows
        if (perf := row.result.perf) is not None and perf.tokens_per_second is not None
    ]
    if not sources:
        return None
    fresh = {source for reference, source in sources if not (reference and source == "cached")}
    return "server" if fresh <= {"server"} else "wall"


def _cached_reference(view: _View) -> bool:
    perf = view.reference.result.perf
    return (
        view.speed is not None
        and perf is not None
        and perf.tokens_per_second is not None
        and perf.source == "cached"
    )


def _speed_header(view: _View) -> str:
    return "tok/s (wall)" if view.speed == "wall" else "tok/s"


def finding_chip(item: ServerFinding) -> str:
    """FAIL or WARN for a finding that may have changed the scores, NOTE for one that did
    not, so a harmless finding never looks like a failure on a shared card."""
    return item.finding.severity.upper() if item.affects_scores else NOTE_CHIP


def _findings(view: _View) -> list[ServerFinding]:
    """Findings that could change the scores first, then by severity."""
    return sorted(
        view.verdict.findings,
        key=lambda f: (not f.affects_scores, _SEVERITY_ORDER.index(f.finding.severity)),
    )


@dataclass(frozen=True, slots=True)
class _Lead:
    """The verdict box: the headline, the remedy for an inconclusive verdict, the rest."""

    headline: str
    remedy: str | None
    details: tuple[str, ...]


def _lead(verdict: Verdict) -> _Lead:
    """The headline, the remedy (what to rerun to settle an inconclusive verdict) on its own
    line, and the remaining details."""
    return _Lead(verdict.headline, verdict.remedy, verdict.details)


def _score_alerts(view: _View) -> list[str]:
    """One line per finding that may have changed the scores, for the verdict box."""
    return [
        f"Server check on {view.who(f)}: {_sentence(f.finding.message, capitalize=False)} "
        "This may affect the scores; see Server checks."
        for f in _findings(view)
        if f.affects_scores
    ]


def _case_counts(report: Report) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for result in (report.reference, *report.candidates):
        for task in result.tasks:
            counts[task.kind] = max(counts.get(task.kind, 0), task.total)
        if result.agreement is not None:
            counts["chat"] = max(counts.get("chat", 0), result.agreement.cases)
    order = ("json", "tools", "code", "chat")
    return [(kind, counts[kind]) for kind in order if kind in counts]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def summary_line(report: Report) -> str:
    """One line of scale, such as "5 models, 115 cases, 24 scoring prompts"."""
    parts = [_plural(1 + len(report.candidates), "model")]
    cases = sum(count for _, count in _case_counts(report))
    if cases:
        parts.append(_plural(cases, "case"))
    prompts = max((c.logit.prompts for c in report.candidates if c.logit is not None), default=0)
    if prompts:
        parts.append(_plural(prompts, "scoring prompt"))
    return ", ".join(parts)


def prompts_file_name(value: str) -> str:
    """The base name of a prompts file. Cards are shared publicly, so never a full path,
    whichever separator the path was written with."""
    return PureWindowsPath(value).name


def _user_case_count(report: Report) -> int | None:
    """How many cases came from the user's prompts file, or None when the report does not
    say. Built-in suites and the prompts file never share a case id, so the user's cases
    are the ones outside every built-in suite that ran."""
    if not report.settings.suites:
        return sum(count for _, count in _case_counts(report)) or None
    seen: set[str] = set()
    for result in (report.reference, *report.candidates):
        seen.update(outcome.case_id for outcome in result.outcomes)
        if result.agreement is not None:
            seen.update(case_id for case_id, _ in result.agreement.per_case)
    if not seen:
        return None
    builtin = {
        case.id
        for name in report.settings.suites
        if name in BUILTIN_SUITES
        for case in load_builtin(name)
    }
    return len(seen - builtin)


def _suites_text(report: Report) -> str:
    """The suites that ran, such as "json, tools + your prompts (12)"."""
    settings = report.settings
    builtin = ", ".join(settings.suites)
    if not settings.prompts_file:
        return builtin or "none"
    count = _user_case_count(report)
    mine = USER_PROMPTS if count is None else f"{USER_PROMPTS} ({count})"
    return f"{builtin} + {mine}" if builtin else mine


def budget_text(report: Report) -> str | None:
    """The --max-size budget as a size, such as "6.0 GB", or None when none was given."""
    budget = report.settings.max_size_bytes
    return None if budget is None else format_size(budget)


def _settings_items(report: Report) -> list[tuple[str, str]]:
    settings = report.settings
    cases = ", ".join(f"{kind} {count}" for kind, count in _case_counts(report))
    items = [
        ("Suites", _suites_text(report)),
        ("Top-k", str(settings.top_k)),
        ("Score tokens", str(settings.score_tokens)),
        ("Cases", cases or "none"),
        ("Code execution", "on" if settings.allow_code_exec else "off"),
        ("Seed", str(settings.seed)),
    ]
    budget = budget_text(report)
    if budget is not None:
        items.append(("Budget", budget))
    if settings.prompts_file:
        items.append(("Prompts file", prompts_file_name(settings.prompts_file)))
    return items


def _settings_line(report: Report) -> str:
    return "; ".join(f"{name} {value}" for name, value in _settings_items(report))


def _notes(view: _View) -> list[str]:
    """Short footnotes that explain how to read this particular card."""
    report = view.report
    shown = {metric.name for metric in view.metrics}
    notes = []
    if any(row.delta(metric.task) for row in view.candidates for metric in view.metrics):
        notes.append(
            "Signed numbers are percentage points against the reference on the same cases."
        )
    if shown & {"KLD mean", "KLD p99"}:
        notes.append(
            f"KLD is a lower bound computed from the top {report.settings.top_k} tokens; "
            "lower means closer to the reference."
        )
        notes.append(kld_band_note(report.settings.top_k))
    approximate = [
        row
        for row in view.candidates
        if row.result.logit is not None and not row.result.logit.exact_token_ids
    ]
    if approximate:
        ollama = all(row.result.spec.kind == "ollama" for row in approximate)
        notes.append(text_forced_note([row.label for row in approximate], ollama=ollama))
    for kind in sorted(view.unreliable):
        name = _SUITE_NAMES[kind]
        if name in shown:
            notes.append(
                f"{name} says little here: the reference itself passes under half of these "
                "cases, so this suite is not used to judge the candidates."
            )
    if "Agree" in shown:
        notes.append("Agree is how similar chat answers are to the reference's answers.")
    if view.speed == "wall":
        notes.append(
            "tok/s (wall) is timed from outside the server and includes prompt processing; "
            "treat it as rough."
        )
    if _cached_reference(view):
        notes.append(
            "The reference tok/s marked cached comes from an earlier run; compare it loosely."
        )
    code = [task_for(row.result, "code") for row in view.rows]
    if not report.settings.allow_code_exec and any(t is not None and t.skipped for t in code):
        notes.append("Code cases were skipped; add --allow-code-exec to score them.")
    notes.extend(report.notes)
    return notes


def kld_band_note(top_k: int = CALIBRATED_TOP_K) -> str:
    """The verdict engine's KLD bands for this --top-k in words, built from its own
    thresholds so the card can never disagree with the verdict."""
    bars = kld_thresholds(top_k)
    parts = []
    floor = 0.0
    for limit in (bars.near_lossless, bars.close, bars.large):
        tag = " (the bar for RUN)" if limit == bars.close else ""
        parts.append(f"under {limit:g} {kld_band(floor, bars)}{tag}")
        floor = limit
    return (
        f"KLD bands (top-{top_k}, calibrated against llama.cpp full-vocabulary KLD): "
        f"{', '.join(parts)}, else {kld_band(floor, bars)}."
    )


def _reasons(view: _View) -> list[tuple[_Row, CandidateVerdict]]:
    """Candidates whose call comes with reasons or caveats, in verdict order."""
    return [
        (row, row.call)
        for row in view.candidates
        if row.call is not None and (row.call.reasons or row.call.caveats)
    ]


def _tagged_chip(row: _Row) -> str:
    """The chip with the near-bar tag, for the plain-text formats."""
    return f"{row.chip} ({NEAR_BAR})" if row.near_bar else row.chip


def _errors(view: _View) -> list[tuple[str, str]]:
    return [(row.label, error) for row in view.candidates for error in row.result.errors]


# Terminal ---------------------------------------------------------------------------------

_RESET: Final = "\x1b[0m"
_BOLD: Final = "1"
_DIM: Final = "2"
_ITALIC: Final = "3"
_GREEN: Final = "32"
_YELLOW: Final = "33"
_RED: Final = "31"
_BLUE: Final = "34"
_STATUS_CODES: Final[dict[Status, tuple[str, ...]]] = {
    "recommended": (_BOLD, _GREEN),
    "ok": (),
    "usable": (_BOLD, _BLUE),
    "avoid": (_BOLD, _RED),
    "inconclusive": (_YELLOW,),
    "failed": (_RED,),
}
_SEVERITY_CODES: Final[dict[str, tuple[str, ...]]] = {
    "fail": (_RED,),
    "warn": (_YELLOW,),
    "skip": (_DIM,),
    "ok": (_DIM,),
}


@dataclass(frozen=True, slots=True)
class _Ansi:
    enabled: bool

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}{_RESET}"


@dataclass(frozen=True, slots=True)
class _TermCell:
    text: str
    delta: str = ""
    codes: tuple[str, ...] = ()
    delta_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TermColumn:
    header: str
    align: Align
    cells: tuple[_TermCell, ...]

    @property
    def text_width(self) -> int:
        return max((len(cell.text) for cell in self.cells), default=0)

    @property
    def delta_width(self) -> int:
        return max((len(cell.delta) for cell in self.cells), default=0)

    @property
    def width(self) -> int:
        delta = self.delta_width
        return max(len(self.header), self.text_width + (delta + 1 if delta else 0))

    def render(self, cell: _TermCell, width: int, paint: _Ansi, row_codes: tuple[str, ...]) -> str:
        if not self.delta_width:
            text = _pad(_truncate(cell.text, width), width, self.align)
            return paint(text, *row_codes, *cell.codes)
        # Values and deltas get sub-columns so the numbers line up across rows.
        used = self.text_width + 1 + self.delta_width
        lead = " " * (width - used) + cell.text.rjust(self.text_width)
        tail = cell.delta.ljust(self.delta_width)
        return (
            paint(lead, *row_codes, *cell.codes) + " " + paint(tail, *row_codes, *cell.delta_codes)
        )


def _pack(facts: Sequence[str], width: int = TERMINAL_WIDTH, gap: str = "   ") -> list[str]:
    """Lay header facts out on as few lines as fit, never splitting one fact across lines."""
    lines: list[str] = []
    for fact in facts:
        if len(fact) > width:
            lines += textwrap.wrap(fact, width, break_on_hyphens=False)
        elif lines and len(lines[-1]) + len(gap) + len(fact) <= width:
            lines[-1] += gap + fact
        else:
            lines.append(fact)
    return lines


def render_terminal(report: Report, *, color: bool = False, verdict: Verdict | None = None) -> str:
    """Render a fixed-width scorecard that fits a 100-column terminal."""
    view = _view(report, verdict or judge(report))
    paint = _Ansi(color)
    facts = [
        f"{name}: {_plain(value)}" + (f" ({backend})" if backend else "")
        for name, value, backend in view.context()
    ]
    lines = [
        paint(_plain(report.title), _BOLD),
        paint(summary_line(report), _DIM),
        *(paint(line, _DIM) for line in _pack(facts)),
        "",
    ]
    lead = _lead(view.verdict)
    lines += [paint(line, _BOLD) for line in _wrap(_plain(lead.headline), first="> ")]
    if lead.remedy is not None:
        lines += [paint(line, _BOLD, _YELLOW) for line in _wrap(_plain(lead.remedy), first="  ")]
    for sentence in (*lead.details, *_score_alerts(view)):
        lines += _wrap(_plain(sentence), first="  ", rest="  ")
    lines += ["", *_terminal_table(view, paint)]
    if view.has_significant:
        lines.append(paint(SIGNIFICANT_FOOTNOTE, _DIM))
    lines += _terminal_reasons(view, paint)
    findings = _findings(view)
    if findings:
        lines += ["", paint("Server checks", _BOLD)]
        for finding in findings:
            lines += _terminal_finding(view, finding, paint)
    errors = _errors(view)
    if errors:
        lines += ["", paint("Errors", _BOLD)]
        for label, error in errors:
            lines += _wrap(f"{_plain(label)}: {_plain(error)}", first="  ", rest="    ")
    lines += ["", *_wrap(f"Settings: {_plain(_settings_line(report))}")]
    notes = _notes(view)
    if notes:
        lines += ["", paint("Notes", _BOLD)]
        for number, note in enumerate(notes, start=1):
            lines += _wrap(_plain(note), first=f"  {number}. ", rest="     ")
    return "\n".join(lines) + "\n"


def _terminal_reasons(view: _View, paint: _Ansi) -> list[str]:
    reasons = _reasons(view)
    if not reasons:
        return []
    width = max(len(_tagged_chip(row)) for row, _ in reasons)
    indent = " " * (width + 4)
    lines = ["", paint("Why", _BOLD)]
    for row, call in reasons:
        tag = paint(_tagged_chip(row).ljust(width), *_STATUS_CODES[call.status])
        body = f"{_plain(row.label)}: {_plain(' '.join(call.reasons))}".rstrip(": ")
        lines += _wrap(body, first=f"  {tag}  ", rest=indent)
        for caveat in call.caveats:
            text = f"{CAVEAT_PREFIX} {_plain(_sentence(caveat, capitalize=False))}"
            lines += [paint(line, _YELLOW) for line in _wrap(text, first=indent, rest=indent)]
    return lines


def _terminal_finding(view: _View, item: ServerFinding, paint: _Ansi) -> list[str]:
    finding = item.finding
    codes = _SEVERITY_CODES[finding.severity] if item.affects_scores else (_DIM,)
    tag = paint(finding_chip(item).ljust(4), *codes)
    message = _sentence(_plain(finding.message), capitalize=False)
    head = f"{_plain(view.who(item))}, {_plain(finding.check)}: {message}"
    indent = "        "
    lines = _wrap(head, first=f"  {tag}  ", rest=indent)
    impact = _wrap(_sentence(_plain(item.impact)), first=indent, rest=indent)
    lines += [paint(line, _DIM) for line in impact]
    if finding.fix:
        lines += _wrap(f"Fix: {_plain(finding.fix)}", first=indent, rest=indent)
    return lines


def _terminal_table(view: _View, paint: _Ansi) -> list[str]:
    columns = _terminal_columns(view, backend=view.backend is None)
    widths = _fit(columns)
    if widths is None:
        # Mixed backends and long labels: the label is worth more than the backend name,
        # which the markdown and HTML cards still show.
        columns = _terminal_columns(view, backend=False)
        widths = _fit(columns) or _squeezed(columns)
    header = _GAP.join(_pad(c.header, w, c.align) for c, w in zip(columns, widths, strict=True))
    lines = [paint(header, _BOLD), paint("-" * len(header), _DIM)]
    for index, row in enumerate(view.rows):
        row_codes = (_DIM, _ITALIC) if row.is_reference else ()
        cells = [
            column.render(column.cells[index], width, paint, row_codes)
            for column, width in zip(columns, widths, strict=True)
        ]
        lines.append(_GAP.join(cells).rstrip())
    if not view.candidates:
        lines.append("(no candidates)")
    return lines


def _terminal_columns(view: _View, *, backend: bool) -> list[_TermColumn]:
    rows = view.rows
    columns = [
        _TermColumn(
            "Call",
            "left",
            tuple(
                _TermCell(row.chip, codes=() if row.status is None else _STATUS_CODES[row.status])
                for row in rows
            ),
        ),
        _TermColumn("Label", "left", tuple(_TermCell(_plain(row.label)) for row in rows)),
    ]
    if backend:
        columns.append(
            _TermColumn("Backend", "left", tuple(_TermCell(row.result.spec.kind) for row in rows))
        )
    if view.show_size:
        columns.append(_TermColumn("Size", "right", tuple(_term_size_cell(view, r) for r in rows)))
    for metric in view.metrics:
        cells = tuple(_term_metric_cell(view.cell(metric, row)) for row in rows)
        columns.append(_TermColumn(metric.short, "right", cells))
    if view.speed is not None:
        cells = tuple(_term_metric_cell(view.speed_cell(row)) for row in rows)
        columns.append(_TermColumn(_speed_header(view), "right", cells))
    return columns


def _term_size_cell(view: _View, row: _Row) -> _TermCell:
    size, change = view.size_cell(row)
    if not row.over_budget:
        return _TermCell(size, change)
    marked = f"{change} {OVER_BUDGET_SHORT}" if change else OVER_BUDGET_SHORT
    return _TermCell(size, marked, (_DIM,), (_DIM,))


def _term_metric_cell(cell: _Cell) -> _TermCell:
    codes = (_DIM,) if cell.muted else ()
    if cell.marker:
        return _TermCell(cell.text, cell.marker, codes, (_DIM,))
    if cell.delta is None:
        return _TermCell(cell.text, codes=codes)
    delta_codes = (_RED,) if is_regression(cell.delta) else (_DIM,)
    return _TermCell(cell.text, format_delta(cell.delta), codes, delta_codes)


def _label_room(columns: Sequence[_TermColumn]) -> int:
    others = sum(column.width for column in columns[2:]) + columns[0].width
    return TERMINAL_WIDTH - others - len(_GAP) * (len(columns) - 1)


def _fit(columns: Sequence[_TermColumn]) -> list[int] | None:
    """Column widths that fit the terminal, or None when the label would get too narrow."""
    needed = columns[1].width
    room = _label_room(columns)
    if room < min(needed, _LABEL_MIN_WIDTH):
        return None
    return [columns[0].width, min(needed, room), *(column.width for column in columns[2:])]


def _squeezed(columns: Sequence[_TermColumn]) -> list[int]:
    label = max(_label_room(columns), len(_ELLIPSIS) + 2)
    return [columns[0].width, label, *(column.width for column in columns[2:])]


def _pad(text: str, width: int, align: Align) -> str:
    return text.rjust(width) if align == "right" else text.ljust(width)


def _truncate(text: str, width: int) -> str:
    """Shorten from the middle: quant labels differ at the end (q4_K_M vs q2_K), not the start."""
    if len(text) <= width:
        return text
    keep = width - len(_ELLIPSIS)
    head = keep // 3
    return text[:head] + _ELLIPSIS + text[len(text) - (keep - head) :]


def _plain(text: str) -> str:
    """Neutralize control characters so labels cannot inject terminal escape sequences."""
    return "".join(char if char.isprintable() else " " for char in text)


def _sentence(text: str, *, capitalize: bool = True) -> str:
    """End with a period, and capitalize unless told not to, so phrases read as sentences."""
    text = " ".join(text.split())
    if not text:
        return text
    if capitalize:
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _wrap(text: str, *, first: str = "", rest: str = "  ") -> list[str]:
    return textwrap.wrap(
        text,
        width=TERMINAL_WIDTH,
        initial_indent=first,
        subsequent_indent=rest,
        break_on_hyphens=False,
    ) or [first.rstrip()]


# Markdown ---------------------------------------------------------------------------------

_MARKDOWN_SPECIAL: Final = frozenset("\\`*_[]<>|~")


def render_markdown(report: Report, *, verdict: Verdict | None = None) -> str:
    """Render a GitHub and Reddit friendly markdown scorecard."""
    view = _view(report, verdict or judge(report))
    context = "; ".join(
        f"{name}: **{_md(value)}**" + (f" ({backend})" if backend else "")
        for name, value, backend in view.context()
    )
    lines = [
        f"## {_md(report.title)}",
        "",
        f"*{_md(summary_line(report))}*",
        "",
        context,
        "",
    ]
    lead = _lead(view.verdict)
    lines.append(f"> **{_md(lead.headline)}**")
    if lead.remedy is not None:
        lines += [">", f"> **{_md(lead.remedy)}**"]
    for sentence in (*lead.details, *_score_alerts(view)):
        lines += [">", f"> {_md(sentence)}"]
    lines += ["", *_markdown_table(view)]
    if view.has_significant:
        lines += ["", _md(SIGNIFICANT_FOOTNOTE)]
    reasons = _reasons(view)
    if reasons:
        lines += ["", "**Why**", ""]
        for row, call in reasons:
            body = f"{_md(row.label)}: {_md(' '.join(call.reasons))}".rstrip(": ")
            lines.append(f"- **{_tagged_chip(row)}** {body}")
            lines += [
                f"  - {CAVEAT_PREFIX} {_md(_sentence(caveat, capitalize=False))}"
                for caveat in call.caveats
            ]
    findings = _findings(view)
    if findings:
        lines += ["", "### Server checks", ""]
        lines += [_markdown_finding(view, finding) for finding in findings]
    errors = _errors(view)
    if errors:
        lines += ["", "### Errors", ""]
        lines += [f"- {_md(label)}: {_md(error)}" for label, error in errors]
    lines += ["", f"**Settings:** {_md(_settings_line(report))}"]
    notes = _notes(view)
    if notes:
        lines += ["", "### Notes", ""]
        lines += [f"{number}. {_md(note)}" for number, note in enumerate(notes, start=1)]
    lines += ["", f"Generated with quantdiff v{_md(report.quantdiff_version)}"]
    return "\n".join(lines) + "\n"


def _markdown_table(view: _View) -> list[str]:
    show_backend = view.backend is None
    headers = ["Call", "Label"]
    aligns = [":--", ":--"]
    if show_backend:
        headers.append("Backend")
        aligns.append(":--")
    if view.show_size:
        headers.append("Size")
        aligns.append("--:")
    headers += [metric.name for metric in view.metrics]
    aligns += ["--:" for _ in view.metrics]
    if view.speed is not None:
        headers.append(_speed_header(view))
        aligns.append("--:")
    lines = [_md_row(headers), _md_row(aligns)]
    for row in view.rows:
        label = _md(row.label)
        chip = f"**{row.chip}**" if row.status == "recommended" else row.chip
        if row.near_bar:
            chip += f" *{NEAR_BAR}*"
        cells = [chip, f"{label} (reference)" if row.is_reference else label]
        if show_backend:
            cells.append(row.result.spec.kind)
        if view.show_size:
            cells.append(_markdown_size(view, row))
        cells += [_markdown_cell(view.cell(metric, row)) for metric in view.metrics]
        if view.speed is not None:
            cells.append(_markdown_cell(view.speed_cell(row)))
        lines.append(_md_row(cells))
    return lines


def _markdown_size(view: _View, row: _Row) -> str:
    size, change = view.size_cell(row)
    if not row.over_budget:
        return f"{size} ({change})" if change else size
    marked = f"{size} ({change}) {OVER_BUDGET_SHORT}" if change else f"{size} {OVER_BUDGET_SHORT}"
    return f"*{marked}*"


def _markdown_cell(cell: _Cell) -> str:
    text = cell.text if cell.delta is None else f"{cell.text} ({_md(format_delta(cell.delta))})"
    if cell.marker:
        text = f"{text} ({cell.marker})"
    return f"*{text}*" if cell.muted and cell.value is not None else text


def _markdown_finding(view: _View, item: ServerFinding) -> str:
    finding = item.finding
    chip = finding_chip(item)
    tag = f"**{chip}**" if item.affects_scores else chip
    message = _sentence(finding.message, capitalize=False)
    line = (
        f"- {tag} {_md(view.who(item))}, {_md(finding.check)}: {_md(message)} "
        f"*{_md(_sentence(item.impact))}*"
    )
    if finding.fix:
        line += f" Fix: {_md(finding.fix)}"
    return line


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md(text: str) -> str:
    """Escape markdown syntax and flatten newlines so text stays inside its table cell."""
    flat = " ".join(text.split())
    return "".join(f"\\{char}" if char in _MARKDOWN_SPECIAL else char for char in flat)


# HTML -------------------------------------------------------------------------------------

_CSS: Final = """
:root {
  --bg: #f3f4f1;
  --surface: #ffffff;
  --text: #16181d;
  --muted: #5d6470;
  --faint: #8a909a;
  --border: #e3e5e8;
  --row-pick: #f0f8f3;
  --row-ref: #f7f8fa;
  --accent: #1f7a4d;
  --accent-soft: #e6f3eb;
  --neutral-soft: #f1f2f4;
  --track: #eceef1;
  --fill: #6f7b8a;
  --fill-ref: #b4bac3;
  --kld-good: #3f9a6b;
  --kld-moderate: #d29a2a;
  --kld-large: #cc4b3c;
  --usable: #1d64b0;
  --usable-soft: #e4eefa;
  --usable-line: #9dbde3;
  --down: #c03a2b;
  --warn: #9a6200;
  --warn-soft: #fdf1d8;
  --fail: #b42318;
  --fail-soft: #fde5e2;
  --t1: #3b5bdb;
  --t1-soft: #e6ebfc;
  --t2: #0b7285;
  --t2-soft: #dff3f6;
  --shadow: 0 1px 2px rgba(16, 24, 40, 0.06), 0 8px 24px rgba(16, 24, 40, 0.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1013;
    --surface: #171a1f;
    --text: #e8eaed;
    --muted: #a3a9b3;
    --faint: #757c87;
    --border: #2a2f37;
    --row-pick: #16261d;
    --row-ref: #1c2026;
    --accent: #5fd08f;
    --accent-soft: #1a2e23;
    --neutral-soft: #1f232a;
    --track: #262b33;
    --fill: #8b95a3;
    --fill-ref: #5a616c;
    --kld-good: #5cc58e;
    --kld-moderate: #e6b450;
    --kld-large: #f07a6b;
    --usable: #7cb6f5;
    --usable-soft: #182a40;
    --usable-line: #2f5a8a;
    --down: #ff8a7d;
    --warn: #f2b84b;
    --warn-soft: #3a2c10;
    --fail: #ff7b6e;
    --fail-soft: #3d1a17;
    --t1: #91a7ff;
    --t1-soft: #1e2645;
    --t2: #66d9e8;
    --t2-soft: #0f3036;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial,
    sans-serif;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
.card {
  max-width: 960px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.section { padding: 20px 28px; border-top: 1px solid var(--border); }
.head { padding: 26px 28px 18px; }
.eyebrow {
  margin: 0 0 6px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: 22px;
  line-height: 1.25;
  letter-spacing: -0.01em;
  overflow-wrap: anywhere;
}
h2 {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.stats { margin: 4px 0 0; color: var(--faint); font-size: 14px; }
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 22px;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
  color: var(--muted);
  font-size: 13px;
}
.meta li { min-width: 0; overflow-wrap: anywhere; }
.meta .mono { color: var(--text); font-weight: 600; }
.mono {
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", "Segoe UI Mono", Consolas, monospace;
  font-size: 0.95em;
  overflow-wrap: anywhere;
}
.verdict {
  margin: 0 28px 22px;
  padding: 20px 24px;
  background: var(--accent-soft);
  border-left: 5px solid var(--accent);
  border-radius: 10px;
}
.verdict.open { background: var(--neutral-soft); border-left-color: var(--faint); }
.verdict.fits { background: var(--usable-soft); border-left-color: var(--usable); }
.verdict p { margin: 0; overflow-wrap: anywhere; }
.verdict .kicker {
  margin-bottom: 6px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.verdict.open .kicker { color: var(--muted); }
.verdict.fits .kicker { color: var(--usable); }
.verdict .headline { font-size: 24px; font-weight: 700; line-height: 1.3; }
.verdict ul { margin: 10px 0 0; padding-left: 18px; color: var(--muted); font-size: 14px; }
.verdict li { overflow-wrap: anywhere; }
.verdict li + li { margin-top: 3px; }
.verdict .remedy {
  margin-top: 12px;
  padding: 10px 14px;
  border-left: 4px solid var(--warn);
  border-radius: 6px;
  background: var(--warn-soft);
  color: var(--text);
  font-size: 15px;
  font-weight: 600;
}
.verdict .alert { margin-top: 10px; color: var(--fail); font-size: 14px; font-weight: 600; }
.table-wrap { overflow-x: auto; padding: 4px 14px 8px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 7px; text-align: right; vertical-align: top; }
th {
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
  border-bottom: 1px solid var(--border);
}
thead tr.groups th { border-bottom: none; padding-bottom: 0; text-align: center; }
td { border-bottom: 1px solid var(--border); font-size: 13px; white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
th.left, td.left { text-align: left; }
td.name { white-space: normal; }
.name-box { min-width: 96px; max-width: 250px; }
tr.pick td { background: var(--row-pick); }
tr.baseline td { background: var(--row-ref); border-bottom: 2px solid var(--border); }
.status {
  display: inline-block;
  min-width: 58px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--track);
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  line-height: 18px;
  text-align: center;
}
.status.recommended { background: var(--accent); color: var(--surface); }
.status.usable { background: var(--usable-soft); color: var(--usable); }
.status.avoid { background: var(--fail-soft); color: var(--fail); }
.status.inconclusive {
  background: var(--warn-soft);
  border: 1px solid var(--warn);
  color: var(--warn);
  line-height: 16px;
}
.status.failed { background: transparent; border: 1px solid var(--fail); color: var(--fail); }
.status.reference {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--faint);
  line-height: 16px;
}
.tag {
  display: block;
  margin-top: 4px;
  color: var(--warn);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-align: center;
  text-transform: uppercase;
  white-space: nowrap;
}
.label { font-weight: 600; }
.sub { display: block; color: var(--faint); font-size: 12px; font-weight: 400; }
.why { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; line-height: 1.35; }
.caveat {
  display: block;
  position: relative;
  margin-top: 4px;
  padding-left: 18px;
  color: var(--warn);
  font-size: 12px;
  line-height: 1.35;
}
.caveat::before {
  content: "!";
  position: absolute;
  left: 0;
  top: 1px;
  width: 13px;
  height: 13px;
  border-radius: 50%;
  background: var(--warn-soft);
  border: 1px solid var(--warn);
  font-size: 9px;
  font-weight: 800;
  line-height: 11px;
  text-align: center;
}
.backend { color: var(--muted); }
.na { color: var(--faint); }
td.base { color: var(--faint); font-size: 12px; font-style: italic; text-align: center; }
.sub.failed { color: var(--fail); font-family: inherit; }
td.grey { opacity: 0.45; }
td.wall { color: var(--muted); }
td.over { color: var(--faint); }
.change.over { color: var(--warn); font-weight: 600; }
.marker { display: block; margin-top: 3px; color: var(--faint); font-size: 11px; }
.bar {
  display: block;
  height: 4px;
  margin-top: 5px;
  margin-left: auto;
  width: 52px;
  border-radius: 2px;
  background: var(--track);
  overflow: hidden;
}
.bar span { display: block; height: 100%; border-radius: 2px; background: var(--fill); }
.bar.near-lossless span, .bar.small span { background: var(--kld-good); }
.bar.moderate span { background: var(--kld-moderate); }
.bar.large span { background: var(--kld-large); }
tr.baseline .bar span { background: var(--fill-ref); }
.delta { display: block; margin-top: 3px; color: var(--faint); font-size: 11px; line-height: 1.2; }
.delta.down { color: var(--down); font-weight: 700; }
.change { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
.footnote { margin: 0; padding: 0 28px 14px; color: var(--faint); font-size: 12px; }
.tier {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.tier.t1 { background: var(--t1-soft); color: var(--t1); }
.tier.t2 { background: var(--t2-soft); color: var(--t2); }
.group-name { margin-left: 6px; color: var(--muted); font-weight: 600; }
.chip {
  display: inline-block;
  min-width: 42px;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-align: center;
  flex: none;
  margin-top: 2px;
}
.chip.fail { background: var(--fail-soft); color: var(--fail); }
.chip.warn { background: var(--warn-soft); color: var(--warn); }
.chip.note { background: var(--track); color: var(--muted); }
.columns { display: grid; grid-template-columns: 3fr 2fr; gap: 28px; }
@media (max-width: 720px) { .columns { grid-template-columns: 1fr; } }
.columns > section { min-width: 0; }
.findings { margin: 0; padding: 0; list-style: none; }
.findings li { display: flex; align-items: flex-start; gap: 12px; padding: 8px 0; }
.findings li + li { border-top: 1px dashed var(--border); }
.findings div { min-width: 0; }
.findings p { margin: 0; overflow-wrap: anywhere; }
.findings .impact { color: var(--muted); font-size: 13px; }
.findings .fix { color: var(--muted); font-size: 13px; }
.findings li.quiet p { color: var(--muted); }
.findings li.quiet .impact, .findings li.quiet .fix { color: var(--faint); }
.muted { margin: 0; color: var(--muted); }
dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
.notes ol { margin: 0; padding-left: 20px; color: var(--muted); font-size: 13px; }
.notes li + li { margin-top: 4px; }
.errors { margin: 0; padding-left: 20px; color: var(--fail); font-size: 13px; }
.errors li { overflow-wrap: anywhere; }
footer {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 28px;
  border-top: 1px solid var(--border);
  color: var(--faint);
  font-size: 12px;
}
footer strong { color: var(--muted); }
"""


def render_html(report: Report, *, verdict: Verdict | None = None) -> str:
    """Render a self-contained HTML scorecard: inline CSS, no scripts, no external assets."""
    view = _view(report, verdict or judge(report))
    footnote = f'<p class="footnote">{_e(SIGNIFICANT_FOOTNOTE)}</p>' if view.has_significant else ""
    sections = [
        '<header class="head">',
        '<p class="eyebrow">quantdiff scorecard</p>',
        f"<h1>{_e(report.title)}</h1>",
        f'<p class="stats">{_e(summary_line(report))}</p>',
        _html_context(view),
        "</header>",
        _html_verdict(view),
        _html_table(view),
        footnote,
        '<div class="section columns">',
        _html_findings(view),
        _html_settings(report),
        "</div>",
        _html_errors(view),
        _html_notes(view),
        "<footer>",
        f"<span>Generated with <strong>quantdiff v{_e(report.quantdiff_version)}</strong></span>",
        f'<time datetime="{_e(report.created_at)}">{_e(report.created_at)}</time>',
        "</footer>",
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="color-scheme" content="light dark">',
            f"<title>{_e(report.title)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            '<main class="card">',
            *(section for section in sections if section),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _html_context(view: _View) -> str:
    items = "".join(
        f'<li>{_e(name)} <span class="mono" title="{_e(value)}">{_e(value)}</span>'
        + (f" on {_e(backend)}" if backend else "")
        + "</li>"
        for name, value, backend in view.context()
    )
    return f'<ul class="meta">{items}</ul>'


def _html_verdict(view: _View) -> str:
    verdict = view.verdict
    lead = _lead(verdict)
    remedy = "" if lead.remedy is None else f'<p class="remedy">{_e(lead.remedy)}</p>'
    items = "".join(f"<li>{_e(sentence)}</li>" for sentence in lead.details)
    alerts = "".join(f'<p class="alert">{_e(alert)}</p>' for alert in _score_alerts(view))
    return (
        f'<section class="verdict{_verdict_tone(verdict)}"><p class="kicker">Verdict</p>'
        f'<p class="headline">{_e(lead.headline)}</p>{remedy}'
        f"{f'<ul>{items}</ul>' if items else ''}{alerts}</section>"
    )


def _verdict_tone(verdict: Verdict) -> str:
    """The verdict box's CSS modifier: "" (green) when a candidate within the budget is
    recommended, " fits" (blue) when the best that fits is a usable candidate, and " open"
    (grey) otherwise, which covers every "Keep <reference>" and all-avoid outcome."""
    if verdict.headline.startswith("Keep "):
        return " open"
    within = [call for call in verdict.candidates if call.fits_budget is not False]
    if any(call.status == "recommended" for call in within):
        return ""
    if any(call.status == "usable" for call in within):
        return " fits"
    return " open"


def _html_table(view: _View) -> str:
    show_backend = view.backend is None
    lead = 2 + show_backend + view.show_size
    logit = sum(1 for metric in view.metrics if metric.group == "logit")
    task = len(view.metrics) - logit
    groups = f'<tr class="groups"><th colspan="{lead}"></th>'
    if logit:
        groups += (
            f'<th colspan="{logit}"><span class="tier t1">T1</span>'
            '<span class="group-name">Logit fidelity</span></th>'
        )
    if task:
        groups += (
            f'<th colspan="{task}"><span class="tier t2">T2</span>'
            '<span class="group-name">Task quality</span></th>'
        )
    if view.speed is not None:
        groups += "<th></th>"
    groups += "</tr>"
    header = '<tr><th class="left">Call</th><th class="left">Label</th>'
    if show_backend:
        header += '<th class="left">Backend</th>'
    if view.show_size:
        header += "<th>Size</th>"
    header += "".join(f"<th>{_e(metric.name)}</th>" for metric in view.metrics)
    if view.speed is not None:
        header += f"<th>{_e(_speed_header(view))}</th>"
    header += "</tr>"
    scales = [_column_max(view, metric) for metric in view.metrics]
    rows = [_html_row(view, row, scales, logit) for row in view.rows]
    if not view.candidates:
        span = lead + len(view.metrics) + (view.speed is not None)
        rows.append(f'<tr><td class="left na" colspan="{span}">No candidates</td></tr>')
    return (
        '<div class="table-wrap"><table>'
        f"<thead>{groups}{header}</thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _column_max(view: _View, metric: _Metric) -> float:
    values = [v for row in view.candidates if (v := metric.value(row.result)) is not None]
    return max(values, default=0.0)


def _html_lead_cells(row: _Row) -> list[str]:
    """The status chip and the label, with the reasons for the call under the label."""
    result = row.result
    sub_html = ""
    if row.status == "failed" and result.errors:
        sub_html = '<span class="sub failed">failed to run</span>'
    elif row.is_reference:
        sub_html = '<span class="sub">reference</span>'
    why = ""
    if row.call is not None and row.call.reasons:
        why = f'<span class="why">{_e(" ".join(row.call.reasons))}</span>'
    if row.call is not None:
        why += "".join(
            f'<span class="caveat">{_e(_sentence(caveat, capitalize=False))}</span>'
            for caveat in row.call.caveats
        )
    tag = f'<span class="tag">{NEAR_BAR}</span>' if row.near_bar else ""
    return [
        f'<td class="left"><span class="status {row.status or "reference"}">{row.chip}</span>'
        f"{tag}</td>",
        f'<td class="left name"><div class="name-box"><span class="label mono" '
        f'title="{_e(result.spec.label)}">{_e(row.label)}</span>{sub_html}{why}</div></td>',
    ]


def _html_row(view: _View, row: _Row, scales: Sequence[float], logit_columns: int) -> str:
    spec = row.result.spec
    cells = _html_lead_cells(row)
    if view.backend is None:
        cells.append(f'<td class="left backend">{_e(spec.kind)}</td>')
    if view.show_size:
        cells.append(_html_size_cell(view, row))
    if row.is_reference and logit_columns:
        cells.append(f'<td class="base" colspan="{logit_columns}">baseline</td>')
    for metric, scale in zip(view.metrics, scales, strict=True):
        if row.is_reference and metric.group == "logit":
            continue
        cells.append(_html_metric_cell(view.cell(metric, row), metric, scale, view.thresholds))
    if view.speed is not None:
        cells.append(_html_speed_cell(view.speed_cell(row)))
    row_class = ""
    if row.is_reference:
        row_class = ' class="baseline"'
    elif row.status == "recommended":
        row_class = ' class="pick"'
    return f"<tr{row_class}>{''.join(cells)}</tr>"


def _html_size_cell(view: _View, row: _Row) -> str:
    size, change = view.size_cell(row)
    change_html = f'<span class="change">{_e(change)}</span>' if change else ""
    if row.size_bytes is None:
        return f'<td class="na">{_e(size)}</td>'
    if row.over_budget:
        over = f'<span class="change over">{OVER_BUDGET}</span>'
        return f'<td class="over">{_e(size)}{change_html}{over}</td>'
    return f"<td>{_e(size)}{change_html}</td>"


def _bar_class(metric: _Metric, value: float, thresholds: KldThresholds) -> str:
    """Rate bars are neutral. The KLD mean bar takes its band's colour, so a near-lossless or
    small KLD never reads as a warning; the p99 tail has no bands and stays neutral."""
    if metric.bar == "kld" and metric.short == "KLD":
        return f"bar {kld_band(value, thresholds)}"
    return "bar"


def _html_metric_cell(cell: _Cell, metric: _Metric, scale: float, thresholds: KldThresholds) -> str:
    if cell.value is None:
        return f'<td class="na">{_e(cell.text)}</td>'
    fraction = cell.value if metric.bar == "rate" else cell.value / scale if scale > 0 else 0.0
    bar = _bar_html(_bar_class(metric, cell.value, thresholds), fraction)
    delta = ""
    if cell.delta is not None:
        direction = " down" if is_regression(cell.delta) else ""
        delta = f'<span class="delta{direction}">{_e(format_interval(cell.delta))}</span>'
    css = ' class="grey"' if cell.muted else ""
    return f'<td{css}><span class="val">{_e(cell.text)}</span>{bar}{delta}</td>'


def _html_speed_cell(cell: _Cell) -> str:
    if cell.marker:
        marker = f'<span class="marker">{_e(cell.marker)}</span>'
        return f'<td class="wall">{_e(cell.text)}{marker}</td>'
    if cell.muted:
        return f'<td class="wall">{_e(cell.text)}</td>'
    if cell.text == MISSING:
        return f'<td class="na">{_e(cell.text)}</td>'
    return f"<td>{_e(cell.text)}</td>"


def _bar_html(css: str, fraction: float) -> str:
    percent = min(max(fraction, 0.0), 1.0) * 100
    return f'<span class="{css}"><span style="width:{percent:.1f}%"></span></span>'


def _html_findings(view: _View) -> str:
    findings = _findings(view)
    if not findings:
        return '<section><h2>Server checks</h2><p class="muted">No problems found.</p></section>'
    items = []
    for item in findings:
        finding = item.finding
        quiet = not item.affects_scores
        chip = "note" if quiet else finding.severity
        fix = f'<p class="fix">Fix: {_e(finding.fix)}</p>' if finding.fix else ""
        items.append(
            ('<li class="quiet">' if quiet else "<li>")
            + f'<span class="chip {chip}">{_e(finding_chip(item))}</span>'
            + f"<div><p>{_html_who(view, item)}, "
            f"{_e(finding.check)} check</p>"
            f"<p>{_e(_sentence(finding.message, capitalize=False))}</p>"
            f'<p class="impact">{_e(_sentence(item.impact))}</p>{fix}</div></li>'
        )
    return f'<section><h2>Server checks</h2><ul class="findings">{"".join(items)}</ul></section>'


def _html_who(view: _View, item: ServerFinding) -> str:
    """Model labels in monospace; "both models" or "all N models" as plain words."""
    who = _e(view.who(item))
    return who if view.covers_everyone(item) else f'<span class="mono">{who}</span>'


def _html_settings(report: Report) -> str:
    rows = "".join(
        f"<dt>{_e(name)}</dt><dd>{_e(value)}</dd>" for name, value in _settings_items(report)
    )
    return f"<section><h2>Settings</h2><dl>{rows}</dl></section>"


def _html_errors(view: _View) -> str:
    items = "".join(
        f'<li><span class="mono">{_e(label)}</span>: {_e(error)}</li>'
        for label, error in _errors(view)
    )
    if not items:
        return ""
    return f'<section class="section"><h2>Errors</h2><ul class="errors">{items}</ul></section>'


def _html_notes(view: _View) -> str:
    notes = _notes(view)
    if not notes:
        return ""
    items = "".join(f"<li>{_e(note)}</li>" for note in notes)
    return f'<section class="section notes"><h2>Notes</h2><ol>{items}</ol></section>'


def _e(text: str) -> str:
    return html.escape(text, quote=True)
