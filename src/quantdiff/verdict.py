"""Turn a Report into a recommendation: which download to run, which to avoid, and why.

This is the contract the scorecards render. The dataclasses are shared with card.py and
the CLI; `judge` applies the rules below, which docs/methodology.md explains for users.

Every comparison is paired, because every model answers the same cases and is scored on
the same prompts: task outcomes are joined by case id and logit results by prompt id. Only
items present on both sides count.

Each candidate is judged on its own against the reference, never against the other
candidates, so adding or removing a candidate never changes another one's status. A
candidate is recommended only on positive evidence that it is close to the reference; thin
evidence leads to inconclusive, not to a pick. The KLD interval below is the 95% bootstrap
interval of the mean per-prompt KLD. Rules, applied in order:

1. failed: the candidate produced no metrics at all.
2. avoid, when any of these holds:
   - a reliable suite (the reference passes at least half of its cases) shows a significant
     paired regression (exact McNemar p < 0.05, candidate below the reference);
   - the reliable suites pooled together show a significant paired regression;
   - with at least MIN_PROMPTS paired prompts, the mean KLD is at least LARGE_KLD and so is
     the lower end of its interval.
3. close (eligible to run), only on positive evidence:
   - with logit metrics, closeness rests on them: at least MIN_PROMPTS paired prompts and
     the upper end of the KLD interval below CLOSE_KLD. Task suites of a few dozen cases
     cannot bound small differences, so with logit evidence they act as a breakage
     detector: any significant loss (rule 2) is avoid, and a wide but not significant
     interval is shown, not used against the candidate. A KLD that is shown small also
     bounds how far the two output distributions can differ.
   - without logit metrics, closeness rests on tasks: the reliable suites pooled, at least
     MIN_CASES cases, and the lower end of the 95% interval of the pass-rate difference at
     or above -TASK_MARGIN points.
   The reasons say which evidence the call rests on.
4. usable: with at least MIN_PROMPTS paired prompts, the lower end of the KLD interval is
   above CLOSE_KLD and the mean is below LARGE_KLD. A measured, moderate loss: the best
   choice when nothing close fits the size budget.
5. inconclusive: everything else, including a large-looking mean on fewer than MIN_PROMPTS
   prompts and a mean at or above LARGE_KLD whose interval reaches below it. The reasons
   name what is missing and estimate how many prompts or cases per suite would decide.
6. The pick. Among the close candidates that fit the --max-size budget (every close one
   when there is no budget) the smallest download is recommended (by size on disk when
   each of them reports one, otherwise the lowest KLD). The other close ones are ok. A
   candidate that is not close is never recommended or ok, so when no close candidate
   fits, the headline names the usable candidate with the lowest KLD that fits instead.

Caveats do not change a status but sit next to it: a KLD interval near a bar (a rerun
could move the candidate across it), and a reliable suite whose estimate is more than
TASK_MARGIN points below the reference without being significant.

A candidate is never described as better than the reference. A significant gain is
reported as a warning sign about the suite or the reference, not as a win.

Ranking: status (recommended, ok, usable, inconclusive, avoid, failed), then mean KLD
ascending, size ascending, and input order.
"""

from __future__ import annotations

import itertools
import math
import os.path
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from quantdiff.stats import (
    Interval,
    bootstrap_mean,
    cases_to_bound_loss,
    items_to_bound_below,
    mcnemar_exact,
    paired_proportion_diff,
)
from quantdiff.suites import load_builtin, load_scoring_prompts
from quantdiff.types import (
    CandidateResult,
    LogitMetrics,
    PreflightFinding,
    Report,
    RunSettings,
    ScoringPrompt,
    TaskKind,
)

__all__ = [
    "CALIBRATED_TOP_K",
    "CLOSE_KLD",
    "FULL_VOCAB_CLOSE",
    "FULL_VOCAB_LARGE",
    "FULL_VOCAB_NEAR_LOSSLESS",
    "LABEL_SEPARATORS",
    "LARGE_KLD",
    "MIN_CASES",
    "MIN_PROMPTS",
    "MIN_SHARED_PREFIX",
    "NEAR_BAR",
    "NEAR_LOSSLESS_KLD",
    "SCORED_TASK_KINDS",
    "TASK_MARGIN",
    "TOP_K_FRACTION",
    "CandidateVerdict",
    "Interval",
    "KldBand",
    "KldThresholds",
    "ServerFinding",
    "Status",
    "TaskDelta",
    "Verdict",
    "describe_delta",
    "display_labels",
    "format_size",
    "judge",
    "kld_band",
    "kld_thresholds",
    "mostly_non_latin",
    "top_k_fraction",
]

Status = Literal["recommended", "ok", "usable", "avoid", "inconclusive", "failed"]
"""recommended: the one to run. ok: also close to the reference, but not the pick (usually
larger). usable: a measured but moderate loss; a sound choice when nothing closer fits.
avoid: a large loss or measured task breakage. inconclusive: not enough evidence either way.
failed: the candidate produced no usable metrics."""

KldBand = Literal["near-lossless", "small", "moderate", "large"]

# Thresholds. Every number the rules use is here, so a calibration can change them in one
# place; docs/methodology.md, docs/calibration.md and README.md quote them.
#
# The KLD bars are set on llama.cpp's full-vocabulary scale and converted to quantdiff's
# top-k lower bound, which reads a fixed fraction of the full value. docs/calibration.md
# measured that fraction on identical positions: 0.24 at k=1, 0.57 at k=5, 0.67 at k=10 and
# 0.76 at k=20 (llama-perplexity --kl-divergence as the full-vocabulary truth).
FULL_VOCAB_NEAR_LOSSLESS: Final = 0.015
"""Full-vocabulary KLD below this is near-lossless (typical of Q8_0)."""
FULL_VOCAB_CLOSE: Final = 0.06
"""Full-vocabulary closeness bar: about Q4_K_M on a 7 to 8B model."""
FULL_VOCAB_LARGE: Final = 0.15
"""Full-vocabulary KLD at or above this is a large loss (Q2_K territory)."""
TOP_K_FRACTION: Final[tuple[tuple[int, float], ...]] = (
    (1, 0.24),
    (5, 0.57),
    (10, 0.67),
    (20, 0.76),
)
"""Measured top-k lower bound as a fraction of full-vocabulary KLD, by k."""
CALIBRATED_TOP_K: Final = 10
"""The default --top-k, at which the module-level KLD constants below apply."""
MIN_PROMPTS: Final = 8
"""Fewer paired prompts than this never prove a candidate close or far on logits: a
bootstrap over a handful of prompts has almost no distinct resamples."""
MIN_CASES: Final = 20
"""Fewer pooled paired task cases than this never prove a candidate close on tasks."""
TASK_MARGIN: Final = 10.0
"""Points of pass rate the pooled task interval may reach below the reference while the
candidate still counts as close."""
NEAR_BAR: Final = 0.1
"""A KLD interval that straddles a bar, or ends within this fraction of it, is near it."""

SCORED_TASK_KINDS: Final[tuple[TaskKind, ...]] = ("json", "tools", "code")
"""Task kinds with a pass/fail check. Chat is scored by agreement instead."""
LABEL_SEPARATORS: Final = ":-_/."
MIN_SHARED_PREFIX: Final = 8

_ALPHA: Final = 0.05
_RELIABLE_REFERENCE_RATE: Final = 0.5
_STATUS_ORDER: Final[dict[Status, int]] = {
    "recommended": 0,
    "ok": 1,
    "usable": 2,
    "inconclusive": 3,
    "avoid": 4,
    "failed": 5,
}
_MAX_CASES_TO_RESOLVE: Final = 5000
"""Larger estimates of the cases that would resolve a task loss are not worth quoting."""
_NON_LATIN_SHARE: Final = 0.5
"""Prompts whose letters are more than this share non-Latin are mostly non-Latin script."""
_SIZE_DIGITS: Final = 3
"""Significant digits of the sizes and budgets that sentences quote."""
_CONFIRM_EXACT: Final = "Confirm with llama-server (exact token ids) before ruling out a download."
_SAME_SIZE: Final = 0.005
"""Size changes under half a percent are reported as the same size."""
_ROUND_NEEDED_TO: Final = 10
"""Evidence estimates are rough, so they are rounded up to a multiple of this."""
_QUANTIZED_REFERENCE: Final = (
    "usually a sign the suite is too small or the reference is quantized itself"
)


@dataclass(frozen=True, slots=True)
class TaskDelta:
    """Candidate minus reference pass rate for one suite, paired case by case, in points."""

    kind: TaskKind
    cases: int
    candidate_rate: float
    reference_rate: float
    delta: Interval
    significant: bool
    """True when the paired test (exact McNemar) rejects equality at the 5% level."""
    reference_reliable: bool
    """False when the reference itself passes under half the cases, so the suite says little
    about this model; such suites are shown but never used to judge a candidate."""


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    label: str
    status: Status
    rank: int | None
    """1 is best; None for failed candidates."""
    kld_band: KldBand | None
    size_bytes: int | None
    size_change: float | None
    """Fractional size change against the reference, e.g. -0.25 for 25% smaller."""
    task_deltas: tuple[TaskDelta, ...]
    reasons: tuple[str, ...]
    """Short plain-English sentences that justify the status, most important first."""
    caveats: tuple[str, ...] = ()
    """Unresolved concerns that do not change the status but a reader must see next to it,
    e.g. "tools -17 unresolved (95% CI -42 to +6); rerun with --max-cases 60"."""
    near_bar: bool = False
    """True when the KLD interval straddles a band edge closely enough that a rerun could
    change the status."""
    fits_budget: bool | None = None
    """Whether the download fits the --max-size budget; None when no budget was given or
    the size is unknown."""


@dataclass(frozen=True, slots=True)
class ServerFinding:
    """A pre-flight finding, with whether it could have changed the scores on this card."""

    finding: PreflightFinding
    labels: tuple[str, ...]
    """Models it applies to; every model when it is a server-wide issue."""
    affects_scores: bool
    impact: str
    """One sentence, e.g. "does not affect these scores: the longest prompt is ~900 tokens"."""


@dataclass(frozen=True, slots=True)
class Verdict:
    headline: str
    """One short sentence that answers "which should I run?", e.g. "Run q4_K_M: 25% smaller
    than q8_0, close on logits (KLD 0.03, CI up to 0.036) on 41 prompts.", "Best that fits
    6 GB: q4_K_M, moderate loss (KLD 0.044)." or an honest "Keep q8_0 for now: no candidate
    is shown to be close on 12 prompts and 24 cases."."""
    details: tuple[str, ...]
    """Supporting sentences: the numbers behind the headline, what to avoid and why. An
    unresolved task loss, when there is one, comes first."""
    candidates: tuple[CandidateVerdict, ...]
    """Every candidate, in rank order, failed ones last."""
    findings: tuple[ServerFinding, ...]
    cases_needed: int | None = None
    """When nothing is recommended and a rerun would likely decide: the value of
    --max-cases (cases per suite and scoring prompts, rounded up to a multiple of 10)."""
    remedy: str | None = None
    """When nothing is recommended: one sentence with the next step, e.g. "Rerun with
    --max-cases 40 to decide." Kept out of the headline and the details."""


# Public helpers ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KldThresholds:
    """The KLD bars on quantdiff's top-k scale for one --top-k setting."""

    near_lossless: float
    close: float
    """A candidate is close on logits when its KLD interval ends below this, and usable at
    best when the interval starts above it. Also the top of the small band."""
    large: float
    """Mean KLD at or above this is the large band; avoided when the interval starts at or
    above it too."""

    @property
    def bar(self) -> str:
        """The closeness bar as sentences quote it."""
        return f"{self.close:.2g}"

    @property
    def large_bar(self) -> str:
        """The large-loss bar as sentences quote it."""
        return f"{self.large:.2f}"


def top_k_fraction(top_k: int) -> float:
    """The measured fraction of full-vocabulary KLD that a top-k lower bound reads,
    interpolated linearly between calibrated k values and held flat outside them."""
    points = TOP_K_FRACTION
    if top_k <= points[0][0]:
        return points[0][1]
    for (k_low, f_low), (k_high, f_high) in itertools.pairwise(points):
        if top_k <= k_high:
            return f_low + (f_high - f_low) * (top_k - k_low) / (k_high - k_low)
    return points[-1][1]


def kld_thresholds(top_k: int = CALIBRATED_TOP_K) -> KldThresholds:
    """The KLD bars for a run with this --top-k, rounded to the precision cards print."""
    fraction = top_k_fraction(top_k)
    return KldThresholds(
        near_lossless=round(FULL_VOCAB_NEAR_LOSSLESS * fraction, 3),
        close=round(FULL_VOCAB_CLOSE * fraction, 3),
        large=round(FULL_VOCAB_LARGE * fraction, 2),
    )


_DEFAULT_THRESHOLDS: Final = kld_thresholds()
NEAR_LOSSLESS_KLD: Final = _DEFAULT_THRESHOLDS.near_lossless
"""Near-lossless bar at the default --top-k (0.01)."""
CLOSE_KLD: Final = _DEFAULT_THRESHOLDS.close
"""Closeness bar at the default --top-k (0.04)."""
LARGE_KLD: Final = _DEFAULT_THRESHOLDS.large
"""Large-loss bar at the default --top-k (0.10)."""


def kld_band(kld_mean: float, thresholds: KldThresholds = _DEFAULT_THRESHOLDS) -> KldBand:
    """quantdiff's band for a mean KLD; see docs/calibration.md for the anchors."""
    if kld_mean < thresholds.near_lossless:
        return "near-lossless"
    if kld_mean < thresholds.close:
        return "small"
    if kld_mean < thresholds.large:
        return "moderate"
    return "large"


def display_labels(report: Report) -> dict[str, str]:
    """Short names for sentences: each full label mapped to the part that tells models apart.

    Quant labels of one model usually differ only after a long common stem, such as
    qwen2.5:7b-instruct-q4_K_M and qwen2.5:7b-instruct-q8_0. When every label (reference
    included) shares a prefix that ends at one of LABEL_SEPARATORS, is at least
    MIN_SHARED_PREFIX characters, and leaves something on every label, the prefix is
    dropped. Otherwise labels are used as they are.
    """
    labels = [result.spec.label for result in (report.reference, *report.candidates)]
    prefix = ""
    if len(labels) >= 2:
        common = os.path.commonprefix(labels)
        end = max((i + 1 for i, char in enumerate(common) if char in LABEL_SEPARATORS), default=0)
        prefix = common[:end]
        if len(prefix) < MIN_SHARED_PREFIX or any(len(label) == len(prefix) for label in labels):
            prefix = ""
    return {label: label[len(prefix) :] for label in labels}


def describe_delta(delta: TaskDelta) -> str:
    """The delta as a reader should take it, e.g. "-30 points (95% CI -48 to -12)".

    A gain that is not significant reads "= reference (within noise)", so a lucky case or
    two is never shown as a candidate beating the reference.
    """
    estimate = round(delta.delta.estimate)
    if not delta.significant:
        if estimate >= 0:
            return "= reference (within noise)"
        return f"{estimate} points (within noise)"
    if estimate > 0:
        return f"+{estimate} points vs reference; {_QUANTIZED_REFERENCE}"
    return f"{estimate} points ({_interval_points(delta.delta)})"


def format_size(size_bytes: int, *, round_up: bool = False) -> str:
    """A download size or budget to three significant digits in decimal units, e.g.
    "7.1 GB", "6.25 GB" or "398 MB". With `round_up`, never less than `size_bytes`, so the
    text can be passed back as a --max-size that the download fits."""
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if size_bytes >= scale:
            value = size_bytes / scale
            decimals = max(0, _SIZE_DIGITS - len(str(int(value))))
            step = 10**decimals
            if round_up:
                value = math.ceil(value * step) / step
            text = f"{value:.{decimals}f}"
            return (text.rstrip("0").rstrip(".") if decimals else text) + f" {unit}"
    return f"{size_bytes} bytes"


def mostly_non_latin(texts: Iterable[str]) -> bool:
    """True when more than half the letters in `texts` are outside the Latin script.

    Only letters count, so digits, punctuation, code symbols and whitespace never tip the
    balance. Text forcing re-tokenizes the reference's text, which is least faithful for
    scripts that byte-level tokenizers split into many pieces.
    """
    letters = non_latin = 0
    for text in texts:
        for char in text:
            if not char.isalpha():
                continue
            letters += 1
            if not (char.isascii() or unicodedata.name(char, "").startswith("LATIN ")):
                non_latin += 1
    return letters > 0 and non_latin > letters * _NON_LATIN_SHARE


# Evidence ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LogitEvidence:
    """Mean per-prompt KLD against the reference, with its bootstrap interval over prompts."""

    prompts: int
    """Prompts with a per-prompt KLD, the unit the interval resamples."""
    mean: float
    interval: Interval | None
    """None when the report has no per-prompt results (reports before schema version 2)."""
    thresholds: KldThresholds
    exact: bool
    """True when the backend scored by token id; False for text forcing."""

    def _bounded(self) -> Interval | None:
        """The interval, when there are enough prompts for it to decide anything."""
        return self.interval if self.prompts >= MIN_PROMPTS else None

    @property
    def close(self) -> bool:
        interval = self._bounded()
        return interval is not None and interval.high < self.thresholds.close

    @property
    def moderate(self) -> bool:
        interval = self._bounded()
        return (
            interval is not None
            and interval.low > self.thresholds.close
            and self.mean < self.thresholds.large
        )

    @property
    def large(self) -> bool:
        interval = self._bounded()
        return (
            interval is not None
            and self.mean >= self.thresholds.large
            and interval.low >= self.thresholds.large
        )

    @property
    def near_bar(self) -> str | None:
        """The name of the bar the interval is near ("closeness" or "large-loss"), if any."""
        interval = self._bounded()
        if interval is None:
            return None
        for name, bar in (
            ("closeness", self.thresholds.close),
            ("large-loss", self.thresholds.large),
        ):
            nearest = min(abs(interval.low - bar), abs(interval.high - bar))
            if interval.low <= bar <= interval.high or nearest <= NEAR_BAR * bar:
                return name
        return None

    @property
    def prompts_needed(self) -> int | None:
        """Prompts that would likely settle which side of the nearest bar the mean is on;
        None if more will not.

        Below the closeness bar that is the interval's upper end dropping under it. At or
        above a bar it is the lower end clearing that bar, estimated on the mirrored
        interval with the same square-root scaling.
        """
        interval = self.interval
        if interval is None:
            return None
        if self.mean < self.thresholds.close:
            needed = items_to_bound_below(interval, self.prompts, self.thresholds.close)
        else:
            bar = (
                self.thresholds.large
                if self.mean >= self.thresholds.large
                else self.thresholds.close
            )
            mirrored = Interval(-interval.estimate, -interval.high, -interval.low)
            needed = items_to_bound_below(mirrored, self.prompts, -bar)
        return None if needed is None else max(MIN_PROMPTS, needed)


@dataclass(frozen=True, slots=True)
class _TaskEvidence:
    """Every reliable suite's paired outcomes pooled into one pass-rate comparison."""

    kinds: tuple[TaskKind, ...]
    cases: int
    delta: Interval
    """Candidate minus reference pass rate in points, with a Newcombe 95% interval."""
    pooled_needed: int | None
    """Pooled cases at which the interval's lower end would likely clear -TASK_MARGIN (the
    current count when it already does); None if more cases will not."""
    p_value: float
    """Exact McNemar p-value of the pooled comparison."""

    @property
    def close(self) -> bool:
        return self.cases >= MIN_CASES and self.delta.low >= -TASK_MARGIN

    @property
    def significant_loss(self) -> bool:
        return self.p_value < _ALPHA and self.delta.estimate < 0.0

    @property
    def cases_needed(self) -> int | None:
        """The pooled requirement spread over the suites, as cases per suite."""
        if self.pooled_needed is None:
            return None
        return math.ceil(max(self.pooled_needed, MIN_CASES) / len(self.kinds))


def _logit_evidence(logit: LogitMetrics | None, thresholds: KldThresholds) -> _LogitEvidence | None:
    if logit is None or logit.prompts == 0:
        return None
    values = [p.kld_mean for p in logit.per_prompt if p.kld_mean is not None]
    if not values:
        return _LogitEvidence(0, logit.kld_mean, None, thresholds, logit.exact_token_ids)
    interval = bootstrap_mean(values)
    return _LogitEvidence(
        len(values), interval.estimate, interval, thresholds, logit.exact_token_ids
    )


def _task_evidence(
    reference: dict[TaskKind, dict[str, bool]],
    candidate: dict[TaskKind, dict[str, bool]],
    reliable: frozenset[TaskKind],
) -> _TaskEvidence | None:
    kinds: list[TaskKind] = []
    ref_pass: list[bool] = []
    cand_pass: list[bool] = []
    for kind in SCORED_TASK_KINDS:
        if kind not in reliable:
            continue
        ours, theirs = _pair_cases(reference[kind], candidate[kind])
        if ours:
            kinds.append(kind)
            ref_pass += ours
            cand_pass += theirs
    if not kinds:
        return None
    return _TaskEvidence(
        kinds=tuple(kinds),
        cases=len(ref_pass),
        delta=paired_proportion_diff(ref_pass, cand_pass),
        pooled_needed=cases_to_bound_loss(ref_pass, cand_pass, TASK_MARGIN),
        p_value=mcnemar_exact(ref_pass, cand_pass),
    )


def _outcomes(result: CandidateResult) -> dict[TaskKind, dict[str, bool]]:
    joined: dict[TaskKind, dict[str, bool]] = {kind: {} for kind in SCORED_TASK_KINDS}
    for outcome in result.outcomes:
        if outcome.kind in joined and outcome.passed is not None:
            joined[outcome.kind][outcome.case_id] = outcome.passed
    return joined


def _pair_cases(first: dict[str, bool], second: dict[str, bool]) -> tuple[list[bool], list[bool]]:
    shared = [case_id for case_id in first if case_id in second]
    return [first[c] for c in shared], [second[c] for c in shared]


# Per-candidate measurements ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Candidate:
    result: CandidateResult
    index: int
    name: str
    kld: float | None
    """Mean KLD over all scored positions, as the card shows it."""
    logit: _LogitEvidence | None
    tasks: _TaskEvidence | None
    size: int | None
    size_change: float | None
    deltas: tuple[TaskDelta, ...]
    has_task_mean: bool
    unresolved: tuple[tuple[TaskDelta, int | None], ...]
    """Reliable suites more than TASK_MARGIN points below the reference whose interval still
    reaches zero and whose loss is not significant, each with the paired cases that would
    likely make the loss significant."""

    @property
    def label(self) -> str:
        return self.result.spec.label

    @property
    def measured(self) -> bool:
        agreement = self.result.agreement
        return (
            self.kld is not None
            or self.has_task_mean
            or (agreement is not None and agreement.cases > 0)
        )

    @property
    def regressions(self) -> list[TaskDelta]:
        return sorted(
            (
                d
                for d in self.deltas
                if d.reference_reliable and d.significant and d.delta.estimate < 0
            ),
            key=lambda d: d.delta.estimate,
        )

    @property
    def gains(self) -> list[TaskDelta]:
        return [d for d in self.deltas if d.significant and d.delta.estimate > 0]

    def rank_key(self) -> tuple[bool, float, bool, int, int]:
        return (
            self.kld is None,
            self.kld or 0.0,
            self.size is None,
            self.size or 0,
            self.index,
        )


def _measure(
    result: CandidateResult,
    index: int,
    name: str,
    *,
    reference: CandidateResult,
    reference_outcomes: dict[TaskKind, dict[str, bool]],
    reliable: frozenset[TaskKind],
    thresholds: KldThresholds,
) -> _Candidate:
    logit = result.logit if result.logit is not None and result.logit.prompts > 0 else None
    size = None if result.info is None else result.info.size_bytes
    reference_size = None if reference.info is None else reference.info.size_bytes
    outcomes = _outcomes(result)
    deltas = []
    unresolved = []
    for kind in SCORED_TASK_KINDS:
        ref_pass, cand_pass = _pair_cases(reference_outcomes[kind], outcomes[kind])
        if not ref_pass:
            continue
        delta = TaskDelta(
            kind=kind,
            cases=len(ref_pass),
            candidate_rate=sum(cand_pass) / len(cand_pass),
            reference_rate=sum(ref_pass) / len(ref_pass),
            delta=paired_proportion_diff(ref_pass, cand_pass),
            significant=mcnemar_exact(ref_pass, cand_pass) < _ALPHA,
            reference_reliable=kind in reliable,
        )
        deltas.append(delta)
        if (
            delta.reference_reliable
            and not delta.significant
            and delta.delta.estimate < -TASK_MARGIN
            and delta.delta.high >= 0.0
        ):
            unresolved.append((delta, _cases_to_resolve(ref_pass, cand_pass)))
    return _Candidate(
        result=result,
        index=index,
        name=name,
        kld=None if logit is None else logit.kld_mean,
        logit=_logit_evidence(logit, thresholds),
        tasks=_task_evidence(reference_outcomes, outcomes, reliable),
        size=size,
        size_change=(None if size is None or not reference_size else size / reference_size - 1.0),
        deltas=tuple(deltas),
        has_task_mean=any(
            task.kind in SCORED_TASK_KINDS and task.rate is not None for task in result.tasks
        ),
        unresolved=tuple(unresolved),
    )


def _cases_to_resolve(ref: Sequence[bool], cand: Sequence[bool]) -> int | None:
    """Roughly how many paired cases would make an observed loss significant.

    The observed shares of the four paired outcomes are held fixed while the case count
    grows, as for the task estimate in stats.cases_to_bound_loss, and the exact McNemar
    test is rerun on the scaled counts. None past _MAX_CASES_TO_RESOLVE cases.
    """
    n = len(ref)
    lost = sum(1 for r, c in zip(ref, cand, strict=True) if r and not c)
    gained = sum(1 for r, c in zip(ref, cand, strict=True) if c and not r)

    def significant(cases: int) -> bool:
        scaled_lost = round(lost * cases / n)
        scaled_gained = round(gained * cases / n)
        reference = [True] * scaled_lost + [False] * scaled_gained
        candidate = [False] * scaled_lost + [True] * scaled_gained
        return mcnemar_exact(reference, candidate) < _ALPHA

    low, high = n, 2 * n
    while not significant(high):
        if high >= _MAX_CASES_TO_RESOLVE:
            return None
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if significant(middle):
            high = middle
        else:
            low = middle
    return high


# Judging ----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Call:
    """Status and reasons for one candidate while the verdict is being built."""

    candidate: _Candidate
    status: Status
    reasons: list[str]
    """Phrases, most important first; turned into sentences at the end."""
    detail: str | None = None
    """The sentence this candidate contributes to Verdict.details, if any."""
    prompts_needed: int | None = None
    cases_needed: int | None = None
    """For an inconclusive candidate that more evidence could decide: scoring prompts and
    cases per suite, rounded up. Both None when no rerun is likely to decide."""
    fits: bool | None = None
    """Whether the download fits the --max-size budget; None without a budget or a size."""
    caveats: list[str] = field(default_factory=list)
    task_caveat: str | None = None
    """The first unresolved task loss, as a caveat phrase."""

    @property
    def name(self) -> str:
        return self.candidate.name


def judge(report: Report, *, scoring: Sequence[ScoringPrompt] = ()) -> Verdict:
    """Build the recommendation for `report`. Deterministic: same report, same verdict.

    `scoring` is the text of the scoring prompts the run used, which reports do not store.
    When it is mostly non-Latin script and a text-forced candidate reads above the
    closeness bar, the details say that text forcing may read high on such prompts.
    """
    names = display_labels(report)
    reference = report.reference
    ref_name = names[reference.spec.label]
    ref_outcomes = _outcomes(reference)
    settings = report.settings
    thresholds = kld_thresholds(settings.top_k)
    reliable = frozenset(
        kind
        for kind in SCORED_TASK_KINDS
        if ref_outcomes[kind]
        and sum(ref_outcomes[kind].values()) / len(ref_outcomes[kind]) >= _RELIABLE_REFERENCE_RATE
    )
    candidates = [
        _measure(
            result,
            index,
            names[result.spec.label],
            reference=reference,
            reference_outcomes=ref_outcomes,
            reliable=reliable,
            thresholds=thresholds,
        )
        for index, result in enumerate(report.candidates)
    ]
    calls = [_assess(candidate, ref_name) for candidate in candidates]
    for call in calls:
        _annotate(call, settings)
    _pick(calls, ref_name, settings.max_size_bytes)
    for call in calls:
        if call.status != "failed":
            call.reasons += [
                f"scored higher than the reference on {d.kind}; {_QUANTIZED_REFERENCE}"
                for d in call.candidate.gains
            ]

    ordered = sorted(
        calls, key=lambda call: (_STATUS_ORDER[call.status], call.candidate.rank_key())
    )
    verdicts = tuple(
        CandidateVerdict(
            label=call.candidate.label,
            status=call.status,
            rank=None if call.status == "failed" else rank,
            kld_band=(
                None if call.candidate.kld is None else kld_band(call.candidate.kld, thresholds)
            ),
            size_bytes=call.candidate.size,
            size_change=call.candidate.size_change,
            task_deltas=call.candidate.deltas,
            reasons=tuple(_sentence(reason) for reason in call.reasons),
            caveats=tuple(call.caveats),
            near_bar=call.candidate.logit is not None and call.candidate.logit.near_bar is not None,
            fits_budget=call.fits,
        )
        for rank, call in enumerate(ordered, start=1)
    )
    summary = _summarize(
        ordered, ref_name, settings, non_latin=mostly_non_latin(p.text for p in scoring)
    )
    return Verdict(
        headline=summary.headline,
        details=summary.details,
        candidates=verdicts,
        findings=_findings(report, names),
        cases_needed=summary.cases_needed,
        remedy=summary.remedy,
    )


def _assess(candidate: _Candidate, ref_name: str) -> _Call:
    """The candidate's status on its own evidence; close candidates come back as ok."""
    if not candidate.measured:
        return _Call(candidate, "failed", [_failure_reason(candidate.result)])
    avoid = _avoid_reasons(candidate, ref_name)
    if avoid:
        return _Call(
            candidate, "avoid", avoid, f"Avoid {candidate.name}: {' and '.join(avoid[:2])}."
        )
    logit, tasks = candidate.logit, candidate.tasks
    # Significant task losses were handled above, so with logit evidence only KLD decides.
    close = logit.close if logit is not None else tasks is not None and tasks.close
    if close:
        return _Call(candidate, "ok", _close_reasons(candidate, ref_name))
    if logit is not None and logit.moderate and logit.interval is not None:
        return _usable(candidate, logit.interval, ref_name)
    return _inconclusive(candidate, ref_name)


def _annotate(call: _Call, settings: RunSettings) -> None:
    """Budget fit and caveats, which sit next to the status without changing it."""
    candidate = call.candidate
    budget = settings.max_size_bytes
    if budget is not None and candidate.size is not None:
        call.fits = candidate.size <= budget
    if call.status == "failed":
        return
    near = None if candidate.logit is None else candidate.logit.near_bar
    if near is not None:
        call.caveats.append(f"near the {near} bar; a rerun could change this")
    tasks = [_task_caveat(delta, needed, settings) for delta, needed in candidate.unresolved]
    call.caveats += tasks
    call.task_caveat = tasks[0] if tasks else None


def _task_caveat(delta: TaskDelta, needed: int | None, settings: RunSettings) -> str:
    """E.g. "tools -17 unresolved (95% CI -42 to +6); rerun with --max-cases 60"."""
    text = (
        f"{delta.kind} {round(delta.delta.estimate)} unresolved ({_interval_points(delta.delta)})"
    )
    if needed is None:
        return text
    cases = _round_up(needed)
    if settings.prompts_file is None and cases <= len(load_builtin(delta.kind)):
        return f"{text}; rerun with --max-cases {cases}"
    return f"{text}; about {cases} {delta.kind} cases would settle it"


def _failure_reason(result: CandidateResult) -> str:
    if result.errors:
        return "no metrics: " + " ".join(result.errors[0].split()).rstrip(".")
    return "no metrics were produced"


def _avoid_reasons(candidate: _Candidate, ref_name: str) -> list[str]:
    reasons = [
        f"{d.kind} drops {-round(d.delta.estimate)} points vs {ref_name} "
        f"({_interval_points(d.delta)})"
        for d in candidate.regressions
    ]
    tasks = candidate.tasks
    if not reasons and tasks is not None and tasks.significant_loss:
        reasons.append(
            f"task scores drop {_points(-tasks.delta.estimate)} points overall vs {ref_name} "
            f"({_interval_points(tasks.delta)} on {_plural(tasks.cases, 'case')})"
        )
    logit = candidate.logit
    if logit is not None and logit.large and logit.interval is not None:
        reasons.append(f"KLD is large ({_kld(logit.mean)}, {_interval_kld(logit.interval)})")
    return reasons


def _close_reasons(candidate: _Candidate, ref_name: str) -> list[str]:
    reasons = []
    logit, tasks = candidate.logit, candidate.tasks
    if logit is not None and logit.interval is not None:
        reasons.append(
            f"KLD {_kld(logit.mean)} ({_interval_kld(logit.interval)}) "
            f"on {_plural(logit.prompts, 'prompt')}"
        )
    if tasks is not None:
        reasons.append(
            _tasks_within(tasks, ref_name) if tasks.close else _no_task_loss(tasks, ref_name)
        )
    note = _evidence_note(candidate, ref_name)
    if note:
        reasons.append(f"rests on {note}")
    return reasons


def _usable(candidate: _Candidate, interval: Interval, ref_name: str) -> _Call:
    loss = f"moderate loss: KLD {_kld(interval.estimate)} ({_interval_kld(interval)})"
    reasons = [loss]
    tasks = candidate.tasks
    if tasks is not None:
        reasons.append(
            _tasks_within(tasks, ref_name) if tasks.close else _no_task_loss(tasks, ref_name)
        )
    return _Call(candidate, "usable", reasons, f"{candidate.name} has a {loss}.")


def _evidence_note(candidate: _Candidate, ref_name: str) -> str | None:
    """Which evidence a close call rests on, when it is only one kind."""
    if candidate.tasks is None:
        if candidate.deltas:
            return (
                f"logit evidence only; {ref_name} passes under half of every task suite, "
                "so the suites cannot judge it"
            )
        return "logit evidence only; no task suites ran"
    if candidate.logit is None:
        return "task evidence only; no logit metrics"
    return None


def _inconclusive(candidate: _Candidate, ref_name: str) -> _Call:
    logit, tasks = candidate.logit, candidate.tasks
    reasons: list[str] = []
    gaps: list[str] = []
    needs: list[int | None] = []
    prompts_needed = cases_needed = None
    if logit is not None:
        if logit.close and logit.interval is not None:
            reasons.append(
                f"KLD {_kld(logit.mean)} is under the {logit.thresholds.bar} closeness bar "
                f"({_interval_kld(logit.interval)})"
            )
        else:
            gaps.append(_logit_gap(logit))
            prompts_needed = logit.prompts_needed
            needs.append(prompts_needed)
    if tasks is not None:
        if tasks.close:
            reasons.append(_tasks_within(tasks, ref_name))
        else:
            gaps.append(_task_gap(tasks, ref_name))
            cases_needed = tasks.cases_needed
            needs.append(cases_needed)
    if logit is None and tasks is None:
        gaps.append("no logit metrics and no reliable task results to judge it by")
        needs.append(None)
    call = _Call(candidate, "inconclusive", gaps + reasons)
    if all(need is not None for need in needs):
        call.prompts_needed = None if prompts_needed is None else _round_up(prompts_needed)
        call.cases_needed = None if cases_needed is None else _round_up(cases_needed)
        call.reasons.append(f"{_wanted(call)} would likely decide")
    above = logit is not None and logit.mean >= logit.thresholds.close
    state = "is undecided" if above else "is not shown to be close"
    call.detail = f"{candidate.name} {state}: {gaps[0]}."
    return call


def _logit_gap(logit: _LogitEvidence) -> str:
    kld = _kld(logit.mean)
    interval = logit.interval
    if interval is None:
        return f"KLD {kld}, but the report has no per-prompt KLD to bound it"
    if logit.prompts < MIN_PROMPTS:
        return _thin_logit_gap(logit)
    thresholds = logit.thresholds
    bar = thresholds.bar
    if logit.mean >= thresholds.large:
        return (
            f"KLD {kld} may be a large loss, but its 95% CI starts at {_kld(interval.low)}, "
            f"under the {thresholds.large_bar} large-loss bar"
        )
    if logit.mean >= thresholds.close:
        return (
            f"KLD {kld} is above the {bar} closeness bar, but its 95% CI reaches down to "
            f"{_kld(interval.low)}"
        )
    return f"KLD {kld}, but its 95% CI reaches {_kld(interval.high)}, above the {bar} closeness bar"


def _thin_logit_gap(logit: _LogitEvidence) -> str:
    """Why fewer than MIN_PROMPTS prompts decide nothing about this KLD."""
    kld = _kld(logit.mean)
    bar = logit.thresholds.bar
    if logit.mean >= logit.thresholds.large:
        return f"KLD {kld} looks large on {_plural(logit.prompts, 'prompt')}, too few to call it"
    if logit.mean >= logit.thresholds.close:
        return f"KLD {kld} is above the {bar} closeness bar"
    return f"KLD {kld}, but {_plural(logit.prompts, 'prompt')} cannot bound it below {bar}"


def _task_gap(tasks: _TaskEvidence, ref_name: str) -> str:
    margin = _points(TASK_MARGIN)
    cases = _plural(tasks.cases, "case")
    if tasks.delta.estimate <= -TASK_MARGIN:
        return f"task scores are {_points(-tasks.delta.estimate)} points lower than {ref_name}"
    if tasks.cases < MIN_CASES:
        return f"{cases} cannot show task scores within {margin} points of {ref_name}"
    return (
        f"task scores could be up to {_points(-tasks.delta.low)} points lower than {ref_name} "
        f"({_interval_points(tasks.delta)} on {cases})"
    )


def _no_task_loss(tasks: _TaskEvidence, ref_name: str) -> str:
    return (
        f"no significant task loss vs {ref_name} on {_plural(tasks.cases, 'case')} "
        f"({_interval_points(tasks.delta)})"
    )


def _tasks_within(tasks: _TaskEvidence, ref_name: str) -> str:
    return (
        f"task scores within {_points(TASK_MARGIN)} points of {ref_name} "
        f"on {_plural(tasks.cases, 'case')}"
    )


def _wanted(call: _Call) -> str:
    """The evidence an inconclusive candidate needs, e.g. "about 20 scoring prompts"."""
    parts = []
    if call.prompts_needed is not None:
        parts.append(f"{call.prompts_needed} scoring prompts")
    if call.cases_needed is not None:
        parts.append(f"{call.cases_needed} cases per suite")
    return "about " + " and ".join(parts)


def _pick(calls: Sequence[_Call], ref_name: str, budget: int | None) -> None:
    """Recommend the smallest close candidate that fits the budget; the other close ones
    stay ok, and those over the budget say so."""
    close = [call for call in calls if call.status == "ok"]
    if budget is not None:
        for call in close:
            if not call.fits:
                call.reasons.insert(0, _over_budget(call, ref_name, budget))
                call.detail = f"{call.name} is close but {_needs(call)}."
    eligible = [call for call in close if budget is None or call.fits]
    if not eligible:
        return
    if all(call.candidate.size is not None for call in eligible):
        pick = min(eligible, key=lambda call: (call.candidate.size, call.candidate.rank_key()))
    else:
        pick = min(eligible, key=lambda call: call.candidate.rank_key())
    pick.status = "recommended"
    if len(eligible) > 1:
        pick.reasons.append(f"smallest download that is close to {ref_name}")
    for call in eligible:
        if call is pick:
            continue
        reason = _also_close(call.candidate, pick.candidate, ref_name)
        call.reasons.insert(0, reason)
        call.detail = f"{call.name} is {reason}."


def _also_close(candidate: _Candidate, pick: _Candidate, ref_name: str) -> str:
    if candidate.size is not None and pick.size:
        larger = candidate.size / pick.size - 1.0
        if larger > _SAME_SIZE:
            return f"also close to {ref_name} but {_percent(larger)} larger"
    return f"also close to {ref_name}"


def _over_budget(call: _Call, ref_name: str, budget: int) -> str:
    size = call.candidate.size
    if size is None:
        return (
            f"close to {ref_name}, but its size is unknown, so it cannot be checked against "
            f"the {format_size(budget)} budget"
        )
    return (
        f"close to {ref_name} but needs {format_size(size)}, over the {format_size(budget)} budget"
    )


def _needs(call: _Call) -> str:
    size = call.candidate.size
    return "its size is unknown" if size is None else f"needs {format_size(size)}"


# Headline and details ---------------------------------------------------------------------


@dataclass(slots=True)
class _Summary:
    headline: str
    details: tuple[str, ...] = ()
    remedy: str | None = None
    cases_needed: int | None = None


def _summarize(
    ordered: Sequence[_Call], ref_name: str, settings: RunSettings, *, non_latin: bool
) -> _Summary:
    if not ordered:
        return _Summary("No candidates were evaluated.")
    live = [call for call in ordered if call.status != "failed"]
    failed = [call.name for call in ordered if call.status == "failed"]
    if not live:
        return _Summary("No candidate produced metrics; see the errors for each candidate.")
    lead = _lead(live, ref_name, settings.max_size_bytes)
    summary = _Summary(lead.headline)
    if live[0].status != "recommended":
        summary.remedy, summary.cases_needed = _next_step(
            live, ref_name, settings, non_latin=non_latin
        )
    details = list(lead.lines)
    caveated = [lead.about, *(call for call in live if call is not lead.about)]
    unresolved = next((call for call in caveated if call.task_caveat), None)
    if unresolved is not None:
        # The headline's own candidate first, right under the headline; another one's
        # unresolved loss after the numbers behind the headline.
        at = 0 if unresolved is lead.about else len(details)
        details.insert(at, f"{unresolved.name}: {unresolved.task_caveat}.")
    details += [
        call.detail
        for call in live
        if call.detail and not (call is lead.about and lead.replaces_detail)
    ]
    if non_latin and _text_forced_above_bar(live):
        note = "Most scoring prompts are not Latin script; text forcing may read higher there."
        details.append(note if summary.remedy == _CONFIRM_EXACT else f"{note} {_CONFIRM_EXACT}")
    details += [
        f"{call.name} scored higher than the reference on {d.kind}; {_QUANTIZED_REFERENCE}."
        for call in live
        for d in call.candidate.gains
    ]
    if failed:
        details.append(f"{_join(failed)} produced no metrics; see the errors.")
    summary.details = tuple(details)
    return summary


@dataclass(frozen=True, slots=True)
class _Lead:
    headline: str
    about: _Call
    """The candidate the headline is about."""
    lines: tuple[str, ...] = ()
    """The numbers behind the headline, first in the details."""
    replaces_detail: bool = False
    """True when the headline or the lines already say what the candidate's detail says."""


def _lead(live: Sequence[_Call], ref_name: str, budget: int | None) -> _Lead:
    """The headline for candidates in rank order, at least one of them not failed."""
    top = live[0]
    if top.status == "recommended":
        return _Lead(
            _run_headline(top.candidate, ref_name),
            top,
            (_run_detail(top.candidate, ref_name),),
            replaces_detail=True,
        )
    close = [call for call in live if call.status == "ok"]
    usable = [
        (call, call.candidate.logit)
        for call in live
        if call.status == "usable" and call.candidate.logit is not None
    ]
    fitting = [(call, logit) for call, logit in usable if budget is None or call.fits]
    if fitting:
        best, logit = fitting[0]
        if budget is None:
            unsettled = any(call.status == "inconclusive" for call in live)
            headline = _smallest_loss_headline(best, logit, ref_name, unsettled=unsettled)
        else:
            headline = (
                f"Best that fits {format_size(budget)}: {best.name}, moderate loss "
                f"(KLD {_kld(logit.mean)})."
            )
        return _Lead(headline, best)
    if budget is not None and (close or usable):
        nearest = close[0] if close else usable[0][0]
        state = "is close" if close else "has a moderate loss"
        headline = (
            f"Nothing that fits {format_size(budget)} is close to {ref_name}; "
            f"{nearest.name} {state} but {_needs(nearest)}."
        )
        return _Lead(headline, nearest, replaces_detail=bool(close))
    if top.status == "inconclusive":
        closest = _closest_detail(top.candidate, ref_name)
        return _Lead(
            _keep_for_now_headline(top.candidate, ref_name),
            top,
            () if closest is None else (closest,),
            replaces_detail=True,
        )
    who = f"{top.name} shows" if len(live) == 1 else "every candidate shows"
    return _Lead(f"Keep {ref_name}: {who} a measured loss.", top)


def _text_forced_above_bar(live: Sequence[_Call]) -> bool:
    """Whether a text-forced candidate reads above the closeness bar."""
    return any(
        call.candidate.logit is not None
        and not call.candidate.logit.exact
        and call.candidate.logit.mean > call.candidate.logit.thresholds.close
        for call in live
    )


def _next_step(
    live: Sequence[_Call], ref_name: str, settings: RunSettings, *, non_latin: bool
) -> tuple[str, int | None]:
    """What to do when nothing is recommended, and the --max-cases value of a rerun."""
    for call in live:
        if call.status == "inconclusive":
            rerun = _remedy(call, settings)
            if rerun is not None:
                return rerun
    return _advice(live, ref_name, settings, non_latin=non_latin), None


def _advice(live: Sequence[_Call], ref_name: str, settings: RunSettings, *, non_latin: bool) -> str:
    """The next step when no rerun of the same kind is likely to decide."""
    text_forced = _text_forced_above_bar(live)
    budget = settings.max_size_bytes
    over = sorted(
        (call.candidate.size, call.name)
        for call in live
        if call.status == "ok" and call.candidate.size is not None
    )
    if non_latin and text_forced:
        return _CONFIRM_EXACT
    if budget is None and any(call.status == "usable" for call in live):
        return (
            "Pass --max-size with the memory you can spare, for example --max-size 6GB, to "
            "pick the best download that fits."
        )
    if budget is not None and over:
        size, name = over[0]
        flag = format_size(size, round_up=True).replace(" ", "")
        return f"Rerun with --max-size {flag} to run {name}, which is close."
    if text_forced:
        return _CONFIRM_EXACT
    return _missing_evidence(live, ref_name)


def _missing_evidence(live: Sequence[_Call], ref_name: str) -> str:
    undecided = [call.candidate for call in live if call.status == "inconclusive"]
    if any(c.logit is None and c.tasks is None for c in undecided):
        return (
            "Rerun on a server that returns logprobs, or with json or tools cases the "
            "reference passes, so there is evidence to judge by."
        )
    if any(c.logit is not None and c.logit.interval is None for c in undecided):
        return (
            "Rerun with this version of quantdiff, which records the per-prompt KLD needed "
            "to decide."
        )
    return f"Try a larger quant against {ref_name}; nothing tested here is close."


def _run_headline(pick: _Candidate, ref_name: str) -> str:
    logit = pick.logit
    if logit is not None and logit.interval is not None:
        evidence = (
            f"close on logits (KLD {_kld(logit.mean)}, CI up to {_kld(logit.interval.high)}) "
            f"on {_plural(logit.prompts, 'prompt')}"
        )
    else:
        cases = 0 if pick.tasks is None else pick.tasks.cases
        evidence = f"close on task scores on {_plural(cases, 'case')} (no logit metrics)"
    change = pick.size_change
    if change is not None and change < -_SAME_SIZE:
        return f"Run {pick.name}: {_percent(-change)} smaller than {ref_name}, {evidence}."
    if change is not None and change > _SAME_SIZE:
        return f"Run {pick.name}: {_percent(change)} larger than {ref_name}, {evidence}."
    return f"Run {pick.name}: {evidence}."


def _smallest_loss_headline(
    best: _Call, logit: _LogitEvidence, ref_name: str, *, unsettled: bool
) -> str:
    """The headline when nothing is close but a candidate has a moderate loss."""
    close = "shown to be close" if unsettled else "close"
    change = best.candidate.size_change
    among = " among the smaller downloads" if change is not None and change < -_SAME_SIZE else ""
    return (
        f"No download is {close} to {ref_name}; {best.name} has the smallest loss{among} "
        f"(moderate, KLD {_kld(logit.mean)})."
    )


def _run_detail(pick: _Candidate, ref_name: str) -> str:
    """The numbers behind a Run headline, as one sentence."""
    logit, tasks = pick.logit, pick.tasks
    parts = []
    if logit is not None and logit.interval is not None:
        parts.append(f"KLD {_kld(logit.mean)} ({_interval_kld(logit.interval)})")
    if tasks is not None:
        parts.append(
            _tasks_within(tasks, ref_name) if tasks.close else _no_task_loss(tasks, ref_name)
        )
    sentence = f"{pick.name}: {' and '.join(parts)}"
    note = _evidence_note(pick, ref_name)
    return f"{sentence}; {note}." if note else f"{sentence}."


def _keep_for_now_headline(closest: _Candidate, ref_name: str) -> str:
    logit, tasks = closest.logit, closest.tasks
    if logit is None and tasks is None:
        return f"Keep {ref_name} for now: no candidate has logit or reliable task results."
    scope = []
    if logit is not None and logit.prompts:
        scope.append(_plural(logit.prompts, "prompt"))
    if tasks is not None:
        scope.append(_plural(tasks.cases, "case"))
    on = f" on {' and '.join(scope)}" if scope else ""
    return f"Keep {ref_name} for now: no candidate is shown to be close{on}."


def _closest_detail(closest: _Candidate, ref_name: str) -> str | None:
    """Why the candidate that looks closest is still not proven close."""
    logit, tasks = closest.logit, closest.tasks
    looks = []
    if logit is not None:
        if logit.interval is None:
            looks.append(f"KLD {_kld(logit.mean)}, with no per-prompt results to bound it")
        elif logit.prompts >= MIN_PROMPTS:
            looks.append(f"KLD {_kld(logit.mean)}, 95% CI up to {_kld(logit.interval.high)}")
        else:
            looks.append(f"KLD {_kld(logit.mean)} on {_plural(logit.prompts, 'prompt')}")
    if tasks is not None:
        lower = round(-tasks.delta.estimate)
        level = f"{lower} points lower" if lower > 0 else f"level with {ref_name}"
        looks.append(f"task scores {level}, 95% CI down to {_signed(tasks.delta.low)}")
    if not looks:
        return None
    return f"{closest.name} looks closest: {'; '.join(looks)}."


def _remedy(call: _Call, settings: RunSettings) -> tuple[str, int] | None:
    """What to rerun with to decide an inconclusive candidate, and the --max-cases value."""
    prompts, cases = call.prompts_needed, call.cases_needed
    if prompts is None and cases is None:
        return None
    flag = max(prompts or 0, cases or 0)
    wanted = _sentence(_wanted(call)).removesuffix(".")
    if settings.prompts_file is not None:
        return f"{wanted} would decide; add them to {settings.prompts_file}.", flag
    kinds = () if cases is None or call.candidate.tasks is None else call.candidate.tasks.kinds
    room = [len(load_builtin(kind)) for kind in kinds]
    if prompts is not None:
        room.append(len(load_scoring_prompts()))
    if flag <= min(room):
        return f"Rerun with --max-cases {flag} to decide.", flag
    return (
        f"{wanted} would decide. That is more than the built-in suites hold, so add your "
        "own with --prompts.",
        flag,
    )


# Server findings --------------------------------------------------------------------------


def _findings(report: Report, names: dict[str, str]) -> tuple[ServerFinding, ...]:
    results = (report.reference, *report.candidates)
    grouped: dict[PreflightFinding, list[str]] = {}
    for result in results:
        failed = {f.check for f in result.preflight if f.severity == "fail"}
        for finding in result.preflight:
            if finding.severity not in ("warn", "fail"):
                continue
            if finding.severity == "warn" and finding.check in failed:
                continue
            labels = grouped.setdefault(finding, [])
            if result.spec.label not in labels:
                labels.append(result.spec.label)
    by_label = {result.spec.label: result for result in results}
    findings = [
        ServerFinding(finding, tuple(labels), *_impact(finding, labels, by_label, report))
        for finding, labels in grouped.items()
    ]
    findings += _same_weights(report, names)
    severity = {"fail": 0, "warn": 1}
    findings.sort(key=lambda f: (not f.affects_scores, severity.get(f.finding.severity, 2)))
    return tuple(findings)


def _impact(
    finding: PreflightFinding,
    labels: Sequence[str],
    by_label: dict[str, CandidateResult],
    report: Report,
) -> tuple[bool, str]:
    if finding.check == "context":
        return _context_impact(labels, by_label, report.settings.longest_prompt_tokens)
    return _FIXED_IMPACT.get(
        finding.check, (True, "may affect scores: a serving problem can lower a model's results")
    )


_FIXED_IMPACT: Final[dict[str, tuple[bool, str]]] = {
    "template": (True, "affects these scores: every task case runs through the chat template"),
    "tokenizer": (True, "affects these scores: logit positions do not line up with the reference"),
    "logprobs": (
        False,
        "does not affect these scores: logit metrics are missing, so only task results count",
    ),
}


def _context_impact(
    labels: Sequence[str], by_label: dict[str, CandidateResult], longest: int | None
) -> tuple[bool, str]:
    if longest is None:
        return True, "may affect scores: the length of the longest prompt is not known"
    infos = [by_label[label].info for label in labels]
    known = [info.context_length for info in infos if info and info.context_length]
    if len(known) < len(infos):
        return True, "may affect scores: the context window is not known"
    window = min(known)
    if window >= longest:
        return False, (
            f"does not affect these scores: the longest prompt is ~{longest} tokens and "
            f"the context window is {window}"
        )
    return True, (
        f"may affect scores: the longest prompt is ~{longest} tokens but the context "
        f"window is {window}"
    )


def _same_weights(report: Report, names: dict[str, str]) -> list[ServerFinding]:
    info = report.reference.info
    reference_id = None if info is None else info.weights_id
    if reference_id is None:
        return []
    return [
        ServerFinding(
            finding=PreflightFinding(
                check="same-weights",
                severity="warn",
                message=f"{names[result.spec.label]} serves the same weights as the reference",
                fix="Point the reference at a different download, such as the BF16 or Q8_0 file.",
            ),
            labels=(result.spec.label,),
            affects_scores=True,
            impact="affects these scores: the candidate is being compared with itself",
        )
        for result in report.candidates
        if result.info is not None and result.info.weights_id == reference_id
    ]


# Formatting -------------------------------------------------------------------------------


def _sentence(phrase: str) -> str:
    # Suite names are identifiers ("json", "tools") and stay lowercase at a sentence start.
    starts_with_suite = phrase.split(" ", 1)[0] in SCORED_TASK_KINDS
    text = phrase if starts_with_suite else phrase[:1].upper() + phrase[1:]
    return text if text.endswith(".") else text + "."


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def _kld(value: float) -> str:
    """Up to three decimals from 0.01 up, which keeps values near the 0.05 bar readable;
    two significant digits below that."""
    if value >= 10:
        return f"{value:.0f}"
    if value >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if 0 < value < 0.0001:
        return "under 0.0001"
    return f"{value:.2g}"


def _interval_kld(interval: Interval) -> str:
    return f"95% CI {_kld(interval.low)} to {_kld(interval.high)}"


def _interval_points(interval: Interval) -> str:
    return f"95% CI {_signed(interval.low)} to {_signed(interval.high)}"


def _signed(points: float) -> str:
    rounded = round(points)
    return f"+{rounded}" if rounded > 0 else str(rounded)


def _points(points: float) -> str:
    return f"{points:.0f}"


def _round_up(count: int) -> int:
    return math.ceil(count / _ROUND_NEEDED_TO) * _ROUND_NEEDED_TO


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _join(items: Sequence[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
