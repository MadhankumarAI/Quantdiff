"""Paired significance tests and confidence intervals for comparing two models.

Every candidate answers the same cases and is scored on the same prompts as the reference,
so comparisons are paired: each case contributes one (reference, candidate) observation.
Pairing removes the case-to-case difficulty variation that dominates small suites, which is
why these tests separate models that an unpaired test calls "within noise".

Everything here is pure and deterministic. Bootstrap resampling uses its own seeded
generator, so the same input always yields the same interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

__all__ = [
    "DEFAULT_RESAMPLES",
    "Interval",
    "bootstrap_mean",
    "cases_to_bound_loss",
    "items_to_bound_below",
    "mcnemar_exact",
    "paired_proportion_diff",
]

DEFAULT_RESAMPLES: Final = 2000
_CONFIDENCE: Final = 0.95
_Z_TWO_SIDED: Final = NormalDist().inv_cdf(1 - (1 - _CONFIDENCE) / 2)
_MAX_CASES_SEARCHED: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a 95% confidence interval, in the metric's own units."""

    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def mcnemar_exact(ref: Sequence[bool], cand: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail outcomes.

    Only discordant pairs carry information: b cases the reference passed and the candidate
    failed, c the other way round. Under equal pass rates each discordant pair is a fair
    coin, so the p-value is the two-sided binomial tail of min(b, c) in b + c trials,
    capped at 1. With no discordant pairs there is no evidence of a difference: 1.0.
    """
    _check_paired(ref, cand)
    lost = sum(1 for r, c in zip(ref, cand, strict=True) if r and not c)
    gained = sum(1 for r, c in zip(ref, cand, strict=True) if c and not r)
    trials = lost + gained
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(min(lost, gained) + 1))
    # Integer division of exact big ints keeps full precision even for thousands of trials.
    return min(1.0, 2 * tail / (1 << trials))


def paired_proportion_diff(ref: Sequence[bool], cand: Sequence[bool]) -> Interval:
    """Candidate minus reference pass rate, in percentage points, with a 95% interval.

    The interval is Newcombe's hybrid score method for paired proportions (method 10 in
    Newcombe, Statistics in Medicine 17:2635, 1998). It combines the Wilson score interval
    of each rate with the observed correlation between the paired outcomes, stays inside
    [-100, 100], and keeps sensible coverage at 0% and 100% where the Wald interval
    collapses to a point.
    """
    _check_paired(ref, cand)
    if not ref:
        raise ValueError("paired_proportion_diff needs at least one pair")
    return _newcombe(*_table(ref, cand))


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Mean of `values` with a 95% percentile bootstrap interval. Deterministic for a seed."""
    if not values:
        raise ValueError("bootstrap_mean needs at least one value")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    n = len(values)
    estimate = math.fsum(values) / n
    rng = random.Random(seed)  # noqa: S311 - seeded for reproducible intervals, not secrecy
    means = sorted(
        math.fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    tail = (1 - _CONFIDENCE) / 2
    return Interval(
        estimate=estimate,
        low=min(estimate, _percentile(means, tail)),
        high=max(estimate, _percentile(means, 1 - tail)),
    )


def items_to_bound_below(interval: Interval, items: int, bound: float) -> int | None:
    """Roughly how many items would put the upper end of `interval` below `bound`.

    `interval` is a 95% interval of a mean over `items` items. Its upper half-width shrinks
    with the square root of the item count, so the estimate keeps the observed mean and
    spread and scales the half-width by sqrt(items / n). None when the observed mean is
    already at or above `bound`, since more items would only narrow the interval around it.
    """
    if items < 1:
        raise ValueError("items must be at least 1")
    room = bound - interval.estimate
    if room <= 0.0:
        return None
    half_width = interval.high - interval.estimate
    return max(items, math.floor(items * (half_width / room) ** 2) + 1)


def cases_to_bound_loss(ref: Sequence[bool], cand: Sequence[bool], margin: float) -> int | None:
    """Roughly how many paired cases would put the lower end of the pass-rate difference
    (candidate minus reference, in points) at or above -`margin` points.

    The observed shares of the four outcomes (both pass, only the reference passes, only
    the candidate passes, neither) are held fixed while the case count grows, and the
    Newcombe interval of paired_proportion_diff is recomputed until its lower bound clears
    the margin. None when the observed difference is already at or below -`margin`, since
    more cases would only narrow the interval around it.
    """
    _check_paired(ref, cand)
    if not ref:
        raise ValueError("cases_to_bound_loss needs at least one pair")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    n = len(ref)
    shares = [count / n for count in _table(ref, cand)]

    def clears(cases: int) -> bool:
        both, lost, gained, neither = (share * cases for share in shares)
        return _newcombe(both, lost, gained, neither).low >= -margin

    if _newcombe(*shares).estimate <= -margin:
        return None
    if clears(n):
        return n
    low, high = n, 2 * n
    while not clears(high):
        if high >= _MAX_CASES_SEARCHED:
            return None
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if clears(middle):
            high = middle
        else:
            low = middle
    return high


def _table(ref: Sequence[bool], cand: Sequence[bool]) -> tuple[int, int, int, int]:
    """Counts of (both pass, only the reference passes, only the candidate passes, neither)."""
    both = sum(1 for r, c in zip(ref, cand, strict=True) if r and c)
    lost = sum(1 for r, c in zip(ref, cand, strict=True) if r and not c)
    gained = sum(1 for r, c in zip(ref, cand, strict=True) if c and not r)
    return both, lost, gained, len(ref) - both - lost - gained


def _newcombe(both: float, lost: float, gained: float, neither: float) -> Interval:
    """Newcombe's hybrid score interval for a paired difference, from (possibly scaled)
    cell counts of the 2x2 table, in percentage points."""
    n = both + lost + gained + neither
    p_ref, p_cand = (both + lost) / n, (both + gained) / n
    low_ref, high_ref = _wilson(both + lost, n)
    low_cand, high_cand = _wilson(both + gained, n)
    margins = (both + lost) * (gained + neither) * (both + gained) * (lost + neither)
    phi = 0.0 if margins == 0 else (both * neither - lost * gained) / math.sqrt(margins)
    estimate = p_cand - p_ref
    below = math.sqrt(
        max(
            0.0,
            (p_cand - low_cand) ** 2
            - 2 * phi * (p_cand - low_cand) * (high_ref - p_ref)
            + (high_ref - p_ref) ** 2,
        )
    )
    above = math.sqrt(
        max(
            0.0,
            (high_cand - p_cand) ** 2
            - 2 * phi * (high_cand - p_cand) * (p_ref - low_ref)
            + (p_ref - low_ref) ** 2,
        )
    )
    return Interval(
        estimate=estimate * 100,
        low=max(-1.0, estimate - below) * 100,
        high=min(1.0, estimate + above) * 100,
    )


def _wilson(successes: float, n: float) -> tuple[float, float]:
    p = successes / n
    z2 = _Z_TWO_SIDED * _Z_TWO_SIDED
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z_TWO_SIDED * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks, as numpy's default percentile does."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _check_paired(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
