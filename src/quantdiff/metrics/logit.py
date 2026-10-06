"""Teacher-forced logit metrics: top-1 agreement and a lower bound on KL divergence.

Servers return only the k most likely tokens at each position, so the full-vocabulary
KL(P_ref || Q_cand) cannot be computed. Both distributions are instead collapsed onto a
coarser partition: every token listed in both top-k lists keeps its own cell, the
reference's tokens missing from the candidate's list share a cell A, and everything else
shares a cell B. Merging outcomes can never increase KL divergence (data processing
inequality), so KL on this partition is a lower bound on the true KL.

The candidate's mass on A is not reported, but it is bounded twice over: an unlisted token
cannot be more likely than the least likely listed one, so Q(A) <= |A| * min listed
probability, and A lies outside the candidate's list, so Q(A) <= 1 - the listed mass.
We take the Q(A) within those bounds that minimizes the partition KL, which keeps the
result a lower bound while still penalizing a candidate whose top-k misses tokens the
reference considers likely.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from typing import Final

from quantdiff.types import (
    LogitMetrics,
    PromptLogit,
    ReferenceTrace,
    TokenProb,
    TokenStep,
    TopK,
)

MASS_FLOOR: Final = 1e-12
"""Smallest mass a cell may hold, so a cell one side assigns ~0 cannot produce log(0)."""

PERCENTILE: Final = 0.99


def partition_kld(ref_top: TopK, cand_top: TopK, *, by_id: bool) -> float:
    """Return a lower bound, in nats, on the full-vocabulary KL(P_ref || Q_cand).

    Tokens are matched by id when `by_id` is True and every token carries an id,
    otherwise by string. `cand_top` must not be empty.
    """
    if not cand_top:
        raise ValueError("candidate top-k is empty; the KL bound needs at least one token")
    use_ids = by_id and _all_have_ids(ref_top, cand_top)
    ref_mass = _mass_by_token(ref_top, use_ids=use_ids)
    cand_mass = _mass_by_token(cand_top, use_ids=use_ids)
    shared = ref_mass.keys() & cand_mass.keys()
    ref_only = ref_mass.keys() - shared

    p_shared = math.fsum(ref_mass[key] for key in shared)
    p_missing = math.fsum(ref_mass[key] for key in ref_only)
    p_rest = max(1.0 - p_shared - p_missing, 0.0)
    q_unlisted = _rest_mass([cand_mass[key] for key in shared])
    # Tokens in A are absent from the candidate's whole list, so they share what that
    # list leaves over, not merely what the shared tokens leave over.
    q_free = _rest_mass(cand_mass.values())

    # Q(A) that minimizes the A and B terms is proportional to P; clamp it to its bounds.
    proportional = q_unlisted * p_missing / max(p_missing + p_rest, MASS_FLOOR)
    q_missing = min(proportional, len(ref_only) * min(cand_mass.values()), q_free, q_unlisted)
    q_rest = q_unlisted - q_missing

    cells = [(ref_mass[key], cand_mass[key]) for key in shared]
    cells += [(p_missing, q_missing), (p_rest, q_rest)]
    divergence = math.fsum(_kl_term(p, q) for p, q in cells)
    # Floors and float rounding can leave a tiny negative residue on near-equal inputs.
    return max(divergence, 0.0)


def top1_match(ref_step: TokenStep, cand_top: TopK, *, by_id: bool) -> bool:
    """Return True if the candidate's most likely token is the reference's greedy choice."""
    if not cand_top:
        return False
    best = max(cand_top, key=lambda token: token.logprob)
    chosen = ref_step.chosen
    if by_id and chosen.token_id is not None and best.token_id is not None:
        return best.token_id == chosen.token_id
    return best.token == chosen.token


def logit_metrics(
    traces: Sequence[ReferenceTrace],
    candidate: Sequence[Sequence[TopK]],
    *,
    exact_token_ids: bool,
) -> LogitMetrics:
    """Aggregate top-1 agreement and partition KL over every teacher-forced position.

    `candidate[j][i]` is the candidate's top-k at position i of `traces[j]`. An empty
    candidate list means the server stopped instead of predicting a token (it put
    end-of-sequence first), so it counts as a top-1 miss and is left out of the KL
    statistics, which need a distribution. Positions where the reference list is empty
    are skipped. `per_prompt` repeats the counts for each trace so two candidates can be
    compared prompt by prompt. Raises ValueError if the candidate shape does not match the
    traces.
    """
    _check_shapes(traces, candidate)
    divergences: list[float] = []
    per_prompt: list[PromptLogit] = []
    for trace, cand_steps in zip(traces, candidate, strict=True):
        prompt = _score_prompt(trace, cand_steps, by_id=exact_token_ids)
        per_prompt.append(prompt.summary)
        divergences.extend(prompt.divergences)

    positions = sum(prompt.positions for prompt in per_prompt)
    if positions == 0:
        return LogitMetrics(
            prompts=0,
            positions=0,
            top1_agreement=0.0,
            kld_mean=0.0,
            kld_p99=0.0,
            kld_max=0.0,
            exact_token_ids=exact_token_ids,
            per_prompt=tuple(per_prompt),
        )
    ordered = sorted(divergences) or [0.0]
    return LogitMetrics(
        prompts=sum(prompt.positions > 0 for prompt in per_prompt),
        positions=positions,
        top1_agreement=sum(prompt.top1_matches for prompt in per_prompt) / positions,
        kld_mean=math.fsum(ordered) / len(ordered),
        kld_p99=_nearest_rank(ordered, PERCENTILE),
        kld_max=ordered[-1],
        exact_token_ids=exact_token_ids,
        per_prompt=tuple(per_prompt),
    )


@dataclass(frozen=True, slots=True)
class _ScoredPrompt:
    summary: PromptLogit
    divergences: list[float]


def _score_prompt(
    trace: ReferenceTrace, cand_steps: Sequence[TopK], *, by_id: bool
) -> _ScoredPrompt:
    divergences: list[float] = []
    matches = 0
    positions = 0
    for ref_step, cand_top in zip(trace.steps, cand_steps, strict=True):
        if not ref_step.top:
            continue
        positions += 1
        matches += top1_match(ref_step, cand_top, by_id=by_id)
        if cand_top:
            divergences.append(partition_kld(ref_step.top, cand_top, by_id=by_id))
    kld_mean = math.fsum(divergences) / len(divergences) if divergences else None
    summary = PromptLogit(trace.prompt_id, positions, matches, kld_mean)
    return _ScoredPrompt(summary, divergences)


def _check_shapes(traces: Sequence[ReferenceTrace], candidate: Sequence[Sequence[TopK]]) -> None:
    if len(traces) != len(candidate):
        raise ValueError(
            f"got candidate scores for {len(candidate)} traces, expected {len(traces)}"
        )
    for trace, cand_steps in zip(traces, candidate, strict=True):
        if len(trace.steps) != len(cand_steps):
            raise ValueError(
                f"trace {trace.prompt_id!r} has {len(trace.steps)} steps but the candidate "
                f"scored {len(cand_steps)} positions"
            )


def _all_have_ids(*tops: TopK) -> bool:
    return all(token.token_id is not None for top in tops for token in top)


def _mass_by_token(top: TopK, *, use_ids: bool) -> dict[Hashable, float]:
    """Map each token key to its probability, summing duplicates of the same key."""
    masses: dict[Hashable, float] = {}
    for token in top:
        key = _token_key(token, use_ids=use_ids)
        masses[key] = masses.get(key, 0.0) + _probability(token.logprob)
    return masses


def _token_key(token: TokenProb, *, use_ids: bool) -> Hashable:
    return token.token_id if use_ids else token.token


def _probability(logprob: float) -> float:
    if math.isnan(logprob):
        raise ValueError("logprob is NaN")
    # Servers occasionally report logprobs a hair above 0 for near-certain tokens.
    return math.exp(min(logprob, 0.0))


def _rest_mass(listed: Collection[float]) -> float:
    """Return the candidate mass outside `listed`, padded by the rounding error of the sum.

    When a cap binds at a tiny true mass, even an error of a few ulps in 1 - sum is a large
    relative error. Padding errs toward more candidate mass, which can only lower the KL.
    """
    rounding = (len(listed) + 1) * sys.float_info.epsilon
    return min(max(1.0 - math.fsum(listed) + rounding, MASS_FLOOR), 1.0)


def _kl_term(p: float, q: float) -> float:
    if p <= 0.0:
        return 0.0
    return p * math.log(p / max(q, MASS_FLOOR))


def _nearest_rank(ordered: Sequence[float], fraction: float) -> float:
    rank = max(math.ceil(fraction * len(ordered)), 1)
    return ordered[rank - 1]
