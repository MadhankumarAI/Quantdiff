from __future__ import annotations

import math
import random

import pytest

from quantdiff.metrics.logit import logit_metrics, partition_kld, top1_match
from quantdiff.types import PromptLogit, ReferenceTrace, TokenProb, TokenStep, TopK
from tests.fakes import make_topk


def _top(*pairs: tuple[str, float], with_ids: bool = True) -> TopK:
    return make_topk(*((token, math.log(prob)) for token, prob in pairs), with_ids=with_ids)


def _step(top: TopK) -> TokenStep:
    return TokenStep(chosen=top[0], top=top)


def _trace(prompt_id: str, *tops: TopK) -> ReferenceTrace:
    return ReferenceTrace(prompt_id=prompt_id, prompt_token_ids=None, steps=tuple(map(_step, tops)))


REF = _top(("a", 0.5), ("b", 0.3), ("c", 0.1))
CAND = _top(("a", 0.4), ("d", 0.3), ("b", 0.2))


def test_partition_kld_matches_hand_computation() -> None:
    # Shared {a, b}, A = {c}. The candidate leaves 1 - 0.6 = 0.4 outside {a, b}, but c is
    # not in its list at all, so Q(c) <= 1 - 0.9 = 0.1 although the proportional split
    # wants 0.2. That leaves 0.3 for B, where the reference has 0.1.
    expected = (
        0.5 * math.log(0.5 / 0.4)
        + 0.3 * math.log(0.3 / 0.2)
        + 0.1 * math.log(0.1 / 0.1)
        + 0.1 * math.log(0.1 / 0.3)
    )
    assert partition_kld(REF, CAND, by_id=True) == pytest.approx(expected, abs=1e-12)
    assert partition_kld(REF, CAND, by_id=True) == pytest.approx(0.1233501, abs=1e-6)


def test_partition_kld_is_zero_for_identical_distributions() -> None:
    assert partition_kld(REF, REF, by_id=True) == 0.0
    assert partition_kld(REF, REF, by_id=False) == 0.0


def test_partition_kld_is_zero_when_shared_masses_agree_but_tails_differ() -> None:
    # The candidate can give b its 0.1 (no more than c's 0.1, and 0.2 is left unlisted).
    ref = _top(("a", 0.7), ("b", 0.1))
    cand = _top(("a", 0.7), ("c", 0.1))
    assert partition_kld(ref, cand, by_id=True) == pytest.approx(0.0, abs=1e-12)


def test_missing_mass_is_capped_by_the_candidates_whole_list() -> None:
    # b is missing from the candidate's list, which already holds 0.9, so Q(b) <= 0.1
    # even though c's probability alone would allow 0.2.
    ref = _top(("a", 0.7), ("b", 0.2))
    cand = _top(("a", 0.7), ("c", 0.2))
    expected = 0.2 * math.log(0.2 / 0.1) + 0.1 * math.log(0.1 / 0.2)
    assert partition_kld(ref, cand, by_id=True) == pytest.approx(expected, abs=1e-12)


def test_disjoint_tops_are_still_penalized() -> None:
    # Nothing is shared. The candidate lists 0.9 on other tokens, so "a" and "b" together
    # get at most 0.1 even though the proportional split wants 0.9.
    ref = _top(("a", 0.6), ("b", 0.3))
    cand = _top(("c", 0.5), ("d", 0.4))
    expected = 0.9 * math.log(0.9 / 0.1) + 0.1 * math.log(0.1 / 0.9)
    assert partition_kld(ref, cand, by_id=True) == pytest.approx(expected, abs=1e-9)
    assert expected > 1.0
    assert not top1_match(_step(ref), cand, by_id=True)


def test_empty_candidate_top_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        partition_kld(REF, (), by_id=False)


def test_full_top_lists_handle_a_rest_mass_of_zero() -> None:
    ref = _top(("a", 0.5), ("b", 0.5))
    cand = _top(("a", 0.25), ("b", 0.75))
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert partition_kld(ref, cand, by_id=True) == pytest.approx(expected, abs=1e-9)


def test_listed_mass_slightly_above_one_is_clamped() -> None:
    ref = (TokenProb("a", math.log(0.6), 1), TokenProb("b", math.log(0.400001), 2))
    cand = (TokenProb("a", 1e-9, 1), TokenProb("b", -math.inf, 2))
    value = partition_kld(ref, cand, by_id=True)
    assert math.isfinite(value)
    assert value > 0.0


def test_nan_logprob_is_rejected() -> None:
    ref = (TokenProb("a", math.nan, 1),)
    with pytest.raises(ValueError, match="NaN"):
        partition_kld(ref, ref, by_id=True)


def test_missing_ids_fall_back_to_string_matching() -> None:
    cand_without_ids = _top(("a", 0.4), ("d", 0.3), ("b", 0.2), with_ids=False)
    by_string = partition_kld(REF, CAND, by_id=False)
    assert partition_kld(REF, cand_without_ids, by_id=True) == pytest.approx(by_string)


def test_by_id_distinguishes_tokens_with_equal_text() -> None:
    ref = (TokenProb(" a", math.log(0.9), 10), TokenProb("b", math.log(0.05), 11))
    cand = (TokenProb(" a", math.log(0.9), 99), TokenProb("b", math.log(0.05), 11))
    assert partition_kld(ref, cand, by_id=False) == pytest.approx(0.0, abs=1e-12)
    # By id only "b" is shared. Token 10 is missing from the candidate, so it can hold at
    # most 0.05 there while the reference gives it 0.9.
    expected = 0.9 * math.log(0.9 / 0.05) + 0.05 * math.log(0.05 / 0.9)
    assert partition_kld(ref, cand, by_id=True) == pytest.approx(expected, abs=1e-9)
    assert top1_match(_step(ref), cand, by_id=False)
    assert not top1_match(_step(ref), cand, by_id=True)


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _truncate(probs: list[float], k: int) -> TopK:
    ranked = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)[:k]
    return tuple(TokenProb(f"t{index}", math.log(probs[index]), index) for index in ranked)


def test_partition_kld_never_exceeds_full_vocabulary_kl() -> None:
    rng = random.Random(1234)  # noqa: S311 - seeded test data, not cryptography
    for _ in range(5000):
        vocab = rng.randint(2, 16)
        scale = rng.choice([0.5, 2.0, 6.0])
        p = _softmax([rng.gauss(0.0, scale) for _ in range(vocab)])
        q = _softmax([rng.gauss(0.0, scale) for _ in range(vocab)])
        full = math.fsum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
        ref_top = _truncate(p, rng.randint(1, vocab))
        cand_top = _truncate(q, rng.randint(1, vocab))
        for by_id in (True, False):
            bound = partition_kld(ref_top, cand_top, by_id=by_id)
            assert 0.0 <= bound <= full + 1e-9


def test_partition_kld_equals_full_kl_when_everything_is_listed() -> None:
    rng = random.Random(7)  # noqa: S311 - seeded test data, not cryptography
    p = _softmax([rng.gauss(0.0, 2.0) for _ in range(6)])
    q = _softmax([rng.gauss(0.0, 2.0) for _ in range(6)])
    full = math.fsum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    assert partition_kld(_truncate(p, 6), _truncate(q, 6), by_id=True) == pytest.approx(full)


def test_top1_match_uses_the_candidate_argmax() -> None:
    assert top1_match(_step(REF), _top(("a", 0.6), ("b", 0.3)), by_id=True)
    assert not top1_match(_step(REF), _top(("b", 0.6), ("a", 0.3)), by_id=True)
    assert not top1_match(_step(REF), (), by_id=True)


def test_logit_metrics_aggregates_positions() -> None:
    traces = [_trace("p1", REF, REF), _trace("p2", REF)]
    candidate = [[REF, CAND], [_top(("b", 0.9))]]
    metrics = logit_metrics(traces, candidate, exact_token_ids=True)

    kld_cand = partition_kld(REF, CAND, by_id=True)
    kld_b = partition_kld(REF, _top(("b", 0.9)), by_id=True)
    assert metrics.prompts == 2
    assert metrics.positions == 3
    assert metrics.top1_agreement == pytest.approx(2 / 3)
    assert metrics.kld_mean == pytest.approx((0.0 + kld_cand + kld_b) / 3)
    assert metrics.kld_max == pytest.approx(max(kld_cand, kld_b))
    assert metrics.kld_p99 == metrics.kld_max
    assert metrics.exact_token_ids is True
    first, second = metrics.per_prompt
    assert (first.prompt_id, first.positions, first.top1_matches) == ("p1", 2, 2)
    assert first.kld_mean == pytest.approx(kld_cand / 2)
    assert (second.prompt_id, second.positions, second.top1_matches) == ("p2", 1, 0)
    assert second.kld_mean == pytest.approx(kld_b)


def test_logit_metrics_p99_uses_nearest_rank() -> None:
    tops = [_top(("a", 0.5), ("b", 0.5 - step / 250)) for step in range(100)]
    traces = [_trace("p", *([REF] * 100))]
    metrics = logit_metrics(traces, [tops], exact_token_ids=True)
    ordered = sorted(partition_kld(REF, top, by_id=True) for top in tops)
    assert metrics.kld_p99 == ordered[98]
    assert metrics.kld_max == ordered[99]


def test_empty_candidate_position_is_a_top1_miss_without_kl() -> None:
    # An empty list means the candidate stopped (end-of-sequence first) where the
    # reference kept going: a disagreement, but with no distribution to measure.
    traces = [_trace("p1", REF, REF), _trace("p2", REF)]
    metrics = logit_metrics(traces, [[(), REF], [()]], exact_token_ids=False)
    assert metrics.positions == 3
    assert metrics.prompts == 2
    assert metrics.top1_agreement == pytest.approx(1 / 3)
    assert metrics.kld_mean == pytest.approx(0.0, abs=1e-12)


def test_logit_metrics_when_candidate_always_stops() -> None:
    metrics = logit_metrics([_trace("p", REF)], [[()]], exact_token_ids=True)
    assert metrics.positions == 1
    assert metrics.top1_agreement == 0.0
    assert metrics.kld_max == 0.0
    assert metrics.per_prompt == (PromptLogit("p", 1, 0, None),)


def test_per_prompt_keeps_traces_without_scored_positions() -> None:
    bare = ReferenceTrace("empty", None, (TokenStep(chosen=REF[0], top=()),))
    metrics = logit_metrics([bare], [[REF]], exact_token_ids=True)
    assert metrics.positions == 0
    assert metrics.per_prompt == (PromptLogit("empty", 0, 0, None),)


def test_reference_positions_without_a_distribution_are_skipped() -> None:
    bare = TokenStep(chosen=REF[0], top=())
    trace = ReferenceTrace(prompt_id="p", prompt_token_ids=None, steps=(bare, _step(REF)))
    metrics = logit_metrics([trace], [[REF, REF]], exact_token_ids=False)
    assert metrics.positions == 1
    assert metrics.top1_agreement == 1.0


@pytest.mark.parametrize(
    "candidate",
    [[[REF]], [[REF], [REF, REF]], [[REF, REF], []]],
)
def test_logit_metrics_rejects_mismatched_shapes(candidate: list[list[TopK]]) -> None:
    traces = [_trace("p1", REF, REF), _trace("p2", REF)]
    with pytest.raises(ValueError, match=r"traces|steps"):
        logit_metrics(traces, candidate, exact_token_ids=True)
