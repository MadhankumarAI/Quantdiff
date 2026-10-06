from __future__ import annotations

import math

import pytest

from quantdiff.stats import (
    Interval,
    bootstrap_mean,
    cases_to_bound_loss,
    items_to_bound_below,
    mcnemar_exact,
    paired_proportion_diff,
)


def pairs(both: int, lost: int, gained: int, neither: int) -> tuple[list[bool], list[bool]]:
    """Paired outcomes from the 2x2 table: (reference passes, candidate passes)."""
    ref = [True] * both + [True] * lost + [False] * gained + [False] * neither
    cand = [True] * both + [False] * lost + [True] * gained + [False] * neither
    return ref, cand


# McNemar ----------------------------------------------------------------------------------


def test_mcnemar_known_value() -> None:
    ref, cand = pairs(20, 10, 2, 8)
    assert mcnemar_exact(ref, cand) == pytest.approx(2 * 79 / 4096)
    assert mcnemar_exact(ref, cand) == pytest.approx(0.0386, abs=1e-4)


def test_mcnemar_is_symmetric_and_ignores_concordant_pairs() -> None:
    ref, cand = pairs(0, 10, 2, 0)
    assert mcnemar_exact(ref, cand) == mcnemar_exact(cand, ref)
    more_ref, more_cand = pairs(500, 10, 2, 500)
    assert mcnemar_exact(ref, cand) == mcnemar_exact(more_ref, more_cand)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ((0, 0, 0, 0), 1.0),
        ((5, 0, 0, 5), 1.0),
        ((0, 1, 1, 0), 1.0),
        ((0, 1, 0, 0), 1.0),
        ((0, 6, 0, 0), 2 / 64),
    ],
    ids=["empty", "all-equal", "balanced", "single-discordant", "six-to-none"],
)
def test_mcnemar_edge_cases(table: tuple[int, int, int, int], expected: float) -> None:
    assert mcnemar_exact(*pairs(*table)) == pytest.approx(expected)


def test_mcnemar_handles_thousands_of_discordant_pairs() -> None:
    p = mcnemar_exact(*pairs(0, 1100, 900, 0))
    assert 0.0 < p < 0.001


def test_paired_tests_reject_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        mcnemar_exact([True], [True, False])
    with pytest.raises(ValueError, match="differ in length"):
        cases_to_bound_loss([True], [True, False], 10.0)


# Paired proportion interval ---------------------------------------------------------------


def test_paired_proportion_diff_is_in_points_and_contains_estimate() -> None:
    interval = paired_proportion_diff(*pairs(24, 12, 0, 4))
    assert interval.estimate == pytest.approx(-30.0)
    assert interval.low < -30.0 < interval.high < 0.0
    assert interval.excludes_zero


def test_paired_proportion_diff_pinned_value() -> None:
    # 12 both pass, 1 lost, 5 gained, 4 neither, worked by hand through Newcombe's method 10.
    interval = paired_proportion_diff(*pairs(12, 1, 5, 4))
    assert interval.estimate == pytest.approx(400 / 22)
    assert interval.low == pytest.approx(-2.44, abs=0.01)
    assert interval.high == pytest.approx(36.94, abs=0.01)


def test_paired_proportion_diff_is_antisymmetric() -> None:
    ref, cand = pairs(10, 7, 2, 3)
    forward, backward = paired_proportion_diff(ref, cand), paired_proportion_diff(cand, ref)
    assert forward.estimate == pytest.approx(-backward.estimate)
    assert forward.low == pytest.approx(-backward.high)


@pytest.mark.parametrize("table", [(10, 0, 0, 0), (0, 0, 0, 10), (1, 0, 0, 0), (0, 0, 1, 0)])
def test_paired_proportion_diff_edges_stay_in_range(table: tuple[int, int, int, int]) -> None:
    interval = paired_proportion_diff(*pairs(*table))
    assert -100.0 <= interval.low <= interval.estimate <= interval.high <= 100.0
    assert interval.low < interval.high


def test_paired_proportion_diff_needs_data() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        paired_proportion_diff([], [])


# Bootstrap --------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    values = [0.01 * i for i in range(30)]
    first = bootstrap_mean(values)
    assert first == bootstrap_mean(values)
    assert first != bootstrap_mean(values, seed=1)
    assert first.estimate == pytest.approx(sum(values) / 30)
    assert first.low <= first.estimate <= first.high


def test_bootstrap_of_identical_values_is_a_point() -> None:
    assert bootstrap_mean([0.25, 0.25, 0.25]) == Interval(0.25, 0.25, 0.25)


def test_bootstrap_single_value_and_errors() -> None:
    assert bootstrap_mean([0.25]) == Interval(0.25, 0.25, 0.25)
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_mean([])
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean([1.0], resamples=0)


# Evidence needed -------------------------------------------------------------------------


def test_items_to_bound_below_scales_the_half_width_by_sqrt_n() -> None:
    # Upper half-width 0.03 over 10 items, 0.02 of room under the bound: the half-width
    # must shrink by 2/3, which takes (3/2)^2 = 2.25 times the items.
    interval = Interval(0.03, 0.01, 0.06)
    needed = items_to_bound_below(interval, 10, 0.05)
    assert needed == 23
    assert needed is not None
    assert 0.03 + 0.03 * math.sqrt(10 / needed) < 0.05


def test_items_to_bound_below_edges() -> None:
    assert items_to_bound_below(Interval(0.02, 0.01, 0.03), 12, 0.05) == 12
    assert items_to_bound_below(Interval(0.05, 0.04, 0.06), 12, 0.05) is None
    assert items_to_bound_below(Interval(0.07, 0.06, 0.08), 12, 0.05) is None
    with pytest.raises(ValueError, match="at least 1"):
        items_to_bound_below(Interval(0.02, 0.01, 0.03), 0, 0.05)


def test_cases_to_bound_loss_is_the_first_count_that_clears_the_margin() -> None:
    ref, cand = pairs(19, 2, 1, 2)
    assert paired_proportion_diff(ref, cand).low < -10
    needed = cases_to_bound_loss(ref, cand, 10.0)
    assert needed is not None
    assert needed > 24

    def scaled(cases: int) -> Interval:
        factor = cases / 24
        return paired_proportion_diff(*pairs(*(round(count * factor) for count in (19, 2, 1, 2))))

    # Rebuilding whole tables only approximates the fixed shares, so check with a margin.
    assert scaled(needed * 2).low >= -10
    assert scaled(needed // 2).low < -10


def test_cases_to_bound_loss_edges() -> None:
    ref, cand = pairs(30, 0, 0, 2)
    assert paired_proportion_diff(ref, cand).low >= -10
    assert cases_to_bound_loss(ref, cand, 10.0) == 32
    lossy_ref, lossy_cand = pairs(20, 5, 0, 5)
    assert cases_to_bound_loss(lossy_ref, lossy_cand, 10.0) is None
    with pytest.raises(ValueError, match="at least one pair"):
        cases_to_bound_loss([], [], 10.0)
    with pytest.raises(ValueError, match="margin"):
        cases_to_bound_loss(ref, cand, 0.0)
