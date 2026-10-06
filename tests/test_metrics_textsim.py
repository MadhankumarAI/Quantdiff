from __future__ import annotations

import pytest

from quantdiff.metrics.textsim import (
    MAX_COMPARED_CHARS,
    agreement_metrics,
    normalize,
    similarity,
    strip_reasoning,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain answer", "plain answer"),
        ("<think>\nplan\n</think>\n\nanswer", "\n\nanswer"),
        ("  <think>a</think>b<think>c</think>", "b<think>c</think>"),
        ("plan only</think>answer", "answer"),
        ("<think>never closed", ""),
        ("mentions </think> after <think>", "mentions </think> after <think>"),
    ],
)
def test_strip_reasoning(text: str, expected: str) -> None:
    assert strip_reasoning(text) == expected


def test_normalize_collapses_whitespace() -> None:
    assert normalize("<think>x</think>\n  Hello,\t\n world!  ") == "Hello, world!"


def test_similarity_bounds() -> None:
    assert similarity("", "") == 1.0
    assert similarity("  same   text", "same text\n") == 1.0
    assert similarity("abc", "") == 0.0
    assert similarity("abcd", "abce") == pytest.approx(0.75)
    assert 0.0 < similarity("the cat sat", "the dog sat") < 1.0


def test_similarity_only_compares_the_truncated_prefix() -> None:
    prefix = "x" * MAX_COMPARED_CHARS
    assert similarity(prefix + "a", prefix + "b") == 1.0


def test_agreement_metrics() -> None:
    metrics = agreement_metrics([("c1", "Paris", " Paris "), ("c2", "abcd", "abce")])
    assert metrics.cases == 2
    assert metrics.exact_match_rate == 0.5
    assert metrics.mean_similarity == pytest.approx((1.0 + 0.75) / 2)
    assert [case_id for case_id, _ in metrics.per_case] == ["c1", "c2"]
    assert [score for _, score in metrics.per_case] == pytest.approx([1.0, 0.75])


def test_agreement_metrics_empty() -> None:
    metrics = agreement_metrics([])
    assert (metrics.cases, metrics.exact_match_rate, metrics.mean_similarity) == (0, 0.0, 0.0)
    assert metrics.per_case == ()
