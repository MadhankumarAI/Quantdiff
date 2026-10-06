"""Text normalization and similarity for comparing free-form chat answers."""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Sequence
from typing import Final

from quantdiff.types import AgreementMetrics

MAX_COMPARED_CHARS: Final = 4000
"""SequenceMatcher is quadratic in the worst case, so long answers are truncated first."""

_OPEN_TAG: Final = "<think>"
_CLOSE_TAG: Final = "</think>"
_LEADING_THINK: Final = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove a leading <think>...</think> block emitted by reasoning models.

    Some chat templates open the block inside the prompt, so the answer starts with the
    reasoning and only contains the closing tag; that prefix is removed too. An opening
    tag that is never closed means generation stopped mid-reasoning, so nothing is left.
    """
    match = _LEADING_THINK.match(text)
    if match:
        return text[match.end() :]
    if text.lstrip().startswith(_OPEN_TAG):
        return ""
    if _CLOSE_TAG in text and _OPEN_TAG not in text:
        return text.split(_CLOSE_TAG, 1)[1]
    return text


def normalize(text: str) -> str:
    """Strip reasoning, trim, and collapse every run of whitespace to one space."""
    return " ".join(strip_reasoning(text).split())


def similarity(a: str, b: str) -> float:
    """Return a similarity ratio in [0, 1] between two answers after normalization."""
    left = normalize(a)[:MAX_COMPARED_CHARS]
    right = normalize(b)[:MAX_COMPARED_CHARS]
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def agreement_metrics(answers: Sequence[tuple[str, str, str]]) -> AgreementMetrics:
    """Score (case id, reference answer, candidate answer) triples by exact match and mean
    similarity, keeping each case's similarity for paired comparisons."""
    if not answers:
        return AgreementMetrics(cases=0, exact_match_rate=0.0, mean_similarity=0.0)
    exact = sum(normalize(reference) == normalize(candidate) for _, reference, candidate in answers)
    per_case = tuple(
        (case_id, similarity(reference, candidate)) for case_id, reference, candidate in answers
    )
    return AgreementMetrics(
        cases=len(answers),
        exact_match_rate=exact / len(answers),
        mean_similarity=math.fsum(score for _, score in per_case) / len(answers),
        per_case=per_case,
    )
