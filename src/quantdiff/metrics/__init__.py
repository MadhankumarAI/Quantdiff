"""Scoring functions: logit divergence, task checks, answer agreement and speed."""

from __future__ import annotations

from quantdiff.metrics.codeexec import extract_code, run_code_case
from quantdiff.metrics.jsonschema import (
    SUPPORTED_KEYWORDS,
    extract_json,
    unsupported_keywords,
    validate,
)
from quantdiff.metrics.logit import logit_metrics, partition_kld, top1_match
from quantdiff.metrics.tasks import evaluate_case, perf_metrics, summarize_tasks
from quantdiff.metrics.textsim import agreement_metrics, normalize, similarity, strip_reasoning
from quantdiff.metrics.toolcheck import arguments_match, check_tool_case

__all__ = [
    "SUPPORTED_KEYWORDS",
    "agreement_metrics",
    "arguments_match",
    "check_tool_case",
    "evaluate_case",
    "extract_code",
    "extract_json",
    "logit_metrics",
    "normalize",
    "partition_kld",
    "perf_metrics",
    "run_code_case",
    "similarity",
    "strip_reasoning",
    "summarize_tasks",
    "top1_match",
    "unsupported_keywords",
    "validate",
]
