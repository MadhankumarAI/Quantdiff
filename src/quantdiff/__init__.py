"""quantdiff: find out which download of a model gives the best answers on your prompts.

Typical use::

    import quantdiff

    report = quantdiff.compare(
        "ollama:qwen2.5:7b-instruct-q8_0",
        ["ollama:qwen2.5:7b-instruct-q4_K_M", "llamacpp:http://127.0.0.1:8080"],
    )
    print(quantdiff.verdict(report))
"""

from __future__ import annotations

from quantdiff._version import __version__
from quantdiff.api import build_plan, compare, write_run
from quantdiff.card import render_html, render_markdown, render_terminal
from quantdiff.errors import (
    BackendError,
    CapabilityError,
    QuantdiffError,
    ReportError,
    SpecError,
    SuiteError,
)
from quantdiff.report import load_report, rank_candidates, save_report, verdict
from quantdiff.spec import parse_spec
from quantdiff.types import CandidateResult, CandidateSpec, Report

__all__ = [
    "BackendError",
    "CandidateResult",
    "CandidateSpec",
    "CapabilityError",
    "QuantdiffError",
    "Report",
    "ReportError",
    "SpecError",
    "SuiteError",
    "__version__",
    "build_plan",
    "compare",
    "load_report",
    "parse_spec",
    "rank_candidates",
    "render_html",
    "render_markdown",
    "render_terminal",
    "save_report",
    "verdict",
    "write_run",
]
