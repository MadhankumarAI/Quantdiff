from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantdiff.api import build_plan, parse_size
from quantdiff.errors import SpecError
from quantdiff.suites import load_scoring_prompts
from quantdiff.types import CandidateSpec

REF = "ollama:qwen2.5:0.5b-instruct-q8_0"
CANDIDATES = ("ollama:qwen2.5:0.5b-instruct-q4_K_M", "ollama:qwen2.5:0.5b-instruct-q2_K")


def _prompts(tmp_path: Path, *lines: dict[str, object]) -> Path:
    path = tmp_path / "private" / "my-prompts.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


def _spec(label: str) -> CandidateSpec:
    return CandidateSpec("ollama", "http://127.0.0.1:11434", label, label)


def test_prompts_file_drives_the_scoring_prompts(tmp_path: Path) -> None:
    path = _prompts(
        tmp_path,
        {"id": "a", "prompt": "Summarize the release notes", "system": "Be brief"},
        {"prompt": "Translate to French: good morning"},
    )
    plan = build_plan(REF, CANDIDATES, prompts=path, preflight=False)
    assert [(p.id, p.text) for p in plan.scoring] == [
        ("a", "Summarize the release notes"),
        ("prompt-002", "Translate to French: good morning"),
    ]


def test_explicit_scoring_prompts_win_over_the_prompts_file(tmp_path: Path) -> None:
    path = _prompts(tmp_path, {"prompt": "Mine"})
    scoring = tmp_path / "scoring.jsonl"
    scoring.write_text('{"prompt": "Score this"}\n', encoding="utf-8")
    plan = build_plan(REF, CANDIDATES, prompts=path, scoring_prompts=scoring, preflight=False)
    assert [p.text for p in plan.scoring] == ["Score this"]


def test_builtin_scoring_prompts_without_user_prompts() -> None:
    plan = build_plan(REF, CANDIDATES, max_cases=2, preflight=False)
    assert plan.scoring == load_scoring_prompts()[:2]


def test_no_scoring_prompts_when_scoring_is_off(tmp_path: Path) -> None:
    path = _prompts(tmp_path, {"prompt": "Mine"})
    assert build_plan(REF, CANDIDATES, prompts=path, score_tokens=0).scoring == ()


def test_settings_keep_only_the_prompts_file_name(tmp_path: Path) -> None:
    path = _prompts(tmp_path, {"prompt": "x" * 41})
    settings = build_plan(REF, CANDIDATES, prompts=path).settings
    assert settings.prompts_file == "my-prompts.jsonl"
    assert settings.longest_prompt_tokens == 11


def test_longest_prompt_counts_every_message_of_a_case(tmp_path: Path) -> None:
    path = _prompts(tmp_path, {"prompt": "x" * 8, "system": "y" * 8}, {"prompt": "z" * 12})
    assert build_plan(REF, CANDIDATES, prompts=path).settings.longest_prompt_tokens == 4


def test_title_names_the_shared_model_and_each_quant() -> None:
    plan = build_plan(REF, CANDIDATES)
    assert plan.title == "qwen2.5:0.5b-instruct: q8_0 vs q4_K_M vs q2_K"


@pytest.mark.parametrize(
    ("labels", "title"),
    [
        (("llama-a", "qwen-b", "phi-c"), "llama-a vs 2 candidates"),
        (("llama3:8b", "qwen2.5:7b"), "llama3:8b vs qwen2.5:7b"),
        (("qwen2.5:", "qwen2.5:q4", "qwen2.5:q2"), "qwen2.5: vs 2 candidates"),
        (("model-q8", "model-q4"), "model-q8 vs model-q4"),
        (("mymodel-q8", "mymodel-q4"), "mymodel: q8 vs q4"),
    ],
)
def test_title_falls_back_without_a_long_shared_prefix(labels: tuple[str, ...], title: str) -> None:
    ref, *candidates = (_spec(label) for label in labels)
    assert build_plan(ref, candidates).title == title


def test_explicit_title_is_kept() -> None:
    assert build_plan(REF, CANDIDATES, title="Mine").title == "Mine"


def test_code_suite_needs_code_execution() -> None:
    with pytest.raises(SpecError, match="add --allow-code-exec"):
        build_plan(REF, CANDIDATES, suites=["json", "code"])
    plan = build_plan(REF, CANDIDATES, suites=["code"], allow_code_exec=True)
    assert plan.settings.suites == ("code",)


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("6GB", 6_000_000_000),
        ("6.5G", 6_500_000_000),
        ("800MB", 800_000_000),
        ("800M", 800_000_000),
        ("6 gb", 6_000_000_000),
        (" 1.25TB ", 1_250_000_000_000),
        ("6000000000", 6_000_000_000),
        ("512KB", 512_000),
    ],
)
def test_max_size_reads_decimal_sizes(text: str, size: int) -> None:
    assert parse_size(text) == size
    assert build_plan(REF, CANDIDATES, max_size=text).settings.max_size_bytes == size


@pytest.mark.parametrize("text", ["", "lots", "6GiB", "-6GB", "6.5", "0GB", "GB", "6 G B", "1e9"])
def test_max_size_rejects_anything_else(text: str) -> None:
    with pytest.raises(SpecError, match="max size"):
        build_plan(REF, CANDIDATES, max_size=text)


def test_max_size_in_bytes_and_none() -> None:
    assert build_plan(REF, CANDIDATES, max_size=7_000).settings.max_size_bytes == 7_000
    assert build_plan(REF, CANDIDATES).settings.max_size_bytes is None
    with pytest.raises(SpecError, match="max_size"):
        build_plan(REF, CANDIDATES, max_size=0)
