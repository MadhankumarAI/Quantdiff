"""Repository-wide text rules that linters do not cover."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".venv",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    "runs",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".jsonl", ".txt", ".cfg", ".html"}
BANNED = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
    chr(0x2018): "curly quote",
    chr(0x2019): "curly quote",
    chr(0x201C): "curly quote",
    chr(0x201D): "curly quote",
}


def _text_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE"}):
            files.append(path)
    return files


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_typographic_punctuation(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    hits = [
        f"line {number}: {name}"
        for number, line in enumerate(text.splitlines(), start=1)
        for char, name in BANNED.items()
        if char in line
    ]
    assert not hits, f"{path.name}: " + "; ".join(hits)
