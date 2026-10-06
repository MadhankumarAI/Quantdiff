from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from quantdiff.errors import ReportError
from quantdiff.report import (
    SCHEMA_VERSION,
    load_report,
    notes_for,
    rank_candidates,
    report_from_dict,
    report_to_dict,
    save_report,
    text_forced_note,
    verdict,
    verdict_sentences,
)
from quantdiff.types import (
    CandidateResult,
    CandidateSpec,
    JSONValue,
    LogitMetrics,
    PreflightFinding,
    Report,
    RunSettings,
    TaskKind,
    TaskMetrics,
)
from quantdiff.verdict import kld_thresholds

FIXTURE = Path(__file__).parent / "data" / "sample_report.json"


def load_fixture() -> Report:
    return load_report(FIXTURE)


def fixture_dict() -> dict[str, JSONValue]:
    data: dict[str, JSONValue] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def candidate(
    label: str,
    *,
    json_cases: tuple[int, int] | None = None,
    tools_cases: tuple[int, int] | None = None,
    code_cases: tuple[int, int] | None = None,
    top1: float | None = None,
    kld: float = 0.01,
    prompts: int = 24,
    exact_token_ids: bool = True,
    preflight: tuple[PreflightFinding, ...] = (),
    errors: tuple[str, ...] = (),
) -> CandidateResult:
    """Build a candidate. Case tuples are (passed, total)."""
    suites: tuple[tuple[TaskKind, tuple[int, int] | None], ...] = (
        ("json", json_cases),
        ("tools", tools_cases),
        ("code", code_cases),
    )
    tasks = tuple(
        TaskMetrics(kind=kind, total=cases[1], passed=cases[0], skipped=0)
        for kind, cases in suites
        if cases is not None
    )
    logit = None
    if top1 is not None:
        logit = LogitMetrics(
            prompts=prompts,
            positions=prompts * 64,
            top1_agreement=top1,
            kld_mean=kld,
            kld_p99=kld * 10,
            kld_max=kld * 50,
            exact_token_ids=exact_token_ids,
        )
    return CandidateResult(
        spec=CandidateSpec(kind="ollama", base_url="http://fake", model=label, label=label),
        info=None,
        logit=logit,
        tasks=tasks,
        agreement=None,
        perf=None,
        preflight=preflight,
        errors=errors,
    )


def make_report(*candidates: CandidateResult, reference: CandidateResult | None = None) -> Report:
    return Report(
        quantdiff_version="0.1.0",
        created_at="2026-10-01T12:00:00Z",
        title="synthetic",
        settings=RunSettings(
            suites=("json", "tools"), top_k=20, score_tokens=64, allow_code_exec=False, seed=0
        ),
        reference=reference or candidate("example-ref"),
        candidates=candidates,
    )


# Serialization ----------------------------------------------------------------------------


def test_fixture_round_trips_through_dict() -> None:
    report = load_fixture()
    assert report_from_dict(report_to_dict(report)) == report
    assert report_to_dict(report) == fixture_dict()


def test_derived_rate_is_written_but_ignored_on_load() -> None:
    data = fixture_dict()
    tasks = data["candidates"][0]["tasks"]
    assert tasks[0]["rate"] == pytest.approx(31 / 40)
    tasks[0]["rate"] = 0.123
    loaded = report_from_dict(data)
    assert loaded.candidates[0].tasks[0].rate == pytest.approx(31 / 40)


def test_save_then_load_is_atomic_and_stable(tmp_path: Path) -> None:
    report = load_fixture()
    target = tmp_path / "report.json"
    target.write_text("stale", encoding="utf-8")
    save_report(report, target)
    assert load_report(target) == report
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]
    text = target.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert text.startswith('{\n  "candidates": [')


def test_save_keeps_non_ascii_text(tmp_path: Path) -> None:
    report = dataclasses.replace(load_fixture(), title="Qwen über alles")
    target = tmp_path / "report.json"
    save_report(report, target)
    assert "Qwen über alles" in target.read_text(encoding="utf-8")


def test_save_rejects_non_finite_numbers(tmp_path: Path) -> None:
    report = make_report(candidate("example-a", top1=0.9, kld=float("nan")))
    with pytest.raises(ReportError, match="JSON cannot represent"):
        save_report(report, tmp_path / "report.json")
    assert list(tmp_path.iterdir()) == []


def test_save_into_missing_directory_raises_report_error(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="cannot write report"):
        save_report(load_fixture(), tmp_path / "missing" / "report.json")


def test_load_rejects_oversized_file(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    save_report(load_fixture(), target)
    with pytest.raises(ReportError, match="larger than 100 bytes"):
        load_report(target, max_bytes=100)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{not json", "not valid JSON"),
        (b"\xff\xfe\x00", "not valid UTF-8"),
        (b'{"schema_version": NaN}', "NaN"),
        (b"[" * 100_000 + b"]" * 100_000, "nested too deeply"),
        (b"[]", "report: expected an object, got array"),
    ],
    ids=["syntax", "encoding", "nan", "nesting", "root-type"],
)
def test_load_rejects_malformed_files(tmp_path: Path, content: bytes, message: str) -> None:
    target = tmp_path / "report.json"
    target.write_bytes(content)
    with pytest.raises(ReportError, match=message):
        load_report(target)


def test_load_missing_file_raises_report_error(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="cannot read report"):
        load_report(tmp_path / "absent.json")


@pytest.mark.parametrize("version", [3, 0, "1", True, None])
def test_unsupported_schema_version_is_rejected(version: object) -> None:
    data = fixture_dict()
    data["schema_version"] = version
    with pytest.raises(ReportError, match="schema_version: unsupported value"):
        report_from_dict(data)
    assert SCHEMA_VERSION == 2


def as_version_1(data: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """A copy of `data` with every field added in schema version 2 removed."""
    data = copy.deepcopy(data)
    data["schema_version"] = 1
    del data["settings"]["longest_prompt_tokens"]
    del data["settings"]["max_size_bytes"]
    for result in (data["reference"], *data["candidates"]):
        del result["outcomes"]
        for part, added in (
            ("info", ("size_bytes", "weights_id")),
            ("logit", ("per_prompt",)),
            ("agreement", ("per_case",)),
            ("perf", ("source",)),
        ):
            if result[part] is not None:
                for name in added:
                    del result[part][name]
    return data


def test_version_1_report_loads_with_empty_per_item_fields() -> None:
    report = report_from_dict(as_version_1(fixture_dict()))
    assert report.schema_version == SCHEMA_VERSION
    assert report.settings.longest_prompt_tokens is None
    for result in (report.reference, *report.candidates):
        assert result.outcomes == ()
        assert result.logit is None or result.logit.per_prompt == ()
        assert result.agreement is None or result.agreement.per_case == ()
        assert result.perf is None or result.perf.source == "wall_clock"
        assert result.info is None or result.info.size_bytes is None
    assert report_to_dict(report)["schema_version"] == SCHEMA_VERSION
    # Without per-prompt results nothing can be proven close, so nothing is recommended.
    assert verdict(report).startswith("Keep bf16 for now: no candidate is shown to be close.")


def test_version_1_report_rejects_version_2_fields() -> None:
    data = as_version_1(fixture_dict())
    data["candidates"][0]["outcomes"] = []
    with pytest.raises(ReportError, match=r"candidates\[0\]: unknown field 'outcomes'"):
        report_from_dict(data)


def _set(path: tuple[str | int, ...], value: object) -> Callable[[JSONValue], None]:
    def mutate(data: JSONValue) -> None:
        for key in path[:-1]:
            data = data[key]
        data[path[-1]] = value

    return mutate


def _delete(path: tuple[str | int, ...]) -> Callable[[JSONValue], None]:
    def mutate(data: JSONValue) -> None:
        for key in path[:-1]:
            data = data[key]
        del data[path[-1]]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set(("candidates", 1, "logit", "kld_mean"), "0.1"),
            r"candidates\[1\]\.logit\.kld_mean: expected a number, got string",
        ),
        (
            _delete(("reference", "spec", "label")),
            r"reference\.spec\.label: missing required field",
        ),
        (_set(("settings", "extra"), 1), r"settings: unknown field 'extra'"),
        (
            _set(("candidates", 0, "tasks", 0, "passed"), True),
            r"candidates\[0\]\.tasks\[0\]\.passed: expected an integer, got boolean",
        ),
        (
            _set(("candidates", 0, "spec", "kind"), "vllm"),
            r"candidates\[0\]\.spec\.kind: expected one of ollama, llamacpp, openai",
        ),
        (
            _set(("candidates", 2, "logit", "top1_agreement"), 1.5),
            r"candidates\[2\]\.logit\.top1_agreement: must be between 0 and 1",
        ),
        (
            _set(("candidates", 0, "tasks", 0, "passed"), 41),
            r"candidates\[0\]\.tasks\[0\]: passed \+ skipped exceeds total",
        ),
        (
            _set(("candidates", 1, "info", "details", 0), ["quantization"]),
            r"candidates\[1\]\.info\.details\[0\]: expected a \[key, value\] pair",
        ),
        (
            _set(("candidates", 1, "preflight", 0, "severity"), "error"),
            r"candidates\[1\]\.preflight\[0\]\.severity: expected one of ok, warn, fail, skip",
        ),
        (_set(("candidates",), {}), r"candidates: expected an array, got object"),
        (_set(("settings", "top_k"), -1), r"settings\.top_k: must be at least 0"),
        (_set(("notes", 0), 7), r"notes\[0\]: expected a string, got integer"),
        (_set(("reference", "perf"), "fast"), r"reference\.perf: expected an object"),
        (
            _set(("candidates", 1, "logit", "per_prompt", 0, "top1_matches"), 65),
            r"candidates\[1\]\.logit\.per_prompt\[0\]: top1_matches exceeds positions",
        ),
        (
            _set(("candidates", 1, "agreement", "per_case", 0, "similarity"), 1.5),
            r"candidates\[1\]\.agreement\.per_case\[0\]\.similarity: must be between 0 and 1",
        ),
        (
            _set(("candidates", 1, "perf", "source"), "guess"),
            r"candidates\[1\]\.perf\.source: expected one of server, wall_clock, cached",
        ),
        (
            _set(("reference", "outcomes", 0, "passed"), "yes"),
            r"reference\.outcomes\[0\]\.passed: expected true or false",
        ),
        (
            _delete(("candidates", 2, "info", "weights_id")),
            r"candidates\[2\]\.info\.weights_id: missing required field",
        ),
        (
            _set(("settings", "longest_prompt_tokens"), -5),
            r"settings\.longest_prompt_tokens: must be at least 0",
        ),
    ],
)
def test_malformed_fields_name_their_path(
    mutate: Callable[[JSONValue], None], message: str
) -> None:
    data = copy.deepcopy(fixture_dict())
    mutate(data)
    with pytest.raises(ReportError, match=message):
        report_from_dict(data)


# Ranking ----------------------------------------------------------------------------------


def test_fixture_ranking_follows_the_verdict() -> None:
    ranked = rank_candidates(load_fixture())
    assert [entry.result.spec.label for entry in ranked] == [
        "example-q4_k_m",
        "example-q8_0",
        "example-q2_k",
        "example-iq1_s",
    ]
    assert [entry.rank for entry in ranked] == [1, 2, 3, 4]
    assert ranked[0].task_rate == pytest.approx((37 / 40 + 34 / 40 + 16 / 20) / 3)
    assert ranked[0].top1 == pytest.approx(0.9407552083333334)
    assert not ranked[3].has_metrics


def test_ranking_does_not_start_with_task_pass_rate() -> None:
    report = make_report(
        candidate("more-passes", json_cases=(9, 10), top1=0.9, kld=0.03),
        candidate("closer", json_cases=(8, 10), top1=0.9, kld=0.01),
        candidate("broken", errors=("connection refused",)),
    )
    labels = [entry.result.spec.label for entry in rank_candidates(report)]
    assert labels == ["closer", "more-passes", "broken"]


# Verdict ----------------------------------------------------------------------------------


def test_verdict_is_headline_then_details_on_one_line() -> None:
    report = load_fixture()
    sentences = verdict_sentences(report)
    assert sentences[0] == (
        "Run q4_k_m: 69% smaller than bf16, close on logits (KLD 0.029, CI up to 0.034) on 24 "
        "prompts."
    )
    assert sentences[-1] == "iq1_s produced no metrics; see the errors."
    line = verdict(report)
    assert line == " ".join(sentences)
    assert "\n" not in line


def test_verdict_edge_cases() -> None:
    assert verdict(make_report()) == "No candidates were evaluated."
    failed = make_report(candidate("example-a", errors=("boom",)))
    assert verdict(failed).startswith("No candidate produced metrics")
    single = make_report(candidate("example-a", top1=0.9123, kld=0.012345))
    assert verdict(single) == (
        "Keep ref for now: no candidate is shown to be close. a looks closest: KLD 0.012, with no "
        "per-prompt results to bound it."
    )


# Notes ------------------------------------------------------------------------------------


def test_notes_explain_tiers_bands_and_text_forcing() -> None:
    notes = notes_for(load_fixture())
    assert notes[0].startswith("T1 metrics")
    assert notes[1].startswith("T2 metrics")
    bars = kld_thresholds(20)
    assert any(
        n.startswith(f"KLD bands: under {bars.near_lossless:g} near-lossless, under {bars.close:g}")
        and "top 20" in n
        for n in notes
    )
    approximate = [note for note in notes if "text-forced" in note]
    assert approximate == [
        "Logit metrics for example-q2_k are text-forced; on English text this matched exact "
        "token-id forcing (docs/calibration.md). Non-Latin text may read higher."
    ]
    assert not any("approximate" in note for note in notes)
    assert any(note.startswith("The ref row is the reference") for note in notes)
    assert notes[-1] == "Synthetic data for tests; not a real measurement."


def test_text_forced_note_for_ollama_candidates_cites_the_calibration() -> None:
    notes = notes_for(
        make_report(candidate("example-a", top1=0.9, kld=0.02, exact_token_ids=False))
    )
    assert (
        "Ollama logit metrics are text-forced; on English text they matched llama-server's "
        "exact token-id forcing (docs/calibration.md). Non-Latin text may read higher."
    ) in notes
    assert text_forced_note(["a", "b"], ollama=False).startswith("Logit metrics for a, b are")


def test_notes_skip_logit_caveats_without_logit_metrics() -> None:
    skipped_code = TaskMetrics(kind="code", total=5, passed=0, skipped=5)
    result = dataclasses.replace(candidate("example-a", json_cases=(1, 2)), tasks=(skipped_code,))
    notes = notes_for(make_report(result))
    assert not any("KLD" in note or "Logit metrics" in note for note in notes[2:])
    assert "Code cases were skipped because code execution was not enabled." in notes
    assert not any("ref row" in note for note in notes)


def test_size_budget_round_trips_and_older_v2_reports_still_load(tmp_path: Path) -> None:
    budgeted = dataclasses.replace(
        load_fixture(),
        settings=dataclasses.replace(load_fixture().settings, max_size_bytes=6_000_000_000),
    )
    path = tmp_path / "report.json"
    save_report(budgeted, path)
    assert load_report(path).settings.max_size_bytes == 6_000_000_000

    data = report_to_dict(load_fixture())
    settings = data["settings"]
    assert isinstance(settings, dict)
    del settings["max_size_bytes"]
    assert report_from_dict(data).settings.max_size_bytes is None
