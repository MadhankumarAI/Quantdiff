from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from quantdiff import cli
from quantdiff.api import RunFiles, build_plan, write_run
from quantdiff.backends._common import RequestFailure
from quantdiff.backends.ollama import explain_ollama_failure
from quantdiff.errors import BackendError, SpecError, SuiteError
from quantdiff.runner import RunPlan, execute
from quantdiff.suites import BUILTIN_SUITES
from quantdiff.types import Report
from quantdiff.verdict import CandidateVerdict, Status, Verdict
from tests.fakes import FakeBackend, text_result
from tests.fixtures.http_fake import FakeServer, closed_port, load, running

HEADLINE = "Run q4: 25% smaller than q8_0, KLD 0.02 (95% CI 0.01 to 0.03) on 8 prompts."
QUICK_RUN = ["--suite", "chat", "--max-cases", "1", "--score-tokens", "0", "--no-preflight"]


def _verdict(*statuses: Status, fits: bool | None = None) -> Verdict:
    candidates = tuple(
        CandidateVerdict(
            label=f"cand{index}",
            status=status,
            rank=index,
            kld_band=None,
            size_bytes=None,
            size_change=None,
            task_deltas=(),
            reasons=(),
            fits_budget=fits,
        )
        for index, status in enumerate(statuses, start=1)
    )
    return Verdict(headline=HEADLINE, details=(), candidates=candidates, findings=())


@pytest.fixture
def verdicts(monkeypatch: pytest.MonkeyPatch) -> list[Verdict]:
    """What `judge` returns: the last verdict in this list."""
    queue = [_verdict("ok")]
    monkeypatch.setattr(cli, "judge", lambda _report, **_options: queue[-1])
    return queue


@pytest.fixture
def fake_execute(monkeypatch: pytest.MonkeyPatch, verdicts: list[Verdict]) -> list[RunPlan]:
    """Route `quantdiff run` through FakeBackends instead of real servers."""
    plans: list[RunPlan] = []
    real_execute: Callable[..., Report] = execute

    def run(plan: RunPlan, **kwargs: object) -> Report:
        plans.append(plan)
        return real_execute(
            plan,
            backend_factory=lambda spec: FakeBackend(
                label=spec.label, chat_handler=lambda *_: text_result('{"a": 1}')
            ),
            cache=None,
            progress=None,
        )

    monkeypatch.setattr(cli, "execute", run)
    return plans


def test_suites_lists_every_builtin(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["suites"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for name in BUILTIN_SUITES:
        assert name in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("quantdiff ")


def test_run_writes_report_and_cards(
    fake_execute: list[RunPlan], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "run",
            "--ref",
            "ollama:model:q8_0",
            "--cand",
            "ollama:model:q4_k_m",
            "--suite",
            "json,chat",
            "--max-cases",
            "2",
            "--score-tokens",
            "0",
            "--no-preflight",
            "--out",
            str(tmp_path),
            "--no-png",
            "-q",
        ]
    )
    assert code == cli.EXIT_OK
    (run_dir,) = tmp_path.iterdir()
    assert {p.name for p in run_dir.iterdir()} == {"report.json", "card.html", "card.md"}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    schema = next(f.default for f in dataclasses.fields(Report) if f.name == "schema_version")
    assert report["schema_version"] == schema
    assert len(fake_execute[0].cases) == 4
    assert "model:q4_k_m" in capsys.readouterr().out


@pytest.mark.parametrize(("fmt", "marker"), [("html", "<html"), ("md", "|"), ("txt", "model")])
def test_card_renders_saved_run(
    fake_execute: list[RunPlan],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
    marker: str,
) -> None:
    out_dir = tmp_path / "runs"
    args = ["run", "--ref", "ollama:model:q8_0", "--cand", "ollama:model:q4"]
    args += ["--suite", "chat", "--max-cases", "1", "--score-tokens", "0", "--no-preflight"]
    cli.main([*args, "--out", str(out_dir), "--no-png", "-q"])
    capsys.readouterr()
    (run_dir,) = out_dir.iterdir()

    target = tmp_path / f"card.{fmt}"
    assert cli.main(["card", str(run_dir), "--format", fmt, "-o", str(target)]) == cli.EXIT_OK
    assert marker in target.read_text(encoding="utf-8")


def test_bad_spec_exits_with_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", "--ref", "nonsense", "--cand", "ollama:x", "--no-preflight"])
    assert code == cli.EXIT_ERROR
    assert capsys.readouterr().err.startswith("quantdiff: error:")


def test_build_plan_defaults_and_labels() -> None:
    plan = build_plan("ollama:m:q8_0", ["ollama:m:q4", "ollama:m:q4"], max_cases=1)
    assert plan.settings.suites == ("json", "tools", "chat")
    assert [spec.label for spec in plan.candidates] == ["m:q4", "m:q4 #2"]
    assert len(plan.scoring) == 1


def test_build_plan_adds_code_suite_only_when_execution_allowed() -> None:
    plan = build_plan("ollama:m:a", ["ollama:m:b"], max_cases=1, allow_code_exec=True)
    assert "code" in plan.settings.suites


def test_build_plan_with_only_user_prompts(tmp_path: Path) -> None:
    prompts = tmp_path / "mine.jsonl"
    prompts.write_text('{"prompt": "hello"}\n{"prompt": "bye"}\n', encoding="utf-8")
    plan = build_plan("ollama:m:a", ["ollama:m:b"], prompts=prompts, score_tokens=0)
    assert plan.settings.suites == ()
    assert len(plan.cases) == 2
    assert plan.scoring == ()


def test_build_plan_validates_inputs() -> None:
    with pytest.raises(SpecError, match="candidate"):
        build_plan("ollama:m:a", [])
    with pytest.raises(SpecError, match="top_k"):
        build_plan("ollama:m:a", ["ollama:m:b"], top_k=50)
    with pytest.raises(SuiteError, match="unknown suite"):
        build_plan("ollama:m:a", ["ollama:m:b"], suites=["nope"])


def test_card_png_uses_the_renderer(
    fake_execute: list[RunPlan], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "runs"
    args = ["run", "--ref", "ollama:model:q8_0", "--cand", "ollama:model:q4"]
    args += ["--suite", "chat", "--max-cases", "1", "--score-tokens", "0", "--no-preflight"]
    cli.main([*args, "--out", str(out_dir), "--no-png", "-q"])
    (run_dir,) = out_dir.iterdir()

    rendered: list[tuple[str, Path]] = []

    def fake_render(html: str, out_path: Path) -> Path:
        rendered.append((html, out_path))
        return out_path

    monkeypatch.setattr(cli, "render_png", fake_render)
    target = tmp_path / "share.png"
    assert cli.main(["card", str(run_dir), "--format", "png", "-o", str(target)]) == cli.EXIT_OK
    assert rendered[0][1] == target
    assert rendered[0][0].startswith("<!DOCTYPE html>") or "<html" in rendered[0][0]


def test_unreachable_server_prints_a_hint(capsys: pytest.CaptureFixture[str]) -> None:
    down = "llamacpp:http://127.0.0.1:9"
    args = ["--no-preflight", "--score-tokens", "0", "--suite", "chat", "-q"]
    code = cli.main(["run", "--ref", down, "--cand", down, *args])
    err = capsys.readouterr().err
    assert code == cli.EXIT_ERROR
    assert "cannot start the comparison" in err
    assert "hint:" in err


def test_verbose_flag_works_before_and_after_the_command() -> None:
    parser = cli._build_parser()
    assert parser.parse_args(["-v", "suites"]).verbose is True
    assert parser.parse_args(["suites", "-v"]).verbose is True
    assert parser.parse_args(["suites"]).verbose is False


# verdict, exit codes and help -------------------------------------------------------------


def _quick_run(tmp_path: Path, *extra: str) -> int:
    args = ["run", "--ref", "ollama:model:q8_0", "--cand", "ollama:model:q4", *QUICK_RUN]
    return cli.main([*args, "--out", str(tmp_path), "--no-png", "-q", *extra])


def test_run_ends_with_the_verdict_headline(
    fake_execute: list[RunPlan], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _quick_run(tmp_path) == cli.EXIT_OK
    assert capsys.readouterr().out.splitlines()[-1] == HEADLINE


def test_run_names_the_card_to_share(
    fake_execute: list[RunPlan],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def with_png(report: Report, out_dir: Path, *, png: bool, verdict: Verdict) -> RunFiles:
        files = write_run(report, out_dir, png=False, verdict=verdict)
        return dataclasses.replace(files, png=files.directory / "card.png")

    monkeypatch.setattr(cli, "write_run", with_png)
    _quick_run(tmp_path)
    (run_dir,) = tmp_path.iterdir()
    assert f"Scorecard image to share: {run_dir / 'card.png'}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("level", "statuses", "code"),
    [
        ("never", ("avoid", "failed"), cli.EXIT_OK),
        ("never", ("inconclusive",), cli.EXIT_OK),
        ("avoid", ("recommended", "inconclusive"), cli.EXIT_OK),
        ("avoid", ("recommended", "avoid"), cli.EXIT_REGRESSION),
        ("avoid", ("failed",), cli.EXIT_REGRESSION),
        ("avoid", ("inconclusive",), cli.EXIT_REGRESSION),
        ("avoid", ("usable", "inconclusive"), cli.EXIT_OK),
        ("inconclusive", ("recommended", "inconclusive"), cli.EXIT_REGRESSION),
        ("inconclusive", ("recommended", "ok"), cli.EXIT_OK),
        ("inconclusive", ("recommended", "usable"), cli.EXIT_REGRESSION),
        ("inconclusive", ("usable",), cli.EXIT_REGRESSION),
    ],
)
def test_fail_on_sets_the_exit_code(
    *,
    fake_execute: list[RunPlan],
    verdicts: list[Verdict],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    level: str,
    statuses: tuple[Status, ...],
    code: int,
) -> None:
    verdicts.append(_verdict(*statuses))
    assert _quick_run(tmp_path, "--fail-on", level) == code
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == HEADLINE
    if code == cli.EXIT_OK:
        assert "--fail-on" not in captured.err
    else:
        assert captured.err.splitlines()[-1].startswith(f"quantdiff: --fail-on {level}: ")


def test_fail_on_message_names_each_flagged_candidate(
    fake_execute: list[RunPlan],
    verdicts: list[Verdict],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verdicts.append(_verdict("recommended", "avoid", "inconclusive", "failed"))
    assert _quick_run(tmp_path, "--fail-on", "inconclusive") == 3
    assert capsys.readouterr().err.splitlines()[-1] == (
        "quantdiff: --fail-on inconclusive: cand2 is avoid, cand3 is inconclusive,"
        " cand4 is failed; exiting with code 3"
    )


def test_fail_on_avoid_fails_when_nothing_is_recommended(
    fake_execute: list[RunPlan],
    verdicts: list[Verdict],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Thin evidence leaves every candidate inconclusive; CI must not read that as a pass.
    verdicts.append(_verdict("inconclusive", "inconclusive"))
    assert _quick_run(tmp_path, "--fail-on", "avoid") == 3
    assert capsys.readouterr().err.splitlines()[-1] == (
        "quantdiff: --fail-on avoid: no candidate is recommended or usable; exiting with code 3"
    )


@pytest.mark.parametrize(
    ("fits", "code"), [(True, cli.EXIT_OK), (False, cli.EXIT_REGRESSION), (None, 3)]
)
def test_fail_on_avoid_passes_a_usable_candidate_only_when_it_fits_the_budget(
    *,
    fake_execute: list[RunPlan],
    verdicts: list[Verdict],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fits: bool | None,
    code: int,
) -> None:
    verdicts.append(_verdict("usable", fits=fits))
    assert _quick_run(tmp_path, "--max-size", "6GB", "--fail-on", "avoid") == code
    assert fake_execute[0].settings.max_size_bytes == 6_000_000_000
    if code == cli.EXIT_REGRESSION:
        assert capsys.readouterr().err.splitlines()[-1] == (
            "quantdiff: --fail-on avoid: no candidate is recommended or usable within "
            "--max-size; exiting with code 3"
        )


def test_bad_max_size_is_an_error_before_anything_runs(
    fake_execute: list[RunPlan], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _quick_run(tmp_path, "--max-size", "lots") == cli.EXIT_ERROR
    assert fake_execute == []
    assert "max size 'lots' is not a size" in capsys.readouterr().err


def test_fail_on_defaults_to_never() -> None:
    args = cli._build_parser().parse_args(["run", "--ref", "ollama:a", "--cand", "ollama:b"])
    assert args.fail_on == "never"


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"]])
def test_help_leads_with_the_happy_path_and_lists_exit_codes(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)
    out = capsys.readouterr().out
    steps = ["1. quantdiff discover", "2. quantdiff run", "3. share runs/<timestamp>/card.png"]
    positions = [out.index(step) for step in steps]
    assert positions == sorted(positions)
    assert positions[0] < out.index("more examples:") < out.index("exit codes:")
    for line in ["0    ok", "1    error", "3    regression found", "130  interrupted"]:
        assert line in out


# did you mean -----------------------------------------------------------------------------


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServer]:
    """A fake Ollama with the real four-model listing, set as OLLAMA_HOST."""
    with running() as server:
        server.reply("GET", "/api/tags", load("ollama_tags"))
        monkeypatch.setenv("OLLAMA_HOST", server.url)
        yield server


def _fail_with_missing(monkeypatch: pytest.MonkeyPatch, base_url: str, *tags: str) -> None:
    """Make `execute` fail the way the Ollama backend does when tags are not installed."""
    problems = []
    for tag in tags:
        failure = RequestFailure(f"{base_url}/api/show", 404, f"model '{tag}' not found")
        problems.append(f"{tag}: {explain_ollama_failure(failure, base_url=base_url, model=tag)}")

    def fail(plan: RunPlan, **kwargs: object) -> Report:
        raise BackendError("cannot start the comparison:\n  " + "\n  ".join(problems))

    monkeypatch.setattr(cli, "execute", fail)


def test_missing_ollama_tag_suggests_installed_ones(
    ollama: FakeServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fail_with_missing(monkeypatch, ollama.url, "qwen2.5")
    args = ["run", "--ref", "ollama:qwen2.5", "--cand", "ollama:qwen2.5:3b", *QUICK_RUN, "-q"]
    assert cli.main(args) == cli.EXIT_ERROR
    err = capsys.readouterr().err.splitlines()
    assert err == [
        "quantdiff: error: cannot start the comparison:",
        "  qwen2.5: model 'qwen2.5' is not available in Ollama;"
        " run `ollama pull qwen2.5` (see `ollama list`)",
        "hint: qwen2.5 is not installed; did you mean: qwen2.5:3b,"
        " qwen2.5:0.5b-instruct-q2_K, qwen2.5:0.5b-instruct-q8_0",
        "hint: `quantdiff discover` lists the models your local Ollama has",
    ]


def test_each_missing_tag_gets_its_own_suggestions(
    ollama: FakeServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fail_with_missing(monkeypatch, ollama.url, "qwen2.5:3b-instrct", "qwen2.5:0.5b-q4_K_M")
    args = ["run", "--ref", "ollama:qwen2.5:3b-instrct", "--cand", "q4=ollama:qwen2.5:0.5b-q4_K_M"]
    cli.main([*args, *QUICK_RUN, "-q"])
    hints = [line for line in capsys.readouterr().err.splitlines() if "did you mean" in line]
    assert len(hints) == 2
    assert hints[0].startswith(
        "hint: qwen2.5:3b-instrct is not installed; did you mean: qwen2.5:3b, "
    )
    assert hints[1].startswith("hint: qwen2.5:0.5b-q4_K_M is not installed; did you mean: ")
    assert "qwen2.5:0.5b-instruct-q4_K_M" in hints[1]
    assert [request.path for request in ollama.requests] == ["/api/tags"]


def test_no_suggestion_when_ollama_cannot_be_listed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"http://127.0.0.1:{closed_port()}"
    monkeypatch.setenv("OLLAMA_HOST", url)
    _fail_with_missing(monkeypatch, url, "qwen2.5")
    args = ["run", "--ref", "ollama:qwen2.5", "--cand", "ollama:x:1b", *QUICK_RUN]
    assert cli.main(args) == cli.EXIT_ERROR
    hints = [line for line in capsys.readouterr().err.splitlines() if line.startswith("hint:")]
    assert hints == ["hint: `quantdiff discover` lists the models your local Ollama has"]


def test_no_suggestion_for_other_errors(
    ollama: FakeServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(plan: RunPlan, **kwargs: object) -> Report:
        raise BackendError("cannot start the comparison:\n  qwen2.5: connection reset")

    monkeypatch.setattr(cli, "execute", fail)
    cli.main(["run", "--ref", "ollama:qwen2.5", "--cand", "ollama:qwen2.5:3b", *QUICK_RUN])
    assert "did you mean" not in capsys.readouterr().err
    assert ollama.requests == []


def test_run_judges_with_the_scoring_prompt_texts(
    fake_execute: list[RunPlan], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def judge(report: Report, *, scoring: object) -> Verdict:
        seen.append(scoring)
        return _verdict("recommended")

    monkeypatch.setattr(cli, "judge", judge)
    assert _quick_run(tmp_path) == cli.EXIT_OK
    assert seen == [fake_execute[0].scoring]
