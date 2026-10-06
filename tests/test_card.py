from __future__ import annotations

import dataclasses
import re
from typing import Literal

import pytest

from quantdiff import card
from quantdiff.card import (
    SIGNIFICANT_FOOTNOTE,
    TERMINAL_WIDTH,
    format_delta,
    format_interval,
    format_size,
    format_size_change,
    prompts_file_name,
    render_html,
    render_markdown,
    render_terminal,
    summary_line,
)
from quantdiff.types import (
    AgreementMetrics,
    BackendKind,
    CandidateResult,
    CandidateSpec,
    CaseOutcome,
    LogitMetrics,
    PerfMetrics,
    PreflightFinding,
    Report,
    RunSettings,
    ServerInfo,
    TaskKind,
    TaskMetrics,
)
from quantdiff.verdict import (
    CLOSE_KLD,
    CandidateVerdict,
    Interval,
    ServerFinding,
    Status,
    TaskDelta,
    Verdict,
    kld_band,
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEM = "qwen2.5:0.5b-instruct-"
MB = 1_000_000
XSS = "<script>alert(1)</script>"
HEADLINE = "Run q4_K_M: 25% smaller than q8_0 with no measurable loss on 30 cases."
DETAIL = "Avoid q2_K: it fails 3 in 10 tool calls the reference gets right."
Source = Literal["server", "wall_clock", "cached"]


# Builders ---------------------------------------------------------------------------------


def model(
    label: str,
    *,
    kind: BackendKind = "ollama",
    size_mb: int | None = None,
    tasks: dict[TaskKind, tuple[int, int]] | None = None,
    skipped_code: int = 0,
    top1: float | None = 0.8,
    positions: int = 1600,
    exact: bool = False,
    agree: float | None = 0.85,
    tok_s: float | None = 40.0,
    source: Source = "server",
    preflight: tuple[PreflightFinding, ...] = (),
    errors: tuple[str, ...] = (),
) -> CandidateResult:
    task_metrics = [
        TaskMetrics(kind=k, total=total, passed=passed, skipped=0)
        for k, (passed, total) in (tasks or {}).items()
    ]
    if skipped_code:
        task_metrics.append(
            TaskMetrics(kind="code", total=skipped_code, passed=0, skipped=skipped_code)
        )
    info = ServerInfo(
        backend=kind,
        model=label,
        context_length=4096,
        chat_template=None,
        template_dialect="go",
        supports_logprobs=True,
        exact_token_ids=exact,
        size_bytes=None if size_mb is None else size_mb * MB,
    )
    logit = None
    if top1 is not None:
        logit = LogitMetrics(
            prompts=10,
            positions=positions,
            top1_agreement=top1,
            kld_mean=0.03,
            kld_p99=0.25,
            kld_max=1.2,
            exact_token_ids=exact,
        )
    return CandidateResult(
        spec=CandidateSpec(kind=kind, base_url="http://fake", model=label, label=label),
        info=info,
        logit=logit,
        tasks=tuple(task_metrics),
        agreement=None if agree is None else AgreementMetrics(10, 0.5, agree),
        perf=None if tok_s is None else PerfMetrics(tok_s, 1.0, source),
        preflight=preflight,
        errors=errors,
    )


def report_of(
    reference: CandidateResult,
    *candidates: CandidateResult,
    title: str = "Qwen2.5 0.5B Instruct on Ollama",
    prompts_file: str | None = None,
    allow_code_exec: bool = True,
    notes: tuple[str, ...] = (),
    max_size_mb: int | None = None,
) -> Report:
    return Report(
        quantdiff_version="0.2.0",
        created_at="2026-10-05T12:00:00Z",
        title=title,
        settings=RunSettings(
            suites=("json", "tools", "code", "chat"),
            top_k=10,
            score_tokens=16,
            allow_code_exec=allow_code_exec,
            seed=0,
            prompts_file=prompts_file,
            max_size_bytes=None if max_size_mb is None else max_size_mb * MB,
        ),
        reference=reference,
        candidates=candidates,
        notes=notes,
    )


def delta(
    kind: TaskKind,
    estimate: float,
    low: float,
    high: float,
    *,
    significant: bool = False,
    reliable: bool = True,
) -> TaskDelta:
    return TaskDelta(
        kind=kind,
        cases=10,
        candidate_rate=0.5,
        reference_rate=0.5,
        delta=Interval(estimate, low, high),
        significant=significant,
        reference_reliable=reliable,
    )


def call(
    label: str,
    status: Status,
    *deltas: TaskDelta,
    rank: int | None = 1,
    size_mb: int | None = None,
    size_change: float | None = None,
    reasons: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
    near_bar: bool = False,
    fits_budget: bool | None = None,
) -> CandidateVerdict:
    return CandidateVerdict(
        label=label,
        status=status,
        rank=rank,
        kld_band="small",
        size_bytes=None if size_mb is None else size_mb * MB,
        size_change=size_change,
        task_deltas=deltas,
        reasons=reasons,
        caveats=caveats,
        near_bar=near_bar,
        fits_budget=fits_budget,
    )


TRUNCATION = PreflightFinding(
    "context", "fail", "front of long prompts is being dropped", "Set num_ctx to 8192."
)
TEMPLATE = PreflightFinding("template", "warn", "chat template differs from upstream")


def qwen() -> tuple[Report, Verdict]:
    """Three Ollama quants: q4_K_M is the pick, q2_K loses tool calls."""
    tasks: dict[TaskKind, tuple[int, int]] = {"json": (9, 10), "tools": (9, 10), "code": (2, 10)}
    reference = model(STEM + "q8_0", size_mb=531, tasks=tasks, top1=None, agree=None, tok_s=33.1)
    q4 = model(
        STEM + "q4_K_M",
        size_mb=398,
        tasks={"json": (9, 10), "tools": (8, 10), "code": (3, 10)},
        top1=0.806,
        tok_s=41.0,
    )
    q2 = model(
        STEM + "q2_K",
        size_mb=339,
        tasks={"json": (10, 10), "tools": (6, 10), "code": (2, 10)},
        top1=0.762,
        tok_s=45.5,
    )
    report = report_of(reference, q2, q4)
    verdict = Verdict(
        headline=HEADLINE,
        details=(DETAIL,),
        candidates=(
            call(
                STEM + "q4_K_M",
                "recommended",
                delta("json", 0, -10, 10),
                delta("tools", -10, -30, 10),
                delta("code", 10, -10, 30, reliable=False),
                size_mb=398,
                size_change=-0.25,
                reasons=("No significant loss on 30 cases.",),
            ),
            call(
                STEM + "q2_K",
                "avoid",
                delta("json", 10, -5, 25),
                delta("tools", -30, -48, -12, significant=True),
                delta("code", 0, -20, 20, reliable=False),
                rank=2,
                size_mb=339,
                size_change=-0.36,
                reasons=("Tool calls drop 30 points.",),
            ),
        ),
        findings=(
            ServerFinding(
                TRUNCATION,
                (STEM + "q8_0", STEM + "q4_K_M", STEM + "q2_K"),
                affects_scores=False,
                impact="does not affect these scores: the longest prompt is ~900 tokens",
            ),
        ),
    )
    return report, verdict


def table_rows(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Call "))
    end = lines.index("", start)
    return [line for line in lines[start:end] if line != SIGNIFICANT_FOOTNOTE]


def md_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| ")]


def all_formats(report: Report, verdict: Verdict) -> list[str]:
    return [
        render_terminal(report, verdict=verdict),
        render_markdown(report, verdict=verdict),
        render_html(report, verdict=verdict),
    ]


# Formatting helpers -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "text"),
    [(398 * MB, "398 MB"), (int(4.7e9), "4.7 GB"), (int(7.5 * MB), "7.5 MB")],
)
def test_format_size(size: int, text: str) -> None:
    assert format_size(size) == text


def test_size_change_and_delta_formatting() -> None:
    assert format_size_change(-0.25) == "-25%"
    assert format_size_change(0.001) == "0%"
    assert format_size_change(0.1) == "+10%"
    significant = delta("tools", -30.4, -48.2, -11.6, significant=True)
    assert format_delta(significant) == "-30*"
    assert format_interval(significant) == "-30* [-48, -12]"
    assert format_delta(delta("json", 0.2, -10, 10)) == "="
    assert format_delta(delta("json", -4, -10, 2)) == "-4"
    assert format_delta(delta("json", 12, 3, 20, significant=True)) == "+12*"
    assert format_interval(delta("json", 5, -4, 14)) == "= [-4, +14]"


@pytest.mark.parametrize(
    "path",
    [
        "C:/Users/someone/secret/prompts.jsonl",
        "C:\\Users\\someone\\secret\\prompts.jsonl",
        "/home/someone/secret/prompts.jsonl",
        "prompts.jsonl",
    ],
)
def test_prompts_file_name_strips_any_directory(path: str) -> None:
    assert prompts_file_name(path) == "prompts.jsonl"


def test_summary_line_counts_models_cases_and_prompts() -> None:
    report, _ = qwen()
    assert summary_line(report) == "3 models, 40 cases, 10 scoring prompts"


# Verdict and status chips -----------------------------------------------------------------


def test_headline_leads_every_format() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    lines = terminal.splitlines()
    assert lines[4] == "> " + HEADLINE
    assert lines[5] == "  " + DETAIL
    assert terminal.index(HEADLINE) < terminal.index("Call ")
    colored = render_terminal(report, verdict=verdict, color=True)
    assert f"\x1b[1m> {HEADLINE}\x1b[0m" in colored
    assert f"> **{card._md(HEADLINE)}**\n>\n> {card._md(DETAIL)}" in markdown
    assert markdown.index("> **") < markdown.index("| Call |")
    assert (
        '<section class="verdict"><p class="kicker">Verdict</p>'
        f'<p class="headline">{HEADLINE}</p><ul><li>{DETAIL}</li></ul></section>'
    ) in page
    assert page.index('class="headline"') < page.index("<table>")


def test_verdict_box_is_neutral_without_a_recommendation() -> None:
    report, verdict = qwen()
    unsure = dataclasses.replace(
        verdict,
        headline="Not enough evidence to pick: run about 120 cases per suite.",
        candidates=tuple(dataclasses.replace(c, status="inconclusive") for c in verdict.candidates),
    )
    page = render_html(report, verdict=unsure)
    assert '<section class="verdict open">' in page
    assert '<tr class="pick">' not in page
    assert page.count('<span class="status inconclusive">UNSURE</span>') == 2


def test_status_chips_in_every_format() -> None:
    report, verdict = qwen()
    rows = table_rows(render_terminal(report, verdict=verdict))
    assert rows[0].startswith("Call   Label ")
    assert [row.split()[0] for row in rows[2:]] == ["REF", "RUN", "AVOID"]
    markdown = md_rows(render_markdown(report, verdict=verdict))
    assert markdown[2].startswith("| REF | q8\\_0 (reference) |")
    assert markdown[3].startswith("| **RUN** | q4\\_K\\_M |")
    assert markdown[4].startswith("| AVOID | q2\\_K |")
    page = render_html(report, verdict=verdict)
    assert '<tr class="pick"><td class="left"><span class="status recommended">RUN</span>' in page
    assert '<span class="status avoid">AVOID</span>' in page
    assert '<span class="status reference">REF</span>' in page


@pytest.mark.parametrize(
    ("status", "chip"),
    [("ok", "OK"), ("inconclusive", "UNSURE"), ("failed", "FAILED")],
)
def test_every_status_has_a_chip(status: Status, chip: str) -> None:
    report, verdict = qwen()
    first, *rest = verdict.candidates
    changed = dataclasses.replace(
        verdict, candidates=(dataclasses.replace(first, status=status), *rest)
    )
    assert f'<span class="status {status}">{chip}</span>' in render_html(report, verdict=changed)
    assert f"| {chip} | q4\\_K\\_M |" in render_markdown(report, verdict=changed)


def test_status_colors_in_terminal() -> None:
    report, verdict = qwen()
    text = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[1;32mRUN  \x1b[0m" in text
    assert "\x1b[1;31mAVOID\x1b[0m" in text
    assert ANSI.sub("", text) == render_terminal(report, verdict=verdict)


def test_rows_follow_the_verdict_order_not_the_report_order() -> None:
    report, verdict = qwen()
    assert [c.spec.label for c in report.candidates] == [STEM + "q2_K", STEM + "q4_K_M"]
    rows = table_rows(render_terminal(report, verdict=verdict))
    assert rows[3].split()[1] == "q4_K_M"
    assert rows[4].split()[1] == "q2_K"
    flipped = dataclasses.replace(verdict, candidates=verdict.candidates[::-1])
    rows = table_rows(render_terminal(report, verdict=flipped))
    assert rows[3].split()[1] == "q2_K"
    page = render_html(report, verdict=verdict)
    assert page.index('title="qwen2.5:0.5b-instruct-q4_K_M"') < page.index(
        'title="qwen2.5:0.5b-instruct-q2_K"'
    )


def test_reasons_are_shown_per_candidate() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert "  RUN    q4_K_M: No significant loss on 30 cases." in terminal
    assert "  AVOID  q2_K: Tool calls drop 30 points." in terminal
    assert "- **AVOID** q2\\_K: Tool calls drop 30 points." in markdown
    assert '<span class="why">Tool calls drop 30 points.</span>' in page


def test_default_verdict_comes_from_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    report, verdict = qwen()
    seen: list[Report] = []

    def judge(given: Report) -> Verdict:
        seen.append(given)
        return verdict

    monkeypatch.setattr(card, "judge", judge)
    assert render_markdown(report) == render_markdown(report, verdict=verdict)
    assert seen == [report]


# Columns ----------------------------------------------------------------------------------


def test_size_column_shows_change_against_the_reference() -> None:
    report, verdict = qwen()
    rows = table_rows(render_terminal(report, verdict=verdict))
    assert "Size" in rows[0]
    assert re.search(r"q8_0\s+531 MB\s{6}", rows[2])
    assert "398 MB -25%" in rows[3]
    assert "339 MB -36%" in rows[4]
    markdown = render_markdown(report, verdict=verdict)
    assert "| 398 MB (-25%) |" in markdown
    assert "| 531 MB |" in markdown
    page = render_html(report, verdict=verdict)
    assert '<td>398 MB<span class="change">-25%</span></td>' in page


def test_size_column_is_hidden_without_sizes() -> None:
    report, verdict = qwen()

    def unsized(result: CandidateResult) -> CandidateResult:
        assert result.info is not None
        return dataclasses.replace(result, info=dataclasses.replace(result.info, size_bytes=None))

    report = dataclasses.replace(
        report,
        reference=unsized(report.reference),
        candidates=tuple(unsized(c) for c in report.candidates),
    )
    verdict = dataclasses.replace(
        verdict,
        candidates=tuple(
            dataclasses.replace(c, size_bytes=None, size_change=None) for c in verdict.candidates
        ),
    )
    for text in all_formats(report, verdict):
        assert "Size" not in text
        assert " MB" not in text


def test_significant_regression_is_marked_and_footnoted() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    rows = table_rows(terminal)
    assert "60% -30*" in rows[4]
    assert "80% -10 " in rows[3]
    assert SIGNIFICANT_FOOTNOTE in terminal.splitlines()
    assert "| 60% (-30\\*) |" in markdown
    assert card._md(SIGNIFICANT_FOOTNOTE) in markdown
    assert '<span class="delta down">-30* [-48, -12]</span>' in page
    assert f'<p class="footnote">{SIGNIFICANT_FOOTNOTE}</p>' in page
    colored = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[31m-30*\x1b[0m" in colored
    assert "\x1b[2m-10 \x1b[0m" in colored


def test_gains_and_noise_are_never_highlighted() -> None:
    report, verdict = qwen()
    gain = delta("json", 10, 2, 18, significant=True)
    first, second = verdict.candidates
    changed = dataclasses.replace(
        verdict, candidates=(first, dataclasses.replace(second, task_deltas=(gain,)))
    )
    page = render_html(report, verdict=changed)
    assert '<span class="delta">+10* [+2, +18]</span>' in page
    assert '<span class="delta">-10 [-30, +10]</span>' in page
    assert "delta down" not in page
    assert "delta up" not in page
    assert 'class="best"' not in page
    colored = render_terminal(report, verdict=changed, color=True)
    assert "\x1b[2m+10*\x1b[0m" in colored
    assert "\x1b[32m+" not in colored


def test_no_footnote_without_significant_deltas() -> None:
    report, verdict = qwen()
    calm = dataclasses.replace(
        verdict,
        candidates=tuple(
            dataclasses.replace(
                c,
                task_deltas=tuple(dataclasses.replace(d, significant=False) for d in c.task_deltas),
            )
            for c in verdict.candidates
        ),
    )
    for text in all_formats(report, calm):
        assert "significant (paired" not in text


def test_unreliable_suite_is_greyed_out_with_a_note() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert '<td class="grey"><span class="val">30%</span>' in page
    assert '<td class="grey"><span class="val">20%</span>' in page
    assert "| *30% (=)* |" in markdown
    note = "Code says little here: the reference itself passes under half of these cases"
    for text in (terminal, markdown, page):
        assert note in " ".join(text.split())
    colored = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[2m30%\x1b[0m \x1b[2m=\x1b[0m" in colored


def test_suites_that_did_not_run_are_hidden() -> None:
    tasks: dict[TaskKind, tuple[int, int]] = {"json": (9, 10), "tools": (9, 10)}
    report = report_of(
        model(STEM + "q8_0", tasks=tasks, top1=None, agree=None, skipped_code=10),
        model(STEM + "q4_K_M", tasks=tasks, skipped_code=10, agree=None),
        allow_code_exec=False,
    )
    verdict = Verdict("Run q4_K_M.", (), (call(STEM + "q4_K_M", "recommended"),), ())
    terminal, markdown, page = all_formats(report, verdict)
    header = table_rows(terminal)[0]
    assert "Code" not in header
    assert "Agree" not in header
    assert "| JSON | Tools |" in markdown
    assert "<th>Code</th>" not in page
    assert "<th>Agree</th>" not in page
    assert "Code cases were skipped; add --allow-code-exec to score them." in terminal


def test_logit_columns_are_hidden_without_logprobs() -> None:
    report = report_of(
        model(STEM + "q8_0", tasks={"json": (9, 10)}, top1=None),
        model(STEM + "q4_K_M", tasks={"json": (9, 10)}, top1=None),
    )
    verdict = Verdict("Run q4_K_M.", (), (call(STEM + "q4_K_M", "recommended"),), ())
    page = render_html(report, verdict=verdict)
    assert "Logit fidelity" not in page
    assert 'class="base"' not in page
    assert "Top-1" not in render_terminal(report, verdict=verdict)


def test_kld_p99_needs_a_thousand_positions() -> None:
    report, verdict = qwen()
    assert "KLD99" in table_rows(render_terminal(report, verdict=verdict))[0]
    assert '<th colspan="3"><span class="tier t1">T1</span>' in render_html(report, verdict=verdict)
    short = dataclasses.replace(
        report,
        candidates=tuple(
            dataclasses.replace(
                c, logit=None if c.logit is None else dataclasses.replace(c.logit, positions=999)
            )
            for c in report.candidates
        ),
    )
    terminal, markdown, page = all_formats(short, verdict)
    assert "KLD99" not in terminal
    assert "KLD p99" not in markdown
    assert "KLD p99" not in page
    assert '<th colspan="2"><span class="tier t1">T1</span>' in page
    assert '<td class="base" colspan="2">baseline</td>' in page


def test_kld_p99_cell_is_missing_for_a_short_candidate() -> None:
    report, verdict = qwen()
    q2, q4 = report.candidates
    assert q2.logit is not None
    short = dataclasses.replace(
        report,
        candidates=(
            dataclasses.replace(q2, logit=dataclasses.replace(q2.logit, positions=200)),
            q4,
        ),
    )
    rows = md_rows(render_markdown(short, verdict=verdict))
    assert "KLD p99" in rows[0]
    assert "| 0.03 | n/a |" in rows[4]


def test_no_warnings_column() -> None:
    report, verdict = qwen()
    for text in all_formats(report, verdict):
        assert "Warn" not in text


# tok/s ------------------------------------------------------------------------------------


def _with_sources(report: Report, reference: Source, *candidates: Source) -> Report:
    def apply(result: CandidateResult, source: Source) -> CandidateResult:
        assert result.perf is not None
        return dataclasses.replace(result, perf=dataclasses.replace(result.perf, source=source))

    return dataclasses.replace(
        report,
        reference=apply(report.reference, reference),
        candidates=tuple(apply(c, s) for c, s in zip(report.candidates, candidates, strict=True)),
    )


def test_server_speeds_are_shown_plainly() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert table_rows(terminal)[0].rstrip().endswith("tok/s")
    assert "tok/s (wall)" not in terminal
    assert "| 33.1 |" in markdown
    assert "<th>tok/s</th>" in page
    assert "<td>41.0</td>" in page


def test_mixed_speed_sources_are_labelled_wall_and_muted() -> None:
    report, verdict = qwen()
    mixed = _with_sources(report, "server", "server", "wall_clock")
    terminal, markdown, page = all_formats(mixed, verdict)
    assert table_rows(terminal)[0].rstrip().endswith("tok/s (wall)")
    assert "| tok/s (wall) |" in markdown
    assert "<th>tok/s (wall)</th>" in page
    assert '<td class="wall">41.0</td>' in page
    assert "tok/s (wall) is timed from outside the server" in " ".join(terminal.split())
    colored = render_terminal(mixed, verdict=verdict, color=True)
    assert re.search(r"\x1b\[2m\s+41\.0\x1b\[0m", colored)


def test_cached_reference_speed_is_shown_muted_and_marked() -> None:
    report, verdict = qwen()
    cached = _with_sources(report, "cached", "server", "server")
    terminal, markdown, page = all_formats(cached, verdict)
    rows = table_rows(terminal)
    assert rows[0].rstrip().endswith("tok/s")
    assert "tok/s (wall)" not in terminal
    assert rows[2].rstrip().endswith("33.1 cached")
    assert re.search(r"\| \*33\.1 \(cached\)\* \|$", md_rows(markdown)[2])
    assert '<td class="wall">33.1<span class="marker">cached</span></td>' in page
    assert "<td>41.0</td>" in page
    note = "The reference tok/s marked cached comes from an earlier run; compare it loosely."
    for text in (terminal, markdown, page):
        assert note in " ".join(text.split())
    colored = render_terminal(cached, verdict=verdict, color=True)
    assert re.search(r"\x1b\[2;3;2m\s*33\.1\x1b\[0m \x1b\[2;3;2mcached", colored)


def test_cached_reference_speed_alone_still_gets_a_column() -> None:
    report, verdict = qwen()
    cached = _with_sources(report, "cached", "server", "server")
    cached = dataclasses.replace(
        cached, candidates=tuple(dataclasses.replace(c, perf=None) for c in cached.candidates)
    )
    rows = table_rows(render_terminal(cached, verdict=verdict))
    assert rows[2].rstrip().endswith("33.1 cached")
    assert rows[3].rstrip().endswith("n/a")


def test_speed_column_is_hidden_without_speeds() -> None:
    report, verdict = qwen()
    report = dataclasses.replace(
        report,
        reference=dataclasses.replace(report.reference, perf=None),
        candidates=tuple(dataclasses.replace(c, perf=None) for c in report.candidates),
    )
    for text in all_formats(report, verdict):
        assert "tok/s" not in text


# Server checks ----------------------------------------------------------------------------


def test_server_checks_list_each_finding_once_and_stay_out_of_the_verdict() -> None:
    report, verdict = qwen()
    for text in all_formats(report, verdict):
        assert text.count("front of long prompts is being dropped") == 1
        assert "all 3 models" in text
        assert "Server checks" in text
        assert "Does not affect these scores: the longest prompt is" in text
        assert "This may affect the scores" not in text
    page = render_html(report, verdict=verdict)
    verdict_box = page[page.index('<section class="verdict"') : page.index("</section>")]
    assert "front of long" not in verdict_box
    assert '<li class="quiet"><span class="chip note">NOTE</span>' in page
    markdown = render_markdown(report, verdict=verdict)
    assert "- NOTE all 3 models, context: front of long prompts is being dropped. *Does" in markdown


def test_findings_that_do_not_affect_scores_get_a_note_chip() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert "  NOTE  all 3 models, context: front of long prompts" in terminal
    for text in (terminal, markdown, page):
        assert "FAIL" not in text
        assert "WARN" not in text
    assert "chip fail" not in page
    assert ".chip.note { background: var(--track); color: var(--muted); }" in page
    colored = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[2mNOTE\x1b[0m" in colored


@pytest.mark.parametrize("severity", ["fail", "warn"])
def test_findings_that_affect_scores_keep_their_severity_chip(
    severity: Literal["fail", "warn"],
) -> None:
    report, verdict = qwen()
    finding = PreflightFinding("template", severity, "chat template differs from upstream")
    item = ServerFinding(finding, (STEM + "q2_K",), affects_scores=True, impact="it may")
    terminal, markdown, page = all_formats(report, dataclasses.replace(verdict, findings=(item,)))
    chip = severity.upper()
    assert f"  {chip}  q2_K, template:" in terminal
    assert f"- **{chip}** q2\\_K, template:" in markdown
    assert f'<li><span class="chip {severity}">{chip}</span>' in page
    assert "NOTE" not in terminal


@pytest.mark.parametrize(
    ("labels", "who", "html"),
    [
        ((STEM + "q8_0", STEM + "q4_K_M"), "both models", "<div><p>both models"),
        ((STEM + "q4_K_M",), "q4_K_M", '<span class="mono">q4_K_M</span>'),
        ((STEM + "q4_K_M", STEM + "q4_K_M"), "q4_K_M", '<span class="mono">q4_K_M</span>'),
    ],
)
def test_finding_names_both_models_or_one_label(
    labels: tuple[str, ...], who: str, html: str
) -> None:
    report, verdict = qwen()
    report = dataclasses.replace(report, candidates=report.candidates[1:])
    verdict = dataclasses.replace(
        verdict,
        candidates=verdict.candidates[:1],
        findings=(dataclasses.replace(verdict.findings[0], labels=labels),),
    )
    terminal, markdown, page = all_formats(report, verdict)
    assert f"  NOTE  {who}, context:" in terminal
    assert f"- NOTE {card._md(who)}, context:" in markdown
    assert f"{html}, context check" in page
    for text in (terminal, markdown, page):
        assert "all 2 models" not in text


def test_findings_that_affect_scores_are_flagged_near_the_verdict() -> None:
    report, verdict = qwen()
    finding = ServerFinding(
        TEMPLATE, (STEM + "q2_K",), affects_scores=True, impact="answers may be malformed"
    )
    flagged = dataclasses.replace(verdict, findings=(*verdict.findings, finding))
    terminal, markdown, page = all_formats(report, flagged)
    alert = (
        "Server check on q2_K: chat template differs from upstream. "
        "This may affect the scores; see Server checks."
    )
    assert alert in " ".join(terminal.split())
    assert f"> {card._md(alert)}" in markdown
    assert f'<p class="alert">{alert}</p></section>' in page
    assert '<li><span class="chip warn">WARN</span>' in page
    # Score-affecting findings are listed before the harmless ones.
    assert page.index("chip warn") < page.index("chip note")
    assert markdown.index("- **WARN** q2\\_K, template") < markdown.index("- NOTE all 3")


def test_no_findings_says_so() -> None:
    report, verdict = qwen()
    clean = dataclasses.replace(verdict, findings=())
    assert "No problems found." in render_html(report, verdict=clean)
    assert "Server checks" not in render_terminal(report, verdict=clean)


# Notes, settings, privacy -----------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["C:/Users/someone/secret/prompts.jsonl", "C:\\Users\\someone\\secret\\prompts.jsonl"]
)
def test_full_prompts_path_never_reaches_a_card(path: str) -> None:
    report, verdict = qwen()
    report = dataclasses.replace(
        report, settings=dataclasses.replace(report.settings, prompts_file=path)
    )
    for text in all_formats(report, verdict):
        assert "someone" not in text
        assert "secret" not in text
        assert "prompts.jsonl" in text


def _with_cases(
    report: Report, suites: tuple[str, ...], prompts_file: str | None, *case_ids: str
) -> Report:
    outcomes = tuple(CaseOutcome(case_id, "json", passed=True) for case_id in case_ids)
    return dataclasses.replace(
        report,
        settings=dataclasses.replace(report.settings, suites=suites, prompts_file=prompts_file),
        reference=dataclasses.replace(report.reference, outcomes=outcomes),
    )


def _settings_suites(report: Report, verdict: Verdict) -> list[str]:
    terminal, markdown, page = all_formats(report, verdict)
    found = re.findall(r"<dt>Suites</dt><dd>(.*?)</dd>", page)
    found += re.findall(r"Settings: Suites (.*?); Top-k", " ".join(terminal.split()))
    found += re.findall(r"\*\*Settings:\*\* Suites (.*?); Top-k", markdown)
    assert len(found) == 3
    return found


def test_suites_setting_names_the_users_prompts_when_no_suite_ran() -> None:
    report, verdict = qwen()
    report = _with_cases(report, (), "C:/Users/someone/mine.jsonl")
    # Without built-in suites every case on the card is one of the user's.
    assert _settings_suites(report, verdict) == ["your prompts (40)"] * 3
    assert "Suites none" not in render_terminal(report, verdict=verdict)
    assert "<dd>mine.jsonl</dd>" in render_html(report, verdict=verdict)


def test_suites_setting_counts_the_users_prompts_next_to_built_in_suites() -> None:
    report, verdict = qwen()
    report = _with_cases(
        report, ("json", "tools"), "mine.jsonl", "json-001", "json-002", "mine-1", "mine-2"
    )
    assert _settings_suites(report, verdict) == ["json, tools + your prompts (2)"] * 3


def test_suites_setting_without_per_case_data_leaves_out_the_count() -> None:
    report, verdict = qwen()
    report = _with_cases(report, ("json", "chat"), "mine.jsonl")
    assert _settings_suites(report, verdict) == ["json, chat + your prompts"] * 3


def test_suites_setting_without_a_prompts_file_lists_the_suites() -> None:
    report, verdict = qwen()
    assert _settings_suites(report, verdict) == ["json, tools, code, chat"] * 3
    empty = _with_cases(report, (), None)
    assert _settings_suites(empty, verdict) == ["none"] * 3


def test_kld_band_note_matches_the_verdict_engine() -> None:
    note = card.kld_band_note()
    limits = [float(value) for value in re.findall(r"under ([0-9.]+) ", note)]
    bands = re.findall(r"under [0-9.]+ ([a-z-]+)", note)
    assert limits == sorted(limits)
    assert len(limits) == len(bands) >= 2
    for limit, band in zip(limits, bands, strict=True):
        assert kld_band(limit * 0.999) == band
        assert kld_band(limit) != band
    assert note.endswith(f", else {kld_band(1e9)}.")
    assert f"under {CLOSE_KLD:g} {kld_band(CLOSE_KLD * 0.999)} (the bar for RUN)," in note
    report, verdict = qwen()
    for text in all_formats(report, verdict):
        assert note in " ".join(text.split())


def test_kld_band_note_needs_a_kld_column() -> None:
    report, verdict = qwen()
    report = dataclasses.replace(
        report, candidates=tuple(dataclasses.replace(c, logit=None) for c in report.candidates)
    )
    for text in all_formats(report, verdict):
        assert "KLD bands" not in text


REMEDY = "Run about 120 cases per suite to tell q4_K_M from q8_0."


def unsure() -> tuple[Report, Verdict]:
    report, verdict = qwen()
    verdict = dataclasses.replace(
        verdict,
        headline="Keep q8_0 for now: nothing separates the candidates on 30 cases.",
        details=("q4_K_M has the highest pass rate, but the gap is within noise.",),
        candidates=tuple(dataclasses.replace(c, status="inconclusive") for c in verdict.candidates),
        cases_needed=120,
        remedy=REMEDY,
    )
    return report, verdict


def test_unsure_verdict_leads_with_the_remedy() -> None:
    report, verdict = unsure()
    terminal, markdown, page = all_formats(report, verdict)
    lines = terminal.splitlines()
    assert lines[4].startswith("> Keep q8_0 for now")
    assert lines[5] == "  " + REMEDY
    assert lines[6].startswith("  q4_K_M has the highest pass rate")
    assert f"\x1b[1;33m  {REMEDY}\x1b[0m" in render_terminal(report, verdict=verdict, color=True)
    assert f"> **{card._md(REMEDY)}**\n>\n> q4\\_K\\_M has" in markdown
    box = page[page.index('<section class="verdict') : page.index("</section>")]
    assert box.startswith('<section class="verdict open">')
    assert f'<p class="remedy">{REMEDY}</p><ul><li>q4_K_M has' in box
    assert box.count(REMEDY) == 1
    assert '<tr class="pick">' not in page
    assert page.count('<span class="status inconclusive">UNSURE</span>') == 2
    assert ".status.inconclusive {\n  background: var(--warn-soft);\n  border:" in page


def test_no_remedy_line_without_a_remedy() -> None:
    report, verdict = unsure()
    settled = dataclasses.replace(verdict, remedy=None)
    page = render_html(report, verdict=settled)
    assert 'class="remedy"' not in page
    assert REMEDY not in page
    assert REMEDY not in render_terminal(report, verdict=settled)


OLLAMA_NOTE = (
    "Ollama logit metrics are text-forced; on English text they matched llama-server's "
    "exact token-id forcing (docs/calibration.md). Non-Latin text may read higher."
)


def test_text_forced_note_cites_the_calibration_for_ollama() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert OLLAMA_NOTE in " ".join(terminal.split())
    assert card._md(OLLAMA_NOTE) in markdown
    assert card._e(OLLAMA_NOTE) in page
    for text in (terminal, markdown, page):
        assert "approximate" not in text
        assert "re-tokenized" not in text


def test_text_forced_note_is_the_same_when_one_ollama_candidate_is_exact() -> None:
    report, verdict = qwen()
    q2, q4 = report.candidates
    assert q4.logit is not None
    exact = dataclasses.replace(q4, logit=dataclasses.replace(q4.logit, exact_token_ids=True))
    page = render_html(dataclasses.replace(report, candidates=(q2, exact)), verdict=verdict)
    assert card._e(OLLAMA_NOTE) in page


def test_text_forced_note_names_non_ollama_candidates() -> None:
    report, verdict = qwen()
    q2, q4 = report.candidates
    other = dataclasses.replace(q2, spec=dataclasses.replace(q2.spec, kind="openai"))
    page = render_html(dataclasses.replace(report, candidates=(other, q4)), verdict=verdict)
    assert "Logit metrics for q4_K_M, q2_K are text-forced;" in page
    assert "Ollama logit metrics" not in page


def test_user_notes_and_errors_are_rendered() -> None:
    report, verdict = qwen()
    broken = model(STEM + "iq1_s", top1=None, agree=None, tok_s=None, errors=("model not found",))
    report = dataclasses.replace(
        report, candidates=(*report.candidates, broken), notes=("Run on a GTX 1650.",)
    )
    verdict = dataclasses.replace(
        verdict,
        candidates=(*verdict.candidates, call(STEM + "iq1_s", "failed", rank=None)),
    )
    terminal, markdown, page = all_formats(report, verdict)
    assert "iq1_s: model not found" in terminal
    assert "- iq1\\_s: model not found" in markdown
    assert '<span class="status failed">FAILED</span>' in page
    assert "failed to run" in page
    for text in (terminal, markdown, page):
        assert "Run on a GTX 1650." in text


# Escaping and width -----------------------------------------------------------------------


def test_html_escapes_every_dynamic_string() -> None:
    report, verdict = qwen()
    finding = PreflightFinding(
        check="<b>check</b>", severity="fail", message="<img onerror=x>", fix='say "hi"'
    )
    report = dataclasses.replace(
        report,
        title='Quants "compared" & <ranked>',
        notes=("<i>note</i>",),
        settings=dataclasses.replace(report.settings, prompts_file="<u>p</u>.jsonl"),
    )
    first, second = verdict.candidates
    verdict = dataclasses.replace(
        verdict,
        headline=XSS,
        details=("<em>detail</em>",),
        candidates=(dataclasses.replace(first, reasons=("<marquee>why</marquee>",)), second),
        findings=(ServerFinding(finding, (XSS,), True, "<s>impact</s>"),),
    )
    page = render_html(report, verdict=verdict)
    for raw in (XSS, "<img", "<b>check", "<i>note", 'say "hi"', "<em>", "<marquee>", "<s>", "<u>"):
        assert raw not in page
    assert '<p class="headline">&lt;script&gt;alert(1)&lt;/script&gt;</p>' in page
    assert "Quants &quot;compared&quot; &amp; &lt;ranked&gt;" in page
    assert "&lt;marquee&gt;why&lt;/marquee&gt;" in page
    markdown = render_markdown(report, verdict=verdict)
    assert "\\<script\\>" in markdown
    assert "<marquee>" not in markdown


def test_html_escapes_labels_in_attributes() -> None:
    stem = '"><img src=x onerror=alert(1)>-'
    report = report_of(model(stem + "ref"), model(stem + "a"))
    verdict = Verdict("Run a.", (), (call(stem + "a", "recommended"),), ())
    page = render_html(report, verdict=verdict)
    assert "<img" not in page
    assert 'title="&quot;&gt;&lt;img src=x onerror=alert(1)&gt;-a"' in page


def test_html_is_self_contained() -> None:
    report, verdict = qwen()
    page = render_html(report, verdict=verdict)
    assert page.startswith("<!doctype html>")
    assert "prefers-color-scheme: dark" in page
    assert "max-width: 960px" in page
    assert "font-variant-numeric: tabular-nums" in page
    lowered = page.lower()
    for forbidden in ("<script", "src=", "href=", "@import", "url(", "http://", "https://"):
        assert forbidden not in lowered


def test_html_bars_are_clamped_percentages() -> None:
    report, verdict = qwen()
    page = render_html(report, verdict=verdict)
    widths = [float(value) for value in re.findall(r'style="width:([0-9.]+)%"', page)]
    assert widths
    assert all(0.0 <= width <= 100.0 for width in widths)


def test_header_hoists_shared_prefix_and_backend() -> None:
    report, verdict = qwen()
    terminal, markdown, page = all_formats(report, verdict)
    assert terminal.splitlines()[2] == (
        "Model: qwen2.5:0.5b-instruct   Reference: qwen2.5:0.5b-instruct-q8_0   Backend: ollama"
    )
    assert "Backend" not in table_rows(terminal)[0]
    assert "Model: **qwen2.5:0.5b-instruct**" in markdown
    assert '<li>Backend <span class="mono" title="ollama">ollama</span></li>' in page


def test_mixed_backends_get_a_column() -> None:
    report, verdict = qwen()
    q2, q4 = report.candidates
    mixed = dataclasses.replace(
        report,
        candidates=(
            q2,
            dataclasses.replace(q4, spec=dataclasses.replace(q4.spec, kind="llamacpp")),
        ),
    )
    assert "Backend" in table_rows(render_terminal(mixed, verdict=verdict))[0]
    assert '<td class="left backend">llamacpp</td>' in render_html(mixed, verdict=verdict)


@pytest.mark.parametrize("backend", ["mixed", "shared"])
def test_terminal_stays_within_width(backend: str) -> None:
    report, verdict = qwen()
    long = {c.label: "org/" + "x" * 90 + c.label for c in verdict.candidates}

    def relabel(result: CandidateResult, label: str, kind: BackendKind) -> CandidateResult:
        return dataclasses.replace(
            result, spec=dataclasses.replace(result.spec, label=label, kind=kind)
        )

    kinds: list[BackendKind] = ["llamacpp", "openai"] if backend == "mixed" else ["ollama"] * 2
    report = dataclasses.replace(
        report,
        reference=relabel(report.reference, "r" * 120, "ollama"),
        candidates=tuple(
            relabel(c, long[c.spec.label], kind)
            for c, kind in zip(report.candidates, kinds, strict=True)
        ),
    )
    verdict = dataclasses.replace(
        verdict,
        headline="x " * 200,
        candidates=tuple(dataclasses.replace(c, label=long[c.label]) for c in verdict.candidates),
    )
    text = ANSI.sub("", render_terminal(report, verdict=verdict, color=True))
    assert all(len(line) <= TERMINAL_WIDTH for line in text.splitlines())
    assert all("..." in row for row in table_rows(text)[2:])


def test_terminal_strips_control_characters() -> None:
    report, verdict = qwen()
    verdict = dataclasses.replace(verdict, headline="Run \x1b[31mq4\x1b[0m now.")
    text = render_terminal(report, verdict=verdict)
    assert "\x1b" not in text


def test_terminal_renders_report_without_candidates() -> None:
    report = report_of(model("example-ref", tasks={"json": (9, 10)}))
    verdict = Verdict("No candidates were evaluated.", (), (), ())
    text = render_terminal(report, verdict=verdict)
    rows = table_rows(text)
    assert rows[2].startswith("REF   example-ref")
    assert rows[3] == "(no candidates)"
    assert "No candidates" in render_html(report, verdict=verdict)


# Usable, caveats, budgets, verdict tone and KLD bands -------------------------------------

CAVEAT = "tools -17 unresolved (95% CI -42 to +6); rerun with --max-cases 60"
FITS_HEADLINE = "Best that fits 400 MB: q2_K, a moderate loss against q8_0."


def fits() -> tuple[Report, Verdict]:
    """q2_K is the usable best that fits a 400 MB budget, with a caveat; q4_K_M is close to
    the bar and over the budget."""
    report, verdict = qwen()
    report = dataclasses.replace(
        report, settings=dataclasses.replace(report.settings, max_size_bytes=400 * MB)
    )
    q4, q2 = verdict.candidates
    verdict = dataclasses.replace(
        verdict,
        headline=FITS_HEADLINE,
        candidates=(
            dataclasses.replace(q2, status="usable", caveats=(CAVEAT,), fits_budget=True),
            dataclasses.replace(q4, status="ok", near_bar=True, fits_budget=False),
        ),
    )
    return report, verdict


def test_usable_chip_in_every_format() -> None:
    report, verdict = fits()
    terminal, markdown, page = all_formats(report, verdict)
    rows = table_rows(terminal)
    assert [row.split()[0] for row in rows[2:]] == ["REF", "USABLE", "OK"]
    assert md_rows(markdown)[3].startswith("| USABLE | q2\\_K |")
    assert '<span class="status usable">USABLE</span>' in page
    assert ".status.usable { background: var(--usable-soft); color: var(--usable); }" in page
    colored = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[1;34mUSABLE\x1b[0m" in colored


def test_caveats_follow_the_reasons_in_every_format() -> None:
    report, verdict = fits()
    terminal, markdown, page = all_formats(report, verdict)
    why = " ".join(terminal[terminal.index("\nWhy\n") :].split())
    assert "USABLE q2_K: Tool calls drop 30 points. caveat: tools -17 unresolved" in why
    assert "rerun with --max-cases 60." in why
    assert "- **USABLE** q2\\_K: Tool calls drop 30 points." in markdown
    assert f"  - caveat: {card._md(CAVEAT)}." in markdown
    caveat = f'<span class="caveat">{card._e(CAVEAT)}.</span>'
    assert f"Tool calls drop 30 points.</span>{caveat}" in page
    colored = render_terminal(report, verdict=verdict, color=True)
    assert re.search(r"\x1b\[33m +caveat: tools -17", colored)


def test_caveats_without_reasons_still_get_a_why_line() -> None:
    report, verdict = fits()
    first, second = verdict.candidates
    bare = dataclasses.replace(verdict, candidates=(dataclasses.replace(first, reasons=()), second))
    terminal = render_terminal(report, verdict=bare)
    assert re.search(r"\n  USABLE +q2_K\n +caveat: tools -17", terminal)
    assert "- **USABLE** q2\\_K\n  - caveat:" in render_markdown(report, verdict=bare)


def test_caveats_are_escaped() -> None:
    report, verdict = fits()
    first, second = verdict.candidates
    hostile = dataclasses.replace(
        verdict, candidates=(dataclasses.replace(first, caveats=(XSS,)), second)
    )
    page = render_html(report, verdict=hostile)
    assert XSS not in page
    assert "&lt;script&gt;" in page
    assert "\\<script\\>" in render_markdown(report, verdict=hostile)


def test_near_bar_tag_sits_next_to_the_chip() -> None:
    report, verdict = fits()
    terminal, markdown, page = all_formats(report, verdict)
    assert "  OK (near bar)  q4_K_M:" in terminal
    assert "| OK *near bar* | q4\\_K\\_M |" in markdown
    assert "- **OK (near bar)** q4\\_K\\_M:" in markdown
    assert '<span class="status ok">OK</span><span class="tag">near bar</span></td>' in page
    assert page.count('<span class="tag">') == 1


def test_budget_is_shown_in_the_header_and_settings() -> None:
    report, verdict = fits()
    terminal, markdown, page = all_formats(report, verdict)
    header = " ".join(terminal[: terminal.index("\n\n")].split())
    assert header.endswith("Backend: ollama Budget: 400 MB")
    assert "Budget 400 MB" in " ".join(terminal.split())
    assert "Budget: **400 MB**" in markdown
    assert "Budget 400 MB" in markdown
    assert 'Budget <span class="mono" title="400 MB">400 MB</span>' in page
    assert "<dt>Budget</dt><dd>400 MB</dd>" in page
    plain, _ = qwen()
    for text in all_formats(plain, verdict):
        assert "Budget" not in text


def test_sizes_over_budget_are_muted_and_marked() -> None:
    report, verdict = fits()
    terminal, markdown, page = all_formats(report, verdict)
    rows = table_rows(terminal)
    assert "398 MB -25% (over)" in rows[4]
    assert "(over)" not in rows[3]
    assert "| *398 MB (-25%) (over)* |" in markdown
    assert "| 339 MB (-36%) |" in markdown
    assert (
        '<td class="over">398 MB<span class="change">-25%</span>'
        '<span class="change over">over budget</span></td>'
    ) in page
    assert page.count("over budget</span>") == 1
    colored = render_terminal(report, verdict=verdict, color=True)
    assert "\x1b[2m398 MB\x1b[0m \x1b[2m-25% (over)\x1b[0m" in colored


@pytest.mark.parametrize(
    ("headline", "statuses", "tone"),
    [
        (HEADLINE, ("recommended", "avoid"), "verdict"),
        (FITS_HEADLINE, ("usable", "avoid"), "verdict fits"),
        ("Keep q8_0 for now: no candidate is proven close.", ("usable", "inconclusive"), "open"),
        ("Keep q8_0: every candidate shows a measured loss.", ("avoid", "avoid"), "open"),
        ("Keep q8_0 for now: nothing fits the budget.", ("recommended", "avoid"), "open"),
        ("No candidate produced metrics.", ("failed", "failed"), "open"),
    ],
)
def test_verdict_box_tone_follows_the_outcome(
    headline: str, statuses: tuple[Status, Status], tone: str
) -> None:
    report, verdict = qwen()
    changed = dataclasses.replace(
        verdict,
        headline=headline,
        candidates=tuple(
            dataclasses.replace(c, status=s)
            for c, s in zip(verdict.candidates, statuses, strict=True)
        ),
    )
    css = "verdict open" if tone == "open" else tone
    assert f'<section class="{css}"><p class="kicker">' in render_html(report, verdict=changed)


def _with_kld(result: CandidateResult, value: float) -> CandidateResult:
    assert result.logit is not None
    return dataclasses.replace(result, logit=dataclasses.replace(result.logit, kld_mean=value))


@pytest.mark.parametrize(
    ("value", "band"),
    [(0.005, "near-lossless"), (0.03, "small"), (0.06, "moderate"), (0.117, "large")],
)
def test_kld_bars_are_coloured_by_band(value: float, band: str) -> None:
    report, verdict = qwen()
    q2, q4 = report.candidates
    changed = dataclasses.replace(report, candidates=(_with_kld(q2, value), q4))
    page = render_html(changed, verdict=verdict)
    assert kld_band(value) == band
    assert f'<span class="bar {band}"><span style="width:' in page
    assert "fill-kld" not in page
    assert ".bar.near-lossless span, .bar.small span { background: var(--kld-good); }" in page


def test_kld_bands_follow_the_runs_top_k() -> None:
    report, verdict = qwen()
    # 0.03 is small at the default top-10 bars but moderate at top-1, whose bars are lower.
    top1 = dataclasses.replace(report, settings=dataclasses.replace(report.settings, top_k=1))
    page = render_html(top1, verdict=verdict)
    assert page.count('<span class="bar moderate">') == 2
    assert '<span class="bar small">' not in page


def test_kld_p99_bars_stay_neutral() -> None:
    report, verdict = qwen()
    page = render_html(report, verdict=verdict)
    assert page.count('<span class="bar small">') == 2
    p99 = re.findall(r'<span class="val">0\.25</span><span class="([^"]*)">', page)
    assert p99 == ["bar", "bar"]


def test_terminal_header_never_splits_a_fact() -> None:
    assert card._pack(["Model: m", "Budget: 380 MB"], width=20) == ["Model: m", "Budget: 380 MB"]
    assert card._pack(["a", "b"], width=20) == ["a   b"]
