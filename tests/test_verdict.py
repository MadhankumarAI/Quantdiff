from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from quantdiff.report import load_report
from quantdiff.stats import Interval, bootstrap_mean, cases_to_bound_loss, items_to_bound_below
from quantdiff.types import (
    AgreementMetrics,
    CandidateResult,
    CandidateSpec,
    CaseOutcome,
    LogitMetrics,
    PreflightFinding,
    PromptLogit,
    Report,
    RunSettings,
    ScoringPrompt,
    ServerInfo,
    TaskKind,
    TaskMetrics,
)
from quantdiff.verdict import (
    CLOSE_KLD,
    LARGE_KLD,
    NEAR_LOSSLESS_KLD,
    CandidateVerdict,
    TaskDelta,
    Verdict,
    describe_delta,
    display_labels,
    format_size,
    judge,
    kld_band,
    mostly_non_latin,
)
from tests.test_report import FIXTURE

DATA = Path(__file__).parent / "data"
STEM = "qwen2.5-7b-instruct-"
POSITIONS = 32
WEIGHTS = (0.3, 1.7, 0.6, 1.2, 0.9, 1.4, 0.5, 1.1, 0.8, 1.5)
"""Per-prompt difficulty: the same prompts are hard for every quant, which pairing removes."""


def suite(passed: int, total: int, *, lose: int = 0, gain: int = 0) -> list[bool]:
    """Outcomes for one suite: the reference's `passed` of `total`, then `lose` of those
    passes flipped to fails and `gain` of the fails flipped to passes."""
    outcomes = [True] * passed + [False] * (total - passed)
    for i in range(lose):
        outcomes[i] = False
    for i in range(gain):
        outcomes[passed + i] = True
    return outcomes


def prompt_klds(mean: float, prompts: int, *, jitter: float = 0.0) -> list[float]:
    weights = [WEIGHTS[i % len(WEIGHTS)] * (1 + jitter * ((-1) ** i)) for i in range(prompts)]
    scale = mean * prompts / sum(weights)
    return [w * scale for w in weights]


def model(
    name: str,
    *,
    outcomes: dict[TaskKind, list[bool]] | None = None,
    klds: Sequence[float] | None = None,
    top1: float = 0.95,
    size: int | None = None,
    weights_id: str | None = None,
    context_length: int | None = 8192,
    agreement: Sequence[float] | None = None,
    preflight: tuple[PreflightFinding, ...] = (),
    errors: tuple[str, ...] = (),
) -> CandidateResult:
    label = STEM + name
    outcomes = outcomes or {}
    cases = tuple(
        CaseOutcome(case_id=f"{kind}-{i:03d}", kind=kind, passed=passed)
        for kind, results in outcomes.items()
        for i, passed in enumerate(results)
    )
    tasks = tuple(
        TaskMetrics(kind=kind, total=len(results), passed=sum(results), skipped=0)
        for kind, results in outcomes.items()
    )
    logit = None
    if klds is not None:
        matches = round(top1 * POSITIONS)
        logit = LogitMetrics(
            prompts=len(klds),
            positions=len(klds) * POSITIONS,
            top1_agreement=matches / POSITIONS,
            kld_mean=sum(klds) / len(klds),
            kld_p99=max(klds),
            kld_max=max(klds),
            exact_token_ids=True,
            per_prompt=tuple(
                PromptLogit(f"prompt-{i:03d}", POSITIONS, matches, kld)
                for i, kld in enumerate(klds)
            ),
        )
    chat = None
    if agreement is not None:
        chat = AgreementMetrics(
            cases=len(agreement),
            exact_match_rate=0.0,
            mean_similarity=sum(agreement) / len(agreement),
            per_case=tuple((f"chat-{i:03d}", value) for i, value in enumerate(agreement)),
        )
    info = ServerInfo(
        backend="llamacpp",
        model=label,
        context_length=context_length,
        chat_template=None,
        template_dialect="jinja",
        supports_logprobs=True,
        exact_token_ids=True,
        size_bytes=size,
        weights_id=weights_id,
    )
    return CandidateResult(
        spec=CandidateSpec(kind="llamacpp", base_url="http://fake", model="", label=label),
        info=None if errors else info,
        logit=logit,
        tasks=tasks,
        agreement=chat,
        perf=None,
        preflight=preflight,
        errors=errors,
        outcomes=cases,
    )


def report(
    reference: CandidateResult, *candidates: CandidateResult, longest: int | None = 900
) -> Report:
    return Report(
        quantdiff_version="0.2.0",
        created_at="2026-10-05T12:00:00Z",
        title="synthetic",
        settings=RunSettings(
            suites=("json", "tools", "code"),
            top_k=10,
            score_tokens=32,
            allow_code_exec=True,
            seed=0,
            longest_prompt_tokens=longest,
        ),
        reference=reference,
        candidates=candidates,
    )


def reference_q8() -> CandidateResult:
    base: dict[TaskKind, list[bool]] = {
        "json": suite(36, 40),
        "tools": suite(36, 40),
        "code": suite(7, 8),
    }
    return model("q8_0", outcomes=base, size=8_000_000_000, weights_id="q8")


def status_of(verdict: Verdict, name: str) -> str:
    return next(c.status for c in verdict.candidates if c.label == STEM + name)


def everything(verdict: Verdict) -> str:
    """Every sentence the verdict can put in front of a reader."""
    reasons = [reason for call in verdict.candidates for reason in call.reasons]
    return " ".join((verdict.headline, *verdict.details, *reasons))


def q4_tasks(*, lose: int = 1) -> dict[TaskKind, list[bool]]:
    """88 paired cases close to reference_q8: `lose` json cases flipped to fails."""
    return {"json": suite(36, 40, lose=lose), "tools": suite(36, 40), "code": suite(7, 8)}


def rounded_up(count: int) -> int:
    return math.ceil(count / 10) * 10


# Run: positive evidence --------------------------------------------------------------------


def test_clear_run_with_41_prompts_and_88_cases() -> None:
    klds = prompt_klds(0.029, 41, jitter=0.3)
    q4 = model("q4_K_M", outcomes=q4_tasks(), klds=klds, size=6_000_000_000)
    verdict = judge(report(reference_q8(), q4))
    interval = bootstrap_mean(klds)
    ci = f"95% CI {interval.low:.3f} to {interval.high:.3f}"
    assert verdict.headline == (
        f"Run q4_K_M: 25% smaller than q8_0, close on logits (KLD 0.029, CI up to "
        f"{interval.high:.3f}) on 41 prompts."
    )
    assert verdict.details == (
        f"q4_K_M: KLD 0.029 ({ci}) and task scores within 10 points of q8_0 on 88 cases.",
    )
    (call,) = verdict.candidates
    assert call.status == "recommended"
    assert call.rank == 1
    assert call.size_change == pytest.approx(-0.25)
    assert call.reasons == (
        f"KLD 0.029 ({ci}) on 41 prompts.",
        "Task scores within 10 points of q8_0 on 88 cases.",
    )
    assert verdict.cases_needed is None
    assert verdict.remedy is None


def test_run_on_logit_evidence_only_says_so() -> None:
    q5 = model("q5_K_M", klds=prompt_klds(0.02, 20), size=5_600_000_000)
    verdict = judge(report(reference_q8(), q5))
    assert verdict.headline.startswith(
        "Run q5_K_M: 30% smaller than q8_0, close on logits (KLD 0.02, CI up to 0.0"
    )
    assert verdict.headline.endswith(") on 20 prompts.")
    assert verdict.details[0].startswith("q5_K_M: KLD 0.02 (95% CI ")
    assert verdict.details[0].endswith("); logit evidence only; no task suites ran.")
    assert verdict.candidates[0].reasons[-1] == (
        "Rests on logit evidence only; no task suites ran."
    )


def test_run_on_task_evidence_only_says_so() -> None:
    q4 = model("q4_K_M", outcomes=q4_tasks(), size=6_000_000_000)
    verdict = judge(report(reference_q8(), q4))
    assert verdict.headline == (
        "Run q4_K_M: 25% smaller than q8_0, close on task scores on 88 cases (no logit metrics)."
    )
    assert verdict.details == (
        "q4_K_M: task scores within 10 points of q8_0 on 88 cases; task evidence only; no "
        "logit metrics.",
    )
    assert status_of(verdict, "q4_K_M") == "recommended"


def test_unreliable_suite_neither_avoids_nor_counts_as_task_evidence() -> None:
    reference = model("q8_0", outcomes={"code": suite(8, 20)})
    candidate = model("q4_K_M", outcomes={"code": suite(8, 20, lose=8)}, klds=prompt_klds(0.03, 20))
    verdict = judge(report(reference, candidate))
    (delta,) = verdict.candidates[0].task_deltas
    assert delta.significant
    assert not delta.reference_reliable
    assert verdict.candidates[0].status == "recommended"
    assert verdict.headline.startswith("Run q4_K_M: close on logits (KLD 0.03, CI up to ")
    assert verdict.details[0].endswith(
        "; logit evidence only; q8_0 passes under half of every task suite, so the suites "
        "cannot judge it."
    )


def test_picks_the_smallest_close_candidate_and_calls_the_others_ok() -> None:
    q8 = model("q8_0", klds=prompt_klds(0.004, 30), size=8_000_000_000)
    q4 = model("q4_K_M", klds=prompt_klds(0.03, 30), size=4_500_000_000)
    reference = model("bf16", size=15_000_000_000)
    verdict = judge(report(reference, q8, q4))
    assert verdict.headline.startswith("Run q4_K_M: 70% smaller than bf16, close on logits")
    assert status_of(verdict, "q8_0") == "ok"
    assert verdict.details[1:] == ("q8_0 is also close to bf16 but 78% larger.",)
    assert verdict.candidates[0].reasons[-1] == "Smallest download that is close to bf16."


def test_without_sizes_the_pick_is_the_lowest_kld() -> None:
    a = model("q5_K_M", klds=prompt_klds(0.02, 20))
    b = model("q4_K_M", klds=prompt_klds(0.03, 20), size=4_500_000_000)
    verdict = judge(report(reference_q8(), b, a))
    assert status_of(verdict, "q5_K_M") == "recommended"
    assert verdict.details[1:] == ("q4_K_M is also close to q8_0.",)


# Usable: a measured, moderate loss ---------------------------------------------------------


@pytest.mark.parametrize("mean", [0.05, 0.06, 0.08])
def test_moderate_kld_is_usable_never_run_or_avoid(mean: float) -> None:
    klds = prompt_klds(mean, 41, jitter=0.3)
    q3 = model("q3_K_M", outcomes=q4_tasks(lose=0), klds=klds, size=3_800_000_000)
    verdict = judge(report(reference_q8(), q3))
    interval = bootstrap_mean(klds)
    (call,) = verdict.candidates
    assert call.kld_band == "moderate"
    assert call.status == "usable"
    assert call.reasons[0] == (
        f"Moderate loss: KLD {mean:g} (95% CI {interval.low:.3f} to {interval.high:.3f})."
    )
    assert verdict.headline == (
        "No download is close to q8_0; q3_K_M has the smallest loss among the smaller "
        f"downloads (moderate, KLD {mean:g})."
    )
    assert verdict.remedy == (
        "Pass --max-size with the memory you can spare, for example --max-size 6GB, to pick "
        "the best download that fits."
    )


def test_usable_and_avoid_boundary_is_the_lower_end_of_the_interval_at_the_large_bar() -> None:
    def call_for(mean: float, jitter: float) -> CandidateVerdict:
        q = model("q3_K_S", klds=prompt_klds(mean, 30, jitter=jitter))
        (call,) = judge(report(reference_q8(), q)).candidates
        return call

    # Interval wholly at or above 0.10: a large loss.
    large = call_for(0.2, 0.0)
    assert large.status == "avoid"
    assert large.reasons[0].startswith("KLD is large (0.2, 95% CI ")
    # Mean just above 0.10 but the interval reaches below it: not proven large.
    straddle = call_for(0.105, 0.5)
    assert straddle.status == "inconclusive"
    assert straddle.near_bar
    assert straddle.reasons[0].startswith("KLD 0.105 may be a large loss, but its 95% CI ")
    assert "near the large-loss bar; a rerun could change this" in straddle.caveats
    # Mean just under 0.10 with the interval above 0.04: usable, flagged near the large bar.
    under = call_for(0.095, 0.5)
    assert under.status == "usable"
    assert under.near_bar


def test_kld_interval_straddling_the_closeness_bar_is_inconclusive_and_near_it() -> None:
    q4 = model("q4_K_M", klds=prompt_klds(0.045, 12, jitter=0.9))
    verdict = judge(report(reference_q8(), q4))
    (call,) = verdict.candidates
    assert call.status == "inconclusive"
    assert call.near_bar
    assert call.caveats == ("near the closeness bar; a rerun could change this",)
    assert call.reasons[0].startswith("KLD 0.045 is above the 0.04 closeness bar, but its 95% CI")


def test_near_bar_counts_an_interval_end_within_ten_percent_of_a_bar() -> None:
    near = model("q4_K_M", klds=[0.0365, 0.0375] * 6)
    clear = model("q5_K_M", klds=[0.02, 0.022] * 6)
    verdict = judge(report(reference_q8(), near, clear))
    calls = {c.label.removeprefix(STEM): c for c in verdict.candidates}
    assert calls["q4_K_M"].status == "ok"
    assert calls["q4_K_M"].near_bar
    assert calls["q5_K_M"].status == "recommended"
    assert not calls["q5_K_M"].near_bar
    assert calls["q5_K_M"].caveats == ()


def test_moderate_kld_on_thin_evidence_is_inconclusive_not_run() -> None:
    q3 = model("q3_K_M", klds=prompt_klds(0.09, 5), size=3_800_000_000)
    verdict = judge(report(reference_q8(), q3))
    assert verdict.candidates[0].status == "inconclusive"
    assert verdict.candidates[0].reasons == (
        "KLD 0.09 is above the 0.04 closeness bar.",
        "About 10 scoring prompts would likely decide.",
    )
    assert verdict.headline == "Keep q8_0 for now: no candidate is shown to be close on 5 prompts."
    assert verdict.details == ("q3_K_M looks closest: KLD 0.09 on 5 prompts.",)
    assert verdict.cases_needed == 10
    assert verdict.remedy == "Rerun with --max-cases 10 to decide."


# Avoid: a measured loss against the reference ---------------------------------------------


def test_large_kld_on_few_prompts_is_inconclusive_not_avoid() -> None:
    q2 = model("q2_K", outcomes={"json": suite(36, 40)}, klds=prompt_klds(0.3, 3))
    verdict = judge(report(reference_q8(), q2))
    (call,) = verdict.candidates
    assert call.status == "inconclusive"
    assert call.kld_band == "large"
    assert call.reasons[0] == "KLD 0.3 looks large on 3 prompts, too few to call it."
    assert (
        verdict.headline
        == "Keep q8_0 for now: no candidate is shown to be close on 3 prompts and 40 cases."
    )
    assert verdict.remedy is not None


def test_significant_suite_regression_is_avoided_even_with_small_kld() -> None:
    q2 = model(
        "q2_K",
        outcomes={"json": suite(36, 40), "tools": suite(36, 40, lose=12), "code": suite(7, 8)},
        klds=prompt_klds(0.03, 30),
        size=3_000_000_000,
    )
    verdict = judge(report(reference_q8(), q2))
    (call,) = verdict.candidates
    assert call.status == "avoid"
    tools = next(d for d in call.task_deltas if d.kind == "tools")
    assert tools.significant
    assert tools.delta.estimate == pytest.approx(-30.0)
    low, high = round(tools.delta.low), round(tools.delta.high)
    assert verdict.details == (
        f"Avoid q2_K: tools drops 30 points vs q8_0 (95% CI {low} to {high}).",
    )
    assert call.reasons == (f"tools drops 30 points vs q8_0 (95% CI {low} to {high}).",)


def test_every_candidate_avoided() -> None:
    q2 = model("q2_K", klds=prompt_klds(0.3, 10))
    q3 = model("q3_K_S", klds=prompt_klds(0.2, 30))
    verdict = judge(report(reference_q8(), q2, q3))
    assert verdict.headline == "Keep q8_0: every candidate shows a measured loss."
    assert [c.status for c in verdict.candidates] == ["avoid", "avoid"]
    assert len(verdict.details) == 2
    assert verdict.remedy == "Try a larger quant against q8_0; nothing tested here is close."


# Inconclusive: not proven close -----------------------------------------------------------


def test_single_candidate_on_thin_evidence_is_not_recommended() -> None:
    q4 = model("q4_K_M", klds=prompt_klds(0.02, 5), size=6_000_000_000)
    verdict = judge(report(reference_q8(), q4))
    (call,) = verdict.candidates
    assert call.status == "inconclusive"
    assert call.reasons[0] == "KLD 0.02, but 5 prompts cannot bound it below 0.04."
    assert verdict.headline == "Keep q8_0 for now: no candidate is shown to be close on 5 prompts."
    assert verdict.details == ("q4_K_M looks closest: KLD 0.02 on 5 prompts.",)
    assert verdict.remedy == "Rerun with --max-cases 10 to decide."
    assert verdict.cases_needed == 10


def test_wide_kld_interval_is_inconclusive() -> None:
    klds = [0.001, 0.002, 0.001, 0.003, 0.002, 0.001, 0.002, 0.3]
    q4 = model("q4_K_M", klds=klds)
    verdict = judge(report(reference_q8(), q4))
    interval = bootstrap_mean(klds)
    (call,) = verdict.candidates
    assert call.status == "inconclusive"
    assert call.reasons[0] == (
        f"KLD {interval.estimate:.3f}, but its 95% CI reaches {interval.high:.3f}, above the "
        "0.04 closeness bar."
    )


def test_cases_needed_combines_prompts_and_cases_per_suite() -> None:
    reference = model("q8_0", outcomes={"json": suite(9, 10), "tools": suite(9, 10)})
    klds = prompt_klds(0.03, 6, jitter=0.6)
    q4 = model("q4_K_M", outcomes={"json": suite(9, 10), "tools": suite(9, 10)}, klds=klds)
    verdict = judge(report(reference, q4))
    prompts = items_to_bound_below(bootstrap_mean(klds), 6, 0.05)
    assert prompts is not None
    ref_pass = suite(9, 10) * 2
    pooled = cases_to_bound_loss(ref_pass, ref_pass, 10.0)
    assert pooled is not None
    per_suite = rounded_up(math.ceil(max(pooled, 20) / 2))
    flag = max(rounded_up(max(prompts, 8)), per_suite)
    assert verdict.cases_needed == flag
    assert verdict.remedy == f"Rerun with --max-cases {flag} to decide."
    assert verdict.headline == (
        "Keep q8_0 for now: no candidate is shown to be close on 6 prompts and 20 cases."
    )
    assert verdict.details == (
        "q4_K_M looks closest: KLD 0.03 on 6 prompts; task scores level with q8_0, 95% CI down "
        "to -13.",
    )
    assert verdict.candidates[0].reasons == (
        "KLD 0.03, but 6 prompts cannot bound it below 0.04.",
        "Task scores could be up to 13 points lower than q8_0 (95% CI -13 to +13 on 20 cases).",
        f"About {rounded_up(max(prompts, 8))} scoring prompts and {per_suite} cases per suite "
        "would likely decide.",
    )


def test_remedy_beyond_the_built_in_suites_asks_for_your_own_prompts() -> None:
    # Without logit metrics, tasks must prove closeness on their own, which takes many cases.
    reference = model("q8_0", outcomes={"json": suite(9, 10), "tools": suite(9, 10)})
    lossy: dict[TaskKind, list[bool]] = {"json": suite(9, 10, lose=1), "tools": suite(9, 10)}
    q4 = model("q4_K_M", outcomes=lossy)
    verdict = judge(report(reference, q4))
    assert verdict.cases_needed is not None
    assert verdict.cases_needed > 34
    assert verdict.remedy == (
        f"About {verdict.cases_needed} cases per suite would decide. That is more than the "
        "built-in suites hold, so add your own with --prompts."
    )


def test_remedy_for_a_prompts_file_names_the_file() -> None:
    q4 = model("q4_K_M", klds=prompt_klds(0.02, 3))
    synthetic = report(reference_q8(), q4)
    mine = dataclasses.replace(
        synthetic, settings=dataclasses.replace(synthetic.settings, prompts_file="mine.jsonl")
    )
    assert judge(mine).remedy == "About 10 scoring prompts would decide; add them to mine.jsonl."


def test_pooled_task_loss_is_avoid_even_when_kld_is_close() -> None:
    # No single suite is significant, but the suites together lose 9 of 80 cases net.
    reference = model("q8_0", outcomes={"json": suite(36, 40), "tools": suite(36, 40)})
    q3 = model(
        "q3_K_M",
        outcomes={"json": suite(36, 40, lose=5), "tools": suite(36, 40, lose=5, gain=1)},
        klds=prompt_klds(0.03, 20),
    )
    verdict = judge(report(reference, q3))
    (call,) = verdict.candidates
    assert call.status == "avoid"
    assert call.reasons[0].startswith("Task scores drop 11 points overall vs q8_0 (95% CI")
    assert verdict.remedy == "Try a larger quant against q8_0; nothing tested here is close."


def test_no_logit_and_no_reliable_tasks_is_inconclusive() -> None:
    q4 = model("q4_K_M", agreement=[0.9] * 10)
    verdict = judge(report(reference_q8(), q4))
    assert verdict.candidates[0].status == "inconclusive"
    assert verdict.headline == (
        "Keep q8_0 for now: no candidate has logit or reliable task results."
    )
    assert verdict.details == ()


def test_logit_without_per_prompt_results_cannot_prove_closeness() -> None:
    q4 = model("q4_K_M", klds=prompt_klds(0.02, 20))
    assert q4.logit is not None
    old = dataclasses.replace(q4, logit=dataclasses.replace(q4.logit, per_prompt=()))
    verdict = judge(report(reference_q8(), old))
    assert verdict.candidates[0].status == "inconclusive"
    assert verdict.candidates[0].reasons[0] == (
        "KLD 0.02, but the report has no per-prompt KLD to bound it."
    )
    assert verdict.headline == "Keep q8_0 for now: no candidate is shown to be close."
    assert verdict.details == (
        "q4_K_M looks closest: KLD 0.02, with no per-prompt results to bound it.",
    )


# Absolute rules: other candidates never change a status ----------------------------------


def test_status_does_not_depend_on_the_other_candidates() -> None:
    q2 = model("q2_K", outcomes=q4_tasks(lose=3), klds=prompt_klds(0.07, 12, jitter=0.4))
    q4 = model("q4_K_M", outcomes=q4_tasks(), klds=prompt_klds(0.029, 41), size=6_000_000_000)
    alone = judge(report(reference_q8(), q2))
    together = judge(report(reference_q8(), q2, q4))
    assert status_of(alone, "q2_K") == status_of(together, "q2_K")
    q2_alone = next(c for c in alone.candidates if c.label == STEM + "q2_K")
    q2_together = next(c for c in together.candidates if c.label == STEM + "q2_K")
    assert q2_alone.reasons == q2_together.reasons


def test_real_q2_k_gets_the_same_status_alone_and_with_q4_k_m() -> None:
    full = load_report(DATA / "default_run_v2.json")
    q2 = next(c for c in full.candidates if c.spec.label.endswith("q2_K"))
    alone = judge(dataclasses.replace(full, candidates=(q2,)))
    together = judge(full)
    statuses = {
        verdict.candidates[[c.label for c in verdict.candidates].index(q2.spec.label)].status
        for verdict in (alone, together)
    }
    assert statuses == {"inconclusive"}


# Real runs --------------------------------------------------------------------------------


def test_critic_single_q2_k_run_is_usable_with_its_caveats_in_view() -> None:
    # KLD 0.098 on 12 prompts: a measured loss, but under the 0.10 large bar, so usable when
    # nothing closer fits, and flagged as near that bar.
    verdict = judge(load_report(DATA / "critic_c2_single_q2k.json"))
    assert verdict.headline == (
        "No download is close to q8_0; q2_K has the smallest loss among the smaller downloads "
        "(moderate, KLD 0.098)."
    )
    (call,) = verdict.candidates
    assert call.status == "usable"
    assert call.near_bar
    assert verdict.details == (
        "q2_K: tools -33 unresolved (95% CI -62 to +6); rerun with --max-cases 30.",
        "q2_K has a moderate loss: KLD 0.098 (95% CI 0.083 to 0.117).",
    )
    assert call.caveats == (
        "near the large-loss bar; a rerun could change this",
        "tools -33 unresolved (95% CI -62 to +6); rerun with --max-cases 30",
    )
    assert verdict.remedy is not None
    assert verdict.remedy.startswith("Pass --max-size")


def test_critic_single_q2_k_with_a_budget_is_the_best_that_fits() -> None:
    full = load_report(DATA / "critic_c2_single_q2k.json")
    budget = dataclasses.replace(full.settings, max_size_bytes=350_000_000)
    verdict = judge(dataclasses.replace(full, settings=budget))
    assert verdict.headline == "Best that fits 350 MB: q2_K, moderate loss (KLD 0.098)."
    assert verdict.candidates[0].status == "usable"
    assert verdict.candidates[0].fits_budget is True
    # Ollama scores by text, and a usable candidate on text forcing is worth confirming.
    assert verdict.remedy == (
        "Confirm with llama-server (exact token ids) before ruling out a download."
    )


def test_critic_three_prompt_run_keeps_the_reference() -> None:
    verdict = judge(load_report(DATA / "critic_c2_three_prompts.json"))
    assert verdict.headline == "Keep q8_0 for now: no candidate is shown to be close on 3 prompts."
    # q2_K's mean KLD (0.108) is above the large bar, but three prompts are too few to call
    # it a large loss.
    assert [(c.label.rsplit("-", 1)[-1], c.status) for c in verdict.candidates] == [
        ("q4_K_M", "inconclusive"),
        ("q2_K", "inconclusive"),
    ]
    assert verdict.details == (
        "q4_K_M looks closest: KLD 0.023 on 3 prompts.",
        "q2_K is undecided: KLD 0.108 looks large on 3 prompts, too few to call it.",
    )
    assert verdict.remedy == "About 10 scoring prompts would decide; add them to mine.jsonl."
    assert verdict.cases_needed == 10


def test_default_run_recommends_q4_k_m_on_kld_with_honest_task_wording() -> None:
    # KLD is proven close (12 prompts, upper bound 0.035). The 24 task cases cannot bound a
    # small difference, but they show no significant loss, and the card says exactly that.
    verdict = judge(load_report(DATA / "default_run_v2.json"))
    # q2_K: mean 0.106 but the interval starts at 0.092, so a large loss is not shown.
    assert [(c.label.rsplit("-", 1)[-1], c.status) for c in verdict.candidates] == [
        ("q4_K_M", "recommended"),
        ("q2_K", "inconclusive"),
    ]
    assert verdict.headline == (
        "Run q4_K_M: 25% smaller than q8_0, close on logits (KLD 0.029, CI up to 0.035) on 12 "
        "prompts."
    )
    assert verdict.details[0] == (
        "q4_K_M: KLD 0.029 (95% CI 0.024 to 0.035) and no significant task loss vs q8_0 on 24 "
        "cases (95% CI -21 to +12)."
    )
    assert "within 10 points" not in " ".join(verdict.details)
    assert verdict.remedy is None


@pytest.mark.parametrize(
    "name",
    [
        "critic_c2_single_q2k.json",
        "critic_c2_three_prompts.json",
        "default_run_v2.json",
        "sample_report.json",
    ],
)
def test_real_runs_never_recommend_q2_k_or_claim_no_loss(name: str) -> None:
    verdict = judge(load_report(DATA / name))
    assert not verdict.headline.lower().startswith("run q2")
    assert all(c.status not in ("recommended", "ok") for c in verdict.candidates if "q2" in c.label)
    assert "no measurable loss" not in everything(verdict)
    assert judge(load_report(DATA / name)) == verdict


# Never better than the reference ----------------------------------------------------------


def test_significant_gain_is_a_warning_not_a_win() -> None:
    reference = model("q8_0", outcomes={"json": suite(20, 40)})
    q4 = model("q4_K_M", outcomes={"json": suite(20, 40, gain=12)}, klds=prompt_klds(0.03, 20))
    verdict = judge(report(reference, q4))
    warning = (
        "q4_K_M scored higher than the reference on json; usually a sign the suite is too "
        "small or the reference is quantized itself."
    )
    assert warning in verdict.details
    assert "better" not in everything(verdict)


def test_describe_delta_never_shows_a_noisy_gain() -> None:
    def delta(estimate: float, *, significant: bool) -> TaskDelta:
        interval = Interval(estimate, estimate - 10, estimate + 10)
        return TaskDelta("json", 40, 0.9, 0.85, interval, significant, reference_reliable=True)

    assert describe_delta(delta(5.0, significant=False)) == "= reference (within noise)"
    assert describe_delta(delta(-5.0, significant=False)) == "-5 points (within noise)"
    assert describe_delta(delta(-30.0, significant=True)) == "-30 points (95% CI -40 to -20)"
    assert describe_delta(delta(30.0, significant=True)).startswith("+30 points vs reference;")


# Findings ---------------------------------------------------------------------------------


def test_same_weights_finding() -> None:
    twin = model("q8_0-copy", klds=prompt_klds(0.0, 10), weights_id="q8")
    other = model("q4_K_M", klds=prompt_klds(0.03, 10), weights_id="q4")
    verdict = judge(report(reference_q8(), twin, other))
    (finding,) = verdict.findings
    assert finding.finding.check == "same-weights"
    assert finding.finding.severity == "warn"
    assert finding.finding.message == "q8_0-copy serves the same weights as the reference"
    assert finding.labels == (STEM + "q8_0-copy",)
    assert finding.affects_scores


def test_context_finding_does_not_affect_short_prompts() -> None:
    warn = PreflightFinding("context", "warn", "context window is 2048 tokens")
    a = model("q4_K_M", klds=prompt_klds(0.03, 10), context_length=2048, preflight=(warn,))
    b = model("q5_K_M", klds=prompt_klds(0.02, 10), context_length=4096, preflight=(warn,))
    (finding,) = judge(report(model("q8_0"), a, b, longest=900)).findings
    assert finding.labels == (STEM + "q4_K_M", STEM + "q5_K_M")
    assert not finding.affects_scores
    assert finding.impact == (
        "does not affect these scores: the longest prompt is ~900 tokens and the context "
        "window is 2048"
    )


@pytest.mark.parametrize(
    ("longest", "context", "impact"),
    [
        (3000, 2048, "may affect scores: the longest prompt is ~3000 tokens but the context"),
        (None, 2048, "may affect scores: the length of the longest prompt is not known"),
        (900, None, "may affect scores: the context window is not known"),
    ],
)
def test_context_finding_affects_scores_when_unknown_or_too_small(
    longest: int | None, context: int | None, impact: str
) -> None:
    warn = PreflightFinding("context", "warn", "context window is small")
    a = model("q4_K_M", klds=prompt_klds(0.03, 10), context_length=context, preflight=(warn,))
    (finding,) = judge(report(model("q8_0"), a, longest=longest)).findings
    assert finding.affects_scores
    assert finding.impact.startswith(impact)


def test_findings_drop_warn_when_same_check_failed_and_template_always_counts() -> None:
    warn = PreflightFinding("context", "warn", "window small")
    fail = PreflightFinding("context", "fail", "front of long prompts is being dropped")
    template = PreflightFinding("template", "warn", "embedded chat template differs")
    ok = PreflightFinding("logprobs", "ok", "logprobs available")
    a = model("q4_K_M", klds=prompt_klds(0.03, 10), preflight=(warn, fail, template, ok))
    findings = judge(report(model("q8_0"), a, longest=None)).findings
    assert [f.finding for f in findings] == [fail, template]
    assert all(f.affects_scores for f in findings)


# Ranking, labels, edge cases --------------------------------------------------------------


def test_ranking_by_status_then_kld_then_size_with_failed_last() -> None:
    broken = model("iq1_S", errors=("connection refused",))
    q2 = model("q2_K", klds=prompt_klds(0.4, 20), size=3_000_000_000)
    q5 = model("q5_K_M", klds=prompt_klds(0.012, 20), size=5_500_000_000)
    q4 = model("q4_K_M", klds=prompt_klds(0.03, 20), size=4_700_000_000)
    q6 = model("q6_K", klds=prompt_klds(0.008, 4), size=6_600_000_000)
    q4s = model("q4_K_S", klds=prompt_klds(0.008, 4), size=4_400_000_000)
    verdict = judge(report(reference_q8(), broken, q2, q6, q4, q5, q4s))
    names = [(c.label.removeprefix(STEM), c.status, c.rank) for c in verdict.candidates]
    assert names == [
        ("q4_K_M", "recommended", 1),
        ("q5_K_M", "ok", 2),
        ("q4_K_S", "inconclusive", 3),
        ("q6_K", "inconclusive", 4),
        ("q2_K", "avoid", 5),
        ("iq1_S", "failed", None),
    ]
    assert verdict.candidates[5].reasons == ("No metrics: connection refused.",)
    assert verdict.details[-1] == "iq1_S produced no metrics; see the errors."


def test_edge_cases_without_results() -> None:
    assert judge(report(model("q8_0"))).headline == "No candidates were evaluated."
    failed = judge(report(model("q8_0"), model("q4_K_M", errors=("boom",))))
    assert failed.headline == "No candidate produced metrics; see the errors for each candidate."


def test_display_labels_drop_shared_stem_only_when_long_enough() -> None:
    names = display_labels(report(model("q8_0"), model("q4_K_M")))
    assert names == {STEM + "q8_0": "q8_0", STEM + "q4_K_M": "q4_K_M"}
    short = dataclasses.replace(
        report(model("q8_0")),
        reference=dataclasses.replace(
            model("x"), spec=CandidateSpec("ollama", "http://fake", "m:a", "m:a")
        ),
        candidates=(
            dataclasses.replace(
                model("y"), spec=CandidateSpec("ollama", "http://fake", "m:b", "m:b")
            ),
        ),
    )
    assert display_labels(short) == {"m:a": "m:a", "m:b": "m:b"}


@pytest.mark.parametrize(
    ("kld", "band"),
    [
        (0.0, "near-lossless"),
        (0.0099, "near-lossless"),
        (0.01, "small"),
        (0.0399, "small"),
        (0.04, "moderate"),
        (0.099, "moderate"),
        (0.1, "large"),
        (3.0, "large"),
    ],
)
def test_kld_bands(kld: float, band: str) -> None:
    assert kld_band(kld) == band


def test_band_edges_are_the_rule_thresholds() -> None:
    assert kld_band(NEAR_LOSSLESS_KLD) == "small"
    assert kld_band(CLOSE_KLD) == "moderate"
    assert kld_band(LARGE_KLD) == "large"


def test_judge_is_deterministic() -> None:
    fixture = load_report(FIXTURE)
    assert judge(fixture) == judge(fixture)


def test_fixture_verdict() -> None:
    verdict = judge(load_report(FIXTURE))
    assert verdict.headline == (
        "Run q4_k_m: 69% smaller than bf16, close on logits (KLD 0.029, CI up to 0.034) on 24 "
        "prompts."
    )
    assert [c.status for c in verdict.candidates] == ["recommended", "ok", "avoid", "failed"]
    assert verdict.details[0] == (
        "q4_k_m: KLD 0.029 (95% CI 0.024 to 0.034) and task scores within 10 points of bf16 on "
        "100 cases."
    )
    assert verdict.details[1] == "q8_0 is also close to bf16 but 74% larger."
    assert verdict.details[2].startswith("Avoid q2_k: code drops 40 points vs bf16 (95% CI")
    (context,) = verdict.findings
    assert not context.affects_scores


def test_headlines_stay_short() -> None:
    for name in ("critic_c2_single_q2k", "critic_c2_three_prompts", "default_run_v2"):
        assert len(judge(load_report(DATA / f"{name}.json")).headline) <= 110
    assert len(judge(load_report(FIXTURE)).headline) <= 110


# Size budget ------------------------------------------------------------------------------


def with_budget(synthetic: Report, budget: int) -> Report:
    return dataclasses.replace(
        synthetic, settings=dataclasses.replace(synthetic.settings, max_size_bytes=budget)
    )


def budget_lineup() -> tuple[CandidateResult, ...]:
    """q5_K_M close at 7.1 GB, q4_K_M usable at 5.4 GB, q3_K_M usable at 4.3 GB."""
    return (
        model("q5_K_M", klds=prompt_klds(0.02, 30), size=7_100_000_000),
        model("q4_K_M", klds=prompt_klds(0.06, 30), size=5_400_000_000),
        model("q3_K_M", klds=prompt_klds(0.08, 30), size=4_300_000_000),
    )


def test_budget_picks_the_smallest_close_download_that_fits() -> None:
    q8 = model("q8_0", klds=prompt_klds(0.004, 30), size=8_500_000_000)
    q5 = model("q5_K_M", klds=prompt_klds(0.02, 30), size=5_500_000_000)
    reference = model("bf16", size=16_000_000_000)
    verdict = judge(with_budget(report(reference, q8, q5), 6_000_000_000))
    assert status_of(verdict, "q5_K_M") == "recommended"
    assert status_of(verdict, "q8_0") == "ok"
    fits = {c.label.removeprefix(STEM): c.fits_budget for c in verdict.candidates}
    assert fits == {"q5_K_M": True, "q8_0": False}
    assert "q8_0 is close but needs 8.5 GB." in verdict.details
    assert verdict.remedy is None


def test_budget_with_no_close_fit_names_the_best_usable_that_fits() -> None:
    verdict = judge(with_budget(report(reference_q8(), *budget_lineup()), 6_000_000_000))
    assert verdict.headline == "Best that fits 6 GB: q4_K_M, moderate loss (KLD 0.06)."
    statuses = {c.label.removeprefix(STEM): c.status for c in verdict.candidates}
    # The best that fits keeps its usable status: it is not relabeled as recommended.
    assert statuses == {"q5_K_M": "ok", "q4_K_M": "usable", "q3_K_M": "usable"}
    assert "q5_K_M is close but needs 7.1 GB." in verdict.details
    q5 = next(c for c in verdict.candidates if c.label.endswith("q5_K_M"))
    assert q5.reasons[0] == "Close to q8_0 but needs 7.1 GB, over the 6 GB budget."
    assert verdict.remedy == "Rerun with --max-size 7.1GB to run q5_K_M, which is close."


def test_budget_that_nothing_fits_says_what_is_close_and_how_big_it_is() -> None:
    verdict = judge(with_budget(report(reference_q8(), *budget_lineup()), 4_000_000_000))
    assert verdict.headline == (
        "Nothing that fits 4 GB is close to q8_0; q5_K_M is close but needs 7.1 GB."
    )
    assert all(c.fits_budget is False for c in verdict.candidates)
    assert not any(c.status == "recommended" for c in verdict.candidates)
    assert "q5_K_M is close but needs 7.1 GB." not in verdict.details
    assert verdict.remedy == "Rerun with --max-size 7.1GB to run q5_K_M, which is close."


def test_budget_ignores_candidates_of_unknown_size() -> None:
    q5 = model("q5_K_M", klds=prompt_klds(0.02, 30))
    verdict = judge(with_budget(report(reference_q8(), q5), 6_000_000_000))
    (call,) = verdict.candidates
    assert call.status == "ok"
    assert call.fits_budget is None
    assert verdict.headline == (
        "Nothing that fits 6 GB is close to q8_0; q5_K_M is close but its size is unknown."
    )


def test_without_a_budget_nothing_close_but_usable_names_the_smallest_loss() -> None:
    verdict = judge(report(reference_q8(), *budget_lineup()[1:]))
    assert verdict.headline == (
        "No download is close to q8_0; q4_K_M has the smallest loss among the smaller "
        "downloads (moderate, KLD 0.06)."
    )
    assert all(c.fits_budget is None for c in verdict.candidates)
    assert verdict.remedy is not None
    assert verdict.remedy.startswith("Pass --max-size")


def test_usable_with_an_undecided_candidate_says_nothing_is_shown_close_yet() -> None:
    thin = model("q5_K_M", klds=prompt_klds(0.02, 4), size=6_000_000_000)
    verdict = judge(report(reference_q8(), thin, budget_lineup()[1]))
    assert verdict.headline.startswith("No download is shown to be close to q8_0; q4_K_M has")
    # More prompts could prove q5_K_M close, so that rerun comes first.
    assert verdict.remedy == "Rerun with --max-cases 10 to decide."


def test_format_size() -> None:
    assert format_size(6_000_000_000) == "6 GB"
    assert format_size(6_250_000_000) == "6.25 GB"
    assert format_size(397_821_319) == "398 MB"
    assert format_size(7_061_000_000, round_up=True) == "7.07 GB"
    assert format_size(512) == "512 bytes"


# Caveats and next steps --------------------------------------------------------------------


def test_unresolved_task_loss_is_a_caveat_right_under_the_headline() -> None:
    # tools: 6 of 40 reference passes lost, 1 gained: -12 points but not significant.
    outcomes: dict[TaskKind, list[bool]] = {
        "json": suite(36, 40),
        "tools": suite(36, 40, lose=6, gain=1),
    }
    reference = model("q8_0", outcomes={"json": suite(36, 40), "tools": suite(36, 40)})
    q4 = model("q4_K_M", outcomes=outcomes, klds=prompt_klds(0.03, 30), size=6_000_000_000)
    verdict = judge(report(reference, q4))
    (call,) = verdict.candidates
    tools = next(d for d in call.task_deltas if d.kind == "tools")
    assert not tools.significant
    assert tools.delta.estimate == pytest.approx(-12.5)
    assert call.status == "recommended"
    (caveat,) = call.caveats
    # 32 built-in tools cases cannot settle it, so the caveat asks for more cases.
    assert caveat == "tools -12 unresolved (95% CI -26 to 0); about 70 tools cases would settle it"
    assert verdict.details[0] == f"q4_K_M: {caveat}."
    assert verdict.details[1].startswith("q4_K_M: KLD 0.03 (95% CI ")


def test_unresolved_caveat_beyond_the_built_in_suite_asks_for_cases() -> None:
    reference = model("q8_0", outcomes={"tools": suite(8, 10)})
    q4 = model("q4_K_M", outcomes={"tools": suite(8, 10, lose=2)}, klds=prompt_klds(0.03, 30))
    (call,) = judge(report(reference, q4)).candidates
    assert call.caveats[-1].startswith("tools -20 unresolved (95% CI ")
    needed = call.caveats[-1].rsplit("; ", 1)[1]
    assert needed.startswith(("rerun with --max-cases ", "about "))


def text_forced(result: CandidateResult) -> CandidateResult:
    assert result.logit is not None
    return dataclasses.replace(
        result, logit=dataclasses.replace(result.logit, exact_token_ids=False)
    )


def test_non_latin_prompts_on_text_forcing_point_to_llama_server() -> None:
    q4 = text_forced(model("q4_K_M", klds=prompt_klds(0.3, 20), size=5_000_000_000))
    synthetic = report(reference_q8(), q4)
    chinese = [ScoringPrompt(f"p{i}", "量化模型的输出 abc") for i in range(3)]
    verdict = judge(synthetic, scoring=chinese)
    note = "Most scoring prompts are not Latin script; text forcing may read higher there."
    assert note in verdict.details
    assert verdict.remedy == (
        "Confirm with llama-server (exact token ids) before ruling out a download."
    )
    english = [ScoringPrompt("p", "The quantized model's output")]
    assert note not in " ".join(judge(synthetic, scoring=english).details)


def test_non_latin_note_carries_the_llama_server_step_when_a_pick_exists() -> None:
    q2 = text_forced(model("q2_K", klds=prompt_klds(0.3, 20), size=3_000_000_000))
    q5 = model("q5_K_M", klds=prompt_klds(0.02, 20), size=5_600_000_000)
    russian = [ScoringPrompt("p", "Квантование")]
    verdict = judge(report(reference_q8(), q2, q5), scoring=russian)
    assert verdict.remedy is None
    assert verdict.details[-1] == (
        "Most scoring prompts are not Latin script; text forcing may read higher there. "
        "Confirm with llama-server (exact token ids) before ruling out a download."
    )


def test_mostly_non_latin_counts_letters_only() -> None:
    assert mostly_non_latin(["日本語の文章"])
    assert not mostly_non_latin(["Café crème brûlée"])
    assert not mostly_non_latin(["def f(x): return x  # λ"])
    assert not mostly_non_latin(["12345 !?", ""])
    assert not mostly_non_latin([])


def test_every_non_recommended_verdict_has_a_next_step() -> None:
    cases = [
        report(reference_q8(), model("q2_K", klds=prompt_klds(0.3, 20))),
        report(reference_q8(), model("q4_K_M", agreement=[0.9] * 10)),
        report(reference_q8(), *budget_lineup()[1:]),
        report(reference_q8(), model("q3_K_M", klds=prompt_klds(0.09, 5))),
    ]
    for synthetic in cases:
        verdict = judge(synthetic)
        assert not any(c.status == "recommended" for c in verdict.candidates)
        assert verdict.remedy
