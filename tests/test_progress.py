from __future__ import annotations

import io
import os

import pytest

from quantdiff.progress import (
    ETA_MIN_SECONDS,
    RateEstimator,
    StatusLine,
    StepLog,
    format_duration,
    make_progress,
)
from quantdiff.types import ProgressEvent, ProgressPhase

REF = "qwen2.5:0.5b-instruct-q8_0"
CAND = "q4_K_M"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((120, 40))
    )


def _event(
    phase: ProgressPhase, model: str, completed: int, total: int = 100, detail: str = ""
) -> ProgressEvent:
    return ProgressEvent(phase=phase, model=model, completed=completed, total=total, detail=detail)


def _visible(output: str) -> list[str]:
    """What a terminal shows: each carriage return rewinds to the start of the line."""
    lines = []
    for line in output.split("\n"):
        screen = ""
        for chunk in line.split("\r"):
            screen = chunk + screen[len(chunk) :]
        lines.append(screen.rstrip())
    return lines


# formatting -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (45.4, "45s"), (59.6, "1m00s"), (185, "3m05s"), (3720, "1h02m"), (-3, "0s")],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


# ETA ---------------------------------------------------------------------------------------


def test_rate_waits_for_enough_seconds_and_units() -> None:
    rate = RateEstimator()
    rate.start(0.0, 0)
    rate.update(4.0, 40)
    assert rate.remaining(40, 100) is None
    rate.update(12.0, 44)
    assert rate.remaining(44, 100) is not None
    rate.start(12.0, 44)
    rate.update(30.0, 48)
    assert rate.remaining(48, 100) is None


def test_rate_estimate_converges_after_a_slow_start() -> None:
    """A model load makes the first unit slow; the ETA follows the steady rate after it."""
    rate = RateEstimator()
    rate.start(0.0, 0)
    now, completed = 20.0, 1
    rate.update(now, completed)
    errors = []
    while completed < 60:
        now += 2.0
        completed += 1
        rate.update(now, completed)
        remaining = rate.remaining(completed, 100)
        if remaining is not None:
            errors.append(abs(remaining - 2.0 * (100 - completed)) / (2.0 * (100 - completed)))
    assert errors[0] < 1.5
    assert errors[-1] < 0.01
    assert errors == sorted(errors, reverse=True)


def test_rate_ignores_the_cost_of_earlier_steps() -> None:
    """The bug this replaced: a slow reference made the ETA of fast scoring 4-6x too long."""
    rate = RateEstimator()
    rate.start(300.0, 50)
    for second in range(1, 21):
        rate.update(300.0 + second, 50 + 10 * second)
    assert rate.remaining(250, 1000) == pytest.approx(75.0)


def test_rate_handles_bursts_and_finished_runs() -> None:
    rate = RateEstimator()
    rate.start(0.0, 0)
    for completed in range(1, 11):
        rate.update(0.0, completed)
    assert rate.remaining(10, 100) is None
    rate.update(ETA_MIN_SECONDS, 20)
    remaining = rate.remaining(20, 100)
    assert remaining is not None
    assert 40.0 < remaining < 80.0
    assert rate.remaining(100, 100) is None
    assert rate.remaining(20, 0) is None


# mode selection ---------------------------------------------------------------------------


def test_auto_mode_follows_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM", raising=False)
    assert isinstance(make_progress(TtyStream()), StatusLine)
    assert isinstance(make_progress(io.StringIO()), StepLog)
    monkeypatch.setenv("TERM", "dumb")
    assert isinstance(make_progress(TtyStream()), StepLog)


# interactive ------------------------------------------------------------------------------


def test_status_line_shows_bar_eta_model_and_detail(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("cases", CAND, 0, detail="0/30"))
    clock.now += 106
    show(_event("cases", CAND, 47, detail="12/30"))
    assert _visible(stream.getvalue()) == [
        "[#########...........]   47%  ETA 2m00s  q4_K_M  cases 12/30"
    ]
    assert "\n" not in stream.getvalue()


def test_status_line_hides_eta_until_enough_units(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("reference", REF, 0))
    clock.now += 1
    show(_event("reference", REF, 2))
    clock.now += 20
    show(_event("reference", REF, 4))
    assert "ETA" not in stream.getvalue()


def test_status_line_measures_eta_per_step(clock: FakeClock) -> None:
    """Ten slow reference cases do not inflate the ETA of fast scoring that follows."""
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    for completed in range(11):
        clock.now += 10
        show(_event("reference", REF, completed, total=1000))
    for completed in range(20, 211, 10):
        clock.now += 1
        show(_event("scoring", CAND, completed, total=1000))
    assert _visible(stream.getvalue())[-1].startswith("[####................]   21%  ETA 1m19s")


def test_status_line_redraws_at_most_ten_times_a_second(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("cases", CAND, 1))
    for completed in range(2, 50):
        clock.now += 0.001
        show(_event("cases", CAND, completed))
    assert stream.getvalue().count("\r") == 1
    clock.now += 0.1
    show(_event("cases", CAND, 50))
    assert stream.getvalue().count("\r") == 2
    assert _visible(stream.getvalue())[-1].startswith("[##########..........]   50%")


def test_status_line_is_truncated_to_the_terminal(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((30, 24))
    )
    stream = io.StringIO()
    make_progress(stream, interactive=True, clock=clock)(_event("reference", REF, 10))
    assert _visible(stream.getvalue()) == ["[##..................]   10%"]


def test_shorter_redraw_erases_the_previous_line(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("cases", CAND, 1, detail="a long detail text"))
    clock.now += 1
    show(_event("cases", CAND, 2))
    assert _visible(stream.getvalue()) == ["[....................]    2%  q4_K_M  cases"]


def test_step_changes_leave_done_lines_and_summary(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("connect", "", 0))
    clock.now += 2
    show(_event("reference", REF, 1))
    clock.now += 38
    show(_event("cases", CAND, 60))
    clock.now += 125
    show(_event("done", "", 100))
    assert _visible(stream.getvalue()) == [
        "  done  connect 1 server  (2s)",
        f"  done  reference {REF}  (38s)",
        "  done  cases q4_K_M  (2m05s)",
        "Finished in 2m45s",
        "",
    ]


def test_failed_step_gets_no_done_line(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("connect", "qwen2.5", 1))
    clock.now += 11
    show(_event("connect", CAND, 2))
    show.close()
    assert "done" not in stream.getvalue()


def test_connecting_to_every_server_is_one_step(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=False, clock=clock)
    show(_event("connect", REF, 1))
    clock.now += 1
    show(_event("connect", CAND, 2))
    clock.now += 1
    show(_event("reference", REF, 3))
    assert stream.getvalue().splitlines() == [
        f"[  1%]  connect  {REF}",
        "  done  connect 2 servers  (2s)",
        f"[  3%]  reference  {REF}",
    ]


def test_connect_done_line_counts_the_servers(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    for completed, label in enumerate((REF, CAND, "q2_K"), start=1):
        show(_event("connect", label, completed))
    clock.now += 3
    show(_event("reference", REF, 4))
    screen = _visible(stream.getvalue())
    assert screen[0] == "  done  connect 3 servers  (3s)"
    assert "done  connect  (" not in stream.getvalue()


def test_zero_total_and_overshoot_are_safe(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("connect", "", 0, total=0))
    clock.now += 1
    show(_event("cases", CAND, 130))
    screen = _visible(stream.getvalue())
    assert screen[0] == "  done  connect 1 server  (1s)"
    assert screen[1].startswith("[####################]  100%  q4_K_M")


def test_close_clears_a_half_drawn_line(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    show(_event("cases", CAND, 3))
    show.close()
    show.close()
    assert _visible(stream.getvalue()) == [""]
    assert stream.getvalue().endswith("\r")


def test_output_is_ascii(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=True, clock=clock)
    for completed in range(0, 101, 7):
        clock.now += 1
        show(_event("cases", CAND, completed))
    show(_event("done", "", 100))
    assert stream.getvalue().isascii()


# non-interactive --------------------------------------------------------------------------


def test_log_prints_steps_and_tenths_without_carriage_returns(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=False, clock=clock)
    show(_event("connect", "", 0))
    clock.now += 1
    for completed in range(41):
        show(_event("reference", REF, completed, detail=f"{completed}/40"))
        clock.now += 1
    show(_event("cases", CAND, 40, detail="0/60"))
    for completed in range(41, 101):
        clock.now += 1
        show(_event("cases", CAND, completed, detail=f"{completed - 40}/60"))
    show(_event("done", "", 100))

    output = stream.getvalue()
    assert "\r" not in output
    assert output.splitlines() == [
        "[  0%]  connect",
        "  done  connect 1 server  (1s)",
        f"[  0%]  reference  {REF}  0/40",
        f"[ 10%]  reference  {REF}  10/40  (ETA 1m33s)",
        f"[ 20%]  reference  {REF}  20/40  (ETA 1m20s)",
        f"[ 30%]  reference  {REF}  30/40  (ETA 1m10s)",
        f"[ 40%]  reference  {REF}  40/40  (ETA 1m00s)",
        f"  done  reference {REF}  (41s)",
        "[ 40%]  cases  q4_K_M  0/60",
        "[ 50%]  cases  q4_K_M  10/60  (ETA 51s)",
        "[ 60%]  cases  q4_K_M  20/60  (ETA 40s)",
        "[ 70%]  cases  q4_K_M  30/60  (ETA 30s)",
        "[ 80%]  cases  q4_K_M  40/60  (ETA 20s)",
        "[ 90%]  cases  q4_K_M  50/60  (ETA 10s)",
        "[100%]  cases  q4_K_M  60/60",
        "  done  cases q4_K_M  (1m00s)",
        "Finished in 1m42s",
    ]


def test_log_with_zero_total_prints_only_step_changes(clock: FakeClock) -> None:
    stream = io.StringIO()
    show = make_progress(stream, interactive=False, clock=clock)
    for completed in range(5):
        show(_event("preflight", CAND, completed, total=0))
    assert stream.getvalue().splitlines() == ["[ --%]  preflight  q4_K_M"]
