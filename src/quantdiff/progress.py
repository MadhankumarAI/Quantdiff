"""Progress displays for long runs.

`make_progress` picks one of two displays for a stream:

- On a terminal, one status line is redrawn in place with a bar, percentage, ETA and the
  current step. Each finished step leaves a permanent `done` line so the scrollback reads
  as a short log. A step is finished only when the next one starts, so a step that fails
  never gets a `done` line, and connecting to every server counts as one step whose
  `done` line says how many servers answered.
- On a pipe or in CI, plain lines are printed when the step changes and at most once per
  tenth of the run, so logs stay short and contain no carriage returns.

The ETA divides the units left by the rate measured in the current step. Work units are
weighted a priori and a scoring unit costs far less than a case, so a rate averaged over
the whole run is badly off; the recent rate of the step at hand is a much better guide.

Output is ASCII only so it renders on any Windows console code page.
"""

from __future__ import annotations

import os
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, TextIO

from quantdiff.types import ProgressEvent

Clock = Callable[[], float]
ProgressFn = Callable[[ProgressEvent], None]

BAR_WIDTH: Final = 20
REDRAW_INTERVAL_SECONDS: Final = 0.1
ETA_MIN_UNITS: Final = 5
ETA_MIN_SECONDS: Final = 10.0
"""Units and seconds into a step before its ETA is shown; earlier estimates swing wildly."""
RATE_WINDOW_EVENTS: Final = 10
"""The measured rate weights roughly the last this many events."""
LOG_STEPS: Final = 10
"""Non-interactive output prints at most once per 1/LOG_STEPS of the run."""


def make_progress(
    stream: TextIO, *, interactive: bool | None = None, clock: Clock = time.monotonic
) -> ProgressDisplay:
    """Return a progress callback writing to `stream`.

    `interactive` None means a live status line when `stream` is a terminal and TERM is not
    "dumb", and plain log lines otherwise. Call `close()` on the result before printing an
    error so a half-drawn status line does not run into it.
    """
    if interactive is None:
        interactive = stream.isatty() and os.environ.get("TERM") != "dumb"
    if interactive:
        return StatusLine(stream, clock)
    return StepLog(stream, clock)


def format_duration(seconds: float) -> str:
    """Format as 45s, 3m05s or 1h02m."""
    whole = max(0, round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class RateEstimator:
    """Units per second within one step, smoothed over the recent events.

    Units and seconds are smoothed separately and divided, so a burst of events that
    arrive together (a cache hit) does not count as an infinitely fast interval.
    """

    def __init__(self) -> None:
        self._alpha = 2 / (RATE_WINDOW_EVENTS + 1)
        self._started = 0.0
        self._start_units = 0
        self._last = 0.0
        self._last_units = 0
        self._units = 0.0
        self._seconds = 0.0

    def start(self, now: float, completed: int) -> None:
        """Begin a new step whose work started at `now` with `completed` units done."""
        self._started = self._last = now
        self._start_units = self._last_units = completed
        self._units = self._seconds = 0.0

    def update(self, now: float, completed: int) -> None:
        units = max(0, completed - self._last_units)
        seconds = max(0.0, now - self._last)
        self._units += self._alpha * (units - self._units)
        self._seconds += self._alpha * (seconds - self._seconds)
        self._last, self._last_units = now, completed

    def remaining(self, completed: int, total: int) -> float | None:
        """Seconds left at the recent rate, or None while too little of the step is done."""
        if total <= 0 or completed >= total:
            return None
        warming_up = (
            self._last - self._started < ETA_MIN_SECONDS
            or self._last_units - self._start_units < ETA_MIN_UNITS
        )
        if warming_up or self._units <= 0 or self._seconds <= 0:
            return None
        return (total - completed) * self._seconds / self._units


class ProgressDisplay(ABC):
    """Shared step tracking: notices when the (phase, model) step changes, times it and
    measures its rate for the ETA."""

    def __init__(self, stream: TextIO, clock: Clock) -> None:
        self._stream = stream
        self._clock = clock
        self._run_started: float | None = None
        self._step: tuple[str, str] | None = None
        self._step_started = 0.0
        self._step_events = 0
        self._rate = RateEstimator()
        self._last_event: tuple[float, int] | None = None

    def __call__(self, event: ProgressEvent) -> None:
        now = self._clock()
        if self._run_started is None:
            self._run_started = now
        if event.phase == "done":
            self._finish_step(now)
            self._finish_run(now - self._run_started)
            return
        step = _step_key(event)
        new_step = step != self._step
        if new_step:
            self._finish_step(now)
            self._step = step
            self._step_started = now
            self._step_events = 0
            # Events report work already done, so the step's first unit began at the
            # previous event.
            self._rate.start(*(self._last_event or (now, event.completed)))
        self._step_events += 1
        self._rate.update(now, event.completed)
        self._last_event = (now, event.completed)
        if new_step:
            self._start_step(event, now)
        else:
            self._advance(event, now)

    @abstractmethod
    def close(self) -> None:
        """Leave the stream at the start of a clean line. Safe to call more than once."""

    def _remaining(self, event: ProgressEvent) -> float | None:
        return self._rate.remaining(event.completed, event.total)

    def _finish_step(self, now: float) -> None:
        if self._step is None:
            return
        self.close()
        phase, model = self._step
        if phase == "connect":
            # The runner sends one connect event per server once that server has answered.
            label = f"connect {self._step_events} server{'' if self._step_events == 1 else 's'}"
        else:
            label = f"{phase} {model}".rstrip()
        self._write_line(f"  done  {label}  ({format_duration(now - self._step_started)})")
        self._step = None

    def _finish_run(self, elapsed: float) -> None:
        self.close()
        self._write_line(f"Finished in {format_duration(elapsed)}")

    @abstractmethod
    def _start_step(self, event: ProgressEvent, now: float) -> None:
        """Show the first event of a new step."""

    @abstractmethod
    def _advance(self, event: ProgressEvent, now: float) -> None:
        """Show a later event of the current step, if it is worth showing."""

    def _write_line(self, text: str) -> None:
        self._stream.write(text + "\n")
        self._stream.flush()


class StatusLine(ProgressDisplay):
    """A single line redrawn in place, for terminals."""

    def __init__(self, stream: TextIO, clock: Clock) -> None:
        super().__init__(stream, clock)
        self._drawn_width = 0
        self._last_draw: float | None = None

    def close(self) -> None:
        if self._drawn_width:
            # Overwrite with spaces rather than an ANSI erase: legacy Windows consoles
            # print escape sequences literally unless virtual terminal mode is enabled.
            self._stream.write("\r" + " " * self._drawn_width + "\r")
            self._stream.flush()
            self._drawn_width = 0

    def _start_step(self, event: ProgressEvent, now: float) -> None:
        self._draw(event, now)

    def _advance(self, event: ProgressEvent, now: float) -> None:
        if self._last_draw is None or now - self._last_draw >= REDRAW_INTERVAL_SECONDS:
            self._draw(event, now)

    def _draw(self, event: ProgressEvent, now: float) -> None:
        width = max(1, shutil.get_terminal_size().columns - 1)
        line = status_text(event, self._remaining(event))[:width]
        padding = " " * max(0, self._drawn_width - len(line))
        self._stream.write("\r" + line + padding)
        self._stream.flush()
        self._drawn_width = len(line)
        self._last_draw = now


class StepLog(ProgressDisplay):
    """Plain lines for pipes and CI logs."""

    def __init__(self, stream: TextIO, clock: Clock) -> None:
        super().__init__(stream, clock)
        self._logged_step = -1

    def close(self) -> None:
        """Nothing to clean up: every log line already ends with a newline."""

    def _start_step(self, event: ProgressEvent, now: float) -> None:
        self._log(event)

    def _advance(self, event: ProgressEvent, now: float) -> None:
        if _log_step(event) > self._logged_step:
            self._log(event)

    def _log(self, event: ProgressEvent) -> None:
        self._logged_step = _log_step(event)
        parts = [f"[{_percent(event):>4}]", event.phase, event.model, event.detail]
        remaining = self._remaining(event)
        if remaining is not None:
            parts.append(f"(ETA {format_duration(remaining)})")
        self._write_line("  ".join(part for part in parts if part))


def status_text(event: ProgressEvent, remaining: float | None) -> str:
    """The interactive status line, before truncation to the terminal width."""
    fraction = _fraction(event)
    filled = 0 if fraction is None else round(fraction * BAR_WIDTH)
    parts = [f"[{'#' * filled}{'.' * (BAR_WIDTH - filled)}]", f"{_percent(event):>4}"]
    if remaining is not None:
        parts.append(f"ETA {format_duration(remaining)}")
    parts.extend((event.model, f"{event.phase} {event.detail}".rstrip()))
    return "  ".join(part for part in parts if part)


def _step_key(event: ProgressEvent) -> tuple[str, str]:
    # The runner connects to every server before any work; a done line per server would
    # time the wrong one, since each event arrives after its connection attempt.
    if event.phase == "connect":
        return ("connect", "")
    return (event.phase, event.model)


def _fraction(event: ProgressEvent) -> float | None:
    if event.total <= 0:
        return None
    return min(1.0, max(0.0, event.completed / event.total))


def _percent(event: ProgressEvent) -> str:
    fraction = _fraction(event)
    return "--%" if fraction is None else f"{int(fraction * 100)}%"


def _log_step(event: ProgressEvent) -> int:
    fraction = _fraction(event)
    return 0 if fraction is None else int(fraction * LOG_STEPS)
