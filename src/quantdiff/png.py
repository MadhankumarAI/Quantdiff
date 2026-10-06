"""PNG export of the HTML scorecard through an installed Chromium-family browser.

A headless browser renders the exact HTML card, so the PNG is pixel-identical to what the
card looks like when opened, and quantdiff keeps zero Python dependencies.

The browser cannot size a screenshot to its content, so rendering takes two launches:

1. Measure: a wrapper page loads the card into an iframe of the target width and, once it
   has loaded, writes the card's full height into a meta tag. ``--dump-dom`` prints the
   resulting DOM and the height is parsed from it. The iframe pins the layout width
   exactly, which a top-level page cannot do because ``--dump-dom`` subtracts window
   decorations from ``--window-size``. The card itself stays free of scripts.
2. Screenshot: the card is captured with a window of exactly that width and height.

Every launch uses a throwaway profile in a private temporary directory and forces the
light color scheme so the output does not depend on the user's settings.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Final

from quantdiff.errors import RenderError

__all__ = [
    "BROWSER_ENV",
    "find_browser",
    "max_height",
    "measure_command",
    "measure_page",
    "parse_height",
    "platform_candidates",
    "render_png",
    "screenshot_command",
]

logger = logging.getLogger(__name__)

BROWSER_ENV: Final = "QUANTDIFF_BROWSER"
MIN_WIDTH: Final = 320
MAX_WIDTH: Final = 4096
MIN_SCALE: Final = 0.5
MAX_SCALE: Final = 4.0
MAX_HEIGHT: Final = 20_000
"""Tallest card, in CSS pixels, that is screenshotted; anything taller is clipped."""
MAX_PNG_PIXELS: Final = 100_000_000
"""Largest image area, in device pixels, so a huge width and scale cannot exhaust memory."""
_STDERR_TAIL_CHARS: Final = 600
# --dump-dom shrinks the viewport by the window frame; the slack keeps the iframe unclipped.
_MEASURE_WINDOW_SLACK: Final = 120
_MEASURE_WINDOW_HEIGHT: Final = 200
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_HEIGHT_META: Final = re.compile(r'<meta name="quantdiff-height" content="(\d+)">')
_NO_BROWSER: Final = (
    "no Chromium-based browser found (Chrome, Edge, Brave or Chromium); "
    f"install one or set {BROWSER_ENV} to its path"
)

_WINDOWS_BROWSERS: Final = (
    ("Microsoft", "Edge", "Application", "msedge.exe"),
    ("Google", "Chrome", "Application", "chrome.exe"),
    ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
    ("Chromium", "Application", "chrome.exe"),
)
_WINDOWS_ROOTS: Final = ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData")
_MACOS_BROWSERS: Final = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
_LINUX_BROWSERS: Final = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "brave-browser",
)

_MEASURE_SCRIPT: Final = """
window.addEventListener("load", function () {
  var card = document.getElementById("card").contentDocument.documentElement;
  var meta = document.createElement("meta");
  meta.name = "quantdiff-height";
  meta.content = String(Math.ceil(card.scrollHeight));
  document.head.appendChild(meta);
});
"""


# Browser discovery ------------------------------------------------------------------------


def find_browser() -> Path | None:
    """Locate a Chromium-family browser, or return None when none is installed.

    The QUANTDIFF_BROWSER environment variable wins when set and must name an existing
    file. Otherwise the standard install locations for the current platform are probed.
    """
    explicit = os.environ.get(BROWSER_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RenderError(f"{BROWSER_ENV} is set to {explicit!r}, which is not a file")
        return path
    for candidate in platform_candidates(sys.platform, os.environ):
        if candidate.is_file():
            return candidate
    return None


def platform_candidates(platform: str, environ: Mapping[str, str]) -> list[Path]:
    """List the usual browser install paths for `platform`, most preferred first."""
    if platform == "win32":
        return _windows_candidates(environ)
    if platform == "darwin":
        return [Path(path) for path in _MACOS_BROWSERS]
    found = (shutil.which(name) for name in _LINUX_BROWSERS)
    return [Path(path) for path in found if path]


def _windows_candidates(environ: Mapping[str, str]) -> list[Path]:
    roots = [Path(environ[name]) for name in _WINDOWS_ROOTS if environ.get(name)]
    return [root.joinpath(*parts) for parts in _WINDOWS_BROWSERS for root in roots]


# Commands and pages -----------------------------------------------------------------------


def _common_flags(profile_dir: Path, scale: float) -> list[str]:
    return [
        "--headless=new",
        # A private profile keeps the user's browser profile, cookies and extensions out.
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-default-apps",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-component-update",
        "--mute-audio",
        "--hide-scrollbars",
        # 1 is Blink's light scheme; the card follows prefers-color-scheme otherwise.
        "--blink-settings=preferredColorScheme=1",
        f"--force-device-scale-factor={scale:g}",
    ]


def measure_command(
    browser: Path, page: Path, profile_dir: Path, *, width: int, scale: float
) -> list[str]:
    """Build the argv that prints the DOM of the measure page after it has loaded."""
    return [
        str(browser),
        *_common_flags(profile_dir, scale),
        f"--window-size={width + _MEASURE_WINDOW_SLACK},{_MEASURE_WINDOW_HEIGHT}",
        "--dump-dom",
        page.as_uri(),
    ]


def screenshot_command(
    browser: Path,
    page: Path,
    output: Path,
    profile_dir: Path,
    *,
    width: int,
    height: int,
    scale: float,
) -> list[str]:
    """Build the argv that screenshots `page` into `output` at exactly width x height."""
    return [
        str(browser),
        *_common_flags(profile_dir, scale),
        f"--window-size={width},{height}",
        f"--screenshot={output}",
        page.as_uri(),
    ]


def measure_page(card_html: str, width: int) -> str:
    """Wrap the card in an iframe `width` CSS pixels wide plus a script that records its height.

    The iframe is 1px tall so the card's scrollHeight is its content height, not the
    viewport's.
    """
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<style>html, body { margin: 0; } "
            f"iframe {{ display: block; border: 0; width: {width}px; height: 1px; }}</style>",
            "</head>",
            "<body>",
            f'<iframe id="card" srcdoc="{escape(card_html, quote=True)}"></iframe>',
            f"<script>{_MEASURE_SCRIPT}</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def parse_height(dom: str) -> int:
    """Read the card height that the measure page wrote into its DOM."""
    match = _HEIGHT_META.search(dom)
    if match is None:
        raise RenderError("the browser did not report the card height; is it Chromium-based?")
    height = int(match.group(1))
    if height <= 0:
        raise RenderError(f"the browser reported a card height of {height}px")
    return height


# Rendering --------------------------------------------------------------------------------


def render_png(
    html: str,
    out_path: Path,
    *,
    width: int = 1040,
    scale: float = 2.0,
    browser: Path | None = None,
    timeout_seconds: float = 60.0,
) -> Path:
    """Render `html` to a PNG at `out_path` and return that path.

    The image is `width * scale` pixels wide and exactly as tall as the page content
    (capped by max_height()). `browser` defaults to find_browser(). Raises
    RenderError when no browser is available, the browser fails or times out, or the
    output is not a PNG.
    """
    _check_size(width, scale)
    executable = browser if browser is not None else find_browser()
    if executable is None:
        raise RenderError(_NO_BROWSER)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="quantdiff-png-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        profile = workdir / "profile"
        measure = workdir / "measure.html"
        card = workdir / "card.html"
        shot = workdir / "card.png"
        measure.write_text(measure_page(html, width), encoding="utf-8")
        card.write_text(html, encoding="utf-8")

        command = measure_command(executable, measure, profile, width=width, scale=scale)
        height = parse_height(_run_browser(command, workdir, timeout_seconds))
        limit = max_height(width, scale)
        if height > limit:
            logger.warning("card is %dpx tall; clipping the PNG to %dpx", height, limit)
            height = limit
        command = screenshot_command(
            executable, card, shot, profile, width=width, height=height, scale=scale
        )
        _run_browser(command, workdir, timeout_seconds)
        _check_png(shot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shot, out_path)
    logger.debug("rendered %s in %.1fs", out_path, time.monotonic() - started)
    return out_path


def _check_size(width: int, scale: float) -> None:
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise RenderError(f"PNG width must be {MIN_WIDTH} to {MAX_WIDTH} CSS pixels, got {width}")
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise RenderError(f"PNG scale must be {MIN_SCALE:g} to {MAX_SCALE:g}, got {scale:g}")


def max_height(width: int, scale: float) -> int:
    """Tallest screenshot, in CSS pixels, that stays within MAX_HEIGHT and MAX_PNG_PIXELS."""
    return min(MAX_HEIGHT, int(MAX_PNG_PIXELS / (width * scale * scale)))


def _run_browser(command: Sequence[str], workdir: Path, timeout_seconds: float) -> str:
    """Run the browser and return its stdout. Output goes to files, not pipes, because
    browser child processes can hold inherited pipes open after the parent is killed."""
    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, no shell; browser path comes from find_browser or the caller
                command,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=_NEW_SESSION,
            )
        except OSError as exc:
            raise RenderError(f"could not start the browser {command[0]}: {exc}") from exc
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill(process)
            process.wait()
            raise RenderError(
                f"the browser did not finish within {timeout_seconds:g}s{_stderr_tail(stderr_path)}"
            ) from None
    if returncode != 0:
        raise RenderError(f"the browser exited with code {returncode}{_stderr_tail(stderr_path)}")
    return stdout_path.read_text(encoding="utf-8", errors="replace")


def _stderr_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return f"; browser output ends with: {text[-_STDERR_TAIL_CHARS:]}"


def _check_png(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(len(_PNG_SIGNATURE))
    except OSError as exc:
        raise RenderError(f"the browser did not write a screenshot: {exc}") from exc
    if header != _PNG_SIGNATURE:
        raise RenderError(f"the browser wrote {path.name}, but it is not a PNG image")


if sys.platform == "win32":
    _NEW_SESSION: Final = False

    def _kill(process: subprocess.Popen[bytes]) -> None:
        # Chromium child processes exit on their own once the browser process is gone.
        process.kill()

else:
    import signal

    _NEW_SESSION: Final = True

    def _kill(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
