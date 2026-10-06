from __future__ import annotations

import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import IO, Any

import pytest

from quantdiff import png
from quantdiff.card import render_html
from quantdiff.errors import RenderError
from quantdiff.png import (
    BROWSER_ENV,
    MAX_HEIGHT,
    find_browser,
    max_height,
    measure_command,
    measure_page,
    parse_height,
    platform_candidates,
    render_png,
    screenshot_command,
)
from quantdiff.report import load_report
from tests.test_report import FIXTURE

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TINY_PNG = (
    PNG_SIGNATURE + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 2, 2) + b"\x08\x06\x00\x00\x00"
)
BROWSER = Path("/opt/browser/chrome")
HARDENING = {
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-gpu",
    "--disable-background-networking",
    "--disable-sync",
    "--disable-component-update",
    "--mute-audio",
    "--hide-scrollbars",
    "--blink-settings=preferredColorScheme=1",
}


def height_dom(height: int | str) -> str:
    return f'<html><head><meta name="quantdiff-height" content="{height}"></head></html>'


class FakeProcess:
    def __init__(self, returncode: int, *, hang: bool) -> None:
        self.returncode = returncode
        self.hang = hang
        self.pid = 4242
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="browser", timeout=timeout or 0)
        return self.returncode


class FakeBrowser:
    """Stands in for subprocess.Popen: answers --dump-dom and --screenshot like a browser."""

    def __init__(
        self,
        *,
        dom: str = height_dom(900),
        image: bytes | None = TINY_PNG,
        returncode: int = 0,
        stderr: bytes = b"",
        hang: bool = False,
    ) -> None:
        self.dom = dom
        self.image = image
        self.returncode = returncode
        self.stderr = stderr
        self.hang = hang
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(
        self, command: list[str], *, stdout: IO[bytes], stderr: IO[bytes], **kwargs: Any
    ) -> FakeProcess:
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert "shell" not in kwargs
        self.commands.append(list(command))
        screenshot = [arg.split("=", 1)[1] for arg in command if arg.startswith("--screenshot=")]
        if not screenshot:
            stdout.write(self.dom.encode())
        elif self.image is not None:
            Path(screenshot[0]).write_bytes(self.image)
        stderr.write(self.stderr)
        process = FakeProcess(self.returncode, hang=self.hang)
        self.processes.append(process)
        return process


@pytest.fixture
def fake_browser(monkeypatch: pytest.MonkeyPatch) -> FakeBrowser:
    fake = FakeBrowser()
    monkeypatch.setattr(subprocess, "Popen", fake)
    return fake


def window_size(command: list[str]) -> str:
    return next(arg for arg in command if arg.startswith("--window-size="))


# find_browser -----------------------------------------------------------------------------


def test_find_browser_uses_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = tmp_path / "my-chrome"
    browser.write_bytes(b"")
    monkeypatch.setenv(BROWSER_ENV, str(browser))
    assert find_browser() == browser


def test_find_browser_rejects_an_override_that_is_not_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BROWSER_ENV, str(tmp_path / "missing"))
    with pytest.raises(RenderError, match=BROWSER_ENV):
        find_browser()
    monkeypatch.setenv(BROWSER_ENV, str(tmp_path))
    with pytest.raises(RenderError, match="not a file"):
        find_browser()


def test_find_browser_returns_the_first_installed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = tmp_path / "chrome.exe"
    installed.write_bytes(b"")
    candidates = [tmp_path / "edge.exe", installed, tmp_path / "brave.exe"]
    monkeypatch.delenv(BROWSER_ENV, raising=False)
    monkeypatch.setattr(png, "platform_candidates", lambda platform, environ: candidates)
    assert find_browser() == installed


def test_find_browser_returns_none_when_nothing_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BROWSER_ENV, "")
    monkeypatch.setattr(png, "platform_candidates", lambda platform, environ: [tmp_path / "x"])
    assert find_browser() is None


def test_windows_candidates_prefer_edge_and_skip_unset_roots() -> None:
    environ = {"ProgramFiles": r"C:\PF", "LocalAppData": r"C:\Users\me\AppData\Local"}
    candidates = platform_candidates("win32", environ)
    assert candidates[:2] == [
        Path(r"C:\PF") / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(r"C:\Users\me\AppData\Local") / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    assert Path(r"C:\PF") / "Google" / "Chrome" / "Application" / "chrome.exe" in candidates
    assert Path(r"C:\PF") / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe" in (
        candidates
    )
    assert Path(r"C:\PF") / "Chromium" / "Application" / "chrome.exe" in candidates
    assert len(candidates) == 8


def test_macos_candidates_are_app_bundle_binaries() -> None:
    candidates = platform_candidates("darwin", {})
    assert candidates[0] == Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    assert all("/Contents/MacOS/" in candidate.as_posix() for candidate in candidates)


def test_linux_candidates_come_from_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"chromium": "/usr/bin/chromium", "brave-browser": "/snap/bin/brave-browser"}
    looked_up: list[str] = []

    def which(name: str) -> str | None:
        looked_up.append(name)
        return found.get(name)

    monkeypatch.setattr(shutil, "which", which)
    candidates = platform_candidates("linux", {})
    assert candidates == [Path("/usr/bin/chromium"), Path("/snap/bin/brave-browser")]
    assert looked_up[0] == "google-chrome"
    assert "microsoft-edge" in looked_up


# Commands and pages -----------------------------------------------------------------------


def test_measure_command_dumps_the_dom_in_a_wider_window(tmp_path: Path) -> None:
    page = tmp_path / "measure.html"
    command = measure_command(BROWSER, page, tmp_path / "profile", width=1040, scale=2.0)
    assert command[0] == str(BROWSER)
    assert command[-1] == page.as_uri()
    assert set(command) >= HARDENING
    assert f"--user-data-dir={tmp_path / 'profile'}" in command
    assert "--force-device-scale-factor=2" in command
    assert "--dump-dom" in command
    width, _ = window_size(command).removeprefix("--window-size=").split(",")
    assert int(width) > 1040


def test_screenshot_command_uses_the_exact_size(tmp_path: Path) -> None:
    page = tmp_path / "card.html"
    output = tmp_path / "card.png"
    command = screenshot_command(
        BROWSER, page, output, tmp_path / "profile", width=1040, height=1224, scale=1.5
    )
    assert set(command) >= HARDENING
    assert window_size(command) == "--window-size=1040,1224"
    assert f"--screenshot={output}" in command
    assert "--force-device-scale-factor=1.5" in command
    assert "--dump-dom" not in command
    assert command[-1] == page.as_uri()


def test_measure_page_embeds_the_escaped_card_at_the_target_width() -> None:
    card = '<!doctype html><body><h1 class="t">A & "B"</h1></body>'
    page = measure_page(card, 800)
    assert "width: 800px" in page
    assert 'srcdoc="&lt;!doctype html&gt;&lt;body&gt;&lt;h1 class=&quot;t&quot;&gt;A &amp;' in page
    assert "<h1" not in page
    assert "quantdiff-height" in page


def test_parse_height_reads_the_meta_tag() -> None:
    assert parse_height(height_dom(1224)) == 1224


def test_parse_height_ignores_the_script_source() -> None:
    dom = '<script>meta.name = "quantdiff-height"; meta.content = "7";</script>'
    with pytest.raises(RenderError, match="did not report the card height"):
        parse_height(dom)


@pytest.mark.parametrize("dom", ["", "Fatal error", height_dom("abc"), height_dom("-5")])
def test_parse_height_rejects_missing_or_garbled_output(dom: str) -> None:
    with pytest.raises(RenderError, match="did not report the card height"):
        parse_height(dom)


def test_parse_height_rejects_zero() -> None:
    with pytest.raises(RenderError, match="0px"):
        parse_height(height_dom(0))


def test_max_height_caps_total_pixels() -> None:
    assert max_height(1040, 2.0) == MAX_HEIGHT
    assert max_height(4096, 4.0) == 1525


# render_png -------------------------------------------------------------------------------


def test_render_png_measures_then_screenshots(fake_browser: FakeBrowser, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "card.png"
    result = render_png("<p>card</p>", out, browser=BROWSER)
    assert result == out
    assert out.read_bytes() == TINY_PNG
    measure, screenshot = fake_browser.commands
    assert "--dump-dom" in measure
    assert window_size(screenshot) == "--window-size=1040,900"
    assert "--force-device-scale-factor=2" in screenshot
    profile = next(arg for arg in measure if arg.startswith("--user-data-dir="))
    assert profile in screenshot
    assert str(tmp_path) not in profile


def test_render_png_clips_very_tall_cards(fake_browser: FakeBrowser, tmp_path: Path) -> None:
    fake_browser.dom = height_dom(MAX_HEIGHT * 3)
    render_png("<p>card</p>", tmp_path / "card.png", browser=BROWSER)
    assert window_size(fake_browser.commands[1]) == f"--window-size=1040,{MAX_HEIGHT}"


@pytest.mark.parametrize(
    ("width", "scale", "message"),
    [(100, 2.0, "width"), (10_000, 2.0, "width"), (1040, 0.1, "scale"), (1040, 9.0, "scale")],
)
def test_render_png_rejects_unreasonable_sizes(
    fake_browser: FakeBrowser, tmp_path: Path, width: int, scale: float, message: str
) -> None:
    with pytest.raises(RenderError, match=message):
        render_png("<p>card</p>", tmp_path / "card.png", width=width, scale=scale, browser=BROWSER)
    assert fake_browser.commands == []


def test_render_png_without_a_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(png, "find_browser", lambda: None)
    with pytest.raises(RenderError, match="no Chromium-based browser found") as caught:
        render_png("<p>card</p>", tmp_path / "card.png")
    assert BROWSER_ENV in str(caught.value)


@pytest.mark.parametrize("image", [b"<html>not a png</html>", b""])
def test_render_png_rejects_output_that_is_not_a_png(
    fake_browser: FakeBrowser, tmp_path: Path, image: bytes
) -> None:
    fake_browser.image = image
    out = tmp_path / "card.png"
    with pytest.raises(RenderError, match="not a PNG"):
        render_png("<p>card</p>", out, browser=BROWSER)
    assert not out.exists()


def test_render_png_reports_a_missing_screenshot(fake_browser: FakeBrowser, tmp_path: Path) -> None:
    fake_browser.image = None
    with pytest.raises(RenderError, match="did not write a screenshot"):
        render_png("<p>card</p>", tmp_path / "card.png", browser=BROWSER)


def test_render_png_reports_the_tail_of_browser_errors(
    fake_browser: FakeBrowser, tmp_path: Path
) -> None:
    fake_browser.returncode = 21
    fake_browser.stderr = b"x" * 5000 + b"\nERROR: cannot open display"
    with pytest.raises(RenderError, match="exited with code 21") as caught:
        render_png("<p>card</p>", tmp_path / "card.png", browser=BROWSER)
    message = str(caught.value)
    assert message.endswith("cannot open display")
    assert len(message) < 1000


def test_render_png_kills_a_browser_that_times_out(
    fake_browser: FakeBrowser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_browser.hang = True

    def kill(process: FakeProcess) -> None:
        process.killed = True

    monkeypatch.setattr(png, "_kill", kill)
    with pytest.raises(RenderError, match=r"did not finish within 1\.5s"):
        render_png("<p>card</p>", tmp_path / "card.png", browser=BROWSER, timeout_seconds=1.5)
    assert [process.killed for process in fake_browser.processes] == [True]


def test_render_png_reports_a_browser_that_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(command: list[str], **kwargs: Any) -> FakeProcess:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    with pytest.raises(RenderError, match="could not start the browser"):
        render_png("<p>card</p>", tmp_path / "card.png", browser=BROWSER)


# live -------------------------------------------------------------------------------------

live = pytest.mark.skipif(os.environ.get("QUANTDIFF_LIVE") != "1", reason="QUANTDIFF_LIVE=1")


@pytest.mark.live
@live
def test_live_render_sample_report(tmp_path: Path) -> None:
    if find_browser() is None:
        pytest.skip("no Chromium-based browser installed")
    out = tmp_path / "card.png"
    started = time.monotonic()
    render_png(render_html(load_report(FIXTURE)), out)
    elapsed = time.monotonic() - started
    data = out.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert width == 1040 * 2
    assert 600 < height < 6000
    assert 20_000 < len(data) < 5_000_000
    assert elapsed < 60
