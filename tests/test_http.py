from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantdiff._http import get_json, post_json, validate_base_url
from quantdiff.errors import BackendError


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        routes = {
            "/ok": (200, b'{"hello": "world"}'),
            "/bad-json": (200, b"not json"),
            "/big": (200, b'"' + b"x" * 2048 + b'"'),
            "/error": (500, b"boom secret-free"),
            "/nan": (200, b'{"x": NaN}'),
            "/deep": (200, b"[" * 100_000 + b"]" * 100_000),
            "/escape": (500, b"bad \x1b]0;pwned\x07 body"),
        }
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/steal")
            self.end_headers()
            return
        status, body = routes.get(self.path, (404, b"missing"))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        body = json.dumps({"echo": payload, "auth": self.headers.get("Authorization")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        return


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_get_json_decodes(server: str) -> None:
    assert get_json(f"{server}/ok") == {"hello": "world"}


def test_post_json_sends_payload_and_headers(server: str) -> None:
    result = post_json(f"{server}/echo", {"a": 1}, headers={"Authorization": "Bearer k"})
    assert result == {"echo": {"a": 1}, "auth": "Bearer k"}


def test_http_error_includes_status(server: str) -> None:
    with pytest.raises(BackendError, match="HTTP 500"):
        get_json(f"{server}/error")


def test_invalid_json_is_rejected(server: str) -> None:
    with pytest.raises(BackendError, match="not valid JSON"):
        get_json(f"{server}/bad-json")


def test_oversized_body_is_rejected(server: str) -> None:
    with pytest.raises(BackendError, match="more than 100 bytes"):
        get_json(f"{server}/big", max_bytes=100)


def test_redirects_are_not_followed(server: str) -> None:
    with pytest.raises(BackendError, match="HTTP 302"):
        get_json(f"{server}/redirect")


def test_unreachable_host_raises_backend_error() -> None:
    with pytest.raises(BackendError, match="failed"):
        get_json("http://127.0.0.1:9/nothing", timeout=2)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///etc/passwd", "scheme"),
        ("ftp://host/x", "scheme"),
        ("http://user:pw@host:1", "credentials"),
        ("http://host:1/?q=1", "query"),
        ("http:///nohost", "no host"),
    ],
)
def test_validate_base_url_rejects_unsafe(url: str, message: str) -> None:
    with pytest.raises(BackendError, match=message):
        validate_base_url(url)


def test_validate_base_url_strips_trailing_slash() -> None:
    assert validate_base_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"


def test_non_standard_json_constants_are_rejected(server: str) -> None:
    with pytest.raises(BackendError, match="not valid JSON"):
        get_json(f"{server}/nan")


def test_deeply_nested_json_is_rejected_not_crashing(server: str) -> None:
    with pytest.raises(BackendError, match="not valid JSON"):
        get_json(f"{server}/deep")


def test_error_preview_strips_terminal_escapes(server: str) -> None:
    with pytest.raises(BackendError) as exc:
        get_json(f"{server}/escape")
    assert "\x1b" not in str(exc.value)
    assert "\x07" not in str(exc.value)
    assert "bad" in str(exc.value)


def test_environment_proxies_are_ignored(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert get_json(f"{server}/ok") == {"hello": "world"}
