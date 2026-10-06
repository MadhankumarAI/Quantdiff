"""A scripted JSON HTTP server that records every request, for backend tests."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from quantdiff.types import JSONValue

FIXTURES = Path(__file__).resolve().parent
_POLL_SECONDS = 0.01
"""Short shutdown poll so per-test servers stop quickly."""


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: JSONValue


Reply = tuple[int, JSONValue]
Responder = Callable[[Request], Reply]


@dataclass
class FakeServer:
    """Routes `(method, path)` to a responder and keeps the requests it received."""

    url: str = ""
    routes: dict[tuple[str, str], Responder] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)

    def reply(self, method: str, path: str, body: JSONValue, *, status: int = 200) -> None:
        self.routes[(method, path)] = lambda _request: (status, body)

    def respond(self, method: str, path: str, responder: Responder) -> None:
        self.routes[(method, path)] = responder

    def bodies(self, path: str) -> list[JSONValue]:
        return [request.body for request in self.requests if request.path == path]

    def handle(self, request: Request) -> Reply:
        self.requests.append(request)
        responder = self.routes.get((request.method, request.path))
        if responder is None:
            return 404, {"error": f"no route for {request.method} {request.path}"}
        return responder(request)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        self._serve(None)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self._serve(json.loads(self.rfile.read(length)))

    def _serve(self, body: JSONValue) -> None:
        request = Request(self.command, self.path, dict(self.headers.items()), body)
        status, reply = self.server.fake.handle(request)
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        return


class _Server(ThreadingHTTPServer):
    def __init__(self, fake: FakeServer) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.fake = fake


@contextmanager
def running() -> Iterator[FakeServer]:
    """Serve a fresh FakeServer on a free local port for the duration of the block."""
    fake = FakeServer()
    httpd = _Server(fake)
    fake.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, args=(_POLL_SECONDS,), daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        httpd.shutdown()
        httpd.server_close()


def load(name: str) -> JSONValue:
    """Load a captured payload from tests/fixtures/<name>.json."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def closed_port() -> int:
    """A local port with nothing listening, for connection refused tests."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port
