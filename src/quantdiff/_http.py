"""Minimal hardened JSON-over-HTTP client built on urllib.

Every network call in quantdiff goes through this module so the safety rules live in one
place: http and https only, no redirects, bounded response size, explicit timeouts, and
error messages that never echo request headers (which may carry an API key).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Final

from quantdiff._text import printable
from quantdiff.errors import BackendError, RequestError
from quantdiff.types import JSONValue

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final = 600.0
DEFAULT_MAX_BYTES: Final = 64 * 1024 * 1024
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_ERROR_BODY_PREVIEW: Final = 300


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a server cannot bounce requests, and their headers, elsewhere."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        # urllib calls this with six fixed arguments; none matter, every redirect is refused.
        return None


# An empty ProxyHandler ignores HTTP(S)_PROXY: requests, and any API key they carry, go only
# to the servers the user named.
_OPENER: Final = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def validate_base_url(url: str) -> str:
    """Return `url` without a trailing slash, or raise BackendError if it is unsafe to use."""
    parsed = _check_url(url)
    if parsed.query or parsed.fragment:
        raise BackendError("base URLs must not contain a query string or fragment")
    return url.rstrip("/")


def _check_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise BackendError(f"unsupported URL scheme {parsed.scheme!r}; use http or https")
    if not parsed.hostname:
        raise BackendError(f"URL has no host: {url!r}")
    if parsed.username or parsed.password:
        raise BackendError("credentials in URLs are not allowed; use an API key variable")
    return parsed


def request_json(
    method: str,
    url: str,
    payload: Mapping[str, JSONValue] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> JSONValue:
    """Send a JSON request and return the decoded JSON response.

    Raises BackendError for transport failures, non-2xx statuses, oversized bodies and
    invalid JSON.
    """
    _check_url(url)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)  # noqa: S310 - scheme checked above
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)

    logger.debug("%s %s", method, url)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        preview = _read_preview(exc)
        raise RequestError(
            f"{method} {url} returned HTTP {exc.code}: {preview}",
            url=url,
            status=exc.code,
            detail=preview,
        ) from None
    except urllib.error.URLError as exc:
        reason = printable(str(exc.reason))
        raise RequestError(
            f"{method} {url} failed: {reason}", url=url, status=None, detail=reason
        ) from None
    except TimeoutError:
        reason = f"timed out after {timeout:g}s"
        raise RequestError(
            f"{method} {url} {reason}", url=url, status=None, detail=reason
        ) from None
    except OSError as exc:
        reason = printable(str(exc))
        raise RequestError(
            f"{method} {url} failed: {reason}", url=url, status=None, detail=reason
        ) from None

    if len(raw) > max_bytes:
        raise BackendError(f"{method} {url} returned more than {max_bytes} bytes")
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise BackendError(f"{method} {url} returned a body that is not valid JSON") from None


def get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> JSONValue:
    return request_json("GET", url, headers=headers, timeout=timeout, max_bytes=max_bytes)


def post_json(
    url: str,
    payload: Mapping[str, JSONValue],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> JSONValue:
    return request_json("POST", url, payload, headers=headers, timeout=timeout, max_bytes=max_bytes)


def _read_preview(exc: urllib.error.HTTPError) -> str:
    try:
        text = exc.read(_ERROR_BODY_PREVIEW).decode("utf-8", errors="replace")
    except OSError:
        return "<unreadable body>"
    return " ".join(printable(text).split()) or "<empty body>"


def _reject_constant(name: str) -> JSONValue:
    raise ValueError(f"non-standard JSON constant {name}")
