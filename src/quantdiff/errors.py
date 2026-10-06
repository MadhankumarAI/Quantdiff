"""Exception hierarchy. Every error quantdiff raises on purpose derives from QuantdiffError."""

from __future__ import annotations


class QuantdiffError(Exception):
    """Base class for all quantdiff errors."""


class SpecError(QuantdiffError, ValueError):
    """A candidate spec string or option could not be parsed."""


class BackendError(QuantdiffError):
    """A model server returned an error, an unexpected payload, or could not be reached."""


class RequestError(BackendError):
    """An HTTP request failed: the server was unreachable or answered with an error status.

    `status` is None for transport failures (refused, timed out); `detail` is the server's
    error body preview or the transport reason, already sanitized for terminals.
    """

    def __init__(self, message: str, *, url: str, status: int | None, detail: str) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.detail = detail


class CapabilityError(BackendError):
    """The backend cannot perform the requested operation, such as returning logprobs."""


class SuiteError(QuantdiffError, ValueError):
    """A prompt suite or user prompts file is missing or malformed."""


class ReportError(QuantdiffError, ValueError):
    """A report file is missing, malformed, or from an unsupported schema version."""


class RenderError(QuantdiffError):
    """A scorecard could not be rendered, for example no browser is available for PNG output."""
