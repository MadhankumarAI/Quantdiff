"""Model server adapters and the factory that picks one for a CandidateSpec."""

from __future__ import annotations

from quantdiff._http import DEFAULT_TIMEOUT_SECONDS
from quantdiff.backends.base import MAX_TOP_K, Backend
from quantdiff.backends.llamacpp import LlamaCppBackend
from quantdiff.backends.ollama import OllamaBackend
from quantdiff.backends.openai_compat import OpenAICompatBackend
from quantdiff.types import CandidateSpec

__all__ = [
    "MAX_TOP_K",
    "Backend",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "open_backend",
]


def open_backend(spec: CandidateSpec, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Backend:
    """Return the adapter for `spec.kind`. No request is sent until the first call."""
    if spec.kind == "ollama":
        return OllamaBackend(spec, timeout=timeout)
    if spec.kind == "llamacpp":
        return LlamaCppBackend(spec, timeout=timeout)
    return OpenAICompatBackend(spec, timeout=timeout)
