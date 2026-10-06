"""Parse candidate spec strings from the command line into CandidateSpec values.

Grammar::

    spec     = [label "="] target
    target   = "ollama:" tag
             | "llamacpp:" base_url
             | "openai:" base_url "#" model ["@env:" env_var]

    label    = any non-empty text before the first "=" that contains no ":"
    env_var  = name of an environment variable holding an API key ([A-Za-z_][A-Za-z0-9_]*)

Examples::

    ollama:qwen2.5:0.5b-instruct-q8_0
    q4=ollama:qwen2.5:7b-instruct-q4_K_M
    llamacpp:http://127.0.0.1:8080
    openai:http://127.0.0.1:1234/v1#qwen2.5-7b-instruct
    vllm=openai:http://gpu-box:8000/v1#Qwen/Qwen2.5-7B-Instruct@env:VLLM_API_KEY

Ollama specs use OLLAMA_HOST (`host`, `host:port` or a full URL) when it is set, and
http://127.0.0.1:11434 otherwise. Default labels are the Ollama tag, `llamacpp@host:port`
and the OpenAI model name. Making labels unique across candidates is the caller's job.
"""

from __future__ import annotations

import dataclasses
import os
import re
import urllib.parse
from typing import Final

from quantdiff._http import validate_base_url
from quantdiff.errors import BackendError, SpecError
from quantdiff.types import CandidateSpec

DEFAULT_OLLAMA_URL: Final = "http://127.0.0.1:11434"
_OLLAMA_DEFAULT_PORT: Final = 11434
_API_KEY_MARKER: Final = "@env:"
_ENV_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OLLAMA_TAG: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
_BIND_ALL_HOSTS: Final = frozenset({"0.0.0.0", "::"})  # noqa: S104 - matched, not bound


def parse_spec(text: str) -> CandidateSpec:
    """Parse one candidate spec string. Raises SpecError if it is malformed."""
    stripped = text.strip()
    if not stripped:
        raise SpecError("empty candidate spec")
    label, target = _split_label(stripped)
    kind, separator, rest = target.partition(":")
    if not separator or not rest:
        raise SpecError(f"{text!r}: expected <kind>:<target>, for example ollama:<tag>")
    if kind == "ollama":
        spec = _ollama_spec(rest)
    elif kind == "llamacpp":
        spec = _llamacpp_spec(rest)
    elif kind == "openai":
        spec = _openai_spec(rest)
    else:
        raise SpecError(f"{text!r}: unknown backend {kind!r}; use ollama, llamacpp or openai")
    return spec if label is None else dataclasses.replace(spec, label=label)


def ollama_base_url(host: str | None) -> str:
    """Resolve an OLLAMA_HOST value the way the Ollama CLI does."""
    value = (host or "").strip()
    if not value:
        return DEFAULT_OLLAMA_URL
    url = value if "://" in value else f"http://{value}"
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    if not hostname:
        raise SpecError(f"OLLAMA_HOST={value!r} has no host")
    # OLLAMA_HOST is often a listen address such as 0.0.0.0, which clients cannot dial.
    if hostname in _BIND_ALL_HOSTS:
        hostname = "127.0.0.1"
    port = _port(parsed, value)
    if port is None and "://" not in value:
        port = _OLLAMA_DEFAULT_PORT
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    return _base_url(urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", "")))


def _split_label(text: str) -> tuple[str | None, str]:
    head, separator, tail = text.partition("=")
    if not separator or ":" in head:
        return None, text
    label = head.strip()
    if not label:
        raise SpecError(f"{text!r}: the label before '=' is empty")
    if not label.isprintable():
        raise SpecError(f"{text!r}: the label contains control characters")
    return label, tail.strip()


def _ollama_spec(tag: str) -> CandidateSpec:
    if not _OLLAMA_TAG.fullmatch(tag):
        raise SpecError(f"invalid Ollama tag {tag!r}")
    return CandidateSpec(
        kind="ollama",
        base_url=ollama_base_url(os.environ.get("OLLAMA_HOST")),
        model=tag,
        label=tag,
    )


def _llamacpp_spec(url: str) -> CandidateSpec:
    base_url = _base_url(url)
    return CandidateSpec(
        kind="llamacpp",
        base_url=base_url,
        model="",
        label=f"llamacpp@{urllib.parse.urlsplit(base_url).netloc}",
    )


def _openai_spec(rest: str) -> CandidateSpec:
    url, separator, model_part = rest.partition("#")
    if not separator:
        raise SpecError(f"openai spec {rest!r} needs a model: openai:<base_url>#<model>")
    model, marker, env_name = model_part.rpartition(_API_KEY_MARKER)
    if not marker:
        model, env_name = model_part, ""
    model = model.strip()
    if not model or any(char.isspace() for char in model):
        raise SpecError(f"invalid model name {model!r} in openai spec")
    if marker and not _ENV_NAME.fullmatch(env_name):
        raise SpecError(f"invalid environment variable name {env_name!r} after {_API_KEY_MARKER}")
    return CandidateSpec(
        kind="openai",
        base_url=_base_url(url),
        model=model,
        label=model,
        api_key_env=env_name or None,
    )


def _base_url(url: str) -> str:
    try:
        return validate_base_url(url.strip())
    except BackendError as exc:
        # The message from validate_base_url never echoes credentials; the raw URL might.
        raise SpecError(f"invalid server URL: {exc}") from None


def _port(parsed: urllib.parse.SplitResult, raw: str) -> int | None:
    try:
        return parsed.port
    except ValueError:
        raise SpecError(f"OLLAMA_HOST={raw!r} has an invalid port") from None
