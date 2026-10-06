"""Adapter for OpenAI-compatible servers such as LM Studio and vLLM.

Scoring uses the legacy /completions endpoint. Its `logprobs` field holds either the
legacy parallel `tokens`, `token_logprobs` and `top_logprobs` arrays, or (llama-server)
a `content` list in the chat format, whose entries also carry each token's bytes. Tokens
are read as text without ids, so teacher forcing works by text and the server
retokenizes each prefix; see `exact_token_ids` in ServerInfo. Prefixes are rebuilt from
token bytes when the server reported them.

Servers that hold back a token ending inside a UTF-8 character (llama-server does) fold
it into the entry of the token that completes the character, an entry whose
alternatives belong to the last folded token. `generate_scored` keeps such a step but
empties its `top`, which marks the position as unscored for every candidate. A
one-token request whose token is such a fragment comes back with the text U+FFFD and no
logprobs; it is returned as an empty TopK, a top-1 miss without a KL value.

An empty TopK from `score_continuation` otherwise means the server ended generation at
that position (end of sequence) without reporting a distribution, or the reference left
the position unscored, in which case no request is sent.

The legacy `top_logprobs` entry is an object keyed by token text, so two distinct tokens
that decode to the same text (for example a byte-fallback token and a merged token)
arrive as one entry and only one of their logprobs survives.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Final

from quantdiff._http import DEFAULT_TIMEOUT_SECONDS, get_json, post_json, validate_base_url
from quantdiff._text import printable
from quantdiff.backends._common import (
    GREEDY_TEMPERATURE,
    SCORING_SEED,
    RequestFailure,
    check_max_tokens,
    check_top_k,
    expect_dict,
    expect_float,
    expect_list,
    expect_str,
    explained_failures,
    forced_texts,
    mentions_missing_model,
    openai_chat_payload,
    optional_bytes,
    optional_int,
    parse_openai_chat,
    sorted_top,
    unfolded_steps,
)
from quantdiff.errors import BackendError, CapabilityError, SpecError
from quantdiff.types import (
    CandidateSpec,
    ChatResult,
    JSONValue,
    Message,
    ServerInfo,
    TokenProb,
    TokenStep,
    ToolSpec,
    TopK,
)

_CONTEXT_FIELDS: Final = ("max_model_len", "loaded_context_length", "context_length")
"""Model-list fields that report context size: vLLM first, then LM Studio."""
_LISTED_MODELS_IN_ERROR: Final = 10
_AUTH_STATUSES: Final = frozenset({401, 403})


class OpenAICompatBackend:
    """A named model on a server that speaks the OpenAI /v1 API."""

    def __init__(self, spec: CandidateSpec, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if spec.kind != "openai":
            raise SpecError(f"OpenAICompatBackend cannot serve a {spec.kind!r} spec")
        if not spec.model:
            raise SpecError("an OpenAI-compatible spec needs a model name")
        self._spec = spec
        self._base_url = validate_base_url(spec.base_url)
        self._timeout = timeout
        self._headers = _auth_headers(spec.api_key_env)
        self._info: ServerInfo | None = None

    @property
    def spec(self) -> CandidateSpec:
        return self._spec

    def info(self) -> ServerInfo:
        """Describe the model. Logprob support is assumed until a scored call proves otherwise."""
        if self._info is None:
            self._info = self._load_info()
        return self._info

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        json_schema: dict[str, JSONValue] | None = None,
        seed: int = 0,
    ) -> ChatResult:
        payload = openai_chat_payload(
            self._spec.model,
            messages,
            max_tokens=max_tokens,
            tools=tools,
            json_schema=json_schema,
            seed=seed,
        )
        started = time.perf_counter()
        body = self._post("/chat/completions", payload)
        return parse_openai_chat(body, seconds=time.perf_counter() - started)

    def tokenize(self, text: str) -> tuple[int, ...] | None:
        return None

    def generate_scored(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        check_max_tokens(max_tokens)
        check_top_k(top_k)
        return unfolded_steps(self._complete(prompt, max_tokens=max_tokens, top_k=top_k))

    def score_continuation(
        self,
        prompt: str,
        continuation: Sequence[TokenStep],
        *,
        top_k: int,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> list[TopK]:
        """Teacher-force by text. `prompt_token_ids` is ignored; see the module docstring."""
        check_top_k(top_k)
        distributions: list[TopK] = []
        for text in forced_texts(prompt, continuation):
            steps = [] if text is None else self._complete(text, max_tokens=1, top_k=top_k)
            distributions.append(steps[0].top if steps else ())
        return distributions

    def close(self) -> None:
        """Nothing to release: every request uses its own connection."""

    def _complete(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        payload: dict[str, JSONValue] = {
            "model": self._spec.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": GREEDY_TEMPERATURE,
            "seed": SCORING_SEED,
            "logprobs": top_k,
            "stream": False,
        }
        body = self._post("/completions", payload)
        choices = expect_list(body.get("choices"), "completions choices")
        if not choices:
            raise BackendError("completions response has no choices")
        choice = expect_dict(choices[0], "completions choice")
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            return _parse_logprobs(logprobs)
        if max_tokens == 1 and _held_back(choice):
            return []
        raise CapabilityError(
            f"{self._base_url} returned no logprobs for /completions; "
            "this server cannot be used for logit metrics"
        )

    def _load_info(self) -> ServerInfo:
        with explained_failures(self._explain):
            listing = get_json(self._url("/models"), headers=self._headers, timeout=self._timeout)
        body = expect_dict(listing, "model list")
        entries = [
            expect_dict(entry, "model list entry")
            for entry in expect_list(body.get("data"), "model list data")
        ]
        model = self._find_model(entries)
        return ServerInfo(
            backend="openai",
            model=self._spec.model,
            context_length=_context_length(model),
            chat_template=None,
            template_dialect="unknown",
            supports_logprobs=True,
            exact_token_ids=False,
            details=_details(model),
        )

    def _find_model(self, entries: list[dict[str, JSONValue]]) -> dict[str, JSONValue]:
        for entry in entries:
            if entry.get("id") == self._spec.model:
                return entry
        if not entries:
            raise BackendError(
                f"model {self._spec.model!r} is not served by {self._base_url}, which lists no "
                "models; load one in LM Studio or start vLLM with --model"
            )
        raise BackendError(
            f"model {self._spec.model!r} is not served by {self._base_url}; "
            f"available: {_model_ids(entries)}. Put one of these after '#' in the spec"
        )

    def _post(self, path: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        with explained_failures(self._explain):
            body = post_json(self._url(path), payload, headers=self._headers, timeout=self._timeout)
        return expect_dict(body, f"{path} response")

    def _explain(self, failure: RequestFailure) -> str | None:
        if failure.status is None:
            return (
                f"cannot reach the server at {self._base_url} ({failure.detail}); "
                "check that LM Studio/vLLM is running and the URL ends in /v1"
            )
        if failure.status in _AUTH_STATUSES:
            return self._auth_advice(failure.status)
        if failure.status != 404:
            return None
        if mentions_missing_model(failure.detail):
            return (
                f"model {self._spec.model!r} is not available on {self._base_url}; "
                f"use a model id listed at {self._base_url}/models"
            )
        if not self._base_url.endswith("/v1"):
            return (
                f"{printable(failure.url)} returned HTTP 404; "
                "OpenAI-compatible URLs usually end in /v1, for example http://127.0.0.1:1234/v1"
            )
        return None

    def _auth_advice(self, status: int) -> str:
        if self._spec.api_key_env is None:
            return (
                f"the server at {self._base_url} requires an API key (HTTP {status}); put the key "
                "in an environment variable and add @env:NAME to the spec"
            )
        return (
            f"the server at {self._base_url} rejected the API key from "
            f"${self._spec.api_key_env} (HTTP {status}); check the key and its permissions"
        )

    def _url(self, path: str) -> str:
        return self._base_url + path


def _auth_headers(api_key_env: str | None) -> Mapping[str, str]:
    if api_key_env is None:
        return {}
    key = os.environ.get(api_key_env)
    if not key:
        raise BackendError(f"environment variable {api_key_env} is not set or is empty")
    return {"Authorization": f"Bearer {key}"}


def _model_ids(entries: list[dict[str, JSONValue]]) -> str:
    """Comma list of served ids, capped so a large hub does not flood the terminal."""
    ids = [printable(str(entry.get("id"))) for entry in entries]
    shown = ", ".join(ids[:_LISTED_MODELS_IN_ERROR])
    hidden = len(ids) - _LISTED_MODELS_IN_ERROR
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def _context_length(model: dict[str, JSONValue]) -> int | None:
    for field in _CONTEXT_FIELDS:
        value = optional_int(model.get(field))
        if value is not None:
            return value
    return None


def _details(model: dict[str, JSONValue]) -> tuple[tuple[str, str], ...]:
    """Owner and creation time, the only identity /models exposes for the reference cache."""
    owner = model.get("owned_by")
    created = optional_int(model.get("created"))
    facts = {
        "owned_by": owner if isinstance(owner, str) else None,
        "created": None if created is None else str(created),
    }
    return tuple((key, value) for key, value in facts.items() if value)


def _held_back(choice: dict[str, JSONValue]) -> bool:
    """True when the server generated a fragment of a UTF-8 character, which it shows as
    U+FFFD, and reported no logprobs for it."""
    text = choice.get("text")
    return isinstance(text, str) and text.endswith("\N{REPLACEMENT CHARACTER}")


def _parse_logprobs(logprobs: dict[str, JSONValue]) -> list[TokenStep]:
    if "content" in logprobs:
        entries = expect_list(logprobs["content"], "logprobs content")
        return [_parse_content_step(entry) for entry in entries]
    tokens = expect_list(logprobs.get("tokens"), "logprobs tokens")
    chosen = expect_list(logprobs.get("token_logprobs"), "logprobs token_logprobs")
    tops = expect_list(logprobs.get("top_logprobs"), "logprobs top_logprobs")
    if not len(tokens) == len(chosen) == len(tops):
        raise BackendError("logprobs arrays have different lengths")
    return [
        TokenStep(
            chosen=TokenProb(
                token=expect_str(token, "logprobs token"),
                logprob=expect_float(logprob, "token logprob"),
            ),
            top=_parse_top(top),
        )
        for token, logprob, top in zip(tokens, chosen, tops, strict=True)
    ]


def _parse_top(value: JSONValue) -> TopK:
    alternatives = expect_dict(value, "top_logprobs entry")
    return sorted_top(
        TokenProb(token=token, logprob=expect_float(logprob, "top logprob"))
        for token, logprob in alternatives.items()
    )


def _parse_content_step(entry: JSONValue) -> TokenStep:
    """Read one entry of the chat-format `content` list. An alternative whose logprob is
    null (minus infinity) has zero probability and is left out."""
    step = expect_dict(entry, "logprobs content entry")
    top = expect_list(step.get("top_logprobs") or [], "content top_logprobs")
    alternatives = (_parse_content_prob(item) for item in top if not _is_impossible(item))
    return TokenStep(chosen=_parse_content_prob(step), top=sorted_top(alternatives))


def _is_impossible(item: JSONValue) -> bool:
    return isinstance(item, dict) and "logprob" in item and item["logprob"] is None


def _parse_content_prob(value: JSONValue) -> TokenProb:
    item = expect_dict(value, "logprobs content token")
    return TokenProb(
        token=expect_str(item.get("token"), "logprobs content token"),
        logprob=expect_float(item.get("logprob"), "token logprob"),
        token_bytes=optional_bytes(item.get("bytes")),
    )
