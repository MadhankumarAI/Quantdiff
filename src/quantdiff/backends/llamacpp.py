"""Adapter for llama.cpp's llama-server (/completion, /tokenize, /props, /v1/chat/completions).

llama-server accepts prompts as token id arrays and reports the id of every token it
scores, so teacher forcing feeds the reference's exact token sequence and never
retokenizes text. When the reference came from a backend without token ids (Ollama or an
OpenAI-compatible server), teacher forcing falls back to text prompts, which llama-server
tokenizes itself. Both the current `completion_probabilities` format (id, token,
logprob, top_logprobs) and the older one (content, probs with tok_str and prob) are read.

llama-server writes a logprob of minus infinity as JSON null. Such alternatives have zero
probability and are left out of the top-k list.

Byte-level tokenizers split many non-Latin characters across tokens, and llama-server
(checked on b11425) holds back a token that ends inside a UTF-8 character:

- In a multi-token completion it reports the held-back tokens and the token that
  completes the character as one entry, with the text and bytes of the whole character
  but the id, logprob and alternatives of the last token only. `generate_scored` keeps
  the leading entries that stand for one token each and, from the first folded entry on,
  generates one token per request, so the trace holds every token id and every position's
  own distribution and exact teacher forcing feeds the true token sequence.
- When the one token a request asks for ends inside a character, the reply has no
  `completion_probabilities` at all. The request is then repeated with a logit bias that
  makes the server pick a newline token instead. Probabilities are reported from the raw
  logits (the default, without `post_sampling_probs`), so the alternatives in that reply
  are the unbiased distribution at the same position, and under greedy decoding its head
  is the token the server held back. Only a server that reports no probabilities even
  for that complete token raises CapabilityError.

When the reference came from a backend without token ids, a position whose text prefix
would end inside a character cannot be sent as a text prompt and is returned as an empty
TopK without a request, as are positions the reference left unscored.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import takewhile
from pathlib import PureWindowsPath
from typing import Final

from quantdiff._http import DEFAULT_TIMEOUT_SECONDS, get_json, post_json, validate_base_url
from quantdiff._text import printable
from quantdiff.backends._common import (
    GREEDY_TEMPERATURE,
    SCORING_SEED,
    RequestFailure,
    check_max_tokens,
    check_top_k,
    decode_rate,
    expect_dict,
    expect_float,
    expect_list,
    expect_str,
    explained_failures,
    folds_tokens,
    forced_texts,
    logprob_from_prob,
    openai_chat_payload,
    optional_bytes,
    optional_int,
    optional_str,
    parse_openai_chat,
    sorted_top,
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

Prompt = str | tuple[int, ...]
"""Raw text, which the server tokenizes with special tokens added, or exact token ids."""

_GREEDY_SAMPLERS: Final = ("temperature",)
"""Only the temperature stage, so server-side penalties cannot override the argmax token."""
_DEFAULT_PORT: Final = 8080
_LOADING_STATUS: Final = 503
_META_FINGERPRINT: Final = ("size", "n_params", "n_vocab")
"""/v1/models meta fields that identify the loaded GGUF file."""
_STOP_TYPES: Final = frozenset({"eos", "word"})
"""`stop_type` values for a completion the server ended itself, before the token limit."""
_COMPLETE_TEXT: Final = "\n"
"""An ASCII character, so its token is complete text. It is forced when the token the
server picked ends inside a character."""
_FORCING_BIAS: Final = 1000.0
"""Logit bias that makes greedy decoding pick the forced token."""


@dataclass(frozen=True, slots=True)
class _Completion:
    steps: tuple[TokenStep, ...]
    reported: bool
    """True when the reply carried `completion_probabilities`, even an empty list."""
    stopped: bool
    """True when the server ended generation itself (end of sequence or a stop word)."""


class LlamaCppBackend:
    """The single model served by a llama-server instance."""

    def __init__(self, spec: CandidateSpec, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if spec.kind != "llamacpp":
            raise SpecError(f"LlamaCppBackend cannot serve a {spec.kind!r} spec")
        self._spec = spec
        self._base_url = validate_base_url(spec.base_url)
        self._timeout = timeout
        self._info: ServerInfo | None = None
        self._complete_token: int | None = None

    @property
    def spec(self) -> CandidateSpec:
        return self._spec

    def info(self) -> ServerInfo:
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
        with explained_failures(self._explain):
            body = post_json(self._url("/v1/chat/completions"), payload, timeout=self._timeout)
        result = parse_openai_chat(body, seconds=time.perf_counter() - started)
        return replace(result, decode_tokens_per_second=_predicted_rate(body))

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize with special tokens (such as BOS) added, exactly as a prompt would be."""
        body = self._post("/tokenize", {"content": text, "add_special": True})
        return tuple(_token_id(token) for token in expect_list(body.get("tokens"), "tokens"))

    def generate_scored(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        """Generate in one request, then token by token from the first folded entry on (see
        the module docstring)."""
        check_max_tokens(max_tokens)
        check_top_k(top_k)
        prompt_ids = self.tokenize(prompt)
        batch = self._complete(prompt_ids, max_tokens=max_tokens, top_k=top_k)
        steps = list(takewhile(lambda step: not folds_tokens(step), batch.steps))
        if batch.stopped and len(steps) == len(batch.steps):
            return steps
        return self._extend(prompt_ids, steps, max_tokens=max_tokens, top_k=top_k)

    def score_continuation(
        self,
        prompt: str,
        continuation: Sequence[TokenStep],
        *,
        top_k: int,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> list[TopK]:
        """Feed token ids when the reference supplied them all, else fall back to text."""
        check_top_k(top_k)
        distributions: list[TopK] = []
        for forced in _forced_prompts(prompt, continuation, prompt_token_ids):
            steps = () if forced is None else self._next_token(forced, top_k=top_k).steps
            distributions.append(steps[0].top if steps else ())
        return distributions

    def close(self) -> None:
        """Nothing to release: every request uses its own connection."""

    def _extend(
        self, prompt_ids: tuple[int, ...], steps: list[TokenStep], *, max_tokens: int, top_k: int
    ) -> list[TokenStep]:
        """Continue greedily one token per request until `max_tokens` steps or a stop.

        A one-token reply never folds tokens, so every step is a single token with its own
        distribution. Steps without ids (the older reply format) cannot be extended exactly
        and are returned as they are.
        """
        ids = list(prompt_ids)
        for step in steps:
            if step.chosen.token_id is None:
                return steps
            ids.append(step.chosen.token_id)
        while len(steps) < max_tokens:
            completion = self._next_token(tuple(ids), top_k=top_k)
            if not completion.steps:
                break
            step = completion.steps[0]
            steps.append(step)
            if completion.stopped or step.chosen.token_id is None:
                break
            ids.append(step.chosen.token_id)
        return steps

    def _next_token(self, prompt: Prompt, *, top_k: int) -> _Completion:
        """Score the next token, recovering the distribution of a held-back token."""
        completion = self._complete(prompt, max_tokens=1, top_k=top_k)
        if completion.reported:
            return completion
        forced = self._complete(prompt, max_tokens=1, top_k=top_k, bias=self._complete_token_id())
        top = forced.steps[0].top if forced.steps else ()
        if not top:
            raise CapabilityError(f"llama-server at {self._base_url} returned no probabilities")
        return _Completion(steps=(TokenStep(chosen=top[0], top=top),), reported=True, stopped=False)

    def _complete_token_id(self) -> int:
        if self._complete_token is None:
            body = self._post("/tokenize", {"content": _COMPLETE_TEXT, "add_special": False})
            tokens = expect_list(body.get("tokens"), "tokens")
            if not tokens:
                raise BackendError(f"/tokenize returned no tokens for {_COMPLETE_TEXT!r}")
            self._complete_token = _token_id(tokens[-1])
        return self._complete_token

    def _complete(
        self, prompt: Prompt, *, max_tokens: int, top_k: int, bias: int | None = None
    ) -> _Completion:
        payload: dict[str, JSONValue] = {
            "prompt": prompt if isinstance(prompt, str) else list(prompt),
            "n_predict": max_tokens,
            "temperature": GREEDY_TEMPERATURE,
            "n_probs": top_k,
            "cache_prompt": True,
            "seed": SCORING_SEED,
            "samplers": list(_GREEDY_SAMPLERS),
        }
        if bias is not None:
            payload["logit_bias"] = [[bias, _FORCING_BIAS]]
        body = self._post("/completion", payload)
        entries = body.get("completion_probabilities")
        steps = (
            ()
            if entries is None
            else tuple(_parse_step(entry) for entry in expect_list(entries, "probabilities"))
        )
        return _Completion(steps=steps, reported=entries is not None, stopped=_stopped(body))

    def _load_info(self) -> ServerInfo:
        props = self._get("/props")
        listing = self._get("/v1/models")
        models = expect_list(listing.get("data") or [], "model list data")
        listed = expect_dict(models[0], "model list entry") if models else {}
        settings = props.get("default_generation_settings")
        settings = settings if isinstance(settings, dict) else {}
        model_file = _file_name(props.get("model_path"))
        meta = listed.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        facts = {
            "model_file": model_file,
            "trained_context_length": _optional_text(optional_int(meta.get("n_ctx_train"))),
            "build": optional_str(props.get("build_info")),
        }
        # Size, parameter and vocabulary counts fingerprint the GGUF for the reference cache.
        facts.update(
            (f"model_{field}", _optional_text(optional_int(meta.get(field))))
            for field in _META_FINGERPRINT
        )
        size, n_params, n_vocab = (optional_int(meta.get(field)) for field in _META_FINGERPRINT)
        known = size is not None and n_params is not None and n_vocab is not None
        return ServerInfo(
            backend="llamacpp",
            model=self._spec.model or optional_str(listed.get("id")) or model_file or "",
            context_length=optional_int(settings.get("n_ctx")),
            chat_template=optional_str(props.get("chat_template")),
            template_dialect="jinja",
            supports_logprobs=True,
            exact_token_ids=True,
            details=tuple((key, value) for key, value in facts.items() if value),
            size_bytes=size,
            weights_id=f"{size}:{n_params}:{n_vocab}" if known else None,
        )

    def _get(self, path: str) -> dict[str, JSONValue]:
        with explained_failures(self._explain):
            body = get_json(self._url(path), timeout=self._timeout)
        return expect_dict(body, f"llama-server {path} response")

    def _post(self, path: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        with explained_failures(self._explain):
            body = post_json(self._url(path), payload, timeout=self._timeout)
        return expect_dict(body, f"llama-server {path} response")

    def _explain(self, failure: RequestFailure) -> str | None:
        if failure.status is None:
            port = urllib.parse.urlsplit(self._base_url).port or _DEFAULT_PORT
            return (
                f"cannot reach llama-server at {self._base_url} ({failure.detail}); "
                f"start it with `llama-server -m model.gguf --port {port}`"
            )
        if failure.status == _LOADING_STATUS and "loading" in failure.detail.lower():
            return (
                f"llama-server at {self._base_url} is still loading the model; "
                "wait until its log says the server is listening, then run again"
            )
        if failure.status == 404:
            return (
                f"{printable(failure.url)} returned HTTP 404; "
                f"check that {self._base_url} is a llama-server and not another kind of server"
            )
        return None

    def _url(self, path: str) -> str:
        return self._base_url + path


def _forced_prompts(
    prompt: str, continuation: Sequence[TokenStep], prompt_token_ids: Sequence[int] | None
) -> list[Prompt | None]:
    """One prompt per continuation position: token ids when every id is known, else text.

    None marks a position that text cannot reproduce; see `forced_texts`.
    """
    chosen_ids = [step.chosen.token_id for step in continuation]
    known = tuple(token_id for token_id in chosen_ids if token_id is not None)
    if prompt_token_ids is None or len(known) != len(chosen_ids):
        return list(forced_texts(prompt, continuation))
    prompt_ids = tuple(prompt_token_ids)
    return [prompt_ids + known[:position] for position in range(len(known))]


def _stopped(body: dict[str, JSONValue]) -> bool:
    """Read `stop_type`, or the `stopped_eos` and `stopped_word` flags of older servers."""
    return (
        body.get("stop_type") in _STOP_TYPES
        or body.get("stopped_eos") is True
        or body.get("stopped_word") is True
    )


def _predicted_rate(body: JSONValue) -> float | None:
    """Read the decode timing llama-server adds to chat responses, when present.

    Computed from predicted_n and predicted_ms (which is how the server derives its own
    predicted_per_second) so a one-token answer can be recognized and left out.
    """
    timings = body.get("timings") if isinstance(body, dict) else None
    if not isinstance(timings, dict):
        return None
    milliseconds = timings.get("predicted_ms")
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)):
        return None
    return decode_rate(optional_int(timings.get("predicted_n")), milliseconds / 1000)


def _optional_text(value: int | None) -> str | None:
    return None if value is None else str(value)


def _token_id(token: JSONValue) -> int:
    """Read one /tokenize entry: a bare id, or {"id", "piece"} when pieces are requested."""
    value = token.get("id") if isinstance(token, dict) else token
    token_id = optional_int(value)
    if token_id is None:
        raise BackendError(f"/tokenize returned an invalid token entry: {token!r}")
    return token_id


def _file_name(path: JSONValue) -> str | None:
    if not isinstance(path, str):
        return None
    # The server may run on Windows or POSIX; PureWindowsPath splits on both separators.
    return PureWindowsPath(path).name or None


def _parse_step(entry: JSONValue) -> TokenStep:
    step = expect_dict(entry, "completion_probabilities entry")
    if "probs" in step:
        return _parse_legacy_step(step)
    top = expect_list(step.get("top_logprobs") or [], "top_logprobs")
    alternatives = (_parse_prob(item) for item in top if not _is_impossible(item))
    return TokenStep(chosen=_parse_prob(step), top=sorted_top(alternatives))


def _is_impossible(item: JSONValue) -> bool:
    """True for an alternative whose logprob is null, the server's encoding of -inf."""
    return isinstance(item, dict) and "logprob" in item and item["logprob"] is None


def _parse_prob(value: JSONValue) -> TokenProb:
    item = expect_dict(value, "token probability")
    return TokenProb(
        token=expect_str(item.get("token"), "token"),
        logprob=expect_float(item.get("logprob"), "logprob"),
        token_id=optional_int(item.get("id")),
        token_bytes=optional_bytes(item.get("bytes")),
    )


def _parse_legacy_step(step: dict[str, JSONValue]) -> TokenStep:
    """Read the older format, which reports probabilities and no token ids."""
    content = expect_str(step.get("content"), "content")
    probs = (_parse_legacy_prob(item) for item in expect_list(step["probs"], "probs"))
    top = sorted_top(prob for prob in probs if prob is not None)
    chosen = next((prob for prob in top if prob.token == content), None)
    if chosen is None:
        raise BackendError(f"chosen token {content!r} is missing from its own probabilities")
    return TokenStep(chosen=chosen, top=top)


def _parse_legacy_prob(value: JSONValue) -> TokenProb | None:
    """Return None for zero-probability filler entries, which have no logarithm."""
    item = expect_dict(value, "probs entry")
    prob = expect_float(item.get("prob"), "prob")
    if prob == 0.0:
        return None
    return TokenProb(
        token=expect_str(item.get("tok_str"), "tok_str"),
        logprob=logprob_from_prob(prob, "prob"),
    )
