"""Adapter for the Ollama native API (/api/generate, /api/chat, /api/show).

Ollama reports logprobs as tokens without ids, so teacher forcing works by text: each
scoring request sends the prompt plus the chosen tokens so far, rebuilt from the bytes
Ollama reports for every token. Ollama retokenizes that text, which can split it
differently from the reference's token sequence (for example around whitespace). That is
why `exact_token_ids` is False for this backend.

Byte-level tokenizers split many non-Latin characters across tokens. Ollama (checked on
0.35.1) holds back a token that ends inside a UTF-8 character:

- While generating, it reports the held-back tokens and the token that completes the
  character as one entry: the text and bytes of the whole character, but the logprob and
  alternatives of the last token only. Those alternatives are character fragments, all
  shown as U+FFFD with the bytes of U+FFFD. That distribution belongs to a later position
  than the one the entry stands for, and no text prompt can reproduce it, because a
  prompt cannot end inside a character. `generate_scored` therefore keeps such a step but
  empties its `top`, which marks the position as unscored for every candidate; metrics
  skip it, so it counts neither as a top-1 miss nor in the KL divergence. The steps after
  it are scored normally.
- When the one token a scoring request asks for ends inside a character, Ollama returns
  no text and no logprobs, exactly as for end of sequence (only `done_reason` differs).
  Either way no distribution is available, so the position is reported as an empty
  TopK: a top-1 miss without a KL value. Against a reference from Ollama that is right,
  because a scored reference step is always a whole token and the candidate picked a
  fragment.

An empty TopK from `score_continuation` also stands for a position the reference left
unscored; no request is sent for it. Ollama leaves out the `logprobs` key entirely when
the first token it picks is end of sequence (or a held-back fragment), so a reply with no
text and no `logprobs` is read as zero steps rather than as missing logprob support.

quantdiff never sets `num_ctx`: it measures the context the user actually gets.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
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
    forced_texts,
    mentions_missing_model,
    openai_messages,
    openai_tool,
    optional_bytes,
    optional_int,
    optional_str,
    sorted_top,
    tool_call,
    unfolded_steps,
)
from quantdiff.errors import CapabilityError, SpecError
from quantdiff.types import (
    CandidateSpec,
    ChatResult,
    JSONValue,
    Message,
    ServerInfo,
    TokenProb,
    TokenStep,
    ToolCall,
    ToolSpec,
    TopK,
)

logger = logging.getLogger(__name__)

KEEP_ALIVE: Final = "10m"
"""Keeps the model loaded between the many small scoring requests."""


class OllamaBackend:
    """A model served by Ollama, addressed by its tag."""

    def __init__(self, spec: CandidateSpec, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if spec.kind != "ollama":
            raise SpecError(f"OllamaBackend cannot serve a {spec.kind!r} spec")
        if not spec.model:
            raise SpecError("an Ollama spec needs a model tag")
        self._spec = spec
        self._base_url = validate_base_url(spec.base_url)
        self._timeout = timeout
        self._info: ServerInfo | None = None

    @property
    def spec(self) -> CandidateSpec:
        return self._spec

    def info(self) -> ServerInfo:
        """Describe the model.

        The effective context length comes from a `num_ctx` set in the Modelfile, or else
        from /api/ps after loading the model, because recent Ollama versions choose the
        default context size at load time from available memory.
        """
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
        check_max_tokens(max_tokens)
        payload: dict[str, JSONValue] = {
            "model": self._spec.model,
            "messages": openai_messages(messages),
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": GREEDY_TEMPERATURE,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = [openai_tool(tool) for tool in tools]
        if json_schema is not None:
            payload["format"] = json_schema
        started = time.perf_counter()
        body = self._post("/api/chat", payload)
        seconds = time.perf_counter() - started
        return _parse_chat(body, seconds=seconds)

    def tokenize(self, text: str) -> tuple[int, ...] | None:
        return None

    def generate_scored(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        check_max_tokens(max_tokens)
        check_top_k(top_k)
        return unfolded_steps(self._generate(prompt, max_tokens=max_tokens, top_k=top_k))

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
            steps = [] if text is None else self._generate(text, max_tokens=1, top_k=top_k)
            distributions.append(steps[0].top if steps else ())
        return distributions

    def close(self) -> None:
        """Nothing to release: every request uses its own connection."""

    def _generate(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        payload: dict[str, JSONValue] = {
            "model": self._spec.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_k,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_predict": max_tokens,
                "temperature": GREEDY_TEMPERATURE,
                "seed": SCORING_SEED,
            },
        }
        body = self._post("/api/generate", payload)
        if "logprobs" not in body:
            if _stopped_without_text(body):
                return []
            raise CapabilityError(
                f"Ollama at {self._base_url} returned no logprobs; upgrade to Ollama 0.12 or newer"
            )
        entries = body["logprobs"] or []
        return [_parse_step(entry) for entry in expect_list(entries, "Ollama logprobs")]

    def _load_info(self) -> ServerInfo:
        show = self._post("/api/show", {"model": self._spec.model})
        version = self._get("/api/version")
        details = expect_dict(show.get("details") or {}, "Ollama model details")
        listed = self._listed_entry("/api/tags") or {}
        digest = optional_str(listed.get("digest"))
        facts = {
            "quantization": optional_str(details.get("quantization_level")),
            "parameter_size": optional_str(details.get("parameter_size")),
            "trained_context_length": _optional_text(_trained_context(show.get("model_info"))),
            "server_version": optional_str(version.get("version")),
            "digest": digest,
            # Only a fallback fingerprint: it changes when a tag is re-pulled, like the digest.
            "modified_at": None if digest else optional_str(show.get("modified_at")),
        }
        context_length = _num_ctx_parameter(show.get("parameters"))
        if context_length is None:
            context_length = self._loaded_context_length()
        return ServerInfo(
            backend="ollama",
            model=self._spec.model,
            context_length=context_length,
            chat_template=optional_str(show.get("template")),
            template_dialect="go",
            supports_logprobs=True,
            exact_token_ids=False,
            details=tuple((key, value) for key, value in facts.items() if value),
            size_bytes=optional_int(listed.get("size")),
            weights_id=digest,
        )

    def _loaded_context_length(self) -> int | None:
        # A generate request without a prompt only loads the model, after which /api/ps
        # reports the context size Ollama actually allocated.
        self._post("/api/generate", {"model": self._spec.model, "keep_alive": KEEP_ALIVE})
        model = self._listed_entry("/api/ps")
        if model is None:
            logger.debug("model %s is not listed by /api/ps", self._spec.model)
            return None
        return optional_int(model.get("context_length"))

    def _listed_entry(self, path: str) -> dict[str, JSONValue] | None:
        """Find this model in a /api/ps or /api/tags listing."""
        listing = self._get(path)
        names = _tag_aliases(self._spec.model)
        for entry in expect_list(listing.get("models") or [], f"Ollama {path} models"):
            model = expect_dict(entry, f"Ollama {path} entry")
            if model.get("name") in names or model.get("model") in names:
                return model
        return None

    def _get(self, path: str) -> dict[str, JSONValue]:
        with explained_failures(self._explain):
            body = get_json(self._url(path), timeout=self._timeout)
        return expect_dict(body, f"Ollama {path} response")

    def _post(self, path: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        with explained_failures(self._explain):
            body = post_json(self._url(path), payload, timeout=self._timeout)
        return expect_dict(body, f"Ollama {path} response")

    def _explain(self, failure: RequestFailure) -> str | None:
        return explain_ollama_failure(failure, base_url=self._base_url, model=self._spec.model)

    def _url(self, path: str) -> str:
        return self._base_url + path


def explain_ollama_failure(
    failure: RequestFailure, *, base_url: str, model: str | None = None
) -> str | None:
    """Say what to do about a failed Ollama request, or return None if there is no advice."""
    if failure.status is None:
        return (
            f"cannot reach Ollama at {base_url} ({failure.detail}); "
            "start it with `ollama serve` or set OLLAMA_HOST"
        )
    if model and failure.status == 404 and mentions_missing_model(failure.detail):
        name = printable(model)
        return (
            f"model {name!r} is not available in Ollama; "
            f"run `ollama pull {name}` (see `ollama list`)"
        )
    return None


def _stopped_without_text(body: dict[str, JSONValue]) -> bool:
    return body.get("done") is True and not body.get("response")


def _parse_chat(body: dict[str, JSONValue], *, seconds: float) -> ChatResult:
    message = expect_dict(body.get("message"), "Ollama chat message")
    raw_calls = expect_list(message.get("tool_calls") or [], "Ollama tool_calls")
    return ChatResult(
        text=optional_str(message.get("content")) or "",
        tool_calls=tuple(_parse_tool_call(call) for call in raw_calls),
        finish_reason=optional_str(body.get("done_reason")),
        prompt_tokens=optional_int(body.get("prompt_eval_count")),
        completion_tokens=optional_int(body.get("eval_count")),
        seconds=seconds,
        decode_tokens_per_second=decode_rate(
            optional_int(body.get("eval_count")), _nanoseconds(body.get("eval_duration"))
        ),
    )


def _nanoseconds(value: JSONValue) -> float | None:
    """Convert one of Ollama's integer nanosecond durations to seconds."""
    duration = optional_int(value)
    return None if duration is None else duration / 1e9


def _parse_tool_call(call: JSONValue) -> ToolCall:
    function = expect_dict(expect_dict(call, "Ollama tool call").get("function"), "function")
    return tool_call(expect_str(function.get("name"), "tool call name"), function.get("arguments"))


def _parse_step(entry: JSONValue) -> TokenStep:
    step = expect_dict(entry, "Ollama logprob entry")
    top = expect_list(step.get("top_logprobs") or [], "Ollama top_logprobs")
    return TokenStep(chosen=_parse_prob(step), top=sorted_top(_parse_prob(item) for item in top))


def _parse_prob(value: JSONValue) -> TokenProb:
    item = expect_dict(value, "Ollama logprob")
    return TokenProb(
        token=expect_str(item.get("token"), "Ollama token"),
        logprob=expect_float(item.get("logprob"), "Ollama logprob"),
        token_bytes=optional_bytes(item.get("bytes")),
    )


def _num_ctx_parameter(parameters: JSONValue) -> int | None:
    """Read `num_ctx` from the Modelfile parameter block, one `name value` pair per line."""
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "num_ctx" and fields[1].isdigit():
            return int(fields[1])
    return None


def _trained_context(model_info: JSONValue) -> int | None:
    if not isinstance(model_info, dict):
        return None
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return optional_int(value)
    return None


def _optional_text(value: int | None) -> str | None:
    return None if value is None else str(value)


def _tag_aliases(tag: str) -> frozenset[str]:
    """Names /api/ps may use for `tag`; Ollama adds ':latest' when no tag is given."""
    _, _, name = tag.rpartition("/")
    if ":" in name:
        return frozenset({tag})
    return frozenset({tag, f"{tag}:latest"})
