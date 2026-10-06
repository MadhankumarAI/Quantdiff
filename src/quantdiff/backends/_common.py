"""Helpers shared by the backend adapters: payload validation, OpenAI chat format, and
turning raw HTTP failures into messages that tell the user what to do."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from quantdiff._text import printable
from quantdiff.backends.base import MAX_TOP_K
from quantdiff.errors import BackendError, CapabilityError, RequestError, SpecError
from quantdiff.types import (
    ChatResult,
    JSONValue,
    Message,
    TokenProb,
    TokenStep,
    ToolCall,
    ToolSpec,
    TopK,
)

GREEDY_TEMPERATURE: Final = 0.0
SCORING_SEED: Final = 0
"""Seed for raw scoring requests. Greedy decoding ignores it, but some servers want one."""
_MAX_BYTE: Final = 255


# Request arguments -----------------------------------------------------------------------


def check_top_k(top_k: int) -> None:
    if not 1 <= top_k <= MAX_TOP_K:
        raise SpecError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")


def check_max_tokens(max_tokens: int) -> None:
    if max_tokens < 1:
        raise SpecError(f"max_tokens must be at least 1, got {max_tokens}")


def forced_texts(prompt: str, continuation: Sequence[TokenStep]) -> list[str | None]:
    """Return the prompt text that teacher-forces each continuation position, or None for a
    position that text cannot reproduce.

    Prefixes are rebuilt from each chosen token's bytes when the server reported them,
    because a token that holds only part of a UTF-8 character has no exact text of its own;
    tokens without bytes contribute their text. A position is None when the reference left
    it unscored (its `top` is empty) or when its prefix ends inside a character, since a
    prompt must be valid text.
    """
    texts: list[str | None] = []
    prefix = bytearray(prompt.encode("utf-8"))
    for step in continuation:
        texts.append(_decoded(prefix) if step.top else None)
        chosen = step.chosen
        prefix += chosen.token.encode("utf-8") if chosen.token_bytes is None else chosen.token_bytes
    return texts


def _decoded(data: bytearray) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# Folded tokens ---------------------------------------------------------------------------


def unfolded_steps(steps: Iterable[TokenStep]) -> list[TokenStep]:
    """Keep every step, but empty the `top` of each step that folds several tokens.

    Ollama and llama-server hold back a token that ends inside a UTF-8 character and report
    it together with the tokens that complete the character, as one entry whose text is
    the whole character but whose logprob and alternatives belong to the last of those
    tokens. That distribution sits one or more tokens after the position the entry stands
    for, so comparing it with a teacher-forced candidate would compare different
    positions. An empty `top` marks the position as unscored; metrics skip it.
    """
    return [TokenStep(step.chosen, ()) if folds_tokens(step) else step for step in steps]


def folds_tokens(step: TokenStep) -> bool:
    """True when the server folded several tokens into this step (see `unfolded_steps`).

    Greedy decoding picks the head of the distribution, so a step's own token is always in
    its `top`. A folded step's text is missing from that list, which holds only the
    character fragments the last folded token chose between, or is listed with different
    bytes when the server matches it by id.
    """
    chosen = step.chosen
    own = next((prob for prob in step.top if _same_token(prob, chosen)), None)
    if own is None:
        return bool(step.top)
    if own.token_bytes is None or chosen.token_bytes is None:
        return False
    return own.token_bytes != chosen.token_bytes


def _same_token(prob: TokenProb, chosen: TokenProb) -> bool:
    if prob.token_id is not None and chosen.token_id is not None:
        return prob.token_id == chosen.token_id
    return prob.token == chosen.token


# Response validation ---------------------------------------------------------------------


def expect_dict(value: JSONValue, where: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise BackendError(f"{where}: expected a JSON object, got {type(value).__name__}")
    return value


def expect_list(value: JSONValue, where: str) -> list[JSONValue]:
    if not isinstance(value, list):
        raise BackendError(f"{where}: expected a JSON array, got {type(value).__name__}")
    return value


def expect_str(value: JSONValue, where: str) -> str:
    if not isinstance(value, str):
        raise BackendError(f"{where}: expected a string, got {type(value).__name__}")
    return value


def expect_float(value: JSONValue, where: str) -> float:
    """Return a finite number. Metrics cannot use NaN or infinity, and json.loads turns an
    out-of-range literal such as 1e999 into infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendError(f"{where}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise BackendError(f"{where}: expected a finite number, got {number}")
    return number


def optional_int(value: JSONValue) -> int | None:
    """Return `value` if it is a JSON integer, else None. Booleans are not integers."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def optional_str(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


def optional_bytes(value: JSONValue) -> bytes | None:
    """Read a token's `bytes` field, a JSON array of byte values, or return None if absent
    or malformed."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    if not all(0 <= item <= _MAX_BYTE for item in value):
        return None
    return bytes(value)


def decode_rate(tokens: int | None, seconds: float | None) -> float | None:
    """Return the server-measured decode speed, or None when it is not meaningful.

    A lone generated token is sampled from the prompt-processing pass, so servers report
    a decode time near zero for it and the quotient would be absurd.
    """
    if tokens is None or tokens < 2 or seconds is None or seconds <= 0:
        return None
    rate = tokens / seconds
    return rate if math.isfinite(rate) else None


def sorted_top(probs: Iterable[TokenProb]) -> TopK:
    """Sort by descending logprob, keeping server order for ties."""
    return tuple(sorted(probs, key=lambda prob: prob.logprob, reverse=True))


def logprob_from_prob(prob: float, where: str) -> float:
    if not 0.0 < prob <= 1.0:
        raise BackendError(f"{where}: probability {prob} is outside (0, 1]")
    return math.log(prob)


# OpenAI chat format ----------------------------------------------------------------------


def openai_tool(tool: ToolSpec) -> dict[str, JSONValue]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def openai_messages(messages: Sequence[Message]) -> list[JSONValue]:
    return [{"role": message.role, "content": message.content} for message in messages]


def openai_chat_payload(
    model: str,
    messages: Sequence[Message],
    *,
    max_tokens: int,
    tools: Sequence[ToolSpec],
    json_schema: dict[str, JSONValue] | None,
    seed: int,
) -> dict[str, JSONValue]:
    """Build a greedy /chat/completions request. An empty `model` is left out."""
    check_max_tokens(max_tokens)
    payload: dict[str, JSONValue] = {
        "messages": openai_messages(messages),
        "max_tokens": max_tokens,
        "temperature": GREEDY_TEMPERATURE,
        "seed": seed,
        "stream": False,
    }
    if model:
        payload["model"] = model
    if tools:
        payload["tools"] = [openai_tool(tool) for tool in tools]
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": json_schema},
        }
    return payload


def parse_openai_chat(body: JSONValue, *, seconds: float) -> ChatResult:
    """Read the first choice of a /chat/completions response."""
    response = expect_dict(body, "chat response")
    choices = expect_list(response.get("choices"), "chat response choices")
    if not choices:
        raise BackendError("chat response has no choices")
    choice = expect_dict(choices[0], "chat choice")
    message = expect_dict(choice.get("message"), "chat message")
    raw_calls = message.get("tool_calls") or []
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return ChatResult(
        text=optional_str(message.get("content")) or "",
        tool_calls=tuple(
            parse_openai_tool_call(call) for call in expect_list(raw_calls, "tool_calls")
        ),
        finish_reason=optional_str(choice.get("finish_reason")),
        prompt_tokens=optional_int(usage.get("prompt_tokens")),
        completion_tokens=optional_int(usage.get("completion_tokens")),
        seconds=seconds,
    )


def parse_openai_tool_call(call: JSONValue) -> ToolCall:
    function = expect_dict(expect_dict(call, "tool call").get("function"), "tool call function")
    name = expect_str(function.get("name"), "tool call name")
    return tool_call(name, function.get("arguments"))


def tool_call(name: str, arguments: JSONValue) -> ToolCall:
    """Build a ToolCall from arguments sent either as a JSON string or as an object."""
    if isinstance(arguments, str):
        return ToolCall(name=name, arguments=parse_arguments(arguments), raw_arguments=arguments)
    raw = json.dumps(arguments)
    return ToolCall(
        name=name,
        arguments=arguments if isinstance(arguments, dict) else None,
        raw_arguments=raw,
    )


def parse_arguments(raw: str) -> dict[str, JSONValue] | None:
    """Decode tool arguments, or return None when they are not a JSON object."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


# Friendly failures -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestFailure:
    """A failed request, taken from the RequestError that quantdiff._http raised."""

    url: str
    status: int | None
    """HTTP status, or None when the server could not be reached at all."""
    detail: str
    """Response body preview for status errors, else the transport error reason."""


def request_failure(exc: BackendError) -> RequestFailure | None:
    """Classify `exc`, or return None for failures that are not about reaching the server."""
    if not isinstance(exc, RequestError):
        return None
    if exc.status is not None:
        return RequestFailure(exc.url, exc.status, exc.detail)
    return RequestFailure(exc.url, None, _short_reason(exc.detail))


Explainer = Callable[[RequestFailure], str | None]


@contextmanager
def explained_failures(explain: Explainer) -> Iterator[None]:
    """Re-raise request failures that `explain` recognizes as a BackendError with its message.

    The original error stays available as `__cause__`. Failures `explain` does not recognize
    propagate unchanged.
    """
    try:
        yield
    except CapabilityError:
        raise
    except BackendError as exc:
        failure = request_failure(exc)
        message = None if failure is None else explain(failure)
        if message is None:
            raise
        raise BackendError(message) from exc


def mentions_missing_model(detail: str) -> bool:
    """True for error bodies such as `model 'x' not found` or `The model x does not exist`."""
    lowered = detail.lower()
    return "model" in lowered and ("not found" in lowered or "does not exist" in lowered)


def _short_reason(reason: str) -> str:
    # Windows reports a refused connection as a long WinError sentence.
    if "refused" in reason.lower():
        return "connection refused"
    return printable(reason)
