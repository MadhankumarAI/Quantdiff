"""In-memory Backend used across the test suite. No network, fully deterministic."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from quantdiff.errors import BackendError, CapabilityError
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

ChatHandler = Callable[[Sequence[Message], Sequence[ToolSpec]], ChatResult]


def text_result(text: str, *, seconds: float = 0.1, completion_tokens: int = 10) -> ChatResult:
    return ChatResult(
        text=text,
        tool_calls=(),
        finish_reason="stop",
        prompt_tokens=20,
        completion_tokens=completion_tokens,
        seconds=seconds,
    )


def make_topk(*pairs: tuple[str, float], with_ids: bool = True) -> TopK:
    """Build a sorted top-k tuple from (token, logprob) pairs. Ids are stable per token."""
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    return tuple(
        TokenProb(token=token, logprob=logprob, token_id=_stable_id(token) if with_ids else None)
        for token, logprob in ordered
    )


def _stable_id(token: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(token))


@dataclass
class FakeBackend:
    """A scripted Backend. Set `chat_handler`, `steps` and `scores` to control outputs."""

    label: str = "fake"
    context_length: int | None = 4096
    chat_template: str | None = "{{ messages }}"
    supports_logprobs: bool = True
    exact_token_ids: bool = True
    weights_id: str | None = None
    chat_handler: ChatHandler | None = None
    steps: list[TokenStep] = field(default_factory=list)
    scores: list[TopK] = field(default_factory=list)
    score_error: BackendError | None = None
    """Raised by score_continuation when set, to simulate a server failure mid-run."""
    chat_calls: list[tuple[Message, ...]] = field(default_factory=list)
    closed: bool = False

    @property
    def spec(self) -> CandidateSpec:
        return CandidateSpec(
            kind="openai", base_url="http://fake", model=self.label, label=self.label
        )

    def info(self) -> ServerInfo:
        return ServerInfo(
            backend="openai",
            model=self.label,
            context_length=self.context_length,
            chat_template=self.chat_template,
            template_dialect="jinja",
            supports_logprobs=self.supports_logprobs,
            exact_token_ids=self.exact_token_ids,
            weights_id=self.weights_id,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        json_schema: dict[str, JSONValue] | None = None,
        seed: int = 0,
    ) -> ChatResult:
        self.chat_calls.append(tuple(messages))
        if self.chat_handler is None:
            return text_result("ok")
        return self.chat_handler(messages, tools)

    def tokenize(self, text: str) -> tuple[int, ...] | None:
        if not self.exact_token_ids:
            return None
        return tuple(_stable_id(word) for word in text.split())

    def generate_scored(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        self._require_logprobs()
        return self.steps[:max_tokens]

    def score_continuation(
        self,
        prompt: str,
        continuation: Sequence[TokenStep],
        *,
        top_k: int,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> list[TopK]:
        self._require_logprobs()
        if self.score_error is not None:
            raise self.score_error
        return self.scores[: len(continuation)]

    def close(self) -> None:
        self.closed = True

    def _require_logprobs(self) -> None:
        if not self.supports_logprobs:
            raise CapabilityError(f"{self.label} does not expose logprobs")
