"""The contract every model server adapter implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from quantdiff.types import (
    CandidateSpec,
    ChatResult,
    JSONValue,
    Message,
    ServerInfo,
    TokenStep,
    ToolSpec,
    TopK,
)

MAX_TOP_K = 20
"""Upper bound on top-k logprobs. Ollama and OpenAI-compatible servers cap at 20."""


@runtime_checkable
class Backend(Protocol):
    """A connection to one served model.

    All generation is greedy (temperature 0) so runs are as repeatable as the server
    allows. Implementations raise BackendError on transport or protocol failures and
    CapabilityError when an operation is unsupported, never bare exceptions.
    """

    @property
    def spec(self) -> CandidateSpec: ...

    def info(self) -> ServerInfo:
        """Describe the served model. May be called many times; implementations cache it."""
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        json_schema: dict[str, JSONValue] | None = None,
        seed: int = 0,
    ) -> ChatResult:
        """Run one chat completion through the server's own chat template."""
        ...

    def tokenize(self, text: str) -> tuple[int, ...] | None:
        """Return token ids for raw `text`, or None if the backend cannot tokenize."""
        ...

    def generate_scored(self, prompt: str, *, max_tokens: int, top_k: int) -> list[TokenStep]:
        """Greedily continue raw `prompt` (no chat template), returning top-k at each step.

        A step whose `top` is empty marks a position no candidate can be compared at, such
        as one whose reported distribution belongs to a later token; metrics skip it.
        `chosen` still holds the token, so teacher forcing can continue past it.

        Raises CapabilityError if the backend does not expose logprobs.
        """
        ...

    def score_continuation(
        self,
        prompt: str,
        continuation: Sequence[TokenStep],
        *,
        top_k: int,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> list[TopK]:
        """Teacher-force `continuation` after raw `prompt` and return the top-k next-token
        distribution at every position, so result[i] is conditioned on continuation[:i].

        When `info().exact_token_ids` is True the implementation must feed token ids
        (`prompt_token_ids` plus each step's `chosen.token_id`) instead of text, so the
        candidate sees exactly the reference's token sequence.

        result[i] is empty when the server stopped there without a distribution, or when
        the position cannot be scored: the reference left it unscored (empty `top`, sent
        without a request), or a text prompt cannot reproduce it.

        Raises CapabilityError if the backend does not expose logprobs.
        """
        ...

    def close(self) -> None:
        """Release resources. Safe to call more than once."""
        ...
