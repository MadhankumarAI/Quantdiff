"""On-disk cache of reference model outputs.

Running the reference is the most expensive part of a comparison, and its outputs do not
change between runs with the same model, suite and settings. Entries are plain JSON, keyed
by a SHA-256 digest of everything that can change the output, and written atomically.
A corrupt or mismatched entry is treated as a miss, never as an error.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from quantdiff.types import (
    ChatResult,
    JSONValue,
    ReferenceTrace,
    ServerInfo,
    TokenProb,
    TokenStep,
    ToolCall,
)

logger = logging.getLogger(__name__)

CACHE_FORMAT: Final = 4
"""Bumped whenever the entry layout changes, so older entries are misses."""
_MAX_ENTRY_BYTES: Final = 256 * 1024 * 1024


def default_cache_dir() -> Path:
    """Return the per-user cache directory, honoring QUANTDIFF_CACHE_DIR first."""
    override = os.environ.get("QUANTDIFF_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "quantdiff" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "quantdiff"


@dataclass(frozen=True, slots=True)
class ReferenceOutputs:
    """Everything the runner needs from the reference model."""

    answers: dict[str, ChatResult]
    """Chat answers keyed by TaskCase id."""
    traces: tuple[ReferenceTrace, ...]
    """Greedy scored continuations, empty when the reference exposes no logprobs."""


def cache_key(
    info: ServerInfo,
    *,
    base_url: str,
    suite_digest: str,
    top_k: int,
    score_tokens: int,
    seed: int,
) -> str:
    """Digest of every input that can change the reference outputs.

    `info.details` carries each backend's fingerprint of the served weights (an Ollama
    digest, a GGUF file size) and the chat template is included directly, so re-pulling a
    fixed upload or a patched template invalidates the entry.
    """
    material = {
        "format": CACHE_FORMAT,
        "backend": info.backend,
        "base_url": base_url,
        "model": info.model,
        "chat_template": info.chat_template,
        "details": sorted(info.details),
        "exact_token_ids": info.exact_token_ids,
        "suite": suite_digest,
        "top_k": top_k,
        "score_tokens": score_tokens,
        "seed": seed,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReferenceCache:
    """A directory of `<key>.json` entries."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory if directory is not None else default_cache_dir()

    def load(self, key: str) -> ReferenceOutputs | None:
        path = self._path(key)
        try:
            if path.stat().st_size > _MAX_ENTRY_BYTES:
                logger.warning("ignoring oversized cache entry %s", path)
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return _outputs_from_dict(data)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("ignoring unreadable cache entry %s: %s", path, exc)
            return None

    def store(self, key: str, outputs: ReferenceOutputs) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_outputs_to_dict(outputs), ensure_ascii=False, separators=(",", ":"))
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".tmp-", suffix=".json")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            tmp_path.replace(self._path(key))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _path(self, key: str) -> Path:
        if len(key) != 64 or not all(char in "0123456789abcdef" for char in key):
            raise ValueError(f"invalid cache key: {key!r}")
        return self.directory / f"{key}.json"


# Serialization ----------------------------------------------------------------------------


def _outputs_to_dict(outputs: ReferenceOutputs) -> dict[str, JSONValue]:
    return {
        "format": CACHE_FORMAT,
        "answers": {case_id: _chat_to_dict(result) for case_id, result in outputs.answers.items()},
        "traces": [_trace_to_dict(trace) for trace in outputs.traces],
    }


def _outputs_from_dict(data: JSONValue) -> ReferenceOutputs:
    if not isinstance(data, dict) or data.get("format") != CACHE_FORMAT:
        raise ValueError("unsupported cache format")
    answers = {str(case_id): _chat_from_dict(item) for case_id, item in data["answers"].items()}
    traces = tuple(_trace_from_dict(item) for item in data["traces"])
    return ReferenceOutputs(answers=answers, traces=traces)


def _chat_to_dict(result: ChatResult) -> dict[str, JSONValue]:
    return {
        "text": result.text,
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments, "raw_arguments": call.raw_arguments}
            for call in result.tool_calls
        ],
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "seconds": result.seconds,
        "decode_tokens_per_second": result.decode_tokens_per_second,
    }


def _chat_from_dict(data: dict[str, JSONValue]) -> ChatResult:
    calls = tuple(
        ToolCall(
            name=str(call["name"]),
            arguments=call["arguments"] if isinstance(call["arguments"], dict) else None,
            raw_arguments=str(call["raw_arguments"]),
        )
        for call in data["tool_calls"]
    )
    return ChatResult(
        text=str(data["text"]),
        tool_calls=calls,
        finish_reason=_optional_str(data["finish_reason"]),
        prompt_tokens=_optional_int(data["prompt_tokens"]),
        completion_tokens=_optional_int(data["completion_tokens"]),
        seconds=float(data["seconds"]),
        decode_tokens_per_second=_optional_float(data["decode_tokens_per_second"]),
    )


def _trace_to_dict(trace: ReferenceTrace) -> dict[str, JSONValue]:
    return {
        "prompt_id": trace.prompt_id,
        "prompt_token_ids": None
        if trace.prompt_token_ids is None
        else list(trace.prompt_token_ids),
        "steps": [
            {"chosen": _prob_to_list(step.chosen), "top": [_prob_to_list(p) for p in step.top]}
            for step in trace.steps
        ],
    }


def _trace_from_dict(data: dict[str, JSONValue]) -> ReferenceTrace:
    ids = data["prompt_token_ids"]
    return ReferenceTrace(
        prompt_id=str(data["prompt_id"]),
        prompt_token_ids=None if ids is None else tuple(int(i) for i in ids),
        steps=tuple(
            TokenStep(
                chosen=_prob_from_list(step["chosen"]),
                top=tuple(_prob_from_list(p) for p in step["top"]),
            )
            for step in data["steps"]
        ),
    )


def _prob_to_list(prob: TokenProb) -> list[JSONValue]:
    """[token, logprob, token id or null, token bytes as base64 or null]."""
    raw = None if prob.token_bytes is None else base64.b64encode(prob.token_bytes).decode("ascii")
    return [prob.token, prob.logprob, prob.token_id, raw]


def _prob_from_list(data: JSONValue) -> TokenProb:
    token, logprob, token_id, raw = data
    return TokenProb(
        token=str(token),
        logprob=float(logprob),
        token_id=None if token_id is None else int(token_id),
        token_bytes=None if raw is None else base64.b64decode(str(raw), validate=True),
    )


def _optional_str(value: JSONValue) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: JSONValue) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: JSONValue) -> float | None:
    return None if value is None else float(value)
