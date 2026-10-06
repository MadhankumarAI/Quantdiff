"""Pre-flight checks that catch setup mistakes before a long comparison run.

Each check is isolated: a server error inside one check becomes a "skip" finding for
that check instead of aborting the others. Checks that do not apply to the given
options (no reference, no Hugging Face repo, probe disabled) produce no finding.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from quantdiff._http import get_json
from quantdiff.backends.base import Backend
from quantdiff.errors import BackendError, SpecError
from quantdiff.types import BackendKind, Message, PreflightFinding, ServerInfo

__all__ = [
    "DEFAULT_CONTEXT_PROBE_TOKENS",
    "HF_BASE_URL",
    "MIN_CONTEXT_PROBE_TOKENS",
    "run_preflight",
]

logger = logging.getLogger(__name__)

HF_BASE_URL: Final = "https://huggingface.co"
DEFAULT_CONTEXT_PROBE_TOKENS: Final = 6000
MIN_CONTEXT_PROBE_TOKENS: Final = 1000
"""Smallest long-probe size worth running; 0 disables the probe instead."""

_CONTROL_PROBE_TOKENS: Final = 200
_WORDS_PER_TOKEN: Final = 0.75
_PROBE_ANSWER_TOKENS: Final = 16
_HF_TIMEOUT_SECONDS: Final = 15.0
_HF_MAX_BYTES: Final = 2 * 1024 * 1024
_HF_REPO_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")

_CONTEXT_FIXES: Final[Mapping[BackendKind, str]] = {
    "ollama": (
        "Set OLLAMA_CONTEXT_LENGTH={size} on the Ollama server, or add "
        "PARAMETER num_ctx {size} to a Modelfile."
    ),
    "llamacpp": "Restart llama-server with -c {size} (--ctx-size).",
    "openai": (
        "Raise the context length to at least {size} tokens in the server settings "
        "(LM Studio: Context Length; vLLM: --max-model-len)."
    ),
}
_TEMPLATE_FIX: Final = (
    "Re-download the model, or pass --chat-template-file with the upstream template."
)
_LOGPROBS_FIX: Final = "Serve the model with llama-server or another backend that returns logprobs."
_TOKENIZER_FIX: Final = "Compare quantizations of the same base model as the reference."

_TOKENIZER_PROBES: Final = (
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces,\ttabs\nand newlines\n\n",
    "def add(a: int, b: int) -> int:\n    return a + b  # sum\n",
    '{"id": 42, "tags": ["x", "y"], "ok": true}',
    "Grüße aus München, café crème brûlée",
    "你好世界。量化模型可以在普通电脑上运行",
    "नमस्ते दुनिया",
    "emoji \U0001f642\U0001f680 and numbers 3.14159 -1024 1e-9",
)

_SYLLABLES: Final = (
    "ka", "lo", "mi", "ru", "te", "va", "zo", "ne",
    "pi", "su", "do", "fa", "gi", "ho", "ju", "be",
)  # fmt: skip
_SUBJECTS: Final = (
    "The harbor master",
    "A patient gardener",
    "The night librarian",
    "An old cartographer",
    "The village baker",
    "A young engineer",
    "The ferry captain",
    "A careful auditor",
    "The museum guide",
    "A retired teacher",
    "The orchard keeper",
    "A traveling violinist",
    "The station clerk",
)
_VERBS: Final = (
    "quietly repaired",
    "carefully counted",
    "slowly painted",
    "proudly displayed",
    "briefly studied",
    "neatly folded",
    "gently polished",
    "patiently sorted",
    "happily described",
    "firmly secured",
    "openly admired",
)
_OBJECTS: Final = (
    "a wooden bench",
    "the brass lanterns",
    "several faded maps",
    "a basket of pears",
    "the winter blankets",
    "an iron gate",
    "a stack of letters",
)
_PLACES: Final = (
    "near the river before noon",
    "beside the old market square",
    "in the garden after the rain",
    "at the edge of the quiet town",
    "under the tall chestnut trees",
)
_SENTENCES_PER_PARAGRAPH: Final = 8


def run_preflight(
    backend: Backend,
    *,
    reference: Backend | None = None,
    hf_repo: str | None = None,
    offline: bool = False,
    context_probe_tokens: int = DEFAULT_CONTEXT_PROBE_TOKENS,
    hf_base_url: str = HF_BASE_URL,
) -> tuple[PreflightFinding, ...]:
    """Run the pre-flight checks for `backend` and return their findings in order.

    `context_probe_tokens` sets the size of the long-prompt truncation probe; 0 turns
    the context check off. `hf_repo` (such as "Qwen/Qwen2.5-7B-Instruct") enables the
    chat template comparison. `reference` enables the tokenizer comparison.
    """
    if hf_repo is not None and not _HF_REPO_PATTERN.fullmatch(hf_repo):
        raise SpecError(f"invalid Hugging Face repo {hf_repo!r}; expected owner/name")
    if context_probe_tokens != 0 and context_probe_tokens < MIN_CONTEXT_PROBE_TOKENS:
        raise SpecError(
            f"context probe must be 0 (off) or at least {MIN_CONTEXT_PROBE_TOKENS} tokens"
        )

    findings = _isolated("logprobs", lambda: (_check_logprobs(backend),))
    if context_probe_tokens:
        findings += _isolated("context", lambda: _check_context(backend, context_probe_tokens))
    if hf_repo is not None:
        findings += _isolated(
            "template",
            lambda: (_check_template(backend, hf_repo, offline=offline, base_url=hf_base_url),),
        )
    if reference is not None:
        findings += _isolated("tokenizer", lambda: (_check_tokenizer(backend, reference),))
    for finding in findings:
        logger.debug("preflight %s %s: %s", finding.check, finding.severity, finding.message)
    return tuple(findings)


def _isolated(check: str, run: Callable[[], Sequence[PreflightFinding]]) -> list[PreflightFinding]:
    try:
        return list(run())
    except BackendError as exc:
        return [PreflightFinding(check, "skip", f"check could not run: {exc}")]


# logprobs --------------------------------------------------------------------------------


def _check_logprobs(backend: Backend) -> PreflightFinding:
    if backend.info().supports_logprobs:
        return PreflightFinding("logprobs", "ok", "logprobs available")
    return PreflightFinding(
        "logprobs", "warn", "logit metrics unavailable; task metrics only", _LOGPROBS_FIX
    )


# context ---------------------------------------------------------------------------------


def _check_context(backend: Backend, probe_tokens: int) -> tuple[PreflightFinding, ...]:
    info = backend.info()
    fix = _CONTEXT_FIXES[info.backend].format(size=_suggested_context(probe_tokens))
    findings = []
    if info.context_length is not None and info.context_length < probe_tokens:
        findings.append(
            PreflightFinding(
                "context",
                "warn",
                f"context window is {info.context_length} tokens, below the "
                f"{probe_tokens}-token probe; long prompts will not fit",
                fix,
            )
        )
    findings.append(_needle_probe(backend, probe_tokens, fix))
    return tuple(findings)


def _needle_probe(backend: Backend, probe_tokens: int, fix: str) -> PreflightFinding:
    """Check that text at the very start of a long prompt still reaches the model.

    Servers that silently drop the front of an over-long prompt return fluent answers
    that are simply wrong, so the probe hides a code word at the start and asks for it
    at the end. A short control probe first rules out models that cannot do the task.
    """
    try:
        control_passed = _recalls_code_word(backend, _CONTROL_PROBE_TOKENS, "control")
    except BackendError as exc:
        return PreflightFinding("context", "skip", f"short probe failed: {exc}")
    if not control_passed:
        return PreflightFinding(
            "context",
            "skip",
            "model could not answer the probe at short length; context check inconclusive",
        )
    try:
        long_passed = _recalls_code_word(backend, probe_tokens, "long")
    except BackendError as exc:
        return PreflightFinding("context", "warn", f"{probe_tokens}-token probe failed: {exc}", fix)
    if not long_passed:
        return PreflightFinding(
            "context", "fail", "front of long prompts is being dropped (silent truncation)", fix
        )
    return PreflightFinding(
        "context", "ok", f"model recalled the start of a {probe_tokens}-token prompt"
    )


def _recalls_code_word(backend: Backend, probe_tokens: int, salt: str) -> bool:
    code_word = _code_word(salt)
    prompt = (
        f"Remember this code word: {code_word}.\n\n"
        f"{_filler(int(probe_tokens * _WORDS_PER_TOKEN))}\n\n"
        "What was the code word given at the very start of this message? "
        "Reply with the code word only."
    )
    result = backend.chat([Message(role="user", content=prompt)], max_tokens=_PROBE_ANSWER_TOKENS)
    return _alphanumeric(code_word) in _alphanumeric(result.text)


def _code_word(salt: str) -> str:
    digest = hashlib.sha256(f"quantdiff-needle-{salt}".encode()).digest()
    word = "".join(_SYLLABLES[byte % len(_SYLLABLES)] for byte in digest[:4])
    number = int.from_bytes(digest[4:6], "big") % 9000 + 1000
    return f"{word.capitalize()}-{number}"


def _filler(word_count: int) -> str:
    """Return about `word_count` words of varied, deterministic, meaningless prose."""
    paragraphs: list[str] = []
    sentences: list[str] = []
    words = 0
    index = 0
    while words < word_count:
        sentence = (
            f"{_SUBJECTS[index % len(_SUBJECTS)]} {_VERBS[index % len(_VERBS)]} "
            f"{_OBJECTS[index % len(_OBJECTS)]} {_PLACES[index % len(_PLACES)]}."
        )
        sentences.append(sentence)
        words += len(sentence.split())
        index += 1
        if len(sentences) == _SENTENCES_PER_PARAGRAPH:
            paragraphs.append(" ".join(sentences))
            sentences = []
    if sentences:
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def _alphanumeric(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _suggested_context(probe_tokens: int) -> int:
    """Smallest power of two that fits the probe plus room for an answer."""
    size = 1024
    while size < probe_tokens + 512:
        size *= 2
    return size


# chat template ---------------------------------------------------------------------------


def _check_template(
    backend: Backend, repo: str, *, offline: bool, base_url: str
) -> PreflightFinding:
    info = backend.info()
    reason = _template_skip_reason(info, repo, offline=offline)
    if reason is not None:
        return PreflightFinding("template", "skip", reason)
    upstream = _upstream_template(repo, base_url)
    if upstream is None:
        return PreflightFinding(
            "template", "skip", f"upstream {repo} tokenizer_config.json has no chat_template"
        )
    if _squash(upstream) == _squash(info.chat_template or ""):
        return PreflightFinding("template", "ok", f"embedded chat template matches upstream {repo}")
    return PreflightFinding(
        "template", "warn", f"embedded chat template differs from upstream {repo}", _TEMPLATE_FIX
    )


def _template_skip_reason(info: ServerInfo, repo: str, *, offline: bool) -> str | None:
    if info.template_dialect == "go":
        return "Ollama templates are Go templates; compare not supported"
    if info.template_dialect != "jinja":
        return "server does not report a Jinja chat template"
    if offline:
        return f"offline; upstream template for {repo} not fetched"
    if info.chat_template is None:
        return "server did not report its chat template"
    return None


def _upstream_template(repo: str, base_url: str) -> str | None:
    url = f"{base_url.rstrip('/')}/{repo}/raw/main/tokenizer_config.json"
    config = get_json(url, timeout=_HF_TIMEOUT_SECONDS, max_bytes=_HF_MAX_BYTES)
    if not isinstance(config, dict):
        raise BackendError(f"{url} did not return a JSON object")
    template = config.get("chat_template")
    if isinstance(template, str):
        return template
    if isinstance(template, list):
        for entry in template:
            if (
                isinstance(entry, dict)
                and entry.get("name") == "default"
                and isinstance(entry.get("template"), str)
            ):
                return str(entry["template"])
    return None


def _squash(text: str) -> str:
    return " ".join(text.split())


# tokenizer -------------------------------------------------------------------------------


def _check_tokenizer(backend: Backend, reference: Backend) -> PreflightFinding:
    for probe in _TOKENIZER_PROBES:
        candidate_ids = backend.tokenize(probe)
        if candidate_ids is None:
            return _tokenizer_skip(backend)
        reference_ids = reference.tokenize(probe)
        if reference_ids is None:
            return _tokenizer_skip(reference)
        if candidate_ids != reference_ids:
            return PreflightFinding(
                "tokenizer",
                "fail",
                "tokenizers differ from the reference; logit metrics are not comparable",
                _TOKENIZER_FIX,
            )
    return PreflightFinding(
        "tokenizer",
        "ok",
        f"tokenizer matches the reference on {len(_TOKENIZER_PROBES)} probe strings",
    )


def _tokenizer_skip(backend: Backend) -> PreflightFinding:
    return PreflightFinding(
        "tokenizer", "skip", f"{backend.spec.label} cannot tokenize; tokenizers not compared"
    )
