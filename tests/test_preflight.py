from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantdiff.errors import BackendError, SpecError
from quantdiff.preflight import run_preflight
from quantdiff.types import (
    BackendKind,
    ChatResult,
    JSONValue,
    Message,
    PreflightFinding,
    ServerInfo,
    TemplateDialect,
    ToolSpec,
)
from tests.fakes import ChatHandler, FakeBackend, text_result

SERVER_TEMPLATE = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
CODE_WORD = re.compile(r"code word: ([A-Za-z]+-\d+)\.")


@dataclass
class ScriptedBackend(FakeBackend):
    """FakeBackend with a configurable server kind, template dialect and tokenizer."""

    kind: BackendKind = "openai"
    dialect: TemplateDialect = "jinja"
    token_offset: int = 0
    info_error: str | None = None
    chat_error_above_words: int | None = None
    max_tokens_seen: list[int] = field(default_factory=list)

    def info(self) -> ServerInfo:
        if self.info_error is not None:
            raise BackendError(self.info_error)
        return replace(super().info(), backend=self.kind, template_dialect=self.dialect)

    def tokenize(self, text: str) -> tuple[int, ...] | None:
        ids = super().tokenize(text)
        return None if ids is None else tuple(token + self.token_offset for token in ids)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        json_schema: dict[str, JSONValue] | None = None,
        seed: int = 0,
    ) -> ChatResult:
        self.max_tokens_seen.append(max_tokens)
        words = len(messages[-1].content.split())
        if self.chat_error_above_words is not None and words > self.chat_error_above_words:
            raise BackendError("HTTP 400: prompt exceeds context window")
        return super().chat(
            messages, max_tokens=max_tokens, tools=tools, json_schema=json_schema, seed=seed
        )


def sees_last_words(window: int | None) -> ChatHandler:
    """A model that answers correctly when the code word is within its last `window` words."""

    def handler(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
        words = messages[-1].content.split()
        visible = " ".join(words if window is None else words[-window:])
        match = CODE_WORD.search(visible)
        return text_result(f"The code word is {match.group(1)}." if match else "I do not know.")

    return handler


def capable(**overrides: object) -> ScriptedBackend:
    backend = ScriptedBackend(
        context_length=32768, chat_template=SERVER_TEMPLATE, chat_handler=sees_last_words(None)
    )
    for name, value in overrides.items():
        setattr(backend, name, value)
    return backend


def only(findings: Sequence[PreflightFinding], check: str) -> PreflightFinding:
    matching = [finding for finding in findings if finding.check == check]
    assert len(matching) == 1, findings
    return matching[0]


# Hugging Face stand-in -------------------------------------------------------------------

HF_ROUTES: dict[str, JSONValue] = {
    "org/same": {
        "chat_template": "  {% for m in messages %}{{ m.role }}:  {{ m.content }}\n\n{% endfor %}"
    },
    "org/different": {"chat_template": "{{ bos_token }}" + SERVER_TEMPLATE},
    "org/named": {
        "chat_template": [
            {"name": "tool_use", "template": "{{ tools }}"},
            {"name": "default", "template": SERVER_TEMPLATE},
        ]
    },
    "org/named-no-default": {"chat_template": [{"name": "rag", "template": "{{ docs }}"}]},
    "org/no-template": {"bos_token": "<s>"},
    "org/not-object": [1, 2, 3],
}


class _HubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        match = re.fullmatch(r"/([^/]+/[^/]+)/raw/main/tokenizer_config\.json", self.path)
        repo = match.group(1) if match else None
        if repo in HF_ROUTES:
            status, body = 200, json.dumps(HF_ROUTES[repo]).encode()
        else:
            status, body = 404, b"Entry not found"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        return


@pytest.fixture(scope="module")
def hub() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def template_finding(
    hub: str, repo: str, backend: ScriptedBackend | None = None
) -> PreflightFinding:
    findings = run_preflight(
        backend or capable(), hf_repo=repo, context_probe_tokens=0, hf_base_url=hub
    )
    return only(findings, "template")


# Whole run -------------------------------------------------------------------------------


def test_healthy_setup_reports_ok_for_every_check_in_order(hub: str) -> None:
    findings = run_preflight(capable(), reference=capable(), hf_repo="org/same", hf_base_url=hub)
    assert [finding.check for finding in findings] == [
        "logprobs",
        "context",
        "template",
        "tokenizer",
    ]
    assert {finding.severity for finding in findings} == {"ok"}


def test_checks_that_do_not_apply_are_omitted() -> None:
    findings = run_preflight(capable(), context_probe_tokens=0)
    assert [finding.check for finding in findings] == ["logprobs"]


def test_server_errors_become_skips_instead_of_aborting() -> None:
    backend = capable(info_error="connection refused")
    findings = run_preflight(
        backend, reference=capable(), hf_repo="org/same", hf_base_url="http://127.0.0.1:9"
    )
    by_check = {finding.check: finding for finding in findings}
    for check in ("logprobs", "context", "template"):
        assert by_check[check].severity == "skip"
        assert "could not run: connection refused" in by_check[check].message
    assert by_check["tokenizer"].severity == "ok"


@pytest.mark.parametrize("tokens", [-5, 1, 999])
def test_rejects_probe_sizes_too_small_to_mean_anything(tokens: int) -> None:
    with pytest.raises(SpecError, match="context probe"):
        run_preflight(capable(), context_probe_tokens=tokens)


# logprobs --------------------------------------------------------------------------------


def test_missing_logprobs_is_a_warning() -> None:
    finding = only(
        run_preflight(capable(supports_logprobs=False), context_probe_tokens=0), "logprobs"
    )
    assert finding.severity == "warn"
    assert finding.message == "logit metrics unavailable; task metrics only"
    assert finding.fix


# context ---------------------------------------------------------------------------------


def test_context_probe_passes_when_the_whole_prompt_is_seen() -> None:
    backend = capable()
    finding = only(run_preflight(backend), "context")
    assert finding.severity == "ok"
    assert backend.max_tokens_seen == [16, 16]


@pytest.mark.parametrize(
    ("kind", "fix_hint"),
    [
        ("ollama", "OLLAMA_CONTEXT_LENGTH=8192"),
        ("llamacpp", "-c 8192"),
        ("openai", "--max-model-len"),
    ],
)
def test_silent_front_truncation_fails_with_backend_specific_fix(
    kind: BackendKind, fix_hint: str
) -> None:
    backend = capable(kind=kind, chat_handler=sees_last_words(1000))
    finding = only(run_preflight(backend), "context")
    assert finding.severity == "fail"
    assert finding.message == "front of long prompts is being dropped (silent truncation)"
    assert finding.fix is not None
    assert fix_hint in finding.fix


def test_model_that_fails_the_short_control_makes_the_check_inconclusive() -> None:
    backend = capable(chat_handler=lambda messages, tools: text_result("No idea."))
    finding = only(run_preflight(backend), "context")
    assert finding.severity == "skip"
    assert "inconclusive" in finding.message
    assert len(backend.chat_calls) == 1


def test_small_reported_context_warns_and_still_probes() -> None:
    backend = capable(context_length=4096, chat_handler=sees_last_words(3000))
    findings = [f for f in run_preflight(backend) if f.check == "context"]
    assert [f.severity for f in findings] == ["warn", "fail"]
    assert "4096" in findings[0].message


def test_unknown_context_length_gives_no_size_warning() -> None:
    findings = [f for f in run_preflight(capable(context_length=None)) if f.check == "context"]
    assert [f.severity for f in findings] == ["ok"]


def test_long_probe_rejected_by_server_is_a_warning() -> None:
    backend = capable(chat_error_above_words=2000)
    finding = only(run_preflight(backend), "context")
    assert finding.severity == "warn"
    assert "prompt exceeds context window" in finding.message
    assert finding.fix


def test_short_probe_error_is_a_skip() -> None:
    backend = capable(chat_error_above_words=0)
    finding = only(run_preflight(backend), "context")
    assert finding.severity == "skip"
    assert "short probe failed" in finding.message


def test_probe_prompts_are_deterministic_sized_and_varied() -> None:
    first, second = capable(), capable()
    run_preflight(first, context_probe_tokens=4000)
    run_preflight(second, context_probe_tokens=4000)
    assert first.chat_calls == second.chat_calls
    (control,), (long,) = first.chat_calls
    control_code = CODE_WORD.search(control.content)
    long_code = CODE_WORD.search(long.content)
    assert control_code is not None
    assert long_code is not None
    assert control_code.group(1) != long_code.group(1)
    assert long.content.startswith("Remember this code word:")
    assert long.content.rstrip().endswith("Reply with the code word only.")
    assert 140 <= len(control.content.split()) <= 260
    assert 3000 <= len(long.content.split()) <= 3100
    sentences = [s for s in re.split(r"(?<=\.)\s+", long.content) if s]
    assert len(set(sentences)) > 0.95 * len(sentences)


def test_disabled_probe_sends_no_chat_requests() -> None:
    backend = capable()
    run_preflight(backend, context_probe_tokens=0)
    assert backend.chat_calls == []


# template --------------------------------------------------------------------------------


def test_template_matching_upstream_after_whitespace_normalization(hub: str) -> None:
    finding = template_finding(hub, "org/same")
    assert finding.severity == "ok"


def test_template_from_named_list_uses_the_default_entry(hub: str) -> None:
    assert template_finding(hub, "org/named").severity == "ok"


def test_template_differing_from_upstream_warns_with_fix(hub: str) -> None:
    finding = template_finding(hub, "org/different")
    assert finding.severity == "warn"
    assert finding.message == "embedded chat template differs from upstream org/different"
    assert finding.fix is not None
    assert "--chat-template-file" in finding.fix


@pytest.mark.parametrize(
    ("repo", "reason"),
    [
        ("org/no-template", "has no chat_template"),
        ("org/named-no-default", "has no chat_template"),
        ("org/not-object", "did not return a JSON object"),
        ("org/missing", "HTTP 404"),
    ],
)
def test_template_upstream_problems_are_skips(hub: str, repo: str, reason: str) -> None:
    finding = template_finding(hub, repo)
    assert finding.severity == "skip"
    assert reason in finding.message


@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        (capable(dialect="go"), "Ollama templates are Go templates; compare not supported"),
        (capable(dialect="unknown"), "does not report a Jinja chat template"),
        (capable(chat_template=None), "did not report its chat template"),
    ],
)
def test_template_skips_without_fetching(backend: ScriptedBackend, reason: str) -> None:
    findings = run_preflight(
        backend, hf_repo="org/same", context_probe_tokens=0, hf_base_url="http://127.0.0.1:9"
    )
    finding = only(findings, "template")
    assert finding.severity == "skip"
    assert reason in finding.message


def test_template_offline_skips_without_fetching() -> None:
    findings = run_preflight(
        capable(),
        hf_repo="org/same",
        offline=True,
        context_probe_tokens=0,
        hf_base_url="http://127.0.0.1:9",
    )
    finding = only(findings, "template")
    assert finding.severity == "skip"
    assert "offline" in finding.message


@pytest.mark.parametrize(
    "repo", ["org", "a/b/c", "../etc/passwd", "-x/y", "org/.hidden", "o/n?x=1"]
)
def test_invalid_hf_repo_is_rejected_before_any_request(repo: str) -> None:
    with pytest.raises(SpecError, match="invalid Hugging Face repo"):
        run_preflight(capable(), hf_repo=repo, context_probe_tokens=0)


# tokenizer -------------------------------------------------------------------------------


def test_different_tokenizer_fails() -> None:
    findings = run_preflight(capable(token_offset=7), reference=capable(), context_probe_tokens=0)
    finding = only(findings, "tokenizer")
    assert finding.severity == "fail"
    assert (
        finding.message == "tokenizers differ from the reference; logit metrics are not comparable"
    )


@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_tokenizer_unavailable_on_either_side_is_a_skip(side: str) -> None:
    blind = capable(label="blind", exact_token_ids=False)
    candidate, reference = (blind, capable()) if side == "candidate" else (capable(), blind)
    finding = only(
        run_preflight(candidate, reference=reference, context_probe_tokens=0), "tokenizer"
    )
    assert finding.severity == "skip"
    assert "blind cannot tokenize" in finding.message
