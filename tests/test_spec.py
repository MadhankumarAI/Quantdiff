from __future__ import annotations

import pytest

from quantdiff.errors import SpecError
from quantdiff.spec import DEFAULT_OLLAMA_URL, ollama_base_url, parse_spec
from quantdiff.types import CandidateSpec


@pytest.fixture(autouse=True)
def _no_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_ollama_tag_keeps_its_colons() -> None:
    assert parse_spec("ollama:qwen2.5:0.5b-instruct-q8_0") == CandidateSpec(
        kind="ollama",
        base_url=DEFAULT_OLLAMA_URL,
        model="qwen2.5:0.5b-instruct-q8_0",
        label="qwen2.5:0.5b-instruct-q8_0",
    )


def test_ollama_accepts_namespaced_and_registry_tags() -> None:
    spec = parse_spec("ollama:hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    assert spec.model == "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1:11500", "http://127.0.0.1:11500"),
        ("gpu-box", "http://gpu-box:11434"),
        ("http://gpu-box", "http://gpu-box"),
        ("https://ollama.example.com/", "https://ollama.example.com"),
        ("0.0.0.0", "http://127.0.0.1:11434"),  # noqa: S104 - listen address under test
        ("0.0.0.0:8000", "http://127.0.0.1:8000"),
        ("[::]:8000", "http://127.0.0.1:8000"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("  ", DEFAULT_OLLAMA_URL),
    ],
)
def test_ollama_host_env_is_resolved_like_the_cli(
    monkeypatch: pytest.MonkeyPatch, host: str, expected: str
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert parse_spec("ollama:llama3").base_url == expected


@pytest.mark.parametrize("host", ["gpu-box:notaport", "ftp://gpu-box", "http://"])
def test_bad_ollama_host_is_a_spec_error(host: str) -> None:
    with pytest.raises(SpecError):
        ollama_base_url(host)


def test_llamacpp_spec_labels_by_host_and_port() -> None:
    assert parse_spec("llamacpp:http://127.0.0.1:8080/") == CandidateSpec(
        kind="llamacpp",
        base_url="http://127.0.0.1:8080",
        model="",
        label="llamacpp@127.0.0.1:8080",
    )


def test_openai_spec_with_model() -> None:
    assert parse_spec("openai:http://127.0.0.1:1234/v1#qwen2.5-7b-instruct") == CandidateSpec(
        kind="openai",
        base_url="http://127.0.0.1:1234/v1",
        model="qwen2.5-7b-instruct",
        label="qwen2.5-7b-instruct",
    )


def test_openai_spec_with_api_key_variable() -> None:
    spec = parse_spec("openai:https://gpu-box:8000/v1#Qwen/Qwen2.5-7B-Instruct@env:VLLM_KEY")
    assert spec.model == "Qwen/Qwen2.5-7B-Instruct"
    assert spec.api_key_env == "VLLM_KEY"
    assert spec.label == "Qwen/Qwen2.5-7B-Instruct"


def test_openai_model_may_contain_at_and_colon() -> None:
    spec = parse_spec("openai:http://h:1/v1#org@team/model:7b@env:KEY")
    assert (spec.model, spec.api_key_env) == ("org@team/model:7b", "KEY")


@pytest.mark.parametrize(
    ("text", "label", "kind"),
    [
        ("q4=ollama:qwen2.5:7b-instruct-q4_K_M", "q4", "ollama"),
        ("my server = llamacpp:http://127.0.0.1:8080", "my server", "llamacpp"),
        ("lms=openai:http://127.0.0.1:1234/v1#m", "lms", "openai"),
    ],
)
def test_label_prefix_overrides_default(text: str, label: str, kind: str) -> None:
    spec = parse_spec(text)
    assert (spec.label, spec.kind) == (label, kind)


def test_equals_after_a_colon_is_not_a_label() -> None:
    spec = parse_spec("openai:http://h:1/v1#name=with-equals")
    assert spec.model == "name=with-equals"
    assert spec.label == "name=with-equals"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("qwen2.5:7b", "unknown backend"),
        ("ollama", "expected <kind>:<target>"),
        ("ollama:", "expected <kind>:<target>"),
        ("ollama:bad tag", "invalid Ollama tag"),
        ("ollama:-leading", "invalid Ollama tag"),
        ("=ollama:llama3", "label before '=' is empty"),
        ("a\tb=ollama:llama3", "control characters"),
        ("llamacpp:127.0.0.1:8080", "scheme"),
        ("llamacpp:file:///etc/passwd", "scheme"),
        ("llamacpp:http://127.0.0.1:8080/?x=1", "query"),
        ("openai:http://127.0.0.1:1234/v1", "needs a model"),
        ("openai:http://127.0.0.1:1234/v1#", "invalid model name"),
        ("openai:http://127.0.0.1:1234/v1#a b", "invalid model name"),
        ("openai:http://127.0.0.1:1234/v1#m@env:", "invalid environment variable"),
        ("openai:http://127.0.0.1:1234/v1#m@env:1BAD", "invalid environment variable"),
        ("openai:http://127.0.0.1:1234/v1#m@env:BAD-NAME", "invalid environment variable"),
    ],
)
def test_malformed_specs_raise_spec_error(text: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_spec(text)


def test_credentials_in_url_are_rejected_without_echoing_them() -> None:
    with pytest.raises(SpecError, match="credentials") as caught:
        parse_spec("openai:http://user:hunter2@127.0.0.1:1234/v1#m")
    assert "hunter2" not in str(caught.value)
