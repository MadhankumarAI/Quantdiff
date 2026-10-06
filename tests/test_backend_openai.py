from __future__ import annotations

from collections.abc import Iterator

import pytest

from quantdiff.backends import OpenAICompatBackend, open_backend
from quantdiff.errors import BackendError, CapabilityError, SpecError
from quantdiff.types import CandidateSpec, Message, TokenProb, TokenStep, ToolSpec
from tests.fixtures.http_fake import FakeServer, Request, closed_port, load, running

MODEL = "qwen2.5-7b-instruct"
CANARY_KEY = "sk-test-0123456789abcdef"
VLLM_MODELS = {
    "object": "list",
    "data": [
        {"id": "other", "object": "model", "owned_by": "vllm", "max_model_len": 4096},
        {
            "id": MODEL,
            "object": "model",
            "created": 1759480000,
            "owned_by": "vllm",
            "max_model_len": 32768,
        },
    ],
}
COMPLETIONS = {
    "id": "cmpl-1",
    "object": "text_completion",
    "choices": [
        {
            "index": 0,
            "text": " Paris.",
            "finish_reason": "length",
            "logprobs": {
                "tokens": [" Paris", "."],
                "token_logprobs": [-0.1, -0.7],
                "top_logprobs": [{" the": -2.5, " Paris": -0.1}, {".": -0.7, ",": -1.1}],
                "text_offset": [0, 6],
            },
        }
    ],
}


def _chat_reply(message: dict[str, object], finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with running() as fake:
        yield fake


def _backend(server: FakeServer, api_key_env: str | None = None) -> OpenAICompatBackend:
    spec = CandidateSpec(
        kind="openai",
        base_url=server.url + "/v1",
        model=MODEL,
        label=MODEL,
        api_key_env=api_key_env,
    )
    backend = open_backend(spec, timeout=5)
    assert isinstance(backend, OpenAICompatBackend)
    return backend


def _step(token: str) -> TokenStep:
    chosen = TokenProb(token=token, logprob=-0.3)
    return TokenStep(chosen=chosen, top=(chosen,))


# info -------------------------------------------------------------------------------------


def test_info_finds_model_and_vllm_context(server: FakeServer) -> None:
    server.reply("GET", "/v1/models", VLLM_MODELS)
    info = _backend(server).info()
    assert info.backend == "openai"
    assert info.model == MODEL
    assert info.context_length == 32768
    assert (info.chat_template, info.template_dialect) == (None, "unknown")
    assert (info.supports_logprobs, info.exact_token_ids) == (True, False)
    assert dict(info.details) == {"owned_by": "vllm", "created": "1759480000"}
    assert (info.size_bytes, info.weights_id) == (None, None)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"id": MODEL, "loaded_context_length": 8192, "context_length": 32768}, 8192),
        ({"id": MODEL, "context_length": 32768}, 32768),
        ({"id": MODEL, "object": "model"}, None),
    ],
)
def test_info_context_fields(
    server: FakeServer, entry: dict[str, object], expected: int | None
) -> None:
    server.reply("GET", "/v1/models", {"data": [entry]})
    assert _backend(server).info().context_length == expected


def test_info_reports_unknown_model(server: FakeServer) -> None:
    server.reply("GET", "/v1/models", {"data": [{"id": "a"}, {"id": "b\x1b[31m"}]})
    with pytest.raises(BackendError) as caught:
        _backend(server).info()
    assert str(caught.value) == (
        f"model '{MODEL}' is not served by {server.url}/v1; available: a, b [31m. "
        "Put one of these after '#' in the spec"
    )


def test_unknown_model_error_caps_the_listed_ids(server: FakeServer) -> None:
    server.reply("GET", "/v1/models", {"data": [{"id": f"m{n}"} for n in range(13)]})
    with pytest.raises(BackendError, match=r"available: m0, m1, .*, m9 and 3 more\."):
        _backend(server).info()


def test_unknown_model_on_an_empty_server(server: FakeServer) -> None:
    server.reply("GET", "/v1/models", {"data": []})
    with pytest.raises(BackendError, match="lists no models; load one in LM Studio"):
        _backend(server).info()


# chat -------------------------------------------------------------------------------------


def test_chat_text_and_request_shape(server: FakeServer) -> None:
    server.reply("POST", "/v1/chat/completions", _chat_reply({"content": "Hi there"}))
    result = _backend(server).chat([Message(role="user", content="hi")], max_tokens=16, seed=4)
    assert (result.text, result.tool_calls, result.finish_reason) == ("Hi there", (), "stop")
    assert (result.prompt_tokens, result.completion_tokens) == (12, 5)
    assert result.decode_tokens_per_second is None
    assert server.bodies("/v1/chat/completions") == [
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 4,
            "stream": False,
        }
    ]


def test_chat_tools_and_schema_passthrough(server: FakeServer) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    tool = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    calls = [
        {"id": "1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}},
        {"id": "2", "type": "function", "function": {"name": "add", "arguments": "{oops"}},
        {"id": "3", "type": "function", "function": {"name": "add", "arguments": "[1, 2]"}},
    ]
    reply = _chat_reply({"content": None, "tool_calls": calls}, finish="tool_calls")
    server.reply("POST", "/v1/chat/completions", reply)

    result = _backend(server).chat(
        [Message(role="user", content="1+2")], max_tokens=32, tools=[tool], json_schema=schema
    )

    assert result.text == ""
    assert [(c.arguments, c.raw_arguments) for c in result.tool_calls] == [
        ({"a": 1}, '{"a": 1}'),
        (None, "{oops"),
        (None, "[1, 2]"),
    ]
    sent = server.bodies("/v1/chat/completions")[0]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {"name": "add", "description": "Add", "parameters": {"type": "object"}},
        }
    ]
    assert sent["response_format"]["json_schema"]["schema"] == schema


def test_chat_without_choices_is_an_error(server: FakeServer) -> None:
    server.reply("POST", "/v1/chat/completions", {"choices": []})
    with pytest.raises(BackendError, match="no choices"):
        _backend(server).chat([Message(role="user", content="x")], max_tokens=4)


# scoring ----------------------------------------------------------------------------------


def test_tokenize_is_unsupported(server: FakeServer) -> None:
    assert _backend(server).tokenize("x") is None


def test_generate_scored_parses_legacy_logprobs(server: FakeServer) -> None:
    server.reply("POST", "/v1/completions", COMPLETIONS)
    steps = _backend(server).generate_scored("The capital is", max_tokens=2, top_k=2)
    assert [step.chosen for step in steps] == [
        TokenProb(" Paris", -0.1),
        TokenProb(".", -0.7),
    ]
    assert steps[0].top == (
        TokenProb(" Paris", -0.1),
        TokenProb(" the", -2.5),
    )
    assert server.bodies("/v1/completions") == [
        {
            "model": MODEL,
            "prompt": "The capital is",
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": 0,
            "logprobs": 2,
            "stream": False,
        }
    ]


def test_missing_logprobs_is_a_capability_error(server: FakeServer) -> None:
    reply = {"choices": [{"index": 0, "text": " Paris", "logprobs": None}]}
    server.reply("POST", "/v1/completions", reply)
    with pytest.raises(CapabilityError, match="no logprobs"):
        _backend(server).generate_scored("x", max_tokens=1, top_k=1)


def test_mismatched_logprob_arrays_are_rejected(server: FakeServer) -> None:
    logprobs = {"tokens": ["a", "b"], "token_logprobs": [-1.0], "top_logprobs": [{}]}
    server.reply("POST", "/v1/completions", {"choices": [{"logprobs": logprobs}]})
    with pytest.raises(BackendError, match="different lengths"):
        _backend(server).generate_scored("x", max_tokens=2, top_k=1)


def test_score_continuation_teacher_forces_by_text(server: FakeServer) -> None:
    def reply(request: Request) -> tuple[int, object]:
        token = f"<{len(request.body['prompt'])}>"
        logprobs = {"tokens": [token], "token_logprobs": [-0.2], "top_logprobs": [{token: -0.2}]}
        return 200, {"choices": [{"logprobs": logprobs}]}

    server.respond("POST", "/v1/completions", reply)
    result = _backend(server).score_continuation("Hi", [_step(" there"), _step("!")], top_k=3)

    sent = server.bodies("/v1/completions")
    assert [body["prompt"] for body in sent] == ["Hi", "Hi there"]
    assert all(body["max_tokens"] == 1 and body["logprobs"] == 3 for body in sent)
    assert [top[0].token for top in result] == ["<2>", "<8>"]


def test_score_continuation_reports_end_of_sequence_as_empty(server: FakeServer) -> None:
    logprobs: dict[str, list[object]] = {"tokens": [], "token_logprobs": [], "top_logprobs": []}
    server.reply("POST", "/v1/completions", {"choices": [{"logprobs": logprobs}]})
    assert _backend(server).score_continuation("x", [_step("a")], top_k=2) == [()]


# characters split across tokens -------------------------------------------------------------
# Payloads captured from llama-server b11425's /v1/completions, continuing a Hindi prompt
# whose next character " \u092f" is split into two tokens.


def test_content_logprobs_with_bytes_are_read(server: FakeServer) -> None:
    reply = {
        "choices": [
            {
                "text": " Paris",
                "logprobs": {
                    "content": [
                        {
                            "id": 12095,
                            "token": " Paris",
                            "bytes": [32, 80, 97, 114, 105, 115],
                            "logprob": -0.1,
                            "top_logprobs": [
                                {
                                    "id": 279,
                                    "token": " the",
                                    "bytes": [32, 116, 104, 101],
                                    "logprob": -2.5,
                                },
                                {
                                    "id": 12095,
                                    "token": " Paris",
                                    "bytes": [32, 80, 97, 114, 105, 115],
                                    "logprob": -0.1,
                                },
                                {"id": 9, "token": "x", "bytes": [120], "logprob": None},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    server.reply("POST", "/v1/completions", reply)
    (step,) = _backend(server).generate_scored("The capital is", max_tokens=1, top_k=3)
    assert step.chosen == TokenProb(" Paris", -0.1, None, b" Paris")
    assert step.top == (
        TokenProb(" Paris", -0.1, None, b" Paris"),
        TokenProb(" the", -2.5, None, b" the"),
    )


def test_generate_scored_leaves_folded_steps_unscored(server: FakeServer) -> None:
    server.reply("POST", "/v1/completions", load("openai_completions_llamacpp"))
    (folded,) = _backend(server).generate_scored("Hindi", max_tokens=2, top_k=3)
    assert folded == TokenStep(
        TokenProb(" \u092f", -1.5552624464035034, None, " \u092f".encode()), ()
    )


def test_score_continuation_reads_a_held_back_fragment_as_no_distribution(
    server: FakeServer,
) -> None:
    server.reply("POST", "/v1/completions", load("openai_completions_partial"))
    assert _backend(server).score_continuation("Hindi", [_step(" \u092e")], top_k=3) == [()]


def test_held_back_fragment_while_generating_is_a_capability_error(server: FakeServer) -> None:
    server.reply("POST", "/v1/completions", load("openai_completions_partial"))
    with pytest.raises(CapabilityError, match="no logprobs"):
        _backend(server).generate_scored("Hindi", max_tokens=4, top_k=3)


def test_score_continuation_rebuilds_prefixes_from_bytes(server: FakeServer) -> None:
    def reply(request: Request) -> tuple[int, object]:
        token = f"<{len(request.body['prompt'])}>"
        logprobs = {"tokens": [token], "token_logprobs": [-0.2], "top_logprobs": [{token: -0.2}]}
        return 200, {"choices": [{"logprobs": logprobs}]}

    server.respond("POST", "/v1/completions", reply)
    folded = TokenStep(TokenProb(" \u092f", -1.6, None, " \u092f".encode()), ())

    result = _backend(server).score_continuation("Hi", [folded, _step("!")], top_k=3)

    assert [body["prompt"] for body in server.bodies("/v1/completions")] == ["Hi \u092f"]
    assert result[0] == ()
    assert result[1] == (TokenProb("<4>", -0.2),)


# API keys ---------------------------------------------------------------------------------


def test_api_key_is_sent_as_bearer_token(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QD_TEST_KEY", CANARY_KEY)
    server.reply("GET", "/v1/models", VLLM_MODELS)
    _backend(server, api_key_env="QD_TEST_KEY").info()
    assert server.requests[0].headers["Authorization"] == f"Bearer {CANARY_KEY}"


def test_no_authorization_header_without_key_env(server: FakeServer) -> None:
    server.reply("GET", "/v1/models", VLLM_MODELS)
    _backend(server).info()
    assert "Authorization" not in server.requests[0].headers


def test_missing_key_variable_names_the_variable_only(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QD_TEST_KEY", raising=False)
    with pytest.raises(BackendError, match="QD_TEST_KEY is not set"):
        _backend(server, api_key_env="QD_TEST_KEY")


def test_key_never_appears_in_errors(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QD_TEST_KEY", CANARY_KEY)
    server.reply("GET", "/v1/models", {"error": "invalid api key"}, status=401)
    server.reply("POST", "/v1/completions", {"choices": [{"logprobs": None}]})
    backend = _backend(server, api_key_env="QD_TEST_KEY")
    with pytest.raises(BackendError) as unauthorized:
        backend.info()
    with pytest.raises(CapabilityError) as no_logprobs:
        backend.generate_scored("x", max_tokens=1, top_k=1)
    for caught in (unauthorized, no_logprobs):
        assert CANARY_KEY not in str(caught.value)
    assert CANARY_KEY not in repr(backend.spec)


# friendly errors --------------------------------------------------------------------------


def test_rejected_key_names_the_variable_only(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QD_TEST_KEY", CANARY_KEY)
    server.reply("GET", "/v1/models", {"error": "invalid api key"}, status=401)
    with pytest.raises(BackendError) as caught:
        _backend(server, api_key_env="QD_TEST_KEY").info()
    assert str(caught.value) == (
        f"the server at {server.url}/v1 rejected the API key from $QD_TEST_KEY (HTTP 401); "
        "check the key and its permissions"
    )


def test_missing_key_suggests_an_env_variable(server: FakeServer) -> None:
    server.reply("POST", "/v1/chat/completions", {"error": "forbidden"}, status=403)
    with pytest.raises(BackendError, match=r"requires an API key \(HTTP 403\).*@env:NAME"):
        _backend(server).chat([Message(role="user", content="x")], max_tokens=4)


def test_model_not_found_points_at_the_model_list(server: FakeServer) -> None:
    error = {"error": {"message": f"The model `{MODEL}` does not exist.", "code": 404}}
    server.reply("POST", "/v1/completions", error, status=404)
    with pytest.raises(BackendError) as caught:
        _backend(server).generate_scored("x", max_tokens=1, top_k=1)
    assert str(caught.value) == (
        f"model '{MODEL}' is not available on {server.url}/v1; "
        f"use a model id listed at {server.url}/v1/models"
    )


def test_404_without_v1_suggests_the_suffix(server: FakeServer) -> None:
    spec = CandidateSpec(kind="openai", base_url=server.url, model=MODEL, label=MODEL)
    with pytest.raises(BackendError, match=r"/models returned HTTP 404; .* usually end in /v1"):
        OpenAICompatBackend(spec, timeout=5).info()


def test_404_with_v1_keeps_the_server_message(server: FakeServer) -> None:
    with pytest.raises(BackendError, match=r"HTTP 404.*no route"):
        _backend(server).info()


def test_unreachable_server_says_what_to_check() -> None:
    url = f"http://127.0.0.1:{closed_port()}/v1"
    backend = OpenAICompatBackend(CandidateSpec(kind="openai", base_url=url, model="m", label="m"))
    with pytest.raises(BackendError) as caught:
        backend.info()
    assert str(caught.value) == (
        f"cannot reach the server at {url} (connection refused); "
        "check that LM Studio/vLLM is running and the URL ends in /v1"
    )


def test_wrong_spec_kind_is_rejected() -> None:
    spec = CandidateSpec(kind="ollama", base_url="http://127.0.0.1:1", model="m", label="m")
    with pytest.raises(SpecError):
        OpenAICompatBackend(spec)
