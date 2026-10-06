from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from quantdiff.backends import LlamaCppBackend, open_backend
from quantdiff.backends._common import expect_float
from quantdiff.errors import BackendError, CapabilityError
from quantdiff.types import CandidateSpec, Message, TokenProb, TokenStep, ToolSpec
from tests.fixtures.http_fake import FakeServer, Request, closed_port, load, running

# Shapes follow tools/server/README.md in ggml-org/llama.cpp.
PROPS = {
    "default_generation_settings": {"n_ctx": 8192, "temperature": 0.8, "n_predict": -1},
    "total_slots": 1,
    "model_path": "C:\\models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
    "build_info": "b6500-abc1234",
}
MODELS = {
    "object": "list",
    "data": [
        {
            "id": "qwen2.5-7b-instruct-q4_k_m",
            "object": "model",
            "owned_by": "llamacpp",
            "meta": {
                "n_vocab": 152064,
                "n_ctx_train": 32768,
                "n_params": 7615616512,
                "size": 4677120000,
            },
        }
    ],
}
COMPLETION = {
    "content": " Paris.",
    "tokens_predicted": 2,
    "completion_probabilities": [
        {
            "id": 12095,
            "token": " Paris",
            "logprob": -0.11,
            "bytes": [32, 80, 97, 114, 105, 115],
            "top_logprobs": [
                {"id": 279, "token": " the", "logprob": -2.6},
                {"id": 12095, "token": " Paris", "logprob": -0.11},
            ],
        },
        {
            "id": 13,
            "token": ".",
            "logprob": -0.5,
            "bytes": [46],
            "top_logprobs": [{"id": 13, "token": ".", "logprob": -0.5}],
        },
    ],
}
LEGACY_COMPLETION = {
    "content": " Paris",
    "completion_probabilities": [
        {
            "content": " Paris",
            "probs": [
                {"tok_str": " the", "prob": 0.25},
                {"tok_str": " Paris", "prob": 0.5},
                {"tok_str": " zero", "prob": 0.0},
            ],
        }
    ],
}
CHAT_TOOL = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
}


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with running() as fake:
        yield fake


def _backend(server: FakeServer) -> LlamaCppBackend:
    spec = CandidateSpec(kind="llamacpp", base_url=server.url + "/", model="", label="llamacpp")
    backend = open_backend(spec, timeout=5)
    assert isinstance(backend, LlamaCppBackend)
    return backend


def _step(token: str, token_id: int | None) -> TokenStep:
    chosen = TokenProb(token=token, logprob=-0.1, token_id=token_id)
    return TokenStep(chosen=chosen, top=(chosen,))


def _echo_length(request: Request) -> tuple[int, object]:
    length = len(request.body["prompt"])
    prob = {"id": length, "token": f"<{length}>", "logprob": -0.2}
    return 200, {"content": "", "completion_probabilities": [{**prob, "top_logprobs": [prob]}]}


# info and tokenize ------------------------------------------------------------------------


def test_info_reads_props_and_models(server: FakeServer) -> None:
    server.reply("GET", "/props", PROPS)
    server.reply("GET", "/v1/models", MODELS)
    info = _backend(server).info()
    assert info.backend == "llamacpp"
    assert info.model == "qwen2.5-7b-instruct-q4_k_m"
    assert info.context_length == 8192
    assert info.chat_template == PROPS["chat_template"]
    assert info.template_dialect == "jinja"
    assert (info.supports_logprobs, info.exact_token_ids) == (True, True)
    assert dict(info.details) == {
        "model_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "trained_context_length": "32768",
        "build": "b6500-abc1234",
        "model_size": "4677120000",
        "model_n_params": "7615616512",
        "model_n_vocab": "152064",
    }
    assert info.size_bytes == 4677120000
    assert info.weights_id == "4677120000:7615616512:152064"


def test_info_falls_back_to_model_file_name(server: FakeServer) -> None:
    server.reply("GET", "/props", {"model_path": "/models/tiny.gguf"})
    server.reply("GET", "/v1/models", {"data": []})
    info = _backend(server).info()
    assert info.model == "tiny.gguf"
    assert info.context_length is None
    assert info.chat_template is None
    assert (info.size_bytes, info.weights_id) == (None, None)


def test_weights_id_needs_every_fingerprint_field(server: FakeServer) -> None:
    server.reply("GET", "/props", PROPS)
    server.reply("GET", "/v1/models", {"data": [{"id": "m", "meta": {"size": 1000}}]})
    info = _backend(server).info()
    assert (info.size_bytes, info.weights_id) == (1000, None)


@pytest.mark.parametrize(
    "tokens",
    [[151643, 785, 6722], [{"id": 151643, "piece": ""}, {"id": 785, "piece": "The"}, 6722]],
)
def test_tokenize_accepts_ids_and_pieces(server: FakeServer, tokens: list[object]) -> None:
    server.reply("POST", "/tokenize", {"tokens": tokens})
    assert _backend(server).tokenize("The capital") == (151643, 785, 6722)
    assert server.bodies("/tokenize") == [{"content": "The capital", "add_special": True}]


def test_tokenize_rejects_bad_entries(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"tokens": ["x"]})
    with pytest.raises(BackendError, match="invalid token entry"):
        _backend(server).tokenize("x")


# chat -------------------------------------------------------------------------------------


def test_chat_tool_call_and_schema(server: FakeServer) -> None:
    schema = {"type": "object"}
    tool = ToolSpec(name="get_weather", description="Weather", parameters={"type": "object"})
    server.reply("POST", "/v1/chat/completions", CHAT_TOOL)

    result = _backend(server).chat(
        [Message(role="user", content="Weather?")],
        max_tokens=64,
        tools=[tool],
        json_schema=schema,
        seed=3,
    )

    assert result.text == ""
    assert result.finish_reason == "tool_calls"
    assert (result.prompt_tokens, result.completion_tokens) == (120, 18)
    assert result.decode_tokens_per_second is None
    assert [(c.name, c.arguments, c.raw_arguments) for c in result.tool_calls] == [
        ("get_weather", {"city": "Paris"}, '{"city":"Paris"}')
    ]
    sent = server.bodies("/v1/chat/completions")[0]
    assert "model" not in sent
    assert sent["temperature"] == 0.0
    assert (sent["seed"], sent["max_tokens"]) == (3, 64)
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }


@pytest.mark.parametrize(
    ("timings", "expected"),
    [
        ({"predicted_n": 18, "predicted_ms": 360.0, "predicted_per_second": 50.0}, 50.0),
        ({"predicted_n": 1, "predicted_ms": 0.01, "predicted_per_second": 100000.0}, None),
        ({"predicted_n": 18, "predicted_ms": "fast"}, None),
        ("not an object", None),
    ],
)
def test_chat_reads_server_decode_speed(
    server: FakeServer, timings: object, expected: float | None
) -> None:
    server.reply("POST", "/v1/chat/completions", {**CHAT_TOOL, "timings": timings})
    result = _backend(server).chat([Message(role="user", content="x")], max_tokens=8)
    assert result.decode_tokens_per_second == (
        None if expected is None else pytest.approx(expected)
    )


# scoring ----------------------------------------------------------------------------------


def test_generate_scored_sends_token_ids(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"tokens": [1, 2, 3]})
    server.reply("POST", "/completion", COMPLETION)

    steps = _backend(server).generate_scored("The capital is", max_tokens=2, top_k=2)

    assert steps[0].chosen == TokenProb(" Paris", -0.11, 12095, b" Paris")
    assert steps[0].top == (
        TokenProb(" Paris", -0.11, 12095),
        TokenProb(" the", -2.6, 279),
    )
    assert steps[1].chosen.token_id == 13
    assert server.bodies("/completion") == [
        {
            "prompt": [1, 2, 3],
            "n_predict": 2,
            "temperature": 0.0,
            "n_probs": 2,
            "cache_prompt": True,
            "seed": 0,
            "samplers": ["temperature"],
        }
    ]


def test_generate_scored_reads_legacy_probabilities(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"tokens": [1]})
    server.reply("POST", "/completion", LEGACY_COMPLETION)
    (step,) = _backend(server).generate_scored("x", max_tokens=1, top_k=3)
    assert step.chosen == TokenProb(" Paris", math.log(0.5))
    assert [prob.token for prob in step.top] == [" Paris", " the"]


def test_generate_scored_without_probabilities(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"tokens": [1]})
    server.reply("POST", "/completion", {"content": " Paris"})
    with pytest.raises(CapabilityError, match="no probabilities"):
        _backend(server).generate_scored("x", max_tokens=1, top_k=3)
    # The batch, the one-token request, and the same request forcing a complete token.
    assert [body.get("logit_bias") for body in server.bodies("/completion")] == [
        None,
        None,
        [[1, 1000.0]],
    ]


def test_score_continuation_feeds_exact_ids(server: FakeServer) -> None:
    server.respond("POST", "/completion", _echo_length)
    continuation = [_step(" Paris", 12095), _step(".", 13), _step("\n", 198)]

    result = _backend(server).score_continuation(
        "ignored", continuation, top_k=5, prompt_token_ids=[7, 8]
    )

    sent = server.bodies("/completion")
    assert [body["prompt"] for body in sent] == [[7, 8], [7, 8, 12095], [7, 8, 12095, 13]]
    assert all(body["n_predict"] == 1 and body["n_probs"] == 5 for body in sent)
    assert [top[0].token_id for top in result] == [2, 3, 4]
    assert server.bodies("/tokenize") == []


@pytest.mark.parametrize(
    ("continuation", "prompt_ids"),
    [
        ([_step(" Paris", None), _step(".", None)], None),
        ([_step(" Paris", 12095), _step(".", None)], [7, 8]),
        ([_step(" Paris", 12095), _step(".", 13)], None),
    ],
    ids=["text-reference", "partial-ids", "no-prompt-ids"],
)
def test_score_continuation_falls_back_to_text(
    server: FakeServer, continuation: list[TokenStep], prompt_ids: list[int] | None
) -> None:
    server.respond("POST", "/completion", _echo_length)
    result = _backend(server).score_continuation(
        "The capital is", continuation, top_k=3, prompt_token_ids=prompt_ids
    )
    sent = server.bodies("/completion")
    assert [body["prompt"] for body in sent] == ["The capital is", "The capital is Paris"]
    assert [top[0].token for top in result] == ["<14>", "<20>"]
    assert server.bodies("/tokenize") == []


def test_null_logprob_alternatives_are_dropped(server: FakeServer) -> None:
    entry = {
        "id": 1,
        "token": "a",
        "logprob": -0.1,
        "top_logprobs": [
            {"id": 1, "token": "a", "logprob": -0.1},
            {"id": 2, "token": "b", "logprob": None},
        ],
    }
    server.reply("POST", "/tokenize", {"tokens": [1]})
    server.reply("POST", "/completion", {"completion_probabilities": [entry]})
    (step,) = _backend(server).generate_scored("x", max_tokens=1, top_k=2)
    assert step.top == (TokenProb("a", -0.1, 1),)


@pytest.mark.parametrize("bad", [float("nan"), float("-inf"), 1e999])
def test_non_finite_numbers_are_rejected(bad: float) -> None:
    with pytest.raises(BackendError, match="finite"):
        expect_float(bad, "logprob")


# friendly errors --------------------------------------------------------------------------


def test_loading_model_says_to_wait(server: FakeServer) -> None:
    server.reply(
        "POST", "/tokenize", {"error": {"code": 503, "message": "Loading model"}}, status=503
    )
    with pytest.raises(BackendError, match="still loading the model") as caught:
        _backend(server).tokenize("x")
    assert "HTTP 503" in str(caught.value.__cause__)


def test_missing_endpoint_suggests_checking_the_server_kind(server: FakeServer) -> None:
    with pytest.raises(BackendError, match=r"/props returned HTTP 404; check that .* llama-server"):
        _backend(server).info()


def test_other_http_errors_keep_the_server_message(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"error": "boom"}, status=500)
    with pytest.raises(BackendError, match=r"HTTP 500.*boom"):
        _backend(server).tokenize("x")


def test_unreachable_server_says_how_to_start_it() -> None:
    port = closed_port()
    url = f"http://127.0.0.1:{port}"
    backend = LlamaCppBackend(CandidateSpec(kind="llamacpp", base_url=url, model="", label="l"))
    with pytest.raises(BackendError) as caught:
        backend.tokenize("x")
    assert str(caught.value) == (
        f"cannot reach llama-server at {url} (connection refused); "
        f"start it with `llama-server -m model.gguf --port {port}`"
    )


# characters split across tokens -----------------------------------------------------------
# Payloads captured from llama-server b11425 serving Qwen2.5 0.5B Instruct q8_0, continuing
# a Hindi prompt whose next character " \u092f" is split into tokens 14925 and 107.

PROMPT_IDS = [1, 2, 3]
NEWLINE_ID = 198


def _tokenize(request: Request) -> tuple[int, object]:
    return 200, {"tokens": [NEWLINE_ID] if request.body["content"] == "\n" else PROMPT_IDS}


def _hindi(request: Request) -> tuple[int, object]:
    """Replay the captured replies, keyed by what each request asked for."""
    body = request.body
    if body["n_predict"] > 1:
        return 200, load("llamacpp_completion_folded")
    if "logit_bias" in body:
        if body["logit_bias"] != [[NEWLINE_ID, 1000.0]]:
            return 400, {"error": f"unexpected logit_bias {body['logit_bias']}"}
        return 200, load("llamacpp_completion_biased")
    fixtures = {
        (): "llamacpp_completion_partial",
        (14925,): "llamacpp_completion_after_partial",
        (14925, 107): "llamacpp_completion_after_folded",
    }
    return 200, load(fixtures[tuple(body["prompt"][len(PROMPT_IDS) :])])


@pytest.fixture
def hindi(server: FakeServer) -> FakeServer:
    server.respond("POST", "/tokenize", _tokenize)
    server.respond("POST", "/completion", _hindi)
    return server


def test_generate_scored_steps_through_folded_tokens(hindi: FakeServer) -> None:
    steps = _backend(hindi).generate_scored("Hindi", max_tokens=3, top_k=3)

    assert [step.chosen.token_id for step in steps] == [14925, 107, 93948]
    lead, tail, plain = steps
    # The held-back lead token comes from the forced reply's unbiased alternatives.
    assert lead.chosen == TokenProb(" ", -0.4418126344680786, 14925, b" \xe0\xa4")
    assert [prob.token_id for prob in lead.top] == [14925, 91217, 47809]
    assert tail.chosen == TokenProb("\ufffd", -1.5552624464035034, 107, b"\xaf")
    assert plain.chosen.token_bytes == "\u0939".encode()
    assert [body["prompt"] for body in hindi.bodies("/completion")] == [
        PROMPT_IDS,
        PROMPT_IDS,
        PROMPT_IDS,
        [*PROMPT_IDS, 14925],
        [*PROMPT_IDS, 14925, 107],
    ]
    assert hindi.bodies("/tokenize") == [
        {"content": "Hindi", "add_special": True},
        {"content": "\n", "add_special": False},
    ]


def test_score_continuation_recovers_a_held_back_token(hindi: FakeServer) -> None:
    lead = TokenProb(" ", -0.44, 14925, b" \xe0\xa4")
    tail = TokenProb("\ufffd", -1.56, 107, b"\xaf")
    continuation = [TokenStep(lead, (lead,)), TokenStep(tail, (tail,))]
    backend = _backend(hindi)

    first = backend.score_continuation("Hindi", continuation, top_k=3, prompt_token_ids=PROMPT_IDS)
    again = backend.score_continuation("Hindi", continuation, top_k=3, prompt_token_ids=PROMPT_IDS)

    assert first == again
    assert [prob.token_id for prob in first[0]] == [14925, 91217, 47809]
    assert [prob.token_id for prob in first[1]] == [107, 229, 113]
    # The newline token id is looked up once per backend.
    assert hindi.bodies("/tokenize") == [{"content": "\n", "add_special": False}]


def test_score_continuation_skips_text_prefixes_inside_a_character(server: FakeServer) -> None:
    server.respond("POST", "/completion", _echo_length)
    lead = TokenProb(" ", -0.44, None, b" \xe0\xa4")
    tail = TokenProb("\ufffd", -1.56, None, b"\xaf")
    folded = TokenStep(TokenProb(" \u092f", -1.6, None, " \u092f".encode()), ())
    continuation = [TokenStep(lead, (lead,)), TokenStep(tail, (tail,)), folded, _step(".", None)]

    result = _backend(server).score_continuation("P", continuation, top_k=3)

    sent = [body["prompt"] for body in server.bodies("/completion")]
    assert sent == ["P", "P \u092f \u092f"]
    assert [top[0].token if top else None for top in result] == ["<1>", None, None, "<5>"]


@pytest.mark.parametrize(
    "reply",
    [
        load("llamacpp_completion_eos"),
        {**COMPLETION, "stop_type": "word"},
        {**COMPLETION, "stopped_eos": True},
        {**COMPLETION, "stopped_word": True},
    ],
    ids=["eos", "stop-word", "legacy-eos", "legacy-word"],
)
def test_generate_scored_ends_where_the_server_stopped(
    server: FakeServer, reply: dict[str, list[object]]
) -> None:
    server.reply("POST", "/tokenize", {"tokens": [1]})
    server.reply("POST", "/completion", reply)
    steps = _backend(server).generate_scored("x", max_tokens=8, top_k=3)
    assert len(steps) == len(reply["completion_probabilities"])
    assert len(server.bodies("/completion")) == 1


def test_generate_scored_continues_after_a_dropped_fragment(server: FakeServer) -> None:
    # At the token limit a trailing fragment is held back, so the batch has fewer entries.
    entries = COMPLETION["completion_probabilities"]
    assert isinstance(entries, list)
    batch = {**COMPLETION, "completion_probabilities": entries[:1]}

    def reply(request: Request) -> tuple[int, object]:
        return (200, batch) if request.body["n_predict"] > 1 else _echo_length(request)

    server.reply("POST", "/tokenize", {"tokens": [1, 2]})
    server.respond("POST", "/completion", reply)

    steps = _backend(server).generate_scored("x", max_tokens=2, top_k=2)

    assert [step.chosen.token_id for step in steps] == [12095, 3]
    assert [body["prompt"] for body in server.bodies("/completion")] == [[1, 2], [1, 2, 12095]]


def test_legacy_steps_without_ids_are_not_extended(server: FakeServer) -> None:
    server.reply("POST", "/tokenize", {"tokens": [1]})
    server.reply("POST", "/completion", LEGACY_COMPLETION)
    steps = _backend(server).generate_scored("x", max_tokens=4, top_k=3)
    assert [step.chosen.token for step in steps] == [" Paris"]
    assert len(server.bodies("/completion")) == 1


def test_newline_token_must_tokenize(server: FakeServer) -> None:
    server.reply("POST", "/completion", load("llamacpp_completion_partial"))
    server.reply("POST", "/tokenize", {"tokens": []})
    with pytest.raises(BackendError, match="no tokens"):
        _backend(server).score_continuation("x", [_step("a", 1)], top_k=3, prompt_token_ids=[1])
