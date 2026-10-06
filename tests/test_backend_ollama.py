from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

from quantdiff.backends import OllamaBackend, open_backend
from quantdiff.errors import BackendError, CapabilityError, SpecError
from quantdiff.metrics.logit import partition_kld
from quantdiff.suites import load_scoring_prompts
from quantdiff.types import CandidateSpec, Message, TokenProb, TokenStep, ToolSpec
from tests.fixtures.http_fake import FakeServer, Request, closed_port, load, running

TAG = "qwen2.5:0.5b-instruct-q4_K_M"
WEATHER = ToolSpec(
    name="get_weather",
    description="Get weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with running() as fake:
        fake.reply("GET", "/api/version", {"version": "0.35.1"})
        fake.reply("GET", "/api/tags", load("ollama_tags"))
        yield fake


def _backend(server: FakeServer, tag: str = TAG) -> OllamaBackend:
    spec = CandidateSpec(kind="ollama", base_url=server.url, model=tag, label=tag)
    backend = open_backend(spec, timeout=5)
    assert isinstance(backend, OllamaBackend)
    return backend


def _step(token: str, logprob: float) -> TokenStep:
    chosen = TokenProb(token=token, logprob=logprob)
    return TokenStep(chosen=chosen, top=(chosen,))


def _one_step_reply(token: str) -> dict[str, object]:
    entry = {"token": token, "logprob": -0.5, "top_logprobs": [{"token": token, "logprob": -0.5}]}
    return {"response": token, "done": True, "logprobs": [entry]}


# info -------------------------------------------------------------------------------------


def test_info_reads_show_and_loaded_context(server: FakeServer) -> None:
    server.reply("POST", "/api/show", load("ollama_show"))
    server.reply("POST", "/api/generate", {"done": True, "done_reason": "load"})
    server.reply("GET", "/api/ps", load("ollama_ps"))
    backend = _backend(server)

    info = backend.info()

    assert info.backend == "ollama"
    assert info.model == TAG
    assert info.context_length == 4096
    assert info.template_dialect == "go"
    assert info.chat_template is not None
    assert info.chat_template.startswith("{{- if .Messages }}")
    assert info.supports_logprobs is True
    assert info.exact_token_ids is False
    assert dict(info.details) == {
        "quantization": "Q4_K_M",
        "parameter_size": "494.03M",
        "trained_context_length": "32768",
        "server_version": "0.35.1",
        "digest": "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67",
    }
    assert info.size_bytes == 397821319
    assert info.weights_id == "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
    assert server.bodies("/api/show") == [{"model": TAG}]
    assert server.bodies("/api/generate") == [{"model": TAG, "keep_alive": "10m"}]


def test_info_is_cached(server: FakeServer) -> None:
    server.reply("POST", "/api/show", load("ollama_show"))
    server.reply("POST", "/api/generate", {"done": True})
    server.reply("GET", "/api/ps", load("ollama_ps"))
    backend = _backend(server)
    assert backend.info() is backend.info()
    assert len(server.bodies("/api/show")) == 1


def test_info_prefers_modelfile_num_ctx(server: FakeServer) -> None:
    show = load("ollama_show")
    show["parameters"] = (
        'num_ctx                        8192\nstop                           "<|im_end|>"'
    )
    server.reply("POST", "/api/show", show)
    assert _backend(server).info().context_length == 8192
    assert server.bodies("/api/generate") == []


def test_info_context_is_none_when_model_is_not_listed(server: FakeServer) -> None:
    server.reply("POST", "/api/show", load("ollama_show"))
    server.reply("POST", "/api/generate", {"done": True})
    server.reply("GET", "/api/ps", {"models": []})
    assert _backend(server).info().context_length is None


def test_info_matches_untagged_name_as_latest(server: FakeServer) -> None:
    running_models = load("ollama_ps")
    running_models["models"][0]["name"] = running_models["models"][0]["model"] = "llama3:latest"
    server.reply("POST", "/api/show", load("ollama_show"))
    server.reply("POST", "/api/generate", {"done": True})
    server.reply("GET", "/api/ps", running_models)
    assert _backend(server, "llama3").info().context_length == 4096


def test_info_falls_back_to_modified_at_without_a_listed_digest(server: FakeServer) -> None:
    server.reply("POST", "/api/show", load("ollama_show"))
    server.reply("POST", "/api/generate", {"done": True})
    server.reply("GET", "/api/ps", load("ollama_ps"))
    server.reply("GET", "/api/tags", {"models": []})
    info = _backend(server).info()
    details = dict(info.details)
    assert "digest" not in details
    assert details["modified_at"] == "2026-10-03T13:58:14.0169168+05:30"
    assert (info.size_bytes, info.weights_id) == (None, None)


# friendly errors --------------------------------------------------------------------------


def _info(backend: OllamaBackend) -> object:
    return backend.info()


def _chat(backend: OllamaBackend) -> object:
    return backend.chat([Message(role="user", content="x")], max_tokens=4)


@pytest.mark.parametrize(("path", "call"), [("/api/show", _info), ("/api/chat", _chat)])
def test_missing_model_says_how_to_pull_it(
    server: FakeServer, path: str, call: Callable[[OllamaBackend], object]
) -> None:
    server.reply("POST", path, {"error": "model 'nope' not found"}, status=404)
    backend = _backend(server, "nope")
    with pytest.raises(BackendError) as caught:
        call(backend)
    assert str(caught.value) == (
        "model 'nope' is not available in Ollama; run `ollama pull nope` (see `ollama list`)"
    )
    assert "HTTP 404" in str(caught.value.__cause__)


def test_unknown_route_404_is_not_reported_as_a_missing_model(server: FakeServer) -> None:
    server.reply("POST", "/api/chat", "404 page not found", status=404)
    with pytest.raises(BackendError, match="HTTP 404"):
        _backend(server).chat([Message(role="user", content="x")], max_tokens=4)


def test_other_http_errors_keep_the_server_message(server: FakeServer) -> None:
    server.reply("POST", "/api/chat", {"error": "out of memory"}, status=500)
    with pytest.raises(BackendError, match=r"HTTP 500.*out of memory"):
        _backend(server).chat([Message(role="user", content="x")], max_tokens=4)


def test_unreachable_server_says_how_to_start_it() -> None:
    url = f"http://127.0.0.1:{closed_port()}"
    backend = OllamaBackend(CandidateSpec(kind="ollama", base_url=url, model=TAG, label=TAG))
    with pytest.raises(BackendError) as caught:
        backend.chat([Message(role="user", content="x")], max_tokens=4)
    assert str(caught.value) == (
        f"cannot reach Ollama at {url} (connection refused); "
        "start it with `ollama serve` or set OLLAMA_HOST"
    )


# chat -------------------------------------------------------------------------------------


def test_chat_text(server: FakeServer) -> None:
    server.reply("POST", "/api/chat", load("ollama_chat"))
    result = _backend(server).chat(
        [Message(role="system", content="Be brief."), Message(role="user", content="hi")],
        max_tokens=8,
        seed=7,
    )
    assert result.text == "Hello."
    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    assert (result.prompt_tokens, result.completion_tokens) == (35, 3)
    assert result.seconds >= 0
    assert result.decode_tokens_per_second == pytest.approx(3 / 0.054066)
    assert server.bodies("/api/chat") == [
        {
            "model": TAG,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.0, "seed": 7, "num_predict": 8},
        }
    ]


@pytest.mark.parametrize(
    "timing",
    [
        {"eval_count": 1, "eval_duration": 1000},
        {"eval_count": 5, "eval_duration": 0},
        {"eval_count": 5},
        {},
    ],
)
def test_chat_decode_speed_needs_a_timed_decode(server: FakeServer, timing: dict[str, int]) -> None:
    reply = {
        k: v for k, v in load("ollama_chat").items() if k not in ("eval_count", "eval_duration")
    }
    server.reply("POST", "/api/chat", {**reply, **timing})
    result = _backend(server).chat([Message(role="user", content="x")], max_tokens=4)
    assert result.decode_tokens_per_second is None


def test_chat_tool_call_arguments_are_an_object(server: FakeServer) -> None:
    server.reply("POST", "/api/chat", load("ollama_chat_tools"))
    result = _backend(server).chat(
        [Message(role="user", content="Weather in Paris?")], max_tokens=60, tools=[WEATHER]
    )
    assert result.text == ""
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Paris"}
    assert call.raw_arguments == '{"city": "Paris"}'
    sent = server.bodies("/api/chat")[0]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": WEATHER.parameters,
            },
        }
    ]
    assert "format" not in sent


def test_chat_tool_call_with_non_object_arguments(server: FakeServer) -> None:
    reply = load("ollama_chat_tools")
    reply["message"]["tool_calls"][0]["function"]["arguments"] = "not json"
    server.reply("POST", "/api/chat", reply)
    call = _backend(server).chat([Message(role="user", content="x")], max_tokens=4).tool_calls[0]
    assert call.arguments is None
    assert call.raw_arguments == "not json"


def test_chat_passes_json_schema_as_format(server: FakeServer) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    server.reply("POST", "/api/chat", load("ollama_chat_format"))
    result = _backend(server).chat(
        [Message(role="user", content="4 as JSON")], max_tokens=20, json_schema=schema
    )
    assert result.text == '{\n  "answer": 4\n}'
    assert server.bodies("/api/chat")[0]["format"] == schema


def test_chat_rejects_malformed_reply(server: FakeServer) -> None:
    server.reply("POST", "/api/chat", {"done": True})
    with pytest.raises(BackendError, match="chat message"):
        _backend(server).chat([Message(role="user", content="x")], max_tokens=4)


# scoring ----------------------------------------------------------------------------------


def test_tokenize_is_unsupported(server: FakeServer) -> None:
    assert _backend(server).tokenize("hello") is None


def test_generate_scored_maps_logprobs(server: FakeServer) -> None:
    server.reply("POST", "/api/generate", load("ollama_generate"))
    steps = _backend(server).generate_scored("The capital of France is", max_tokens=3, top_k=3)

    assert [step.chosen.token for step in steps] == [" ______", ".", " A"]
    assert steps[0].chosen == TokenProb(" ______", -1.4817551374435425, None, b" ______")
    assert [prob.token for prob in steps[0].top] == [" ______", " Paris", " ____"]
    assert all(
        [prob.logprob for prob in step.top] == sorted((p.logprob for p in step.top), reverse=True)
        for step in steps
    )
    assert server.bodies("/api/generate") == [
        {
            "model": TAG,
            "prompt": "The capital of France is",
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 3,
            "keep_alive": "10m",
            "options": {"num_predict": 3, "temperature": 0.0, "seed": 0},
        }
    ]


def test_generate_stopping_at_once_returns_no_steps(server: FakeServer) -> None:
    server.reply("POST", "/api/generate", load("ollama_generate_eos"))
    assert _backend(server).generate_scored("x", max_tokens=4, top_k=3) == []


def test_score_continuation_reads_omitted_logprobs_as_end_of_sequence(
    server: FakeServer,
) -> None:
    server.reply("POST", "/api/generate", load("ollama_generate_eos"))
    assert _backend(server).score_continuation("x", [_step("a", -1.0)], top_k=2) == [()]


def test_generate_without_logprobs_is_a_capability_error(server: FakeServer) -> None:
    server.reply("POST", "/api/generate", {"response": " Paris", "done": True})
    with pytest.raises(CapabilityError, match=r"0\.12"):
        _backend(server).generate_scored("x", max_tokens=1, top_k=1)


@pytest.mark.parametrize(("max_tokens", "top_k"), [(0, 5), (4, 0), (4, 21)])
def test_generate_rejects_bad_arguments(server: FakeServer, max_tokens: int, top_k: int) -> None:
    with pytest.raises(SpecError):
        _backend(server).generate_scored("x", max_tokens=max_tokens, top_k=top_k)
    assert server.requests == []


def test_score_continuation_teacher_forces_by_text(server: FakeServer) -> None:
    def reply(request: Request) -> tuple[int, object]:
        prompt = request.body["prompt"]
        return 200, _one_step_reply(f"<{len(prompt)}>")

    server.respond("POST", "/api/generate", reply)
    continuation = [_step(" Paris", -0.4), _step(".", -0.9), _step(" It", -1.2)]

    result = _backend(server).score_continuation("The capital is", continuation, top_k=4)

    sent = server.bodies("/api/generate")
    assert [body["prompt"] for body in sent] == [
        "The capital is",
        "The capital is Paris",
        "The capital is Paris.",
    ]
    assert all(body["options"]["num_predict"] == 1 for body in sent)
    assert all(body["top_logprobs"] == 4 and body["raw"] is True for body in sent)
    assert [top[0].token for top in result] == ["<14>", "<20>", "<21>"]


def test_score_continuation_reports_end_of_sequence_as_empty(server: FakeServer) -> None:
    server.reply("POST", "/api/generate", {"response": "", "done": True, "logprobs": []})
    assert _backend(server).score_continuation("x", [_step("a", -1.0)], top_k=2) == [()]


# characters split across tokens -------------------------------------------------------------


def test_generate_scored_leaves_folded_steps_unscored(server: FakeServer) -> None:
    # Captured from Ollama 0.35.1: the first entry folds two tokens of " \u092f" and lists
    # the alternatives of the second one, all shown as U+FFFD.
    server.reply("POST", "/api/generate", load("ollama_generate_folded"))
    folded, plain = _backend(server).generate_scored("Hindi", max_tokens=3, top_k=3)

    assert folded == TokenStep(
        TokenProb(" \u092f", -1.5935224294662476, None, " \u092f".encode()), ()
    )
    assert plain.chosen == TokenProb("\u0939", -0.42422962188720703, None, "\u0939".encode())
    assert [prob.token for prob in plain.top] == ["\u0939", "\u0947", "\u094b"]
    assert plain.top[1].token_bytes == "\u0947".encode()


def test_score_continuation_rebuilds_prefixes_from_bytes(server: FakeServer) -> None:
    def reply(request: Request) -> tuple[int, object]:
        return 200, _one_step_reply(f"<{len(request.body['prompt'])}>")

    server.respond("POST", "/api/generate", reply)
    lead = TokenProb(" ", -0.4, None, b" \xe0\xa4")
    tail = TokenProb("\ufffd", -1.6, None, b"\xaf")
    continuation = [TokenStep(lead, (lead,)), TokenStep(tail, (tail,)), _step(".", -0.2)]

    result = _backend(server).score_continuation("P", continuation, top_k=2)

    # The prefix after the lead bytes ends inside a character, so it is not sent.
    assert [body["prompt"] for body in server.bodies("/api/generate")] == ["P", "P \u092f"]
    assert [top[0].token if top else None for top in result] == ["<1>", None, "<3>"]


def test_score_continuation_sends_nothing_for_unscored_positions(server: FakeServer) -> None:
    server.respond("POST", "/api/generate", lambda request: (200, _one_step_reply("\u0939")))
    folded = TokenStep(TokenProb(" \u092f", -1.6, None, " \u092f".encode()), ())

    result = _backend(server).score_continuation("P", [folded, _step("\u0939", -0.4)], top_k=2)

    assert [body["prompt"] for body in server.bodies("/api/generate")] == ["P \u092f"]
    assert result[0] == ()
    assert result[1] == (TokenProb("\u0939", -0.5),)


def test_score_continuation_reads_a_held_back_fragment_as_no_distribution(
    server: FakeServer,
) -> None:
    # Captured from Ollama 0.35.1: the one requested token ends inside a character, so the
    # reply has no text and no logprobs, like end of sequence but with done_reason "length".
    server.reply("POST", "/api/generate", load("ollama_generate_partial"))
    assert _backend(server).score_continuation("P", [_step("\u0939", -0.4)], top_k=3) == [()]


def test_score_continuation_of_nothing_sends_nothing(server: FakeServer) -> None:
    assert _backend(server).score_continuation("x", [], top_k=2) == []
    assert server.requests == []


def test_wrong_spec_kind_is_rejected() -> None:
    spec = CandidateSpec(kind="llamacpp", base_url="http://127.0.0.1:1", model="", label="l")
    with pytest.raises(SpecError):
        OllamaBackend(spec)


# live -------------------------------------------------------------------------------------

live = pytest.mark.skipif(os.environ.get("QUANTDIFF_LIVE") != "1", reason="QUANTDIFF_LIVE=1")


def _live_backend() -> OllamaBackend:
    url = os.environ.get("QUANTDIFF_OLLAMA_URL", "http://127.0.0.1:11434")
    tag = os.environ.get("QUANTDIFF_OLLAMA_MODEL", TAG)
    return OllamaBackend(CandidateSpec(kind="ollama", base_url=url, model=tag, label=tag))


@pytest.mark.live
@live
def test_live_info_and_scoring() -> None:
    backend = _live_backend()
    assert backend.info().context_length is not None
    steps = backend.generate_scored("The capital of France is", max_tokens=2, top_k=3)
    assert len(steps) == 2
    scores = backend.score_continuation("The capital of France is", steps[:1], top_k=3)
    assert scores[0][0].token == steps[0].chosen.token


@pytest.mark.live
@live
def test_live_split_characters_agree_with_themselves() -> None:
    # The built-in Hindi prompt splits characters across tokens. Against itself the model
    # must agree wherever the reference had a clear winner, and stay near zero KLD overall.
    # Near-ties may flip: Ollama computes a position slightly differently inside a batch
    # (generation) than as a single forced token (docs/calibration.md, finding 1).
    backend = _live_backend()
    prompt = next(p.text for p in load_scoring_prompts(None) if p.id == "score-038")
    steps = backend.generate_scored(prompt, max_tokens=32, top_k=3)
    scores = backend.score_continuation(prompt, steps, top_k=3)
    scored = [(step, top) for step, top in zip(steps, scores, strict=True) if step.top and top]
    assert scored
    clear = [(step, top) for step, top in scored if step.top[0].logprob - step.top[1].logprob > 0.5]
    assert all(top[0].token == step.chosen.token for step, top in clear)
    mean_kld = sum(partition_kld(step.top, top, by_id=False) for step, top in scored) / len(scored)
    assert mean_kld < 0.02


@pytest.mark.live
@live
def test_live_weights_identity_and_decode_speed() -> None:
    backend = _live_backend()
    info = backend.info()
    assert info.size_bytes is not None
    assert info.size_bytes > 0
    assert info.weights_id is not None
    assert len(info.weights_id) == 64
    result = backend.chat([Message(role="user", content="Count from 1 to 10.")], max_tokens=16)
    assert result.completion_tokens is not None
    assert result.completion_tokens > 1
    assert result.decode_tokens_per_second is not None
    assert result.decode_tokens_per_second > 0


@pytest.mark.live
@live
def test_live_tool_call() -> None:
    result = _live_backend().chat(
        [Message(role="user", content="What is the weather in Paris?")],
        max_tokens=60,
        tools=[WEATHER],
    )
    assert [call.name for call in result.tool_calls] == ["get_weather"]
