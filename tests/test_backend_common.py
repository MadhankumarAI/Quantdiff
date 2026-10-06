"""Shared backend helpers for text teacher forcing and folded multi-byte tokens."""

from __future__ import annotations

import pytest

from quantdiff.backends._common import folds_tokens, forced_texts, optional_bytes, unfolded_steps
from quantdiff.types import JSONValue, TokenProb, TokenStep
from tests.fixtures.http_fake import load

YA = " \u092f"
"""Devanagari YA after a space: bytes 20 E0 A4 AF, split by Qwen2.5 into two tokens."""
YA_LEAD = YA.encode("utf-8")[:3]
YA_TAIL = YA.encode("utf-8")[3:]


def _token(token: str, token_bytes: bytes | None = None, token_id: int | None = None) -> TokenStep:
    chosen = TokenProb(token=token, logprob=-0.5, token_id=token_id, token_bytes=token_bytes)
    return TokenStep(chosen=chosen, top=(chosen,))


def _unscored(token: str, token_bytes: bytes | None = None) -> TokenStep:
    return TokenStep(TokenProb(token=token, logprob=-0.5, token_bytes=token_bytes), ())


def _ollama_step(entry: JSONValue) -> TokenStep:
    def prob(item: JSONValue) -> TokenProb:
        return TokenProb(item["token"], item["logprob"], None, optional_bytes(item["bytes"]))

    return TokenStep(prob(entry), tuple(prob(item) for item in entry["top_logprobs"]))


def _llamacpp_step(entry: JSONValue) -> TokenStep:
    def prob(item: JSONValue) -> TokenProb:
        return TokenProb(item["token"], item["logprob"], item["id"], optional_bytes(item["bytes"]))

    return TokenStep(prob(entry), tuple(prob(item) for item in entry["top_logprobs"]))


# optional_bytes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([32, 224, 164, 175], YA.encode("utf-8")),
        ([], b""),
        (None, None),
        ("abc", None),
        ([True, 1], None),
        ([256], None),
        ([-1], None),
        ([1.0], None),
    ],
)
def test_optional_bytes(value: JSONValue, expected: bytes | None) -> None:
    assert optional_bytes(value) == expected


# forced_texts -----------------------------------------------------------------------------


def test_forced_texts_join_token_text_without_bytes() -> None:
    steps = [_token(" Paris"), _token("."), _token(" It")]
    assert forced_texts("The capital is", steps) == [
        "The capital is",
        "The capital is Paris",
        "The capital is Paris.",
    ]


def test_forced_texts_rebuild_a_split_character_from_bytes() -> None:
    # llama-server reports the lead bytes of YA as the text " " and the tail as U+FFFD.
    steps = [_token(" ", YA_LEAD, 14925), _token("\ufffd", YA_TAIL, 107), _token("x", b"x")]
    assert forced_texts("P", steps) == ["P", None, "P" + YA]


def test_forced_texts_skip_positions_the_reference_left_unscored() -> None:
    steps = [_unscored(YA, YA.encode("utf-8")), _token("\u0939", "\u0939".encode())]
    assert forced_texts("P", steps) == [None, "P" + YA]


def test_forced_texts_mix_bytes_and_text() -> None:
    steps = [_token(" a", b" a"), _token(" b"), _token(" c", b" c")]
    assert forced_texts("", steps) == ["", " a", " a b"]


# folded tokens ----------------------------------------------------------------------------


def test_ollama_folded_entry_is_detected_by_its_missing_text() -> None:
    folded, plain = (_ollama_step(entry) for entry in load("ollama_generate_folded")["logprobs"])
    assert folded.chosen.token == YA
    assert {prob.token for prob in folded.top} == {"\ufffd"}
    assert folds_tokens(folded)
    assert not folds_tokens(plain)


def test_llamacpp_folded_entry_is_detected_by_its_own_bytes() -> None:
    entries = load("llamacpp_completion_folded")["completion_probabilities"]
    folded, plain = (_llamacpp_step(entry) for entry in entries)
    # The folded entry carries the id of the last token (107), whose own bytes are only AF.
    assert folded.chosen.token_id == folded.top[0].token_id == 107
    assert folds_tokens(folded)
    assert not folds_tokens(plain)


def test_steps_without_bytes_or_distribution_are_not_folded() -> None:
    assert not folds_tokens(_token(" Paris"))
    assert not folds_tokens(_unscored(" Paris"))
    other = TokenStep(TokenProb("a", -0.1, 1), (TokenProb("a", -0.1, 1, b"a"),))
    assert not folds_tokens(other)


def test_unfolded_steps_empty_only_folded_distributions() -> None:
    steps = [_ollama_step(entry) for entry in load("ollama_generate_folded")["logprobs"]]
    unfolded = unfolded_steps(steps)
    assert unfolded[0] == TokenStep(steps[0].chosen, ())
    assert unfolded[1] == steps[1]
