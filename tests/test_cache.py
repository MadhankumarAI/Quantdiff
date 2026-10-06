from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantdiff.cache import (
    CACHE_FORMAT,
    ReferenceCache,
    ReferenceOutputs,
    cache_key,
    default_cache_dir,
)
from quantdiff.types import ChatResult, ReferenceTrace, ServerInfo, TokenProb, TokenStep, ToolCall


def _info(model: str = "m") -> ServerInfo:
    return ServerInfo(
        backend="ollama",
        model=model,
        context_length=4096,
        chat_template=None,
        template_dialect="go",
        supports_logprobs=True,
        exact_token_ids=False,
        details=(("quantization", "Q8_0"),),
    )


def _outputs() -> ReferenceOutputs:
    step = TokenStep(
        chosen=TokenProb(" Paris", -0.4, 12, b" Paris"),
        top=(TokenProb(" Paris", -0.4, 12, b" Paris"), TokenProb(" Lyon", -2.1, None)),
    )
    # A token holding only the lead bytes of a character, and a folded step left unscored.
    fragment = TokenProb(" ", -0.44, 14925, b" \xe0\xa4")
    split = TokenStep(chosen=fragment, top=(fragment, TokenProb("", -3.0, 151645, b"")))
    unscored = TokenStep(chosen=TokenProb(" \u092f", -1.6, None, " \u092f".encode()), top=())
    answer = ChatResult(
        text="hi",
        tool_calls=(ToolCall("get_weather", {"city": "Oslo"}, '{"city": "Oslo"}'),),
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=None,
        seconds=0.25,
        decode_tokens_per_second=41.5,
    )
    return ReferenceOutputs(
        answers={"json-001": answer},
        traces=(
            ReferenceTrace("score-001", (1, 2, 3), (step,)),
            ReferenceTrace("score-037", None, (split, unscored)),
        ),
    )


def _key(
    *,
    model: str = "m",
    base_url: str = "http://127.0.0.1:11434",
    suite_digest: str = "abc",
    top_k: int = 10,
    score_tokens: int = 32,
    seed: int = 0,
) -> str:
    return cache_key(
        _info(model),
        base_url=base_url,
        suite_digest=suite_digest,
        top_k=top_k,
        score_tokens=score_tokens,
        seed=seed,
    )


def test_round_trip(tmp_path: Path) -> None:
    cache = ReferenceCache(tmp_path)
    key = _key()
    cache.store(key, _outputs())
    assert cache.load(key) == _outputs()


def test_entry_from_an_older_format_is_a_miss(tmp_path: Path) -> None:
    key = _key()
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"format": CACHE_FORMAT - 1, "answers": {}, "traces": []}), encoding="utf-8"
    )
    assert ReferenceCache(tmp_path).load(key) is None


def test_missing_entry_is_a_miss(tmp_path: Path) -> None:
    assert ReferenceCache(tmp_path).load(_key()) is None


def test_corrupt_entry_is_a_miss(tmp_path: Path) -> None:
    key = _key()
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert ReferenceCache(tmp_path).load(key) is None


def test_token_bytes_are_stored_as_base64(tmp_path: Path) -> None:
    key = _key()
    ReferenceCache(tmp_path).store(key, _outputs())
    data = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    fragment = data["traces"][1]["steps"][0]["chosen"]
    assert fragment == [" ", -0.44, 14925, "IOCk"]
    assert data["traces"][0]["steps"][0]["top"][1] == [" Lyon", -2.1, None, None]


@pytest.mark.parametrize("raw", ["not base64!", 7])
def test_bad_token_bytes_are_a_miss(tmp_path: Path, raw: object) -> None:
    key = _key()
    ReferenceCache(tmp_path).store(key, _outputs())
    path = tmp_path / f"{key}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["traces"][0]["steps"][0]["chosen"][3] = raw
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ReferenceCache(tmp_path).load(key) is None


def test_entry_without_token_bytes_is_a_miss(tmp_path: Path) -> None:
    key = _key()
    ReferenceCache(tmp_path).store(key, _outputs())
    path = tmp_path / f"{key}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["traces"][0]["steps"][0]["chosen"] = [" Paris", -0.4, 12]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ReferenceCache(tmp_path).load(key) is None


def test_key_changes_with_every_input() -> None:
    base = _key()
    assert _key(top_k=5) != base
    assert _key(score_tokens=8) != base
    assert _key(seed=1) != base
    assert _key(suite_digest="def") != base
    assert _key(model="other") != base
    assert _key(base_url="http://10.0.0.2:11434") != base


def test_rejects_keys_that_could_escape_the_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid cache key"):
        ReferenceCache(tmp_path).load("../../etc/passwd")


def test_store_leaves_no_temp_files(tmp_path: Path) -> None:
    ReferenceCache(tmp_path).store(_key(), _outputs())
    assert [p.name for p in tmp_path.iterdir()] == [f"{_key()}.json"]


def test_default_dir_honors_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUANTDIFF_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path
