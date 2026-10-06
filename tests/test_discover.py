from __future__ import annotations

from collections.abc import Iterator

import pytest

from quantdiff.discover import (
    Member,
    ModelFamily,
    closest_tags,
    discover_ollama,
    format_discovery,
    format_size,
    installed_tags,
    precision_bits,
    source_of,
    suggest_command,
)
from quantdiff.errors import BackendError
from tests.fixtures.http_fake import FakeServer, Reply, Request, closed_port, load, running

SMALL = "qwen2.5:0.5b-instruct"
UNSLOTH = "hf.co/unsloth/Qwen3-8B-GGUF"
BARTOWSKI = "hf.co/bartowski/Qwen_Qwen3-8B-GGUF"


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with running() as fake:
        yield fake


def _tag(name: str, *, quant: str | None = None, size: int = 1, digest: str = "") -> object:
    details = {} if quant is None else {"quantization_level": quant}
    return {"name": name, "model": name, "size": size, "digest": digest or name, "details": details}


def _discover(server: FakeServer, *models: object) -> list[ModelFamily]:
    server.reply("GET", "/api/tags", {"models": list(models)})
    return discover_ollama(server.url)


def _serve_show(server: FakeServer, fixture: str) -> None:
    """Answer /api/show like Ollama: the captured body, or 404 for an unknown model."""
    bodies = load(fixture)

    def show(request: Request) -> Reply:
        model = request.body["model"]
        if model not in bodies:
            return 404, {"error": f"model '{model}' not found"}
        return 200, bodies[model]

    server.respond("POST", "/api/show", show)


def _uploaders(server: FakeServer, *, metadata: bool = True) -> list[ModelFamily]:
    server.reply("GET", "/api/tags", load("ollama_tags_uploaders"))
    if metadata:
        _serve_show(server, "ollama_tags_uploaders_show")
    return discover_ollama(server.url)


def _by_name(families: list[ModelFamily]) -> dict[str, ModelFamily]:
    return {family.name: family for family in families}


# grouping ---------------------------------------------------------------------------------


def test_real_listing_groups_quants_of_one_model(server: FakeServer) -> None:
    server.reply("GET", "/api/tags", load("ollama_tags"))
    families = discover_ollama(server.url, timeout=2)

    assert [family.name for family in families] == [SMALL, "qwen2.5:3b"]
    small, big = families
    assert small.members == (
        (f"{SMALL}-q8_0", "Q8_0", 531081605),
        (f"{SMALL}-q4_K_M", "Q4_K_M", 397821319),
        (f"{SMALL}-q2_K", "Q2_K", 338620805),
    )
    assert small.suggested_reference == f"{SMALL}-q8_0"
    assert small.suggested_candidates == (f"{SMALL}-q4_K_M", f"{SMALL}-q2_K")
    assert big.members == (("qwen2.5:3b", "Q4_K_M", 1929912432),)
    assert big.suggested_candidates == ()


def test_suffixes_are_parsed_when_details_are_missing(server: FakeServer) -> None:
    families = _discover(
        server,
        _tag("llama3:8b-instruct-Q4_K_M"),
        _tag("llama3:8b-instruct-fp16"),
        _tag("llama3:8b-instruct-iq4_xs"),
        _tag("llama3:8b-instruct-q6_K"),
        _tag("llama3:8b-instruct-bf16", size=2),
    )
    (family,) = families
    assert family.name == "llama3:8b-instruct"
    assert [(tag.rpartition("-")[2], quant) for tag, quant, _ in family.members] == [
        ("bf16", "BF16"),
        ("fp16", "FP16"),
        ("q6_K", "Q6_K"),
        ("Q4_K_M", "Q4_K_M"),
        ("iq4_xs", "IQ4_XS"),
    ]


def test_bare_tag_joins_the_family_of_its_suffixed_quants(server: FakeServer) -> None:
    families = _discover(
        server, _tag("qwen2.5:3b", quant="Q4_K_M", digest="a"), _tag("qwen2.5:3b-q8_0", digest="b")
    )
    (family,) = families
    assert family.name == "qwen2.5:3b"
    assert family.suggested_reference == "qwen2.5:3b-q8_0"


def test_tags_sharing_a_digest_are_listed_once(server: FakeServer) -> None:
    families = _discover(
        server,
        _tag(f"{SMALL}", quant="Q4_K_M", digest="same"),
        _tag(f"{SMALL}-q4_K_M", quant="Q4_K_M", digest="same"),
    )
    assert families[0].members == ((f"{SMALL}-q4_K_M", "Q4_K_M", 1),)


def test_hugging_face_tags_group_by_repository(server: FakeServer) -> None:
    repo = "hf.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    (family,) = _discover(server, _tag(f"{repo}:Q4_K_M"), _tag(f"{repo}:Q8_0"))
    assert family.name == repo
    assert family.suggested_reference == f"{repo}:Q8_0"


def test_unknown_quantization_ranks_last(server: FakeServer) -> None:
    (family,) = _discover(server, _tag("m:7b", size=9), _tag("m:7b-q4_0"))
    assert family.members[-1] == ("m:7b", "unknown", 9)


@pytest.mark.parametrize(
    ("quant", "bits"),
    [("BF16", 16), ("F16", 16), ("FP16", 16), ("Q8_0", 8), ("IQ4_XS", 4), ("Q2_K", 2), ("?", 0)],
)
def test_precision_bits(quant: str, bits: int) -> None:
    assert precision_bits(quant) == bits


# errors -----------------------------------------------------------------------------------


def test_unreachable_ollama_says_how_to_start_it() -> None:
    url = f"http://127.0.0.1:{closed_port()}"
    with pytest.raises(BackendError, match=r"cannot reach Ollama at .*`ollama serve`"):
        discover_ollama(url)


def test_default_url_comes_from_ollama_host(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.reply("GET", "/api/tags", {"models": []})
    monkeypatch.setenv("OLLAMA_HOST", server.url)
    assert discover_ollama() == []
    assert [request.path for request in server.requests] == ["/api/tags"]


def test_malformed_listing_is_a_backend_error(server: FakeServer) -> None:
    server.reply("GET", "/api/tags", {"models": [{"size": 3}]})
    with pytest.raises(BackendError, match="model name"):
        discover_ollama(server.url)


# suggestions and output -------------------------------------------------------------------


def test_suggest_command() -> None:
    family = ModelFamily(
        SMALL, (Member(f"{SMALL}-q8_0", "Q8_0", 1), Member(f"{SMALL}-q4_K_M", "Q4_K_M", 1))
    )
    assert suggest_command(family) == (
        f"quantdiff run --ref ollama:{SMALL}-q8_0 --cand ollama:{SMALL}-q4_K_M"
    )
    assert suggest_command(ModelFamily("x", (Member("x:1b", "Q4_0", 1),))) is None


@pytest.mark.parametrize(
    ("size", "text"), [(338620805, "339 MB"), (1929912432, "1.9 GB"), (0, "0 MB")]
)
def test_format_size(size: int, text: str) -> None:
    assert format_size(size) == text


def test_format_discovery_of_the_real_listing(server: FakeServer) -> None:
    server.reply("GET", "/api/tags", load("ollama_tags"))
    assert format_discovery(discover_ollama(server.url)) == "\n".join(
        [
            "Found 4 downloads of 2 models in Ollama.",
            "",
            SMALL,
            f"  Q8_0     531 MB  {SMALL}-q8_0  (reference)",
            f"  Q4_K_M   398 MB  {SMALL}-q4_K_M",
            f"  Q2_K     339 MB  {SMALL}-q2_K",
            "  Compare them:",
            f"    quantdiff run --ref ollama:{SMALL}-q8_0 --cand ollama:{SMALL}-q4_K_M"
            f" --cand ollama:{SMALL}-q2_K",
            "",
            "qwen2.5:3b",
            "  Q4_K_M   1.9 GB  qwen2.5:3b",
            "  Only one download. Pull a q8_0 or fp16 download of this model from its tag",
            "  list to use as a reference, then run quantdiff discover again.",
        ]
    )


def test_single_low_bit_download_suggests_pulling_q8() -> None:
    family = ModelFamily("llama3:8b-instruct", (Member("llama3:8b-instruct-q4_K_M", "Q4_K_M", 1),))
    assert format_discovery([family]).splitlines()[-2:] == [
        "  Only one download. To get a reference to compare against:",
        "    ollama pull llama3:8b-instruct-q8_0",
    ]


def test_single_high_bit_download_suggests_pulling_q4() -> None:
    family = ModelFamily("repo", (Member("hf.co/a/b-GGUF:Q8_0", "Q8_0", 1),))
    assert format_discovery([family]).splitlines()[-2:] == [
        "  Only one download. To get a smaller download to compare:",
        "    ollama pull hf.co/a/b-GGUF:Q4_K_M",
    ]


def test_low_bit_reference_gets_a_warning() -> None:
    family = ModelFamily(
        "m:7b", (Member("m:7b-q4_K_M", "Q4_K_M", 2), Member("m:7b-q2_K", "Q2_K", 1))
    )
    assert "most precise download here is Q4_K_M" in format_discovery([family])


def test_comparable_families_are_listed_first() -> None:
    single = ModelFamily("a:1b", (Member("a:1b", "Q4_0", 1),))
    pair = ModelFamily("z:1b", (Member("z:1b-q8_0", "Q8_0", 2), Member("z:1b-q4_0", "Q4_0", 1)))
    text = format_discovery([single, pair])
    assert text.index("z:1b") < text.index("a:1b")


def test_no_models_suggests_what_to_pull() -> None:
    text = format_discovery([])
    assert text.startswith("No models found in Ollama.")
    assert "ollama pull qwen2.5:0.5b-instruct-q8_0" in text
    assert "ollama pull qwen2.5:0.5b-instruct-q4_K_M" in text


# metadata and uploaders -------------------------------------------------------------------


def test_real_metadata_keeps_the_real_grouping(server: FakeServer) -> None:
    server.reply("GET", "/api/tags", load("ollama_tags"))
    _serve_show(server, "ollama_tags_show")
    families = discover_ollama(server.url)
    assert [family.name for family in families] == [SMALL, "qwen2.5:3b"]
    assert len(families[0].members) == 3
    assert [request.body["model"] for request in server.requests[1:]] == [
        f"{SMALL}-q2_K",
        f"{SMALL}-q4_K_M",
        f"{SMALL}-q8_0",
        "qwen2.5:3b",
    ]


def test_library_tag_and_uploaders_form_one_family(server: FakeServer) -> None:
    families = _by_name(_uploaders(server))
    assert sorted(families) == [
        "hf.co/Qwen/Qwen2.5-3B-GGUF",
        "mychat:latest",
        "qwen2.5:3b",
        "qwen3:8b",
    ]
    qwen3 = families["qwen3:8b"]
    assert qwen3.members == (
        (f"{BARTOWSKI}:Q8_0", "Q8_0", 8709519584),
        ("qwen3:8b", "Q4_K_M", 5225388164),
        (f"{UNSLOTH}:UD-Q4_K_XL", "UD-Q4_K_XL", 5135890208),
        (f"{BARTOWSKI}:Q4_K_M", "Q4_K_M", 5027785440),
        (f"{UNSLOTH}:Q4_K_M", "Q4_K_M", 5027783456),
    )
    assert qwen3.sources == ("bartowski", "ollama", "unsloth")


def test_metadata_never_merges_a_custom_name_or_a_different_finetune(server: FakeServer) -> None:
    families = _by_name(_uploaders(server))
    # Same weights and metadata as qwen3:8b, but the name says nothing about the model.
    assert families["mychat:latest"].suggested_reference == "mychat:latest"
    # Same name tokens as the library qwen2.5:3b, but a base model, not Instruct.
    assert len(families["qwen2.5:3b"].members) == 1
    assert len(families["hf.co/Qwen/Qwen2.5-3B-GGUF"].members) == 1


def test_one_show_request_per_file(server: FakeServer) -> None:
    _uploaders(server)
    shown = [request.body["model"] for request in server.requests if request.path == "/api/show"]
    assert "qwen3:latest" in shown
    assert "qwen3:8b" not in shown
    assert len(shown) == len(set(shown)) == 8


def test_without_metadata_uploaders_still_group_by_name(server: FakeServer) -> None:
    families = _by_name(_uploaders(server, metadata=False))
    assert families["Qwen3-8B"].sources == ("bartowski", "unsloth")
    assert len(families["Qwen3-8B"].members) == 4
    # Library names cannot be told apart from base repos, so they stay separate.
    assert len(families["qwen3:8b"].members) == 1
    assert len(families["qwen2.5:3b"].members) == 1


def test_format_discovery_labels_sources(server: FakeServer) -> None:
    text = format_discovery(_uploaders(server))
    assert "\n".join(text.splitlines()[2:10]) == "\n".join(
        [
            "qwen3:8b  (from bartowski, ollama, unsloth)",
            f"  bartowski  Q8_0         8.7 GB  {BARTOWSKI}:Q8_0  (reference)",
            "  ollama     Q4_K_M       5.2 GB  qwen3:8b",
            f"  unsloth    UD-Q4_K_XL   5.1 GB  {UNSLOTH}:UD-Q4_K_XL",
            f"  bartowski  Q4_K_M       5.0 GB  {BARTOWSKI}:Q4_K_M",
            f"  unsloth    Q4_K_M       5.0 GB  {UNSLOTH}:Q4_K_M",
            "  Compare them:",
            f"    quantdiff run --ref ollama:{BARTOWSKI}:Q8_0 --cand ollama=ollama:qwen3:8b"
            f" --cand unsloth-UD-Q4_K_XL=ollama:{UNSLOTH}:UD-Q4_K_XL"
            f" --cand bartowski=ollama:{BARTOWSKI}:Q4_K_M"
            f" --cand unsloth-Q4_K_M=ollama:{UNSLOTH}:Q4_K_M",
        ]
    )


def test_unsloth_dynamic_quants_are_quantizations(server: FakeServer) -> None:
    (family,) = _discover(
        server,
        _tag(f"{UNSLOTH}:UD-Q4_K_XL", quant="Q4_K_M"),
        _tag(f"{UNSLOTH}:UD-IQ2_M", quant="IQ2_M"),
        _tag(f"{UNSLOTH}:Q8_0", quant="Q8_0"),
    )
    assert family.name == UNSLOTH
    assert [member.quantization for member in family.members] == ["Q8_0", "UD-Q4_K_XL", "UD-IQ2_M"]
    assert suggest_command(family) == (
        f"quantdiff run --ref ollama:{UNSLOTH}:Q8_0"
        f" --cand ollama:{UNSLOTH}:UD-Q4_K_XL --cand ollama:{UNSLOTH}:UD-IQ2_M"
    )


def test_single_unsloth_dynamic_quant_suggests_its_q8(server: FakeServer) -> None:
    families = _discover(server, _tag(f"{UNSLOTH}:UD-Q4_K_XL", quant="Q4_K_M"))
    assert format_discovery(families).splitlines()[-1] == f"    ollama pull {UNSLOTH}:Q8_0"


@pytest.mark.parametrize(
    ("tag", "source"),
    [
        ("qwen3:8b", "ollama"),
        ("registry.ollama.ai/library/qwen3:8b", "ollama"),
        ("someone/qwen3:8b", "someone"),
        (f"{UNSLOTH}:Q4_K_M", "unsloth"),
        ("huggingface.co/bartowski/x-GGUF", "bartowski"),
    ],
)
def test_source_of(tag: str, source: str) -> None:
    assert source_of(tag) == source


@pytest.mark.parametrize(("quant", "bits"), [("UD-Q4_K_XL", 4), ("UD-IQ2_M", 2), ("Q4_K_XL", 4)])
def test_precision_bits_of_dynamic_quants(quant: str, bits: int) -> None:
    assert precision_bits(quant) == bits


# suggestions for mistyped tags ------------------------------------------------------------

INSTALLED = [f"{SMALL}-q2_K", f"{SMALL}-q4_K_M", f"{SMALL}-q8_0", "qwen2.5:3b"]


def test_a_bare_name_suggests_its_installed_tags() -> None:
    assert closest_tags("qwen2.5", INSTALLED) == [
        "qwen2.5:3b",
        f"{SMALL}-q2_K",
        f"{SMALL}-q8_0",
    ]


def test_a_longer_misspelled_tag_suggests_the_tag_it_extends() -> None:
    assert closest_tags("qwen2.5:3b-instrct", INSTALLED)[0] == "qwen2.5:3b"


def test_near_spellings_are_suggested_and_unrelated_names_are_not() -> None:
    assert closest_tags("qwen2.5:0.5b-instruct-q4_k_m", INSTALLED)[0] == f"{SMALL}-q4_K_M"
    assert closest_tags("llama3", INSTALLED) == []
    assert "qwen2.5:3b" not in closest_tags("qwen2.5:3b", INSTALLED)


def test_installed_tags_lists_the_names(server: FakeServer) -> None:
    server.reply("GET", "/api/tags", load("ollama_tags"))
    assert installed_tags(server.url) == INSTALLED
