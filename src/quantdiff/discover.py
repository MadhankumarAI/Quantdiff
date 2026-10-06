"""Find the models installed in Ollama and suggest a ready-to-run comparison.

Downloads are grouped into families of the same model in two steps.

1. Every tag gets a name key: the model name and tag with the quantization suffix removed,
   split into lowercase word and number tokens so "-", "_", "." and case do not matter.
   For `hf.co/<uploader>/<repo>:<quant>` the uploader is dropped, and so are a trailing
   `-GGUF`, an imatrix `-i1` and a leading `<org>_` prefix (bartowski names repos
   `Qwen_Qwen3-8B-GGUF`). `qwen3:8b`, `hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL` and
   `hf.co/bartowski/Qwen_Qwen3-8B-GGUF:Q4_K_M` all get the tokens qwen, 3, 8b.
2. Ollama's /api/show reports the GGUF metadata of each file. When it has an architecture,
   basename and size label, and the name key contains a number and only tokens that the
   metadata (basename, size label, fine-tune) also has, the download is keyed by that
   metadata. So `qwen2.5:3b`, whose metadata says Qwen2.5 3B Instruct, joins
   `hf.co/x/Qwen2.5-3B-Instruct-GGUF:Q8_0`, while a custom `mychat:latest` built on the
   same weights stays on its own. Without usable metadata a download is keyed by its name
   key alone, and Ollama library tags never share a name key with Hugging Face downloads:
   library tags often leave out "instruct", so `qwen2.5:3b` and a base `Qwen2.5-3B` repo
   have the same name.

Within a family the most precise download is the suggested reference. A family with
downloads from more than one source (the Ollama library and Hugging Face uploaders) gets a
command with one labelled candidate per source, so the uploads can be compared directly.
"""

from __future__ import annotations

import difflib
import functools
import logging
import os
import re
import shlex
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from quantdiff._http import get_json, post_json, validate_base_url
from quantdiff._text import printable
from quantdiff.backends._common import (
    expect_dict,
    expect_list,
    expect_str,
    explained_failures,
    optional_int,
    optional_str,
)
from quantdiff.backends.ollama import explain_ollama_failure
from quantdiff.errors import BackendError
from quantdiff.spec import ollama_base_url
from quantdiff.types import JSONValue

logger = logging.getLogger(__name__)

_QUANT_SUFFIX: Final = re.compile(
    r"(?:^|[-_.])(?P<quant>(?:ud-)?i?q\d+(?:_[0-9a-z]+)*|bf16|fp16|f16|fp32|f32)$",
    re.IGNORECASE,
)
_BITS: Final = re.compile(r"^(?:ud-)?(?:i?q|mxfp|b?f|fp)(?P<bits>\d+)", re.IGNORECASE)
_TOKEN: Final = re.compile(r"[a-z]+|\d+(?:\.\d+)*(?:[a-z](?![a-z]))?")
_HF_HOSTS: Final = ("hf.co/", "huggingface.co/")
_REPO_SUFFIXES: Final = (
    re.compile(r"[-_.]gguf$", re.IGNORECASE),
    re.compile(r"-i1$", re.IGNORECASE),
)
_ORG_PREFIX: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*_(?=[A-Za-z])")
_IGNORED_TOKENS: Final = frozenset({"latest"})
_REFERENCE_MIN_BITS: Final = 8
"""Below this, a download is a poor stand-in for the original weights."""
_DECIMAL_GB: Final = 1000**3
_DECIMAL_MB: Final = 1000**2
_SUGGESTION_CUTOFF: Final = 0.6
UNKNOWN_QUANTIZATION: Final = "unknown"
LIBRARY_SOURCE: Final = "ollama"
"""Source name for tags from the Ollama library, as opposed to a Hugging Face uploader."""


class Member(NamedTuple):
    """One download in a family."""

    tag: str
    quantization: str
    """Such as Q4_K_M or UD-Q4_K_XL."""
    size: int
    """Bytes."""

    @property
    def source(self) -> str:
        return source_of(self.tag)


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """Downloads of one model that differ in quantization or uploader.

    `members` is sorted from most to least precise, larger files first on ties.
    """

    name: str
    members: tuple[Member, ...]

    @property
    def suggested_reference(self) -> str:
        """The most precise download, the best stand-in for the original weights."""
        return self.members[0].tag

    @property
    def suggested_candidates(self) -> tuple[str, ...]:
        return tuple(member.tag for member in self.members[1:])

    @property
    def sources(self) -> tuple[str, ...]:
        """Where the downloads came from, in member order, each once."""
        return tuple(dict.fromkeys(member.source for member in self.members))


@dataclass(frozen=True, slots=True)
class _Download:
    tag: str
    display: str
    """The family name this tag alone suggests, such as `qwen2.5:0.5b-instruct`."""
    name_tokens: tuple[str, ...]
    quantization: str
    size: int
    digest: str
    explicit_suffix: bool
    """True when the tag itself names the quantization, as in `-q4_K_M`."""

    @property
    def member(self) -> Member:
        return Member(self.tag, self.quantization, self.size)


@dataclass(frozen=True, slots=True)
class _SplitTag:
    display: str
    stem: str
    """The tag up to and including the separator before the quantization suffix."""
    suffix: str | None
    name_tokens: tuple[str, ...]


def discover_ollama(base_url: str | None = None, *, timeout: float = 5.0) -> list[ModelFamily]:
    """List installed Ollama models grouped into families, sorted by family name.

    `base_url` defaults to OLLAMA_HOST, resolved the way the Ollama CLI does. Sends one
    /api/show request per distinct file; a failed one only means that file is grouped by
    its name.
    """
    url = _resolve(base_url)
    downloads = [_read_download(entry) for entry in _listing(url, timeout)]
    identities: dict[str, frozenset[str] | None] = {}
    for download in downloads:
        if download.digest not in identities:
            identities[download.digest] = _show_identity(url, download.tag, timeout)
    return _group(downloads, identities)


def installed_tags(base_url: str | None = None, *, timeout: float = 5.0) -> list[str]:
    """Every tag Ollama has installed, in its listing order."""
    url = _resolve(base_url)
    return [_tag_name(entry) for entry in _listing(url, timeout)]


def closest_tags(wanted: str, installed: Iterable[str], *, limit: int = 3) -> list[str]:
    """Up to `limit` installed tags that `wanted` was probably meant to be.

    Tags that extend `wanted` or that `wanted` extends (`qwen2.5` and `qwen2.5:3b-instrct`
    both suggest `qwen2.5:3b`) come first, then near spellings.
    """
    tags = list(dict.fromkeys(installed))
    folded = wanted.casefold()

    def similarity(tag: str) -> float:
        return difflib.SequenceMatcher(None, folded, tag.casefold()).ratio()

    def related(tag: str) -> bool:
        other = tag.casefold()
        return other.startswith(folded) or folded.startswith(other)

    extending = sorted(
        (tag for tag in tags if related(tag)), key=lambda tag: (-similarity(tag), tag)
    )
    by_folded = {tag.casefold(): tag for tag in tags}
    similar = [
        by_folded[match]
        for match in difflib.get_close_matches(
            folded, by_folded, n=limit, cutoff=_SUGGESTION_CUTOFF
        )
    ]
    return list(dict.fromkeys(tag for tag in [*extending, *similar] if tag != wanted))[:limit]


def source_of(tag: str) -> str:
    """The Hugging Face uploader or Ollama namespace a tag comes from, else `ollama`."""
    name = _name_part(tag)
    hf = _hf_repo(name)
    if hf is not None:
        return hf[0]
    namespace, separator, _ = name.rpartition("/")
    owner = namespace.rpartition("/")[2]
    if not separator or owner == "library":
        return LIBRARY_SOURCE
    return owner


def suggest_command(family: ModelFamily) -> str | None:
    """A `quantdiff run` command comparing every download in the family, if it has two.

    Candidates are labelled by source when the family has more than one.
    """
    if len(family.members) < 2:
        return None
    labels = _candidate_labels(family)
    parts = ["quantdiff run", f"--ref ollama:{shlex.quote(family.suggested_reference)}"]
    for member in family.members[1:]:
        label = labels.get(member.tag)
        spec = f"ollama:{member.tag}" if label is None else f"{label}=ollama:{member.tag}"
        parts.append(f"--cand {shlex.quote(spec)}")
    return " ".join(parts)


def format_discovery(families: Iterable[ModelFamily]) -> str:
    """Describe the families for a terminal, with a next step for each."""
    ordered = sorted(families, key=lambda family: (len(family.members) < 2, family.name))
    if not ordered:
        return (
            "No models found in Ollama. Pull two downloads of the same model to compare,"
            " for example:\n"
            "  ollama pull qwen2.5:0.5b-instruct-q8_0\n"
            "  ollama pull qwen2.5:0.5b-instruct-q4_K_M\n"
            "then run quantdiff discover again."
        )
    downloads = sum(len(family.members) for family in ordered)
    header = f"Found {downloads} downloads of {len(ordered)} models in Ollama."
    return "\n\n".join([header, *(_format_family(family) for family in ordered)])


def precision_bits(quantization: str) -> int:
    """Bits per weight named by a quantization such as Q4_K_M or BF16; 0 when unknown."""
    match = _BITS.match(quantization)
    return 0 if match is None else int(match["bits"])


def format_size(size: int) -> str:
    """Decimal units, matching `ollama list`."""
    if size >= _DECIMAL_GB:
        return f"{size / _DECIMAL_GB:.1f} GB"
    return f"{round(size / _DECIMAL_MB)} MB"


# Reading Ollama ---------------------------------------------------------------------------


def _resolve(base_url: str | None) -> str:
    return validate_base_url(base_url or ollama_base_url(os.environ.get("OLLAMA_HOST")))


def _listing(url: str, timeout: float) -> list[JSONValue]:
    with explained_failures(functools.partial(explain_ollama_failure, base_url=url)):
        body = get_json(f"{url}/api/tags", timeout=timeout)
    listing = expect_dict(body, "Ollama /api/tags response")
    return expect_list(listing.get("models") or [], "Ollama /api/tags models")


def _tag_name(entry: JSONValue) -> str:
    model = expect_dict(entry, "Ollama /api/tags entry")
    return printable(expect_str(model.get("name") or model.get("model"), "Ollama model name"))


def _read_download(entry: JSONValue) -> _Download:
    model = expect_dict(entry, "Ollama /api/tags entry")
    tag = _tag_name(entry)
    details = model.get("details")
    listed = details.get("quantization_level") if isinstance(details, dict) else None
    split = _split_tag(tag)
    # The tag is more specific than the file type Ollama lists: Unsloth's UD-Q4_K_XL is
    # listed as Q4_K_M.
    named = None if split.suffix is None else split.suffix.upper()
    return _Download(
        tag=tag,
        display=split.display,
        name_tokens=split.name_tokens,
        quantization=printable(named or optional_str(listed) or UNKNOWN_QUANTIZATION),
        size=optional_int(model.get("size")) or 0,
        digest=optional_str(model.get("digest")) or tag,
        explicit_suffix=split.suffix is not None,
    )


def _show_identity(url: str, tag: str, timeout: float) -> frozenset[str] | None:
    """Tokens naming the model in a file's GGUF metadata, or None if Ollama has too little."""
    try:
        body = expect_dict(
            post_json(f"{url}/api/show", {"model": tag}, timeout=timeout), "Ollama /api/show"
        )
    except BackendError as exc:
        logger.debug("no metadata for %s: %s", tag, exc)
        return None
    info = body.get("model_info")
    if not isinstance(info, dict):
        return None
    architecture = optional_str(info.get("general.architecture"))
    basename = optional_str(info.get("general.basename"))
    size_label = optional_str(info.get("general.size_label"))
    if not (architecture and basename and size_label):
        return None
    finetune = optional_str(info.get("general.finetune")) or ""
    return frozenset(
        {f"arch={architecture.casefold()}", *_tokens(f"{basename} {size_label} {finetune}")}
    )


# Grouping ---------------------------------------------------------------------------------


def _split_tag(tag: str) -> _SplitTag:
    name, separator, version = tag.rpartition(":")
    if not separator or "/" in version:
        name, version = tag, "latest"
    match = _QUANT_SUFFIX.search(version)
    base = version if match is None else version[: match.start()]
    suffix = None if match is None else match["quant"]
    stem = "" if suffix is None else tag[: len(tag) - len(suffix)]
    return _SplitTag(
        display=f"{name}:{base}" if base else name,
        stem=stem,
        suffix=suffix,
        name_tokens=_name_tokens(name, base),
    )


def _name_part(tag: str) -> str:
    name, separator, version = tag.rpartition(":")
    return tag if not separator or "/" in version else name


def _hf_repo(name: str) -> tuple[str, str] | None:
    """(uploader, repo) for `hf.co/<uploader>/<repo>`, else None."""
    folded = name.casefold()
    for host in _HF_HOSTS:
        if folded.startswith(host):
            uploader, separator, repo = name[len(host) :].partition("/")
            if separator and uploader and repo:
                return uploader, repo
    return None


def _model_name(name: str) -> str:
    """The model's own name: the Hugging Face repo without packaging marks, or the last
    path segment of an Ollama name."""
    hf = _hf_repo(name)
    if hf is None:
        return name.rpartition("/")[2]
    repo = hf[1]
    for suffix in _REPO_SUFFIXES:
        repo = suffix.sub("", repo)
    return _ORG_PREFIX.sub("", repo)


def _name_tokens(name: str, base: str) -> tuple[str, ...]:
    tokens = _tokens(f"{_model_name(name)} {base}")
    return tuple(token for token in tokens if token not in _IGNORED_TOKENS)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _family_key(download: _Download, identity: frozenset[str] | None) -> tuple[str, ...]:
    tokens = download.name_tokens
    numbered = any(char.isdigit() for token in tokens for char in token)
    if identity is not None and numbered and identity.issuperset(tokens):
        return ("metadata", *sorted(identity))
    hub = "hf" if _hf_repo(_name_part(download.tag)) is not None else "ollama"
    return ("name", hub, *tokens)


def _group(
    downloads: Sequence[_Download], identities: dict[str, frozenset[str] | None]
) -> list[ModelFamily]:
    by_family: dict[tuple[str, ...], dict[str, _Download]] = {}
    for download in downloads:
        key = _family_key(download, identities.get(download.digest))
        # Tags that share a digest are the same file; keep the most descriptive tag.
        same_family = by_family.setdefault(key, {})
        kept = same_family.get(download.digest)
        if kept is None or _tag_preference(download) > _tag_preference(kept):
            same_family[download.digest] = download
    families = [
        ModelFamily(name=_family_name(list(kept.values())), members=_by_precision(kept.values()))
        for kept in by_family.values()
    ]
    return sorted(families, key=lambda family: (family.name, family.members))


def _tag_preference(download: _Download) -> tuple[bool, bool]:
    return download.explicit_suffix, not download.tag.endswith(":latest")


def _family_name(downloads: list[_Download]) -> str:
    sources = {source_of(item.tag) for item in downloads}
    if len(sources) > 1:
        library = [item for item in downloads if source_of(item.tag) == LIBRARY_SOURCE]
        if not library:
            return _most_common(_model_name(_name_part(item.tag)) for item in downloads)
        downloads = library
    return _most_common(item.display for item in downloads)


def _most_common(names: Iterable[str]) -> str:
    counts = Counter(names)
    return min(counts, key=lambda name: (-counts[name], name))


def _by_precision(downloads: Iterable[_Download]) -> tuple[Member, ...]:
    ranked = sorted(
        downloads,
        key=lambda item: (precision_bits(item.quantization), item.size),
        reverse=True,
    )
    return tuple(item.member for item in ranked)


def _candidate_labels(family: ModelFamily) -> dict[str, str]:
    """Labels for candidates of a family with several sources: the source, plus the
    quantization when a source has more than one candidate."""
    if len(family.sources) < 2:
        return {}
    candidates = family.members[1:]
    per_source = Counter(member.source for member in candidates)
    return {
        member.tag: member.source
        if per_source[member.source] == 1
        else f"{member.source}-{member.quantization}"
        for member in candidates
    }


# Output -----------------------------------------------------------------------------------


def _format_family(family: ModelFamily) -> str:
    quant_width = max(len(member.quantization) for member in family.members)
    several_sources = len(family.sources) > 1
    source_width = max(len(source) for source in family.sources)
    lines = [
        f"{family.name}  (from {', '.join(family.sources)})" if several_sources else family.name
    ]
    for index, member in enumerate(family.members):
        marker = "  (reference)" if index == 0 and len(family.members) > 1 else ""
        source = f"{member.source:<{source_width}}  " if several_sources else ""
        lines.append(
            f"  {source}{member.quantization:<{quant_width}}  {format_size(member.size):>7}"
            f"  {member.tag}{marker}"
        )
    command = suggest_command(family)
    if command is None:
        lines.extend(_single_download_hint(family.members[0]))
    else:
        lines.extend(["  Compare them:", f"    {command}"])
        lines.extend(_weak_reference_note(family.members[0].quantization))
    return "\n".join(lines)


def _weak_reference_note(quantization: str) -> list[str]:
    if precision_bits(quantization) >= _REFERENCE_MIN_BITS:
        return []
    return [
        f"  The most precise download here is {quantization}; pull a q8_0 or fp16 download",
        "  of this model for a reference closer to the original weights.",
    ]


def _single_download_hint(member: Member) -> list[str]:
    split = _split_tag(member.tag)
    if split.suffix is None:
        return [
            "  Only one download. Pull a q8_0 or fp16 download of this model from its tag",
            "  list to use as a reference, then run quantdiff discover again.",
        ]
    upper = split.suffix[0].isupper()
    if precision_bits(member.quantization) >= _REFERENCE_MIN_BITS:
        pull, purpose = ("Q4_K_M" if upper else "q4_K_M"), "a smaller download to compare"
    else:
        pull, purpose = ("Q8_0" if upper else "q8_0"), "a reference to compare against"
    return [f"  Only one download. To get {purpose}:", f"    ollama pull {split.stem}{pull}"]
