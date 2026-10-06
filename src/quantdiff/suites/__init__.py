"""Built-in prompt suites and strict loaders for user prompt files.

Suites are JSON Lines files: one task case (or scoring prompt) per line. Built-in
suites ship inside the package and are read with importlib.resources, so they load the
same way from a source checkout and from an installed wheel.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import logging
import os
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Final, Protocol, TypeVar

from quantdiff.errors import SuiteError
from quantdiff.metrics.jsonschema import SUPPORTED_KEYWORDS, check_schema
from quantdiff.types import JSONValue, Message, Role, ScoringPrompt, TaskCase, TaskKind, ToolSpec

__all__ = [
    "BUILTIN_SUITES",
    "MAX_CASES",
    "MAX_FILE_BYTES",
    "SCHEMA_KEYWORDS",
    "case_to_dict",
    "load_builtin",
    "load_cases_file",
    "load_scoring_prompts",
    "scoring_prompts_from_cases",
    "suite_digest",
]

logger = logging.getLogger(__name__)

BUILTIN_SUITES: Final[tuple[str, ...]] = ("json", "tools", "code", "chat")
MAX_FILE_BYTES: Final = 20 * 1024 * 1024
MAX_CASES: Final = 10_000

_DATA_PACKAGE: Final = "quantdiff.suites"
_DEFAULT_MAX_TOKENS: Final = 512
_MAX_MAX_TOKENS: Final = 32_768
_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TOOL_NAME_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_ROLES: Final[Mapping[str, Role]] = {"system": "system", "user": "user", "assistant": "assistant"}
_KINDS: Final[Mapping[str, TaskKind]] = {
    "json": "json",
    "tools": "tools",
    "code": "code",
    "chat": "chat",
}
_COMMON_FIELDS: Final = frozenset({"id", "kind", "messages", "max_tokens"})
_KIND_REQUIRED: Final[Mapping[TaskKind, frozenset[str]]] = {
    "json": frozenset({"json_schema"}),
    "tools": frozenset({"tools", "expected_tool"}),
    "code": frozenset({"entry_point", "tests"}),
    "chat": frozenset(),
}
_KIND_OPTIONAL: Final[Mapping[TaskKind, frozenset[str]]] = {
    "json": frozenset(),
    "tools": frozenset({"expected_arguments"}),
    "code": frozenset(),
    "chat": frozenset(),
}
_PROMPT_FIELDS: Final = frozenset({"prompt", "id", "system", "max_tokens"})
_MESSAGES_FIELDS: Final = frozenset({"messages", "id", "max_tokens"})
_SCORING_FIELDS: Final = frozenset({"id", "text"})


class _Identified(Protocol):
    @property
    def id(self) -> str: ...


_T = TypeVar("_T", bound=_Identified)
Record = dict[str, JSONValue]


# Public API ------------------------------------------------------------------------------


def load_builtin(name: str) -> tuple[TaskCase, ...]:
    """Load one of the suites listed in BUILTIN_SUITES."""
    if name not in BUILTIN_SUITES:
        raise SuiteError(f"unknown suite {name!r}; choose from: {', '.join(BUILTIN_SUITES)}")
    return _parse_records(_builtin_text(f"{name}.jsonl"), f"built-in suite {name!r}", _build_case)


def load_cases_file(path: str | os.PathLike[str]) -> tuple[TaskCase, ...]:
    """Load and validate a user prompts file.

    Besides full task cases, each line may be the shorthand `{"prompt": "..."}` (with
    optional "id", "system" and "max_tokens") or `{"messages": [...]}`; both become chat
    cases.
    """
    file_path = Path(path)
    return _parse_records(_read_text(file_path), str(file_path), _build_case)


def load_scoring_prompts(path: str | os.PathLike[str] | None = None) -> tuple[ScoringPrompt, ...]:
    """Load raw-completion prompts for logit metrics; None selects the built-in set.

    Each line is `{"text": "..."}` with an optional "id", or any line a prompts file
    accepts, in which case the last user message is scored as raw text. So one file can
    serve as both --prompts and --scoring-prompts.
    """
    if path is None:
        return _parse_records(
            _builtin_text("scoring.jsonl"), "built-in scoring prompts", _build_scoring_prompt
        )
    file_path = Path(path)
    return _parse_records(_read_text(file_path), str(file_path), _build_scoring_prompt)


def scoring_prompts_from_cases(cases: Sequence[TaskCase]) -> tuple[ScoringPrompt, ...]:
    """Score each case's last user message as raw text, keeping the case id."""
    return tuple(_scoring_prompt_from_case(case) for case in cases)


def case_to_dict(case: TaskCase) -> dict[str, JSONValue]:
    """Return the canonical file form of `case`; load_cases_file accepts it unchanged."""
    out: dict[str, JSONValue] = {
        "id": case.id,
        "kind": case.kind,
        "messages": [{"role": m.role, "content": m.content} for m in case.messages],
        "max_tokens": case.max_tokens,
    }
    if case.json_schema is not None:
        out["json_schema"] = case.json_schema
    if case.tools:
        out["tools"] = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in case.tools
        ]
    if case.kind == "tools":
        out["expected_tool"] = case.expected_tool
    if case.expected_arguments is not None:
        out["expected_arguments"] = case.expected_arguments
    if case.entry_point is not None:
        out["entry_point"] = case.entry_point
    if case.tests is not None:
        out["tests"] = case.tests
    return out


def suite_digest(cases: Sequence[TaskCase], scoring: Sequence[ScoringPrompt]) -> str:
    """Return a sha256 hex digest that changes whenever any case or prompt changes."""
    payload = {
        "cases": [case_to_dict(case) for case in cases],
        "scoring": [{"id": prompt.id, "text": prompt.text} for prompt in scoring],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Reading ---------------------------------------------------------------------------------


def _builtin_text(filename: str) -> str:
    return resources.files(_DATA_PACKAGE).joinpath("data").joinpath(filename).read_text("utf-8")


def _read_text(path: Path) -> str:
    try:
        if not path.exists():
            raise SuiteError(f"prompts file not found: {path}")
        if not path.is_file():
            raise SuiteError(f"{path} is not a file")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SuiteError(f"{path} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        data = path.read_bytes()
    except OSError as exc:
        raise SuiteError(f"cannot read {path}: {exc.strerror or exc}") from None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SuiteError(f"{path} is not valid UTF-8 (byte offset {exc.start})") from None


def _parse_records(text: str, source: str, build: Callable[[Record, int], _T]) -> tuple[_T, ...]:
    items: list[_T] = []
    first_seen: dict[str, int] = {}
    # str.splitlines would also split on U+2028 and friends, which JSON allows raw in strings.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if len(items) == MAX_CASES:
            raise SuiteError(f"{source} has more than {MAX_CASES} entries")
        try:
            item = build(_decode_object(line), number)
        except SuiteError as exc:
            raise SuiteError(f"{source}, line {number}: {exc}") from None
        if item.id in first_seen:
            raise SuiteError(
                f"{source}, line {number}: duplicate id {item.id!r} "
                f"(first used on line {first_seen[item.id]})"
            )
        first_seen[item.id] = number
        items.append(item)
    if not items:
        raise SuiteError(f"{source} contains no entries")
    logger.debug("loaded %d entries from %s", len(items), source)
    return tuple(items)


def _decode_object(line: str) -> Record:
    try:
        value = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise SuiteError(f"invalid JSON: {exc.msg} (column {exc.colno})") from None
    if not isinstance(value, dict):
        raise SuiteError("each line must be a JSON object")
    return value


def _unique_keys(pairs: list[tuple[str, JSONValue]]) -> Record:
    record: Record = {}
    for key, value in pairs:
        if key in record:
            raise SuiteError(f"duplicate key {key!r}")
        record[key] = value
    return record


def _reject_constant(name: str) -> JSONValue:
    raise SuiteError(f"{name} is not valid JSON")


# Task cases ------------------------------------------------------------------------------


def _build_case(record: Record, line: int) -> TaskCase:
    if "kind" in record:
        return _full_case(record)
    if "prompt" in record:
        return _prompt_case(record, line)
    if "messages" in record:
        return _messages_case(record, line)
    raise SuiteError('expected a "kind", "prompt" or "messages" field')


def _full_case(record: Record) -> TaskCase:
    kind = _kind(record["kind"])
    required = frozenset({"id", "kind", "messages"}) | _KIND_REQUIRED[kind]
    _check_fields(
        record, allowed=_COMMON_FIELDS | required | _KIND_OPTIONAL[kind], required=required
    )
    base = TaskCase(
        id=_case_id(record["id"]),
        kind=kind,
        messages=_messages(record["messages"]),
        max_tokens=_max_tokens(record.get("max_tokens", _DEFAULT_MAX_TOKENS)),
    )
    if kind == "json":
        return _with_json_schema(base, record)
    if kind == "tools":
        return _with_tools(base, record)
    if kind == "code":
        return _with_code(base, record)
    return base


def _prompt_case(record: Record, line: int) -> TaskCase:
    _check_fields(record, allowed=_PROMPT_FIELDS, required={"prompt"})
    messages = [Message(role="user", content=_text(record["prompt"], "prompt"))]
    if "system" in record:
        messages.insert(0, Message(role="system", content=_text(record["system"], "system")))
    return TaskCase(
        id=_case_id(record.get("id", f"prompt-{line:03d}")),
        kind="chat",
        messages=tuple(messages),
        max_tokens=_max_tokens(record.get("max_tokens", _DEFAULT_MAX_TOKENS)),
    )


def _messages_case(record: Record, line: int) -> TaskCase:
    _check_fields(record, allowed=_MESSAGES_FIELDS, required={"messages"})
    return TaskCase(
        id=_case_id(record.get("id", f"prompt-{line:03d}")),
        kind="chat",
        messages=_messages(record["messages"]),
        max_tokens=_max_tokens(record.get("max_tokens", _DEFAULT_MAX_TOKENS)),
    )


def _with_json_schema(base: TaskCase, record: Record) -> TaskCase:
    schema = _object_schema(record["json_schema"], "json_schema")
    return replace(base, json_schema=schema)


def _with_tools(base: TaskCase, record: Record) -> TaskCase:
    tools = _tools(record["tools"])
    expected_tool = record["expected_tool"]
    expected_arguments = record.get("expected_arguments")
    if expected_tool is None:
        if expected_arguments is not None:
            raise SuiteError('"expected_arguments" requires a non-null "expected_tool"')
        return replace(base, tools=tools)
    by_name = {tool.name: tool for tool in tools}
    if not isinstance(expected_tool, str) or expected_tool not in by_name:
        raise SuiteError(f'"expected_tool" must be null or one of: {", ".join(by_name)}')
    if expected_arguments is not None:
        _check_expected_arguments(expected_arguments, by_name[expected_tool])
    return replace(
        base, tools=tools, expected_tool=expected_tool, expected_arguments=expected_arguments
    )


def _with_code(base: TaskCase, record: Record) -> TaskCase:
    entry_point = record["entry_point"]
    if (
        not isinstance(entry_point, str)
        or not entry_point.isidentifier()
        or keyword.iskeyword(entry_point)
    ):
        raise SuiteError('"entry_point" must be a valid Python function name')
    return replace(base, entry_point=entry_point, tests=_text(record["tests"], "tests"))


def _kind(value: JSONValue) -> TaskKind:
    kind = _KINDS.get(value) if isinstance(value, str) else None
    if kind is None:
        raise SuiteError(f'"kind" must be one of: {", ".join(_KINDS)}')
    return kind


def _case_id(value: JSONValue) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise SuiteError(
            '"id" must be 1 to 128 characters of letters, digits, ".", "_" or "-", '
            "starting with a letter or digit"
        )
    return value


def _max_tokens(value: JSONValue) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= _MAX_MAX_TOKENS:
        return value
    raise SuiteError(f'"max_tokens" must be an integer from 1 to {_MAX_MAX_TOKENS}')


def _text(value: JSONValue, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteError(f'"{field}" must be a non-empty string')
    return value


def _messages(value: JSONValue) -> tuple[Message, ...]:
    if not isinstance(value, list) or not value:
        raise SuiteError('"messages" must be a non-empty array')
    messages = tuple(_message(item, index) for index, item in enumerate(value))
    if messages[-1].role != "user":
        raise SuiteError('the last message must have role "user"')
    return messages


def _message(item: JSONValue, index: int) -> Message:
    where = f"messages[{index}]"
    if not isinstance(item, dict):
        raise SuiteError(f"{where} must be an object")
    _check_fields(item, allowed={"role", "content"}, required={"role", "content"}, where=where)
    role = _ROLES.get(item["role"]) if isinstance(item["role"], str) else None
    if role is None:
        raise SuiteError(f"{where}.role must be one of: {', '.join(_ROLES)}")
    return Message(role=role, content=_text(item["content"], f"{where}.content"))


def _check_fields(
    record: Mapping[str, JSONValue],
    *,
    allowed: Collection[str],
    required: Collection[str],
    where: str = "",
) -> None:
    prefix = f"{where}: " if where else ""
    unknown = sorted(set(record) - set(allowed))
    if unknown:
        raise SuiteError(f"{prefix}unknown field {unknown[0]!r}")
    missing = sorted(set(required) - set(record))
    if missing:
        raise SuiteError(f"{prefix}missing field {missing[0]!r}")


# Tools -----------------------------------------------------------------------------------


def _tools(value: JSONValue) -> tuple[ToolSpec, ...]:
    if not isinstance(value, list) or not value:
        raise SuiteError('"tools" must be a non-empty array')
    tools = tuple(_tool(item, index) for index, item in enumerate(value))
    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SuiteError(f"duplicate tool name {duplicates[0]!r}")
    return tools


def _tool(item: JSONValue, index: int) -> ToolSpec:
    where = f"tools[{index}]"
    if not isinstance(item, dict):
        raise SuiteError(f"{where} must be an object")
    fields = {"name", "description", "parameters"}
    _check_fields(item, allowed=fields, required=fields, where=where)
    name = item["name"]
    if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
        raise SuiteError(f"{where}.name must be a function name such as get_weather")
    if not isinstance(item["description"], str):
        raise SuiteError(f"{where}.description must be a string")
    parameters = _object_schema(item["parameters"], f"{where}.parameters")
    if parameters.get("type") != "object":
        raise SuiteError(f'{where}.parameters must have "type": "object"')
    return ToolSpec(name=name, description=item["description"], parameters=parameters)


def _check_expected_arguments(value: JSONValue, tool: ToolSpec) -> None:
    if not isinstance(value, dict):
        raise SuiteError('"expected_arguments" must be an object or null')
    properties = tool.parameters.get("properties", {})
    unknown = sorted(set(value) - set(properties))
    if unknown:
        raise SuiteError(f"expected_arguments key {unknown[0]!r} is not a parameter of {tool.name}")


# JSON Schema subset ----------------------------------------------------------------------


SCHEMA_KEYWORDS: Final = SUPPORTED_KEYWORDS
"""JSON Schema keywords the task metrics can check. Anything else is rejected up front."""


def _object_schema(value: JSONValue, where: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise SuiteError(f"{where} must be a JSON object")
    check_schema(value, where)
    return value


# Scoring prompts -------------------------------------------------------------------------


def _build_scoring_prompt(record: Record, line: int) -> ScoringPrompt:
    if "text" in record:
        _check_fields(record, allowed=_SCORING_FIELDS, required={"text"})
        return ScoringPrompt(
            id=_case_id(record.get("id", f"score-{line:03d}")),
            text=_text(record["text"], "text"),
        )
    if not record.keys() & {"kind", "prompt", "messages"}:
        raise SuiteError('expected a "text" or "prompt" field')
    return _scoring_prompt_from_case(_build_case(record, line))


def _scoring_prompt_from_case(case: TaskCase) -> ScoringPrompt:
    user_texts = [message.content for message in case.messages if message.role == "user"]
    if not user_texts:
        raise SuiteError(f"case {case.id!r} has no user message to score")
    return ScoringPrompt(id=case.id, text=user_texts[-1])
