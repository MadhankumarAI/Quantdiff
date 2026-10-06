from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from quantdiff import suites
from quantdiff.errors import SuiteError
from quantdiff.suites import (
    BUILTIN_SUITES,
    SCHEMA_KEYWORDS,
    case_to_dict,
    load_builtin,
    load_cases_file,
    load_scoring_prompts,
    scoring_prompts_from_cases,
    suite_digest,
)
from quantdiff.types import JSONValue, Message, TaskCase, ToolSpec

DATA_DIR = Path(__file__).parent / "data"
SOLUTIONS: dict[str, str] = json.loads(
    (DATA_DIR / "code_reference_solutions.json").read_text(encoding="utf-8")
)
CODE_CASES = load_builtin("code")
MIN_SUITE_SIZES = {"json": 30, "tools": 30, "code": 30, "chat": 20}
LINE_SEPARATOR = chr(0x2028)
BANNED_PUNCTUATION = tuple(map(chr, (0x2014, 0x2013, 0x2018, 0x2019, 0x201C, 0x201D)))
EXPECTED_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "pattern",
    "anyOf",
    "oneOf",
    "allOf",
    "description",
    "title",
    "default",
    "$schema",
}


def _all_builtin_cases() -> list[TaskCase]:
    return [case for name in BUILTIN_SUITES for case in load_builtin(name)]


def _write_lines(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _schema_keywords(schema: JSONValue) -> Iterator[str]:
    """Yield every keyword used anywhere in `schema`, skipping property names and data."""
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        yield key
        if key == "properties":
            for subschema in value.values():
                yield from _schema_keywords(subschema)
        elif key in {"items", "additionalProperties"}:
            yield from _schema_keywords(value)
        elif key in {"anyOf", "oneOf", "allOf"}:
            for subschema in value:
                yield from _schema_keywords(subschema)


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _violations(value: JSONValue, schema: dict[str, JSONValue], path: str = "$") -> list[str]:
    """A deliberately small validator covering the keywords the tool suite uses."""
    expected_type = schema.get("type")
    if expected_type is not None and not _TYPE_CHECKS[expected_type](value):
        return [f"{path}: expected {expected_type}"]
    if isinstance(value, dict):
        return _object_violations(value, schema, path)
    if isinstance(value, list):
        errors = [f"{path}: too few items"] if len(value) < schema.get("minItems", 0) else []
        for index, item in enumerate(value):
            errors += _violations(item, schema.get("items", {}), f"{path}[{index}]")
        return errors
    return _scalar_violations(value, schema, path)


def _object_violations(
    value: dict[str, JSONValue], schema: dict[str, JSONValue], path: str
) -> list[str]:
    properties = schema.get("properties", {})
    errors = [f"{path}: missing {key}" for key in schema.get("required", []) if key not in value]
    if schema.get("additionalProperties") is False:
        errors += [f"{path}: unexpected {key}" for key in value if key not in properties]
    for key, item in value.items():
        if key in properties:
            errors += _violations(item, properties[key], f"{path}.{key}")
    return errors


def _scalar_violations(value: JSONValue, schema: dict[str, JSONValue], path: str) -> list[str]:
    checks: list[tuple[str, Callable[[JSONValue], bool]]] = [
        ("enum", lambda bound: value in bound),
        ("pattern", lambda bound: re.search(bound, value) is not None),
        ("minimum", lambda bound: value >= bound),
        ("maximum", lambda bound: value <= bound),
        ("exclusiveMinimum", lambda bound: value > bound),
    ]
    return [
        f"{path}: {value!r} violates {keyword}"
        for keyword, holds in checks
        if keyword in schema and not holds(schema[keyword])
    ]


def _word_count(text: str) -> int:
    """Count words, treating each CJK ideograph as a word since Chinese has no spaces."""
    return len(re.findall(r"\w+", re.sub(r"[一-鿿]", " x ", text)))


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", source],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# Built-in suites -------------------------------------------------------------------------


@pytest.mark.parametrize("name", BUILTIN_SUITES)
def test_builtin_suite_loads_with_prefixed_unique_ids(name: str) -> None:
    cases = load_builtin(name)
    assert len(cases) >= MIN_SUITE_SIZES[name]
    assert {case.kind for case in cases} == {name}
    assert all(case.id.startswith(f"{name}-") for case in cases)
    assert len({case.id for case in cases}) == len(cases)


def test_case_ids_are_unique_across_builtin_suites() -> None:
    ids = [case.id for case in _all_builtin_cases()]
    assert len(ids) == len(set(ids))


def test_builtin_cases_end_with_a_user_message() -> None:
    for case in _all_builtin_cases():
        assert case.messages[-1].role == "user", case.id
        assert 16 <= case.max_tokens <= 1024, case.id


def test_json_cases_have_strict_object_schemas_and_ask_for_json_only() -> None:
    for case in load_builtin("json"):
        assert case.json_schema is not None
        assert case.json_schema["type"] == "object", case.id
        assert case.json_schema.get("additionalProperties") is False, case.id
        assert "JSON only" in case.messages[-1].content, case.id
        assert not case.tools
        assert case.entry_point is None


def test_builtin_schemas_use_only_supported_keywords() -> None:
    schemas: list[JSONValue] = []
    for case in _all_builtin_cases():
        schemas.append(case.json_schema)
        schemas.extend(tool.parameters for tool in case.tools)
    used = {keyword for schema in schemas for keyword in _schema_keywords(schema)}
    assert used <= EXPECTED_SCHEMA_KEYWORDS


def test_schema_keyword_allowlist_matches_the_metrics_validator() -> None:
    assert SCHEMA_KEYWORDS == EXPECTED_SCHEMA_KEYWORDS


def test_tools_cases_expect_a_listed_tool_or_none() -> None:
    cases = load_builtin("tools")
    for case in cases:
        names = [tool.name for tool in case.tools]
        assert 1 <= len(names) <= 6, case.id
        assert case.expected_tool is None or case.expected_tool in names, case.id
        if case.expected_tool is None:
            assert case.expected_arguments is None, case.id
    no_call = [case for case in cases if case.expected_tool is None]
    assert len(no_call) >= 6
    assert sum(len(case.tools) >= 2 for case in cases if case.expected_tool) >= 15


def test_expected_arguments_validate_against_tool_schemas() -> None:
    for case in load_builtin("tools"):
        if case.expected_tool is None:
            continue
        assert case.expected_arguments is not None, case.id
        tool: ToolSpec = next(t for t in case.tools if t.name == case.expected_tool)
        assert _violations(case.expected_arguments, tool.parameters) == [], case.id


def test_local_validator_catches_bad_arguments() -> None:
    schema = {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["c", "f"]}, "n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    assert _violations({"n": 1, "unit": "c"}, schema) == []
    assert len(_violations({"unit": "k", "extra": 1}, schema)) == 3


def test_chat_cases_carry_no_checks() -> None:
    for case in load_builtin("chat"):
        assert case.json_schema is None
        assert not case.tools
        assert case.entry_point is None
        assert case.tests is None


def test_builtin_text_has_no_typographic_punctuation() -> None:
    texts = [json.dumps(case_to_dict(case), ensure_ascii=False) for case in _all_builtin_cases()]
    texts += [prompt.text for prompt in load_scoring_prompts()]
    for text in texts:
        assert not any(char in text for char in BANNED_PUNCTUATION), text[:80]


# Code suite ------------------------------------------------------------------------------


def test_code_cases_name_the_function_and_need_no_imports() -> None:
    for case in CODE_CASES:
        assert case.entry_point is not None
        assert case.tests is not None
        assert f"`{case.entry_point}(" in case.messages[-1].content, case.id
        assert "import" not in case.tests, case.id
        checks = case.tests.count("assert") + case.tests.count("raise AssertionError")
        assert checks >= 3, case.id


def test_every_code_case_has_exactly_one_reference_solution() -> None:
    assert set(SOLUTIONS) == {case.id for case in CODE_CASES}


@pytest.mark.parametrize("case", CODE_CASES, ids=lambda case: case.id)
def test_reference_solution_passes_case_asserts(case: TaskCase) -> None:
    result = _run_python(f"{SOLUTIONS[case.id]}\n{case.tests}")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("case", CODE_CASES, ids=lambda case: case.id)
def test_stub_solution_fails_case_asserts(case: TaskCase) -> None:
    stub = f"def {case.entry_point}(*args, **kwargs):\n    return None\n"
    assert _run_python(f"{stub}\n{case.tests}").returncode != 0


PLAUSIBLE_WRONG_SOLUTIONS = {
    "is_palindrome": "def is_palindrome(text):\n    return text == text[::-1]\n",
    "roman_to_int": (
        "def roman_to_int(numeral):\n"
        "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
        "    return sum(values[c] for c in numeral)\n"
    ),
    "merge_intervals": (
        "def merge_intervals(intervals):\n"
        "    merged = []\n"
        "    for start, end in intervals:\n"
        "        if merged and start < merged[-1][1]:\n"
        "            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n"
        "        else:\n"
        "            merged.append((start, end))\n"
        "    return merged\n"
    ),
    "camel_to_snake": (
        "def camel_to_snake(name):\n"
        "    return ''.join('_' + c.lower() if c.isupper() else c for c in name).lstrip('_')\n"
    ),
    "compare_versions": ("def compare_versions(a, b):\n    return (a > b) - (a < b)\n"),
    "format_bytes": (
        "def format_bytes(size):\n"
        "    if size < 1000:\n"
        "        return f'{size} B'\n"
        "    return f'{size / 1000:.1f} KiB'\n"
    ),
}


@pytest.mark.parametrize("entry_point", sorted(PLAUSIBLE_WRONG_SOLUTIONS))
def test_plausible_wrong_solution_fails_case_asserts(entry_point: str) -> None:
    case = next(case for case in CODE_CASES if case.entry_point == entry_point)
    wrong = PLAUSIBLE_WRONG_SOLUTIONS[entry_point]
    assert _run_python(f"{wrong}\n{case.tests}").returncode != 0


# Scoring prompts -------------------------------------------------------------------------


def test_builtin_scoring_prompts_are_diverse_and_sized() -> None:
    prompts = load_scoring_prompts()
    assert len(prompts) >= 40
    assert len({prompt.id for prompt in prompts}) == len(prompts)
    assert all(prompt.id.startswith("score-") for prompt in prompts)
    for prompt in prompts:
        assert 30 <= _word_count(prompt.text) <= 200, prompt.id
    joined = "".join(prompt.text for prompt in prompts)
    assert re.search(r"[一-鿿]", joined), "Chinese prompt missing"
    assert re.search(r"[ऀ-ॿ]", joined), "Hindi prompt missing"
    assert "def " in joined
    assert "fn " in joined or "func " in joined


def test_scoring_prompts_file_defaults_ids(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "s.jsonl", '{"text": "Once upon a time"}', '{"id": "b", "text": "x y z"}'
    )
    prompts = load_scoring_prompts(path)
    assert [prompt.id for prompt in prompts] == ["score-001", "b"]


def test_scoring_prompts_file_accepts_prompts_file_lines(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "s.jsonl",
        '{"prompt": "Write a haiku", "system": "Be terse"}',
        '{"id": "m", "messages": [{"role": "user", "content": "a"}, '
        '{"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]}',
    )
    prompts = load_scoring_prompts(path)
    assert [(prompt.id, prompt.text) for prompt in prompts] == [
        ("prompt-001", "Write a haiku"),
        ("m", "c"),
    ]


def test_scoring_prompts_from_cases_use_the_last_user_message() -> None:
    case = TaskCase(
        id="c1",
        kind="chat",
        messages=(Message("system", "s"), Message("user", "first"), Message("user", "last")),
    )
    assert [(p.id, p.text) for p in scoring_prompts_from_cases([case])] == [("c1", "last")]
    no_user = TaskCase(id="c2", kind="chat", messages=(Message("system", "s"),))
    with pytest.raises(SuiteError, match="no user message"):
        scoring_prompts_from_cases([no_user])


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "a"}', 'expected a "text" or "prompt" field'),
        ('{"text": ""}', "non-empty string"),
        ('{"text": "x", "prompt": "y"}', "unknown field 'prompt'"),
    ],
)
def test_scoring_prompts_file_rejects_bad_lines(tmp_path: Path, line: str, message: str) -> None:
    path = _write_lines(tmp_path / "s.jsonl", line)
    with pytest.raises(SuiteError, match=f"line 1: .*{re.escape(message)}"):
        load_scoring_prompts(path)


# User prompt files -----------------------------------------------------------------------


def test_load_builtin_rejects_unknown_name() -> None:
    with pytest.raises(SuiteError, match="unknown suite 'math'"):
        load_builtin("math")


def test_shorthand_prompt_becomes_chat_case(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "p.jsonl",
        '{"prompt": "Name a prime."}',
        "",
        '{"prompt": "Hi", "id": "greet", "system": "Be brief.", "max_tokens": 32}',
        '{"messages": [{"role": "user", "content": "Hello"}]}',
    )
    first, second, third = load_cases_file(path)
    assert first.id == "prompt-001"
    assert first.kind == "chat"
    assert [m.role for m in first.messages] == ["user"]
    assert second.id == "greet"
    assert second.max_tokens == 32
    assert [(m.role, m.content) for m in second.messages] == [
        ("system", "Be brief."),
        ("user", "Hi"),
    ]
    assert third.id == "prompt-004"
    assert third.kind == "chat"


def test_full_cases_round_trip_through_case_to_dict(tmp_path: Path) -> None:
    cases = _all_builtin_cases()
    path = tmp_path / "all.jsonl"
    path.write_text(
        "".join(json.dumps(case_to_dict(case)) + "\n" for case in cases), encoding="utf-8"
    )
    assert load_cases_file(path) == tuple(cases)


def test_load_cases_file_accepts_bom_crlf_and_raw_line_separators(tmp_path: Path) -> None:
    line = '{"prompt": "a' + LINE_SEPARATOR + 'b"}'
    path = tmp_path / "p.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + line.encode("utf-8") + b"\r\n")
    (case,) = load_cases_file(path)
    assert case.messages[0].content == f"a{LINE_SEPARATOR}b"


VALID_TOOL = {
    "name": "get_weather",
    "description": "Weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _tools_line(**overrides: JSONValue) -> str:
    record: dict[str, JSONValue] = {
        "id": "t1",
        "kind": "tools",
        "messages": [{"role": "user", "content": "Weather in Oslo?"}],
        "tools": [VALID_TOOL],
        "expected_tool": "get_weather",
    }
    record.update(overrides)
    return json.dumps(record)


def _json_line(schema: JSONValue) -> str:
    return json.dumps(
        {
            "id": "j1",
            "kind": "json",
            "messages": [{"role": "user", "content": "Give JSON."}],
            "json_schema": schema,
        }
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"prompt": "a", "prompt": "b"}', "duplicate key 'prompt'"),
        ('{"prompt": "a", "max_tokens": NaN}', "NaN is not valid JSON"),
        ('{"text": "a"}', 'expected a "kind", "prompt" or "messages" field'),
        ('{"prompt": "a", "temperature": 0}', "unknown field 'temperature'"),
        ('{"prompt": "   "}', '"prompt" must be a non-empty string'),
        ('{"prompt": "a", "max_tokens": true}', '"max_tokens" must be an integer'),
        ('{"prompt": "a", "max_tokens": 0}', '"max_tokens" must be an integer'),
        ('{"prompt": "a", "id": "has space"}', '"id" must be'),
        ('{"kind": "poem", "id": "x", "messages": []}', '"kind" must be one of'),
        ('{"kind": "chat", "id": "x"}', "missing field 'messages'"),
        ('{"messages": []}', '"messages" must be a non-empty array'),
        ('{"messages": [{"role": "robot", "content": "a"}]}', "role must be one of"),
        ('{"messages": [{"role": "user"}]}', "messages[0]: missing field 'content'"),
        (
            '{"messages": [{"role": "user", "content": "a"},'
            ' {"role": "assistant", "content": "b"}]}',
            'last message must have role "user"',
        ),
        (
            '{"kind": "chat", "id": "x", "messages": [{"role": "user", "content": "a"}],'
            ' "tests": "assert True"}',
            "unknown field 'tests'",
        ),
        (
            '{"kind": "code", "id": "x", "messages": [{"role": "user", "content": "a"}],'
            ' "entry_point": "class", "tests": "assert True"}',
            '"entry_point" must be a valid Python function name',
        ),
        (
            '{"kind": "code", "id": "x", "messages": [{"role": "user", "content": "a"}],'
            ' "entry_point": "f"}',
            "missing field 'tests'",
        ),
        (_tools_line(expected_tool="get_time"), '"expected_tool" must be null or one of'),
        (
            _tools_line(expected_tool=None, expected_arguments={"city": "Oslo"}),
            "requires a non-null",
        ),
        (_tools_line(expected_arguments={"town": "Oslo"}), "key 'town' is not a parameter"),
        (_tools_line(expected_arguments=["Oslo"]), "must be an object or null"),
        (_tools_line(tools=[]), '"tools" must be a non-empty array'),
        (_tools_line(tools=[VALID_TOOL, VALID_TOOL]), "duplicate tool name 'get_weather'"),
        (_tools_line(tools=[{**VALID_TOOL, "name": "get weather"}]), "tools[0].name"),
        (_tools_line(tools=[{**VALID_TOOL, "strict": True}]), "unknown field 'strict'"),
        (
            _tools_line(tools=[{**VALID_TOOL, "parameters": {"type": "string"}}]),
            'must have "type": "object"',
        ),
        (_tools_line(), None),
        (_json_line({"type": "object", "if": {}}), "unsupported JSON Schema keyword 'if'"),
        (
            _json_line({"type": "object", "properties": {"a": {"format": "date"}}}),
            "json_schema.properties.a uses unsupported JSON Schema keyword 'format'",
        ),
        (_json_line({"anyOf": [{"$ref": "#/x"}]}), "json_schema.anyOf[0] uses unsupported"),
        (_json_line({"type": "decimal"}), "json_schema.type must be one of"),
        (_json_line({"type": "string", "pattern": "("}), "not a valid regular expression"),
        (
            _json_line({"type": "string", "pattern": r"^(\w+\s?)*$"}),
            "json_schema.pattern repeats",
        ),
        (_json_line({"type": "object", "properties": {"a": True, "b": False}}), None),
        (_json_line({"type": "array", "items": False}), None),
        (
            _json_line({"type": "object", "properties": {"a": 3}}),
            "must be a JSON object or boolean",
        ),
        (_json_line({"type": "array", "minItems": -1}), "must be a non-negative integer"),
        (_json_line({"type": "number", "maximum": "9"}), "json_schema.maximum must be a number"),
        (_json_line({"type": "object", "required": [1]}), "must be an array of strings"),
        (_json_line({"enum": []}), "must be a non-empty array"),
        (
            _json_line({"additionalProperties": {"not": {}}}),
            "unsupported JSON Schema keyword 'not'",
        ),
        (_json_line([]), "json_schema must be a JSON object"),
    ],
)
def test_load_cases_file_reports_line_numbers(
    tmp_path: Path, line: str, message: str | None
) -> None:
    path = _write_lines(tmp_path / "p.jsonl", '{"prompt": "first line is fine"}', line)
    if message is None:
        assert len(load_cases_file(path)) == 2
        return
    with pytest.raises(SuiteError, match=f"p.jsonl, line 2: .*{re.escape(message)}"):
        load_cases_file(path)


def test_duplicate_ids_name_both_lines(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "p.jsonl", '{"prompt": "a", "id": "x"}', "", '{"prompt": "b", "id": "x"}'
    )
    with pytest.raises(SuiteError, match=r"line 3: duplicate id 'x' \(first used on line 1\)"):
        load_cases_file(path)


def test_load_cases_file_rejects_missing_empty_and_non_utf8_files(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not found"):
        load_cases_file(tmp_path / "missing.jsonl")
    with pytest.raises(SuiteError, match="is not a file"):
        load_cases_file(tmp_path)
    with pytest.raises(SuiteError, match="contains no entries"):
        load_cases_file(_write_lines(tmp_path / "blank.jsonl", "", "   "))
    latin1 = tmp_path / "latin1.jsonl"
    latin1.write_bytes('{"prompt": "café"}'.encode("latin-1"))
    with pytest.raises(SuiteError, match="not valid UTF-8"):
        load_cases_file(latin1)


def test_load_cases_file_enforces_size_and_count_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_lines(tmp_path / "p.jsonl", *(f'{{"prompt": "q{n}"}}' for n in range(5)))
    monkeypatch.setattr(suites, "MAX_CASES", 4)
    with pytest.raises(SuiteError, match="more than 4 entries"):
        load_cases_file(path)
    monkeypatch.setattr(suites, "MAX_FILE_BYTES", 10)
    with pytest.raises(SuiteError, match="larger than"):
        load_cases_file(path)


# Digest ----------------------------------------------------------------------------------


def test_suite_digest_is_stable_and_sensitive_to_content() -> None:
    cases = load_builtin("chat")
    scoring = load_scoring_prompts()
    digest = suite_digest(cases, scoring)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert suite_digest(load_builtin("chat"), load_scoring_prompts()) == digest
    assert suite_digest(cases[1:], scoring) != digest
    assert suite_digest(cases, scoring[1:]) != digest
    assert suite_digest(cases, ()) != digest


def test_suite_digest_ignores_schema_key_order() -> None:
    def json_case(schema: dict[str, JSONValue]) -> TaskCase:
        return TaskCase(id="j", kind="json", messages=(), json_schema=schema)

    first = json_case({"type": "object", "required": []})
    second = json_case({"required": [], "type": "object"})
    assert suite_digest([first], []) == suite_digest([second], [])
