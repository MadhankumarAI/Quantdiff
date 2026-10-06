from __future__ import annotations

import re
import time
from typing import Any

import pytest

from quantdiff.errors import SuiteError
from quantdiff.metrics.jsonschema import (
    MAX_PATTERN_INPUT,
    SUPPORTED_KEYWORDS,
    check_schema,
    extract_json,
    unsupported_keywords,
    validate,
)

PERSON: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 20},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": "^[^@]+@[^@]+$"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "role": {"enum": ["admin", "user"]},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_supported_keywords_cover_the_documented_subset() -> None:
    assert {"type", "properties", "anyOf", "$schema", "pattern"} <= SUPPORTED_KEYWORDS
    assert "$ref" not in SUPPORTED_KEYWORDS


def test_valid_instance_has_no_errors() -> None:
    instance = {"name": "Ada", "age": 36, "email": "ada@example.com", "tags": ["x"], "role": "user"}
    assert validate(instance, PERSON) == []


def test_errors_carry_json_pointer_paths() -> None:
    instance = {"name": "", "age": -1, "email": "nope", "tags": [1], "role": "root", "x": 1}
    errors = validate(instance, PERSON)
    assert "/name: must be at least 1 characters" in errors
    assert "/age: must be >= 0" in errors
    assert "/email: does not match pattern '^[^@]+@[^@]+$'" in errors
    assert "/tags/0: expected string, got integer" in errors
    assert '/role: must be one of ["admin", "user"]' in errors
    assert "/: unexpected property 'x'" in errors
    assert len(errors) == 6


def test_missing_required_property() -> None:
    assert validate({"name": "Ada"}, PERSON) == ["/: missing required property 'age'"]


@pytest.mark.parametrize(
    ("instance", "type_name", "valid"),
    [
        (3, "integer", True),
        (3.0, "integer", True),
        (3.5, "integer", False),
        (True, "integer", False),
        (True, "number", False),
        (10**400, "integer", True),
        (2.5, "number", True),
        (False, "boolean", True),
        (None, "null", True),
        ("3", "integer", False),
        ([], "array", True),
        ({}, "object", True),
        ({}, "array", False),
    ],
)
def test_type_semantics_follow_json(instance: object, type_name: str, valid: bool) -> None:
    assert (validate(instance, {"type": type_name}) == []) is valid


def test_type_may_be_a_list() -> None:
    schema = {"type": ["string", "null"]}
    assert validate(None, schema) == []
    assert validate("x", schema) == []
    assert validate(1, schema) == ["/: expected string or null, got integer"]


def test_enum_and_const_do_not_confuse_booleans_with_numbers() -> None:
    assert validate(True, {"enum": [1]}) != []
    assert validate(1.0, {"enum": [1]}) == []
    assert validate(1, {"const": True}) != []
    assert validate({"a": [1, 2]}, {"const": {"a": [1.0, 2]}}) == []
    assert validate({"a": [1]}, {"const": {"a": [1], "b": 2}}) != []


def test_exclusive_bounds() -> None:
    schema = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
    assert validate(0.5, schema) == []
    assert validate(0, schema) == ["/: must be > 0"]
    assert validate(1, schema) == ["/: must be < 1"]


def test_additional_properties_schema_applies_to_extra_keys() -> None:
    schema = {"type": "object", "properties": {"a": {}}, "additionalProperties": {"type": "number"}}
    assert validate({"a": "x", "b": 1}, schema) == []
    assert validate({"b": "x"}, schema) == ["/b: expected number, got string"]


def test_combinators() -> None:
    any_of = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert validate(1, any_of) == []
    assert validate(1.5, any_of) == ["/: does not match any schema in anyOf"]

    one_of = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert validate(1.5, one_of) == []
    assert validate(1, one_of) == ["/: matches 2 schemas in oneOf, expected exactly 1"]

    all_of = {"allOf": [{"type": "string"}, {"maxLength": 2}]}
    assert validate("ab", all_of) == []
    assert validate("abc", all_of) == ["/: must be at most 2 characters"]


def test_boolean_schemas() -> None:
    assert validate(1, True) == []
    assert validate(1, False) == ["/: no value is allowed here"]


def test_pattern_uses_search_semantics() -> None:
    assert validate("xx123yy", {"pattern": "[0-9]+"}) == []


def test_pattern_skips_oversized_strings() -> None:
    errors = validate("a" * (MAX_PATTERN_INPUT + 1), {"pattern": "a"})
    assert errors == [f"/: is longer than {MAX_PATTERN_INPUT} characters"]


def test_dollar_does_not_match_before_a_trailing_newline() -> None:
    schema = {"pattern": "^abc$"}
    assert validate("abc", schema) == []
    assert validate("abc\n", schema) == ["/: does not match pattern '^abc$'"]
    assert validate("a$b", {"pattern": r"^a\$b$"}) == []
    assert validate("$", {"pattern": "^[$]$"}) == []


def test_character_classes_are_ascii() -> None:
    arabic_indic_three = chr(0x663)
    assert validate("3", {"pattern": r"^\d$"}) == []
    assert validate(arabic_indic_three, {"pattern": r"^\d$"}) != []
    assert validate(chr(0xE9), {"pattern": r"^\w$"}) != []


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(\w+\s?)*$",
        "(a+)+",
        "(a|aa)*",
        "(?:x*y)+",
        "((ab)*c)*",
        "(a?){2,}",
        "(a+){2}",
        "(?P<word>[a-z]+)*",
        "x(a{1,3})*",
    ],
)
def test_nested_quantifiers_are_refused(pattern: str) -> None:
    with pytest.raises(SuiteError, match="exponential time"):
        check_schema({"type": "string", "pattern": pattern})
    with pytest.raises(SuiteError, match="exponential time"):
        validate("x", {"pattern": pattern})


@pytest.mark.parametrize(
    "pattern",
    [
        "^(FRA|OSL)$",
        r"^([01]\d|2[0-3]):[0-5]\d$",
        r"^(\d{3}-)*\d{4}$",
        "^(ab)+$",
        "^(a+)?b$",
        "^[(a+)]*$",
        r"^\(a+\)+$",
        "(?i)^[a-z]+$",
        r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$",
        "^a{,}b{2,5}$",
    ],
)
def test_unambiguous_patterns_are_accepted(pattern: str) -> None:
    check_schema({"type": "string", "pattern": pattern})


def test_refused_pattern_fails_fast_instead_of_backtracking() -> None:
    started = time.monotonic()
    with pytest.raises(SuiteError):
        validate("word " * 6 + "!" * 30, {"pattern": r"^(\w+\s?)*$"})
    assert time.monotonic() - started < 1.0


def test_check_schema_accepts_boolean_subschemas() -> None:
    check_schema(True)
    check_schema(
        {
            "type": "object",
            "properties": {"a": False, "b": True},
            "additionalProperties": {"type": "string"},
            "items": True,
            "anyOf": [True, {"type": "object"}],
        }
    )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"format": "date"}, "schema uses unsupported JSON Schema keyword 'format'"),
        ({"properties": {"a": {"$ref": "#"}}}, "schema.properties.a uses unsupported"),
        ({"items": 3}, "schema.items must be a JSON object or boolean"),
        ({"type": "decimal"}, "schema.type must be one of"),
        ({"pattern": "("}, "schema.pattern is not a valid regular expression"),
        ({"pattern": 3}, "schema.pattern must be a string"),
        ({"anyOf": []}, "schema.anyOf must be a non-empty array of schemas"),
        ({"minLength": True}, "schema.minLength must be a non-negative integer"),
        ({"maximum": "9"}, "schema.maximum must be a number"),
        ([], "schema must be a JSON object or boolean"),
    ],
)
def test_check_schema_rejects_malformed_schemas(schema: Any, message: str) -> None:
    with pytest.raises(SuiteError, match=re.escape(message)):
        check_schema(schema)


def test_paths_escape_pointer_characters() -> None:
    schema = {"properties": {"a/b~c": {"type": "string"}}}
    assert validate({"a/b~c": 1}, schema) == ["/a~1b~0c: expected string, got integer"]


@pytest.mark.parametrize(
    ("schema", "instance"),
    [
        ({"type": "decimal"}, "x"),
        ({"enum": "a"}, "x"),
        ({"pattern": "("}, "x"),
        ({"minLength": -1}, "x"),
        ({"minLength": True}, "x"),
        ({"anyOf": []}, "x"),
        ({"minimum": "0"}, 1),
        ({"required": "name"}, {}),
        ({"properties": []}, {}),
        ({"items": 3}, ["x"]),
    ],
)
def test_malformed_schemas_raise_suite_error(schema: dict[str, Any], instance: object) -> None:
    with pytest.raises(SuiteError):
        validate(instance, schema)


def test_unsupported_keywords_reports_nested_paths() -> None:
    schema = {
        "type": "object",
        "format": "x",
        "properties": {
            "a": {"type": "string", "format": "email"},
            "b": {"items": {"$ref": "#/defs/x"}},
        },
        "additionalProperties": {"multipleOf": 2},
        "anyOf": [{"type": "object"}, {"dependentRequired": {}}],
    }
    assert sorted(unsupported_keywords(schema)) == [
        "/additionalProperties/multipleOf",
        "/anyOf/1/dependentRequired",
        "/format",
        "/properties/a/format",
        "/properties/b/items/$ref",
    ]


def test_unsupported_keywords_accepts_the_supported_subset() -> None:
    assert unsupported_keywords(PERSON) == []
    assert unsupported_keywords(True) == []


def test_unsupported_keywords_flags_non_schema_nodes() -> None:
    assert unsupported_keywords({"items": [{"type": "string"}]}) == ["/items"]
    assert unsupported_keywords("string") == ["/"]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('{"a": 1}', {"a": 1}),
        ("  \n[1, 2]\n ", [1, 2]),
        ("42", 42),
        ('Here you go:\n```json\n{"a": 1}\n```\nDone.', {"a": 1}),
        ('```\n{"a": 2}\n```', {"a": 2}),
        ('```JSON\n{"a": 3}\n```', {"a": 3}),
        ('<think>maybe {"a": 0}</think>\n{"a": 4}', {"a": 4}),
        ('the user wants JSON</think>{"a": 5}', {"a": 5}),
        ('```json\nnot json\n```\n```json\n{"a": 6}\n```', {"a": 6}),
        ("null", None),
    ],
)
def test_extract_json_accepts(text: str, value: object) -> None:
    assert extract_json(text) == (value, None)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "response is empty"),
        ("<think>still thinking", "response is empty"),
        ("Sure! {'a': 1}", "invalid JSON"),
        ('```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```', "found 2 fenced JSON blocks"),
        ('```python\n{"a": 1}\n```', "invalid JSON"),
    ],
)
def test_extract_json_rejects(text: str, reason: str) -> None:
    value, problem = extract_json(text)
    assert value is None
    assert problem is not None
    assert problem.startswith(reason)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('{"n": NaN}', "invalid JSON: NaN is not a JSON value"),
        ('{"n": Infinity}', "invalid JSON: Infinity is not a JSON value"),
        ("[-Infinity]", "invalid JSON: -Infinity is not a JSON value"),
        ('{"n": 1e400}', "invalid JSON: number 1e400 is too large"),
        ('{"n": 1, "n": 2}', "invalid JSON: duplicate key 'n'"),
        ('```json\n{"a": {"b": 1, "b": 1}}\n```', "invalid JSON"),
    ],
)
def test_extract_json_rejects_what_json_does_not_allow(text: str, reason: str) -> None:
    value, problem = extract_json(text)
    assert value is None
    assert problem is not None
    assert problem.startswith(reason)


def test_extract_json_survives_pathological_nesting() -> None:
    assert extract_json("[" * 100_000) == (None, "invalid JSON: nesting is too deep")
