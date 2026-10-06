"""A small JSON Schema validator for the keyword subset quantdiff suites use.

Suite loaders call `check_schema` on every schema they read. It rejects any keyword outside
`SUPPORTED_KEYWORDS`, so a schema is never silently checked only in part, and any malformed
keyword value, so suite bugs surface at load time. `validate` raises SuiteError for the same
problems when handed a schema that skipped that check.

`pattern` follows ECMA 262 regular expressions where Python's dialect differs in ways a
schema author would not expect: `\\d`, `\\w` and `\\b` match ASCII only, and `$` matches only
at the very end of the string (Python's `$` also matches before a trailing newline).
Patterns that repeat a group which itself contains a quantifier or an alternation, such as
`(a+)+` or `(a|aa)*`, are refused because they can backtrack for exponential time.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Final

from quantdiff.errors import SuiteError
from quantdiff.metrics.textsim import strip_reasoning
from quantdiff.types import JSONValue

MAX_PATTERN_INPUT: Final = 10_000
"""Longer strings are not matched against `pattern`.

This bounds the regex engine's input. It does not by itself bound backtracking time; refusing
nested quantifiers does that for the exponential cases.
"""

_SUBSCHEMA_KEYWORDS: Final = ("additionalProperties", "items")
_SCHEMA_LIST_KEYWORDS: Final = ("anyOf", "oneOf", "allOf")
_FENCE: Final = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_JSON_FENCE_TAGS: Final = frozenset({"", "json"})

_JSON_TYPE_NAMES: Final = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)


def _is_number(value: JSONValue) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: JSONValue) -> bool:
    """JSON has one number type, so 3.0 is an integer; float() would overflow on big ints."""
    if isinstance(value, float):
        return value.is_integer()
    return isinstance(value, int) and not isinstance(value, bool)


_TYPE_CHECKS: Final[Mapping[str, Callable[[JSONValue], bool]]] = {
    "null": lambda value: value is None,
    "boolean": lambda value: isinstance(value, bool),
    "integer": _is_integer,
    "number": _is_number,
    "string": lambda value: isinstance(value, str),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}

# Regular expression tokens, as produced by _regex_tokens.
_ATOM: Final = "atom"
_OPEN: Final = "open"
_CLOSE: Final = "close"
_ALTERNATION: Final = "alternation"
_QUANTIFIER: Final = "quantifier"
_END_ANCHOR: Final = "end"
_QUANTIFIER_SYNTAX: Final = re.compile(r"(?:[*+?]|\{(?:\d+(?:,\d*)?|,\d*)\})[?+]?")
_GROUP_START: Final = re.compile(
    r"\((?:\?(?:P<\w+>|<\w+>|<[=!]|[:=!>]|[aiLmsux]*(?:-[imsx]+)?:?))?"
)
_GROUP_ATOM: Final = re.compile(r"\(\?(?:P=\w+|#[^)]*)\)")
_SINGLE_CHAR_TOKENS: Final = {")": _CLOSE, "|": _ALTERNATION, "$": _END_ANCHOR}


class _StrictJSONError(ValueError):
    """Input that Python's json module accepts but JSON does not allow."""


class _NestedQuantifierError(Exception):
    """A pattern repeats a group that can match the same text in more than one way."""

    def __init__(self, group: str) -> None:
        super().__init__(group)
        self.group = group


# Schema checks ---------------------------------------------------------------------------


def check_schema(schema: JSONValue, where: str = "schema") -> None:
    """Raise SuiteError unless `schema` uses only supported keywords with valid values.

    `where` names the schema in error messages; nested locations are appended to it.
    """
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SuiteError(f"{where} must be a JSON object or boolean")
    for key, value in schema.items():
        check = _KEYWORD_CHECKS.get(key)
        if check is None:
            raise SuiteError(f"{where} uses unsupported JSON Schema keyword {key!r}")
        check(value, f"{where}.{key}")


def _schema_properties(value: JSONValue, where: str) -> None:
    if not isinstance(value, dict):
        raise SuiteError(f"{where} must be an object")
    for name, subschema in value.items():
        check_schema(subschema, f"{where}.{name}")


def _schema_list(value: JSONValue, where: str) -> None:
    if not isinstance(value, list) or not value:
        raise SuiteError(f"{where} must be a non-empty array of schemas")
    for index, subschema in enumerate(value):
        check_schema(subschema, f"{where}[{index}]")


def _schema_type(value: JSONValue, where: str) -> None:
    names = value if isinstance(value, list) else [value]
    if not names or not all(isinstance(name, str) and name in _TYPE_CHECKS for name in names):
        raise SuiteError(f"{where} must be one of {sorted(_TYPE_CHECKS)} or a list of them")


def _schema_required(value: JSONValue, where: str) -> None:
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise SuiteError(f"{where} must be an array of strings")


def _schema_enum(value: JSONValue, where: str) -> None:
    if not isinstance(value, list) or not value:
        raise SuiteError(f"{where} must be a non-empty array")


def _schema_number(value: JSONValue, where: str) -> None:
    if not _is_number(value):
        raise SuiteError(f"{where} must be a number")


def _schema_count(value: JSONValue, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SuiteError(f"{where} must be a non-negative integer")


def _schema_pattern(value: JSONValue, where: str) -> None:
    if not isinstance(value, str):
        raise SuiteError(f"{where} must be a string")
    _compile(value, where)


def _schema_string(value: JSONValue, where: str) -> None:
    if not isinstance(value, str):
        raise SuiteError(f"{where} must be a string")


def _schema_any(value: JSONValue, where: str) -> None:
    return


_KEYWORD_CHECKS: Final[Mapping[str, Callable[[JSONValue, str], None]]] = {
    "type": _schema_type,
    "properties": _schema_properties,
    "required": _schema_required,
    "additionalProperties": check_schema,
    "items": check_schema,
    "enum": _schema_enum,
    "const": _schema_any,
    "minLength": _schema_count,
    "maxLength": _schema_count,
    "minimum": _schema_number,
    "maximum": _schema_number,
    "exclusiveMinimum": _schema_number,
    "exclusiveMaximum": _schema_number,
    "minItems": _schema_count,
    "maxItems": _schema_count,
    "pattern": _schema_pattern,
    "anyOf": _schema_list,
    "oneOf": _schema_list,
    "allOf": _schema_list,
    "description": _schema_string,
    "title": _schema_string,
    "default": _schema_any,
    "$schema": _schema_string,
}
SUPPORTED_KEYWORDS: Final = frozenset(_KEYWORD_CHECKS)
"""JSON Schema keywords the validator checks. Anything else is rejected up front."""


def unsupported_keywords(schema: JSONValue, path: str = "") -> list[str]:
    """Return the JSON-pointer path of every keyword this validator cannot check."""
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, dict):
        return [path or "/"]
    found = [_pointer(path, key) for key in schema if key not in SUPPORTED_KEYWORDS]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, subschema in properties.items():
            found += unsupported_keywords(subschema, _pointer(_pointer(path, "properties"), name))
    for keyword in _SUBSCHEMA_KEYWORDS:
        if keyword in schema:
            found += unsupported_keywords(schema[keyword], _pointer(path, keyword))
    for keyword in _SCHEMA_LIST_KEYWORDS:
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for index, branch in enumerate(branches):
                found += unsupported_keywords(branch, _pointer(_pointer(path, keyword), index))
    return found


# Validation ------------------------------------------------------------------------------


def validate(instance: JSONValue, schema: JSONValue) -> list[str]:
    """Return one readable error per violation, each prefixed with its path; [] if valid."""
    return _validate(instance, schema, "")


def extract_json(text: str) -> tuple[JSONValue | None, str | None]:
    """Parse a model answer as JSON, returning (value, None) or (None, reason).

    The whole answer must be JSON, or contain exactly one ```json or bare ``` fenced
    block that parses. A leading <think> block is ignored. NaN, Infinity, numbers too large
    for a float and duplicate object keys are rejected: Python's json module accepts them,
    but they are not valid JSON and would let a value slip past numeric bounds.
    """
    answer = strip_reasoning(text).strip()
    if not answer:
        return None, "response is empty"
    value, error = _parse(answer)
    if error is None:
        return value, None

    parsed = []
    for match in _FENCE.finditer(answer):
        if match.group(1).lower() not in _JSON_FENCE_TAGS:
            continue
        block_value, block_error = _parse(match.group(2).strip())
        if block_error is None:
            parsed.append(block_value)
    if len(parsed) == 1:
        return parsed[0], None
    if parsed:
        return None, f"found {len(parsed)} fenced JSON blocks, expected exactly one"
    return None, f"invalid JSON: {error}"


def _parse(text: str) -> tuple[JSONValue, str | None]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (json.JSONDecodeError, _StrictJSONError) as exc:
        return None, str(exc)
    except RecursionError:
        return None, "nesting is too deep"
    return value, None


def _unique_keys(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    record: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in record:
            raise _StrictJSONError(f"duplicate key {key!r}")
        record[key] = value
    return record


def _reject_constant(name: str) -> JSONValue:
    raise _StrictJSONError(f"{name} is not a JSON value")


def _finite_float(text: str) -> float:
    value = float(text)
    if math.isinf(value):
        raise _StrictJSONError(f"number {text} is too large")
    return value


def _validate(instance: JSONValue, schema: JSONValue, path: str) -> list[str]:
    if schema is True:
        return []
    if schema is False:
        return [_error(path, "no value is allowed here")]
    if not isinstance(schema, dict):
        raise SuiteError(f"schema at {path or '/'} must be an object or boolean")

    type_errors = _check_type(instance, schema, path)
    if type_errors:
        # Further keywords would only restate the type mismatch.
        return type_errors
    errors = _check_values(instance, schema, path)
    if isinstance(instance, str):
        errors += _check_string(instance, schema, path)
    elif _is_number(instance):
        errors += _check_number(instance, schema, path)
    elif isinstance(instance, list):
        errors += _check_array(instance, schema, path)
    elif isinstance(instance, dict):
        errors += _check_object(instance, schema, path)
    return errors + _check_combinators(instance, schema, path)


def _check_type(instance: JSONValue, schema: dict[str, JSONValue], path: str) -> list[str]:
    if "type" not in schema:
        return []
    declared = schema["type"]
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if name not in _TYPE_CHECKS:
            raise SuiteError(f"schema at {path or '/'} has unknown type {name!r}")
    if any(_TYPE_CHECKS[name](instance) for name in names):
        return []
    expected = " or ".join(names)
    return [_error(path, f"expected {expected}, got {_json_type(instance)}")]


def _check_values(instance: JSONValue, schema: dict[str, JSONValue], path: str) -> list[str]:
    errors = []
    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(_error(path, f"must equal {_preview(schema['const'])}"))
    if "enum" in schema:
        options = schema["enum"]
        if not isinstance(options, list):
            raise SuiteError(f"schema at {path or '/'} has a non-array enum")
        if not any(_json_equal(instance, option) for option in options):
            errors.append(_error(path, f"must be one of {_preview(options)}"))
    return errors


def _check_string(instance: str, schema: dict[str, JSONValue], path: str) -> list[str]:
    errors = []
    min_length = _count(schema, "minLength", path)
    max_length = _count(schema, "maxLength", path)
    if min_length is not None and len(instance) < min_length:
        errors.append(_error(path, f"must be at least {min_length} characters"))
    if max_length is not None and len(instance) > max_length:
        errors.append(_error(path, f"must be at most {max_length} characters"))
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            raise SuiteError(f"schema at {path or '/'} has a non-string pattern")
        compiled = _compile(pattern, f"schema at {path or '/'}")
        if len(instance) > MAX_PATTERN_INPUT:
            errors.append(_error(path, f"is longer than {MAX_PATTERN_INPUT} characters"))
        elif not compiled.search(instance):
            errors.append(_error(path, f"does not match pattern {pattern!r}"))
    return errors


def _check_number(instance: float, schema: dict[str, JSONValue], path: str) -> list[str]:
    errors = []
    bounds: tuple[tuple[str, Callable[[float, float], bool], str], ...] = (
        ("minimum", lambda value, limit: value >= limit, ">="),
        ("maximum", lambda value, limit: value <= limit, "<="),
        ("exclusiveMinimum", lambda value, limit: value > limit, ">"),
        ("exclusiveMaximum", lambda value, limit: value < limit, "<"),
    )
    for keyword, holds, symbol in bounds:
        if keyword not in schema:
            continue
        limit = schema[keyword]
        if not _is_number(limit):
            raise SuiteError(f"schema at {path or '/'} has a non-numeric {keyword}")
        if not holds(instance, limit):
            errors.append(_error(path, f"must be {symbol} {limit}"))
    return errors


def _check_array(instance: list[JSONValue], schema: dict[str, JSONValue], path: str) -> list[str]:
    errors = []
    min_items = _count(schema, "minItems", path)
    max_items = _count(schema, "maxItems", path)
    if min_items is not None and len(instance) < min_items:
        errors.append(_error(path, f"must have at least {min_items} items"))
    if max_items is not None and len(instance) > max_items:
        errors.append(_error(path, f"must have at most {max_items} items"))
    if "items" in schema:
        for index, item in enumerate(instance):
            errors += _validate(item, schema["items"], _pointer(path, index))
    return errors


def _check_object(
    instance: dict[str, JSONValue], schema: dict[str, JSONValue], path: str
) -> list[str]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict):
        raise SuiteError(f"schema at {path or '/'} has non-object properties")
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise SuiteError(f"schema at {path or '/'} has a required list that is not strings")

    errors = [
        _error(path, f"missing required property {name!r}")
        for name in required
        if name not in instance
    ]
    for name, value in instance.items():
        child = _pointer(path, name)
        if name in properties:
            errors += _validate(value, properties[name], child)
        elif "additionalProperties" in schema:
            if schema["additionalProperties"] is False:
                errors.append(_error(path, f"unexpected property {name!r}"))
            else:
                errors += _validate(value, schema["additionalProperties"], child)
    return errors


def _check_combinators(instance: JSONValue, schema: dict[str, JSONValue], path: str) -> list[str]:
    errors = []
    for branch in _branches(schema, "allOf", path):
        errors += _validate(instance, branch, path)
    if "anyOf" in schema:
        branches = _branches(schema, "anyOf", path)
        if not any(not _validate(instance, branch, path) for branch in branches):
            errors.append(_error(path, "does not match any schema in anyOf"))
    if "oneOf" in schema:
        branches = _branches(schema, "oneOf", path)
        matched = sum(not _validate(instance, branch, path) for branch in branches)
        if matched != 1:
            errors.append(_error(path, f"matches {matched} schemas in oneOf, expected exactly 1"))
    return errors


def _branches(schema: dict[str, JSONValue], keyword: str, path: str) -> list[JSONValue]:
    branches = schema.get(keyword, [])
    if not isinstance(branches, list) or (keyword in schema and not branches):
        raise SuiteError(f"schema at {path or '/'} needs a non-empty array for {keyword}")
    return branches


def _count(schema: dict[str, JSONValue], keyword: str, path: str) -> int | None:
    value = schema.get(keyword)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SuiteError(f"schema at {path or '/'} needs a non-negative integer for {keyword}")
    count: int = value
    return count


# Patterns --------------------------------------------------------------------------------


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return _compile_cached(pattern)
    except re.error as exc:
        raise SuiteError(f"{where} is not a valid regular expression: {exc}") from None
    except _NestedQuantifierError as exc:
        raise SuiteError(
            f"{where} repeats {exc.group!r}, which itself contains a quantifier or an "
            "alternation; such patterns can take exponential time to fail, so use a "
            "character class or an unrepeated group instead"
        ) from None


@functools.lru_cache(maxsize=256)
def _compile_cached(pattern: str) -> re.Pattern[str]:
    # Compiling first rejects invalid syntax, so the scanner below only sees valid patterns.
    re.compile(pattern)
    group = _nested_quantifier(pattern)
    if group is not None:
        raise _NestedQuantifierError(group)
    ecma = "".join(
        r"\Z" if kind == _END_ANCHOR else pattern[start:end]
        for kind, start, end in _regex_tokens(pattern)
    )
    return re.compile(ecma, re.ASCII)


def _nested_quantifier(pattern: str) -> str | None:
    """Return the first group repeated more than once that contains a quantifier or `|`.

    That shape lets the engine split one input many ways, which is what makes failing
    matches take exponential time. Fixed counts such as `{3}` are not quantifiers here
    because they leave only one way to split.
    """
    enclosing: list[tuple[int, bool]] = []
    ambiguous = False
    closed: tuple[int, bool] | None = None
    for kind, start, end in _regex_tokens(pattern):
        just_closed, closed = closed, None
        if kind == _OPEN:
            enclosing.append((start, ambiguous))
            ambiguous = False
        elif kind == _CLOSE and enclosing:
            group_start, outer = enclosing.pop()
            closed = (group_start, ambiguous)
            ambiguous = outer or ambiguous
        elif kind == _ALTERNATION:
            ambiguous = True
        elif kind == _QUANTIFIER:
            low, high = _repeat_bounds(pattern[start:end])
            if just_closed is not None and just_closed[1] and (high is None or high > 1):
                return pattern[just_closed[0] : end]
            ambiguous = ambiguous or low != high
    return None


def _repeat_bounds(quantifier: str) -> tuple[int, int | None]:
    """Return (minimum, maximum) repeats for a quantifier token; None means unbounded."""
    symbol = quantifier[0]
    if symbol == "*":
        return 0, None
    if symbol == "+":
        return 1, None
    if symbol == "?":
        return 0, 1
    low, comma, high = quantifier[1 : quantifier.index("}")].partition(",")
    minimum = int(low or "0")
    if not comma:
        return minimum, minimum
    return minimum, int(high) if high else None


def _regex_tokens(pattern: str) -> Iterator[tuple[str, int, int]]:
    """Split a valid Python regular expression into (kind, start, end) tokens."""
    index = 0
    while index < len(pattern):
        kind, end = _next_token(pattern, index)
        yield kind, index, end
        index = end


def _next_token(pattern: str, index: int) -> tuple[str, int]:
    char = pattern[index]
    if char in _SINGLE_CHAR_TOKENS:
        return _SINGLE_CHAR_TOKENS[char], index + 1
    if char == "\\":
        return _ATOM, index + 2
    if char == "[":
        return _ATOM, _class_end(pattern, index)
    if char == "(":
        atom = _GROUP_ATOM.match(pattern, index)
        if atom:
            return _ATOM, atom.end()
        group = _GROUP_START.match(pattern, index)
        return _OPEN, group.end() if group else index + 1
    quantifier = _QUANTIFIER_SYNTAX.match(pattern, index)
    return (_QUANTIFIER, quantifier.end()) if quantifier else (_ATOM, index + 1)


def _class_end(pattern: str, start: int) -> int:
    index = start + 1
    if pattern.startswith("^", index):
        index += 1
    # A "]" right after the opening bracket (or "[^") is a literal member.
    if pattern.startswith("]", index):
        index += 1
    while index < len(pattern) and pattern[index] != "]":
        index += 2 if pattern[index] == "\\" else 1
    return index + 1


# Helpers ---------------------------------------------------------------------------------


def _json_equal(left: JSONValue, right: JSONValue) -> bool:
    """Equality under JSON rules: true is not 1, but 1 equals 1.0."""
    if _is_number(left) and _is_number(right):
        return bool(left == right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(map(_json_equal, left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    return bool(left == right)


def _json_type(value: JSONValue) -> str:
    if value is None:
        return "null"
    # bool is checked before int because bool is a subclass of int.
    for python_type, name in _JSON_TYPE_NAMES:
        if isinstance(value, python_type):
            return name
    return type(value).__name__


def _pointer(path: str, token: str | int) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _error(path: str, message: str) -> str:
    return f"{path or '/'}: {message}"


def _preview(value: JSONValue, limit: int = 80) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
