# Contributing to quantdiff

Thanks for helping. Bug reports with a `report.json` attached, new backends, and new suite cases
are all welcome.

## Dev setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer.

```
git clone https://github.com/quantdiff/quantdiff
cd quantdiff
uv venv
uv pip install -e ".[dev]"
```

Activate the venv (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) or prefix
commands with `uv run`.

## Checks

CI runs these on Linux, macOS, and Windows with Python 3.10 and 3.13. Run them before opening a
pull request:

```
ruff check
ruff format --check
mypy
pytest
```

The default test run uses fake backends and needs no model server. Live tests talk to a real
local Ollama and are deselected unless you opt in:

```
ollama pull qwen2.5:0.5b-instruct
QUANTDIFF_LIVE=1 pytest -m live
```

## Ground rules

- **No runtime dependencies.** `dependencies = []` in `pyproject.toml` stays empty. If you need
  something, write the small stdlib version. Dev dependencies are pinned exactly.
- **No em dashes, en dashes, or curly quotes**, in code, docs, YAML, or TOML. Use plain ASCII
  punctuation. `tests/test_style.py` enforces this.
- Write docs and messages plainly. Say what something does and what its limits are.
- Never commit benchmark numbers that did not come from a real run, with the `report.json` to
  back them.
- Type everything. `mypy --strict` must pass.
- Errors raised to users should say what went wrong and what to do about it.

## Adding a backend

Backends live in `src/quantdiff/backends/` and implement the `Backend` protocol in
`backends/base.py`. Read that file first; the docstrings are the contract.

1. Implement `info`, `chat`, `tokenize`, `generate_scored`, and `score_continuation`.
2. Use `quantdiff._http` for every request. Do not use `urllib` directly and do not add a client
   library; `_http` enforces the scheme, redirect, credential, and size rules described in
   `SECURITY.md`.
3. Generation is always greedy. Pass the seed if the server accepts one.
4. Raise `CapabilityError` for unsupported operations (for example no logprobs) and
   `BackendError` for transport or protocol failures. Never let a bare exception escape.
5. Set `ServerInfo.exact_token_ids` to True only if `score_continuation` feeds token ids. If it
   concatenates text, the results are labeled approximate, which is correct.
6. Add the spec prefix to the spec parser and the README's spec grammar section.
7. Add tests against a fake server that replays recorded responses, including malformed ones. A
   live test marked `@pytest.mark.live` is welcome too.

## Adding suite cases

Suite data lives in `src/quantdiff/suites/data/` as JSONL, one `TaskCase` per line.

- **Deterministic answers only.** Every case must have one checkable correct outcome. If two
  reasonable experts could disagree on the answer, the case measures taste, not quantization
  damage.
- **json cases** need a `json_schema` that a correct answer satisfies and a plausible wrong
  answer does not.
- **tools cases** need `tools`, `expected_tool`, and `expected_arguments` (subset match). Avoid
  arguments with many valid spellings unless the schema pins them down.
- **code cases** need `entry_point` and `tests` (plain asserts), and you must include a reference
  solution in the pull request description or the test fixtures showing the asserts pass. Tests
  must be fast, pure, and use no network, filesystem, or randomness.
- **chat cases** are scored by agreement with the reference, so prefer prompts with short, stable
  answers.
- Use a stable, descriptive `id`, unique within the suite. Reports and cached reference results
  refer to cases by id, so do not rename existing ones casually.
- Keep prompts free of personal data and of text you do not have the right to redistribute.

## Pull requests

- One logical change per pull request, with tests.
- Add a line to `CHANGELOG.md` under `Unreleased`.
- Do not bump the version; maintainers do that at release time.
