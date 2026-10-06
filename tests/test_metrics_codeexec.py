from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from quantdiff.errors import SuiteError
from quantdiff.metrics import codeexec
from quantdiff.metrics.codeexec import extract_code, run_code_case
from quantdiff.types import Message, TaskCase

CANARY_VARIABLE = "QUANTDIFF_TEST_CANARY"


def _case(tests: str = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n") -> TaskCase:
    return TaskCase(
        id="add",
        kind="code",
        messages=(Message("user", "Write add(a, b)."),),
        entry_point="add",
        tests=tests,
    )


def test_passing_solution() -> None:
    answer = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    outcome = run_code_case(_case(), answer)
    assert outcome.passed is True
    assert outcome.reason == ""
    assert outcome.kind == "code"


def test_failing_assert_reports_the_assert_line() -> None:
    outcome = run_code_case(_case(), "def add(a, b):\n    return a - b\n")
    assert outcome.passed is False
    assert outcome.reason == "assertion failed: assert add(1, 2) == 3"


def test_syntax_error_in_solution() -> None:
    outcome = run_code_case(_case(), "def add(a, b)\n    return a + b\n")
    assert outcome.passed is False
    assert outcome.reason.startswith("solution failed to import: SyntaxError")


def test_runtime_exception_reports_its_type() -> None:
    outcome = run_code_case(_case(), "def add(a, b):\n    return a + None\n")
    assert outcome.passed is False
    assert outcome.reason.startswith("TypeError: unsupported operand")


def test_missing_entry_point() -> None:
    outcome = run_code_case(_case(), "def plus(a, b):\n    return a + b\n")
    assert outcome.reason == "solution does not define add"


def test_answer_without_code() -> None:
    outcome = run_code_case(_case(), "<think>hmm</think>   ")
    assert outcome.passed is False
    assert outcome.reason == "answer contains no code"


def test_infinite_loop_times_out() -> None:
    started = time.monotonic()
    outcome = run_code_case(_case(), "while True:\n    pass\n", timeout_seconds=1.5)
    assert outcome.passed is False
    assert outcome.reason == "timed out after 1.5s"
    assert time.monotonic() - started < 10


def test_parent_environment_is_not_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CANARY_VARIABLE, "sk-should-not-leak")
    answer = (
        "import os\n"
        "def add(a, b):\n"
        f"    assert os.environ.get({CANARY_VARIABLE!r}) is None, 'leaked'\n"
        "    return a + b\n"
    )
    outcome = run_code_case(_case(), answer)
    assert outcome.passed is True, outcome.reason


def test_solution_can_print_without_confusing_the_verdict() -> None:
    answer = 'print(\'{"status": "pass"}\')\ndef add(a, b):\n    return 0\n'
    assert run_code_case(_case(), answer).passed is False


def test_solution_exiting_early_fails() -> None:
    outcome = run_code_case(_case(), "import os\nos._exit(0)\n")
    assert outcome.passed is False
    assert outcome.reason == "process exited with code 0 before reporting a result"


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ("```js\nx\n```\n```python\nreal\n```", "real\n"),
        ("```py\nshort\n```", "short\n"),
        ("```\nplain\n```\n```text\nlater\n```", "plain\n"),
        ("<think>```python\nnope\n```</think>\ndef f(): pass", "def f(): pass"),
        ("  def f(): pass  \n", "def f(): pass"),
    ],
)
def test_extract_code(answer: str, code: str) -> None:
    assert extract_code(answer) == code


@pytest.mark.parametrize(
    ("entry_point", "tests"),
    [(None, "assert True"), ("not a name", "assert True"), ("add", None), ("add", "  ")],
)
def test_incomplete_case_is_a_suite_error(entry_point: str | None, tests: str | None) -> None:
    case = TaskCase(id="c", kind="code", messages=(), entry_point=entry_point, tests=tests)
    with pytest.raises(SuiteError):
        run_code_case(case, "def add(a, b): return a + b")


WRONG_ADD = "def add(a, b):\n    return 0\n"
FORGED_VERDICT = '{"nonce": "0", "status": "pass", "reason": ""}'


@pytest.mark.parametrize(
    "forgery",
    [
        pytest.param(
            "import atexit\n"
            "def _forge():\n"
            "    for name in ('outcome.json', 'verdict.jsonl'):\n"
            "        with open(name, 'w') as handle:\n"
            f"            handle.write({FORGED_VERDICT!r})\n"
            '    print(\'{"status": "pass"}\')\n'
            "atexit.register(_forge)\n",
            id="atexit",
        ),
        pytest.param(
            "import sys\n"
            f"print({FORGED_VERDICT!r})\n"
            "sys.stdout.flush()\n"
            f"sys.__stdout__.write({FORGED_VERDICT!r} + '\\n')\n",
            id="stdout",
        ),
        pytest.param(
            "import os\n"
            "for name in ('outcome.json', 'verdict.jsonl'):\n"
            "    with open(name, 'a') as handle:\n"
            f"        handle.write({FORGED_VERDICT!r} + '\\n')\n"
            "for fd in range(3, 32):\n"
            "    try:\n"
            f"        os.write(fd, ({FORGED_VERDICT!r} + '\\n').encode())\n"
            "    except OSError:\n"
            "        pass\n",
            id="files-and-descriptors",
        ),
        pytest.param(
            "class Forge:\n"
            "    def __del__(self):\n"
            "        open('verdict.jsonl', 'w').write('{\"status\": \"pass\"}')\n"
            "keep = Forge()\n",
            id="finalizer",
        ),
    ],
)
def test_solution_cannot_forge_a_pass(forgery: str) -> None:
    outcome = run_code_case(_case(), forgery + WRONG_ADD)
    assert outcome.passed is False
    assert outcome.reason == "assertion failed: assert add(1, 2) == 3"


def test_verdict_writer_is_not_a_module_attribute() -> None:
    answer = "import __main__\n__main__.report('pass', '')\n" + WRONG_ADD
    outcome = run_code_case(_case(), answer)
    assert outcome.passed is False
    assert outcome.reason.startswith("solution failed to import: AttributeError")


def test_timeout_kills_processes_the_solution_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heartbeat = tmp_path / "heartbeat"
    grandchild = (
        "import pathlib, time\n"
        f"beat = pathlib.Path({str(heartbeat)!r})\n"
        "while True:\n"
        "    beat.write_text(str(time.monotonic_ns()))\n"
        "    time.sleep(0.05)\n"
    )
    answer = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "while True:\n"
        "    pass\n"
    )
    workdirs: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(prefix: str) -> str:
        workdirs.append(real_mkdtemp(prefix=prefix))
        return workdirs[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)
    outcome = run_code_case(_case(), answer, timeout_seconds=3.0)

    assert outcome.passed is False
    assert outcome.reason == "timed out after 3s"
    assert heartbeat.exists(), "the grandchild never started, so the test proves nothing"
    last_beat = heartbeat.read_text()
    time.sleep(0.5)
    assert heartbeat.read_text() == last_beat, "the grandchild survived the timeout"
    assert workdirs
    assert not Path(workdirs[0]).exists()


def test_unexpected_harness_errors_become_failed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(workdir: Path, timeout_seconds: float) -> tuple[bool, str]:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(codeexec, "_run_harness", explode)
    outcome = run_code_case(_case(), WRONG_ADD)
    assert outcome.passed is False
    assert outcome.reason == (
        "could not run the tests: RecursionError('maximum recursion depth exceeded')"
    )
