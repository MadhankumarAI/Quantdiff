"""Run model-written code against a case's assert-based tests in a child process.

This is isolation for honest mistakes, not a security sandbox. The code runs in a fresh
interpreter inside a throwaway directory, with an empty environment (so API keys in the
parent's environment are not visible), a timeout, and on POSIX some resource limits. On
timeout every process the code started is killed, not only the interpreter. A deliberately
hostile program can still read your files or reach the network. quantdiff therefore never
calls this unless the user passes --allow-code-exec.

The verdict is guarded against solutions that try to report a pass for themselves. The
parent sends the harness a random nonce on stdin before the solution is imported, and only
accepts a verdict that carries it. The harness writes that verdict to a private duplicate
of its original stdout, after pointing stdout at the output log, then ends the process
with os._exit so no exit handler, finally block or finalizer from the solution runs after
it. Files the solution writes and anything it prints are never read as a verdict. The
solution shares the harness's interpreter, so code written to dig the nonce out of the
harness's stack frames can still forge a pass; that takes deliberate effort aimed at
quantdiff, and SECURITY.md lists it as a known limit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

from quantdiff.errors import SuiteError
from quantdiff.metrics.textsim import strip_reasoning
from quantdiff.types import CaseOutcome, TaskCase

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final = 10.0
OUTPUT_PREVIEW_CHARS: Final = 2000

_FENCE: Final = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_PYTHON_TAGS: Final = frozenset({"python", "python3", "py"})
_SOLUTION_FILE: Final = "solution.py"
_TESTS_FILE: Final = "case_tests.py"
_HARNESS_FILE: Final = "harness.py"
_VERDICT_FILE: Final = "verdict.jsonl"
_OUTPUT_FILE: Final = "output.txt"
_MAX_VERDICT_BYTES: Final = 64 * 1024
_NONCE_BYTES: Final = 16

# The harness runs in the child. Everything that can produce a verdict lives in main()'s
# locals, and the functions it calls are bound before the solution can replace them.
_HARNESS: Final = """\
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, {tests_file!r})
ENTRY_POINT = {entry_point!r}


def one_line(text, limit=200):
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def describe(exc):
    if isinstance(exc, AssertionError):
        line = ""
        for frame in reversed(traceback.extract_tb(exc.__traceback__)):
            if frame.filename == TESTS:
                line = (frame.line or "").strip()
                break
        detail = ": " + one_line(exc) if str(exc) else ""
        return one_line("assertion failed: " + (line or "assert") + detail)
    return one_line(type(exc).__name__ + (": " + str(exc) if str(exc) else ""))


def main():
    nonce = sys.stdin.readline().strip()
    write, exit_now, dumps = os.write, os._exit, json.dumps
    verdict_fd = os.dup(1)
    os.dup2(2, 1)

    def report(status, reason):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except BaseException:
                pass
        line = dumps({{"nonce": nonce, "status": status, "reason": reason}}) + "\\n"
        write(verdict_fd, line.encode("utf-8"))
        exit_now(0)

    sys.path.insert(0, HERE)
    try:
        import solution
    except BaseException as exc:
        report("fail", "solution failed to import: " + describe(exc))
    if not callable(getattr(solution, ENTRY_POINT, None)):
        report("fail", "solution does not define " + ENTRY_POINT)
    namespace = dict(vars(solution))
    try:
        with open(TESTS, encoding="utf-8") as handle:
            source = handle.read()
        exec(compile(source, TESTS, "exec"), namespace)
    except BaseException as exc:
        report("fail", describe(exc))
    report("pass", "")


main()
"""


def extract_code(answer: str) -> str:
    """Return the code in a model answer.

    Prefers the first ```python fenced block, then the first fenced block of any kind,
    then the whole answer. A leading <think> block is ignored.
    """
    text = strip_reasoning(answer)
    blocks = [(match.group(1).lower(), match.group(2)) for match in _FENCE.finditer(text)]
    for tag, body in blocks:
        if tag in _PYTHON_TAGS:
            return body
    if blocks:
        return blocks[0][1]
    return text.strip()


def run_code_case(
    case: TaskCase, answer: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> CaseOutcome:
    """Run the code in `answer` against `case.tests` and report pass or fail.

    Raises SuiteError if the case lacks a valid entry point or tests. Any other problem,
    including one in quantdiff's own process handling, becomes a failed outcome with a
    reason, so one misbehaving answer cannot abort a run.
    """
    entry_point, tests = _require_code_fields(case)
    code = extract_code(answer)
    if not code.strip():
        return _outcome(case, passed=False, reason="answer contains no code")
    try:
        passed, reason = _run_isolated(
            code=code, tests=tests, entry_point=entry_point, timeout_seconds=timeout_seconds
        )
    except Exception as exc:
        logger.exception("could not run code case %s", case.id)
        return _outcome(case, passed=False, reason=f"could not run the tests: {exc!r}")
    return _outcome(case, passed=passed, reason=reason)


def _require_code_fields(case: TaskCase) -> tuple[str, str]:
    if not case.entry_point or not case.entry_point.isidentifier():
        raise SuiteError(f"code case {case.id!r} needs an entry_point that is a Python name")
    if not case.tests or not case.tests.strip():
        raise SuiteError(f"code case {case.id!r} has no tests")
    return case.entry_point, case.tests


def _run_isolated(
    *, code: str, tests: str, entry_point: str, timeout_seconds: float
) -> tuple[bool, str]:
    # Not TemporaryDirectory: on CPython 3.10 and 3.11 its ignore_cleanup_errors path
    # recurses without bound when Windows refuses to delete a directory still in use.
    workdir = Path(tempfile.mkdtemp(prefix="quantdiff-code-"))
    try:
        _write_files(workdir, code=code, tests=tests, entry_point=entry_point)
        return _run_harness(workdir, timeout_seconds)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _write_files(workdir: Path, *, code: str, tests: str, entry_point: str) -> None:
    harness = _HARNESS.format(tests_file=_TESTS_FILE, entry_point=entry_point)
    (workdir / _SOLUTION_FILE).write_text(code, encoding="utf-8")
    (workdir / _TESTS_FILE).write_text(tests, encoding="utf-8")
    (workdir / _HARNESS_FILE).write_text(harness, encoding="utf-8")


def _run_harness(workdir: Path, timeout_seconds: float) -> tuple[bool, str]:
    # -I would make Python ignore the PYTHON* variables below. The environment is built
    # from scratch instead, so -E adds nothing; -s and -S keep site-packages out.
    command = [sys.executable, "-s", "-S", str(workdir / _HARNESS_FILE)]
    nonce = secrets.token_hex(_NONCE_BYTES)
    verdict_path = workdir / _VERDICT_FILE
    output_path = workdir / _OUTPUT_FILE
    with verdict_path.open("wb") as verdict, output_path.open("wb") as output:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell; running the code is the point
            command,
            cwd=workdir,
            env=_child_environment(),
            stdin=subprocess.PIPE,
            stdout=verdict,
            stderr=output,
            preexec_fn=_resource_limiter(timeout_seconds),  # noqa: PLW1509 - only sets rlimits, no locks or threads
            start_new_session=_NEW_SESSION,
        )
        with _contained(process):
            # The harness waits for this line before importing the solution, so the
            # process is already contained by the time any model code runs.
            _send_nonce(process, nonce)
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                return False, f"timed out after {timeout_seconds:g}s"

    logger.debug("code case output: %s", _read_preview(output_path))
    return _read_verdict(verdict_path, nonce, returncode)


def _send_nonce(process: subprocess.Popen[bytes], nonce: str) -> None:
    if process.stdin is None:
        raise RuntimeError("harness stdin is not a pipe")
    try:
        process.stdin.write(f"{nonce}\n".encode("ascii"))
        process.stdin.close()
    except OSError:
        # The interpreter died before reading; the missing verdict reports it.
        logger.debug("code harness exited before it read its nonce")


def _child_environment() -> dict[str, str]:
    environment = {
        "PATH": "",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if sys.platform == "win32":
        # The Windows C runtime and the interpreter fail to start without it.
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return environment


def _read_verdict(verdict_path: Path, nonce: str, returncode: int) -> tuple[bool, str]:
    """Return the harness's verdict, ignoring any line that lacks this run's nonce."""
    with verdict_path.open("rb") as handle:
        raw = handle.read(_MAX_VERDICT_BYTES)
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            verdict = json.loads(line)
        except ValueError:
            continue
        if not isinstance(verdict, dict) or verdict.get("nonce") != nonce:
            continue
        if verdict.get("status") not in {"pass", "fail"}:
            return False, "harness reported a malformed result"
        return verdict["status"] == "pass", str(verdict.get("reason", ""))
    return False, f"process exited with code {returncode} before reporting a result"


def _read_preview(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(OUTPUT_PREVIEW_CHARS * 4)
    return raw.decode("utf-8", errors="replace")[:OUTPUT_PREVIEW_CHARS]


def _outcome(case: TaskCase, *, passed: bool, reason: str) -> CaseOutcome:
    return CaseOutcome(case_id=case.id, kind=case.kind, passed=passed, reason=reason)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _NEW_SESSION: Final = False
    _EXTENDED_LIMIT_INFORMATION: Final = 9
    _KILL_ON_JOB_CLOSE: Final = 0x2000
    _PROCESS_TERMINATE_AND_SET_QUOTA: Final = 0x0001 | 0x0100
    _NOT_INHERITED: Final = 0

    class _BasicLimits(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        _fields_ = (
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        )

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    def _resource_limiter(timeout_seconds: float) -> Callable[[], None] | None:
        return None

    @contextlib.contextmanager
    def _contained(process: subprocess.Popen[bytes]) -> Iterator[None]:
        """Kill `process` and every process it started when the block exits.

        A job object follows the whole process tree, which a plain kill does not: a
        grandchild would survive it and keep the working directory in use.
        """
        job = _job_for(process)
        try:
            yield
        finally:
            if job is None:
                process.kill()
            else:
                _kernel32.TerminateJobObject(job, 1)
                _kernel32.CloseHandle(job)
            process.wait()

    def _job_for(process: subprocess.Popen[bytes]) -> int | None:
        job: int | None = _kernel32.CreateJobObjectW(None, None)
        if not job:
            _warn_uncontained()
            return None
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        handle: int | None = _kernel32.OpenProcess(
            _PROCESS_TERMINATE_AND_SET_QUOTA, _NOT_INHERITED, process.pid
        )
        assigned = bool(
            handle
            and _kernel32.SetInformationJobObject(
                job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
            )
            and _kernel32.AssignProcessToJobObject(job, handle)
        )
        if handle:
            _kernel32.CloseHandle(handle)
        if not assigned:
            _warn_uncontained()
            _kernel32.CloseHandle(job)
            return None
        return job

    def _warn_uncontained() -> None:
        logger.warning(
            "could not put the code harness in a Windows job object (error %d); "
            "processes started by model code may outlive a timeout",
            ctypes.get_last_error(),
        )

else:
    import resource
    import signal

    _NEW_SESSION: Final = True
    _MEMORY_BYTES: Final = 1024**3
    _FILE_BYTES: Final = 16 * 1024**2
    _OPEN_FILES: Final = 64

    def _resource_limiter(timeout_seconds: float) -> Callable[[], None] | None:
        limits = (
            (resource.RLIMIT_CPU, math.ceil(timeout_seconds) + 1),
            (resource.RLIMIT_AS, _MEMORY_BYTES),
            (resource.RLIMIT_FSIZE, _FILE_BYTES),
            (resource.RLIMIT_NOFILE, _OPEN_FILES),
        )

        def apply_limits() -> None:
            for which, wanted in limits:
                _, hard = resource.getrlimit(which)
                value = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
                try:
                    resource.setrlimit(which, (value, value))
                except (ValueError, OSError):
                    # Some kernels (macOS for RLIMIT_AS) refuse a limit; the timeout still applies.
                    continue

        return apply_limits

    @contextlib.contextmanager
    def _contained(process: subprocess.Popen[bytes]) -> Iterator[None]:
        """Kill `process` and every process in its session's group when the block exits."""
        try:
            yield
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                # The group is already empty (ESRCH), or only zombies remain (EPERM on macOS).
                logger.debug("code harness process group already gone")
            process.wait()
