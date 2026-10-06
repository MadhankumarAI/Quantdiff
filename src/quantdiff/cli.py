"""Command line interface: `quantdiff run | card | discover | suites`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from quantdiff._text import printable_lines
from quantdiff._version import __version__
from quantdiff.api import RunFiles, build_plan, write_run
from quantdiff.cache import ReferenceCache
from quantdiff.card import render_html, render_markdown, render_terminal
from quantdiff.discover import closest_tags, discover_ollama, format_discovery, installed_tags
from quantdiff.errors import BackendError, QuantdiffError, SpecError
from quantdiff.png import render_png
from quantdiff.preflight import DEFAULT_CONTEXT_PROBE_TOKENS
from quantdiff.progress import make_progress
from quantdiff.report import load_report
from quantdiff.runner import execute
from quantdiff.spec import parse_spec
from quantdiff.suites import BUILTIN_SUITES, load_builtin
from quantdiff.verdict import Status, Verdict, judge

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REGRESSION = 3
EXIT_INTERRUPTED = 130

_FAILING_STATUSES: dict[str, frozenset[Status]] = {
    "never": frozenset(),
    "avoid": frozenset({"avoid", "failed"}),
    "inconclusive": frozenset({"avoid", "failed", "inconclusive", "usable"}),
}
"""Candidate statuses that make `run --fail-on LEVEL` exit with EXIT_REGRESSION. Any level
but never also fails when no candidate is recommended, so thin evidence never passes CI;
avoid lets that through only when a usable candidate fits the --max-size budget."""

_DISCOVER_HINT = "`quantdiff discover` lists the models your local Ollama has"
_SUGGESTION_TIMEOUT_SECONDS = 2.0
_MISSING_MODEL_WORDS = ("not available", "not found", "does not exist")

_SUITE_SUMMARIES = {
    "json": "structured output that must match a JSON Schema",
    "tools": "picking the right tool with valid, correct arguments (or none)",
    "code": "Python functions checked by hidden asserts (needs --allow-code-exec)",
    "chat": "short answers scored by agreement with the reference",
}

_EPILOG = """\
quick start:
  1. quantdiff discover
       lists your Ollama models and prints a ready-to-run command for each model
  2. quantdiff run --ref ollama:qwen2.5:7b-instruct-q8_0 --cand ollama:qwen2.5:7b-instruct-q4_K_M
       compares each candidate with the reference and ends with the verdict
  3. share runs/<timestamp>/card.png
       the scorecard image; card.md in the same folder pastes into Reddit or GitHub

more examples:
  quantdiff run --ref ollama:qwen2.5:7b-instruct-q8_0 \\
                --cand ollama:qwen2.5:7b-instruct-q4_K_M --cand ollama:qwen2.5:7b-instruct-q3_K_M
  quantdiff run --ref ollama:qwen3:8b-q8_0 --cand ollama=ollama:qwen3:8b \\
                --cand unsloth=ollama:hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M
  quantdiff run --ref ollama:qwen2.5:7b-instruct-q8_0 --cand gguf=llamacpp:http://127.0.0.1:8080
  quantdiff run ... --max-size 6GB     pick the closest download that fits in 6 GB
  quantdiff run ... --fail-on avoid    for CI: exit 3 on any avoid, or when nothing is
                                       recommended and nothing usable fits
  quantdiff card runs/20261003T090504Z --format png -o card.png

exit codes:
  0    ok
  1    error
  3    regression found or nothing recommended (run --fail-on)
  130  interrupted
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except BackendError as exc:
        _error(str(exc), hints=[*_did_you_mean(str(exc), args), _DISCOVER_HINT])
        return EXIT_ERROR
    except QuantdiffError as exc:
        _error(str(exc))
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nquantdiff: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantdiff",
        description="Find out which download of a model gives the best answers on your prompts.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"quantdiff {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS keeps a root-level -v from being reset to False by the subcommand parser.
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="debug logging"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_run(commands, common)
    _add_card(commands, common)

    discover = commands.add_parser(
        "discover", parents=[common], help="list local Ollama models and suggest a comparison"
    )
    discover.add_argument("--host", metavar="URL", help="Ollama URL (default: OLLAMA_HOST)")
    discover.set_defaults(handler=_cmd_discover)

    suites = commands.add_parser("suites", parents=[common], help="list built-in suites")
    suites.set_defaults(handler=_cmd_suites)
    return parser


def _add_run(
    commands: argparse._SubParsersAction[argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    run = commands.add_parser(
        "run",
        parents=[common],
        help="compare candidates against a reference",
        description="Compare candidate downloads of a model against a reference download.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    models = run.add_argument_group("models")
    models.add_argument("--ref", required=True, metavar="SPEC", help="reference model spec")
    models.add_argument(
        "--cand", required=True, action="append", metavar="SPEC", help="candidate spec, repeatable"
    )
    work = run.add_argument_group("what to run")
    work.add_argument("--suite", type=_csv, metavar="NAMES", help="comma list of built-in suites")
    work.add_argument("--prompts", metavar="FILE", help="your own prompts as JSONL")
    work.add_argument("--scoring-prompts", metavar="FILE", help="raw prompts for logit metrics")
    work.add_argument(
        "--max-cases",
        type=int,
        metavar="N",
        help="cap each suite, your prompts file and the scoring prompts at N",
    )
    work.add_argument(
        "--allow-code-exec", action="store_true", help="run model-written code for the code suite"
    )
    tuning = run.add_argument_group("tuning")
    tuning.add_argument("--top-k", type=int, default=10, help="logprobs per position, 1-20")
    tuning.add_argument(
        "--score-tokens", type=int, default=32, help="tokens per scoring prompt; 0 disables"
    )
    tuning.add_argument("--seed", type=int, default=0, help="sampling seed sent to servers")
    checks = run.add_argument_group("pre-flight checks")
    checks.add_argument("--hf-repo", metavar="OWNER/NAME", help="upstream repo for template check")
    checks.add_argument("--offline", action="store_true", help="never contact huggingface.co")
    checks.add_argument("--no-preflight", action="store_true", help="skip pre-flight checks")
    checks.add_argument(
        "--context-probe-tokens",
        type=int,
        default=DEFAULT_CONTEXT_PROBE_TOKENS,
        metavar="N",
        help="length of the truncation probe; 0 disables",
    )
    output = run.add_argument_group("output")
    output.add_argument("--title", help="scorecard title")
    output.add_argument("--out", default="runs", metavar="DIR", help="output directory")
    output.add_argument("--no-png", action="store_true", help="skip card.png")
    output.add_argument("--open", action="store_true", help="open the scorecard when done")
    output.add_argument("--no-cache", action="store_true", help="always rerun the reference")
    output.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    verdict = run.add_argument_group("verdict")
    verdict.add_argument(
        "--max-size",
        metavar="SIZE",
        help="largest download you can run, e.g. 6GB, 6.5G or 800MB (decimal units); the"
        " pick is the closest download that fits",
    )
    verdict.add_argument(
        "--fail-on",
        choices=tuple(_FAILING_STATUSES),
        default="never",
        help="avoid: exit 3 when any candidate is avoid or failed, or nothing is recommended"
        " and nothing usable fits; inconclusive: also when any candidate is inconclusive or"
        " usable, or nothing is recommended; default: never",
    )
    run.set_defaults(handler=_cmd_run)


def _add_card(
    commands: argparse._SubParsersAction[argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    card = commands.add_parser("card", parents=[common], help="render a saved report")
    card.add_argument("path", metavar="PATH", help="run directory or report.json")
    card.add_argument("--format", choices=("txt", "md", "html", "png"), default="txt")
    card.add_argument("-o", "--output", metavar="FILE", help="write to FILE instead of stdout")
    card.set_defaults(handler=_cmd_card)


def _cmd_run(args: argparse.Namespace) -> int:
    plan = build_plan(
        args.ref,
        args.cand,
        suites=args.suite,
        prompts=args.prompts,
        scoring_prompts=args.scoring_prompts,
        top_k=args.top_k,
        score_tokens=args.score_tokens,
        max_cases=args.max_cases,
        allow_code_exec=args.allow_code_exec,
        seed=args.seed,
        hf_repo=args.hf_repo,
        offline=args.offline,
        preflight=not args.no_preflight,
        context_probe_tokens=args.context_probe_tokens,
        title=args.title,
        max_size=args.max_size,
    )
    progress = None if args.quiet else make_progress(sys.stderr)
    cache = None if args.no_cache else ReferenceCache()
    try:
        report = execute(plan, cache=cache, progress=progress)
    finally:
        if progress is not None:
            progress.close()
    verdict = judge(report, scoring=plan.scoring)
    files = write_run(report, args.out, png=not args.no_png, verdict=verdict)
    print(render_terminal(report, color=_use_color(sys.stdout), verdict=verdict), flush=True)
    _print_saved(files, sys.stderr)
    # Repeated as the last line of stdout because people skim the end of the output.
    print(f"\n{verdict.headline}", flush=True)
    if args.open:
        webbrowser.open(files.html.resolve().as_uri())
    regression = _regression(verdict, args.fail_on, budget=report.settings.max_size_bytes)
    if regression is None:
        return EXIT_OK
    print(f"quantdiff: {regression}", file=sys.stderr)
    return EXIT_REGRESSION


def _cmd_card(args: argparse.Namespace) -> int:
    path = Path(args.path)
    report = load_report(path / "report.json" if path.is_dir() else path)
    if args.format == "png":
        target = Path(args.output or "card.png")
        render_png(render_html(report), target)
        print(f"Saved {target}", file=sys.stderr)
        return EXIT_OK
    if args.format == "html":
        text = render_html(report)
    elif args.format == "md":
        text = render_markdown(report)
    else:
        text = render_terminal(report, color=args.output is None and _use_color(sys.stdout))
    if args.output is None:
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    return EXIT_OK


def _cmd_discover(args: argparse.Namespace) -> int:
    print(format_discovery(discover_ollama(args.host)))
    return EXIT_OK


def _cmd_suites(_: argparse.Namespace) -> int:
    for name in BUILTIN_SUITES:
        print(f"{name:<6} {len(load_builtin(name)):>3} cases  {_SUITE_SUMMARIES[name]}")
    return EXIT_OK


def _regression(verdict: Verdict, level: str, *, budget: int | None) -> str | None:
    """Why `--fail-on level` fails the run, or None when it passes."""
    failing = _FAILING_STATUSES[level]
    found = [
        f"{item.label} is {item.status}" for item in verdict.candidates if item.status in failing
    ]
    if failing and not any(item.status == "recommended" for item in verdict.candidates):
        usable_fits = any(
            item.status == "usable" and (budget is None or item.fits_budget is True)
            for item in verdict.candidates
        )
        if "usable" in failing:
            found.append("no candidate is recommended")
        elif not usable_fits:
            within = "" if budget is None else " within --max-size"
            found.append(f"no candidate is recommended or usable{within}")
    if not found:
        return None
    return f"--fail-on {level}: {', '.join(found)}; exiting with code {EXIT_REGRESSION}"


def _did_you_mean(message: str, args: argparse.Namespace) -> list[str]:
    """Installed Ollama tags close to each tag that `message` says is missing.

    Best effort: when Ollama cannot be listed there are simply no suggestions.
    """
    if args.command != "run":
        return []
    lines = message.lower().splitlines()
    listings: dict[str, list[str]] = {}
    hints = []
    for text in dict.fromkeys([args.ref, *args.cand]):
        try:
            spec = parse_spec(text)
        except SpecError:
            continue
        if spec.kind != "ollama" or not _says_missing(lines, spec.model):
            continue
        if spec.base_url not in listings:
            try:
                listings[spec.base_url] = installed_tags(
                    spec.base_url, timeout=_SUGGESTION_TIMEOUT_SECONDS
                )
            except QuantdiffError:
                listings[spec.base_url] = []
        close = closest_tags(spec.model, listings[spec.base_url])
        if close:
            hints.append(f"{spec.model} is not installed; did you mean: {', '.join(close)}")
    return hints


def _says_missing(lines: Sequence[str], model: str) -> bool:
    quoted = repr(model).lower()
    return any(
        quoted in line and any(words in line for words in _MISSING_MODEL_WORDS) for line in lines
    )


def _print_saved(files: RunFiles, stream: TextIO) -> None:
    lines = []
    if files.png is not None:
        lines.append(f"\nScorecard image to share: {files.png}")
    lines.append(f"\nSaved to {files.directory}")
    if files.png is not None:
        lines.append("  card.png     share on Reddit or X")
    lines.append("  card.md      paste into a Reddit post or GitHub issue")
    lines.append("  card.html    open in a browser")
    lines.append("  report.json  raw results; re-render with `quantdiff card`")
    if files.png_error is not None:
        lines.append(f"No card.png: {files.png_error}")
    print("\n".join(lines), file=stream)


def _error(message: str, *, hints: Sequence[str] = ()) -> None:
    print(f"quantdiff: error: {printable_lines(message)}", file=sys.stderr)
    for hint in hints:
        print(f"hint: {printable_lines(hint)}", file=sys.stderr)


def _csv(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected a comma separated list")
    return names


def _use_color(stream: TextIO) -> bool:
    return stream.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


if __name__ == "__main__":
    raise SystemExit(main())
