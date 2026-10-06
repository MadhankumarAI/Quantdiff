"""Plain data types shared by every quantdiff module.

All types are frozen dataclasses so results can be cached and serialized without
defensive copies. Nothing here performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JSONValue = Any
"""A value produced by json.loads. Kept as Any because JSON is recursive."""

BackendKind = Literal["ollama", "llamacpp", "openai"]
TaskKind = Literal["json", "tools", "code", "chat"]
Role = Literal["system", "user", "assistant", "tool"]
Severity = Literal["ok", "warn", "fail", "skip"]
TemplateDialect = Literal["jinja", "go", "unknown"]
PerfSource = Literal["server", "wall_clock", "cached"]
ProgressPhase = Literal["connect", "reference", "preflight", "cases", "scoring", "done"]


# Model servers ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Where a model is served and how to label it on the scorecard.

    `model` is the Ollama tag or the served model name. It is empty for llama-server,
    which serves exactly one model. `api_key_env` names an environment variable that
    holds the key; the key itself never lives in a spec, a report, or a log line.
    """

    kind: BackendKind
    base_url: str
    model: str
    label: str
    api_key_env: str | None = None


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """What a backend reports about the model it serves."""

    backend: BackendKind
    model: str
    context_length: int | None
    chat_template: str | None
    template_dialect: TemplateDialect
    supports_logprobs: bool
    exact_token_ids: bool
    """True when the backend scores by token id, so teacher forcing never retokenizes."""
    details: tuple[tuple[str, str], ...] = ()
    """Extra key/value facts worth showing, such as quantization type or file size."""
    size_bytes: int | None = None
    """Size of the served weights on disk, when the server reports it."""
    weights_id: str | None = None
    """Stable identity of the served weights (Ollama digest, GGUF size and params), used to
    spot a candidate that is the same file as the reference."""


# Tokens and distributions ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenProb:
    """One token and its natural-log probability."""

    token: str
    logprob: float
    token_id: int | None = None
    token_bytes: bytes | None = None
    """The token's raw bytes when the server reports them. A token can hold part of a
    multi-byte character, so text rebuilt from `token` strings can be corrupted while text
    rebuilt from bytes is exact."""


@dataclass(frozen=True, slots=True)
class TokenStep:
    """One generated position: the token chosen greedily and the top-k alternatives.

    `top` is sorted by descending logprob and includes the chosen token when it is in
    the top k.
    """

    chosen: TokenProb
    top: tuple[TokenProb, ...]


TopK = tuple[TokenProb, ...]
"""A next-token distribution truncated to the k most likely tokens, sorted descending."""


@dataclass(frozen=True, slots=True)
class ScoringPrompt:
    """A raw-completion prompt used for logit metrics. No chat template is applied."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReferenceTrace:
    """The reference model's greedy continuation of one scoring prompt."""

    prompt_id: str
    prompt_token_ids: tuple[int, ...] | None
    steps: tuple[TokenStep, ...]


# Chat and task cases ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A function tool in OpenAI format. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, JSONValue] | None
    """Parsed arguments, or None when `raw_arguments` was not a JSON object."""
    raw_arguments: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    decode_tokens_per_second: float | None = None
    """Generation speed as measured by the server itself (excludes load and prompt time)."""


@dataclass(frozen=True, slots=True)
class TaskCase:
    """One chat prompt plus the checks that decide whether an answer passes.

    Which optional fields are required depends on `kind`:
    json needs `json_schema`; tools needs `tools` and `expected_tool`;
    code needs `entry_point` and `tests`; chat needs nothing and is scored by
    agreement with the reference answer only.
    """

    id: str
    kind: TaskKind
    messages: tuple[Message, ...]
    max_tokens: int = 512
    json_schema: dict[str, JSONValue] | None = None
    tools: tuple[ToolSpec, ...] = ()
    expected_tool: str | None = None
    expected_arguments: dict[str, JSONValue] | None = None
    """Subset match: every key here must be present in the call with an equal value."""
    entry_point: str | None = None
    tests: str | None = None
    """Python source with plain assert statements that exercise `entry_point`."""


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    kind: TaskKind
    passed: bool | None
    """None means the case was skipped, for example code cases without --allow-code-exec."""
    reason: str = ""


# Metrics and report ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogitMetrics:
    """Teacher-forced comparison against the reference distribution.

    `kld_*` values are lower bounds on the true KL(reference || candidate): they are
    computed on the partition {top-k tokens, everything else}, which can only
    shrink KL divergence. See docs/methodology.md.
    """

    prompts: int
    positions: int
    top1_agreement: float
    kld_mean: float
    kld_p99: float
    kld_max: float
    exact_token_ids: bool
    per_prompt: tuple[PromptLogit, ...] = ()
    """Per scoring prompt results, in trace order, so comparisons can be paired by prompt."""


@dataclass(frozen=True, slots=True)
class PromptLogit:
    """Logit metrics for one scoring prompt. `kld_mean` is None when no position had a
    candidate distribution (the candidate stopped immediately)."""

    prompt_id: str
    positions: int
    top1_matches: int
    kld_mean: float | None


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    kind: TaskKind
    total: int
    passed: int
    skipped: int
    failures: tuple[CaseOutcome, ...] = ()

    @property
    def rate(self) -> float | None:
        scored = self.total - self.skipped
        return None if scored == 0 else self.passed / scored


@dataclass(frozen=True, slots=True)
class AgreementMetrics:
    """How often the candidate's chat answers match the reference's answers."""

    cases: int
    exact_match_rate: float
    mean_similarity: float
    per_case: tuple[tuple[str, float], ...] = ()
    """(case id, similarity) pairs, so agreement can be compared case by case."""


@dataclass(frozen=True, slots=True)
class PerfMetrics:
    tokens_per_second: float | None
    mean_latency_seconds: float | None
    source: PerfSource = "wall_clock"
    """Where tokens_per_second came from: the server's own decode timing (comparable across
    models), wall-clock request time (includes prompt processing), or a cached earlier run
    (not comparable with fresh numbers)."""


@dataclass(frozen=True, slots=True)
class PreflightFinding:
    check: str
    severity: Severity
    message: str
    fix: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateResult:
    spec: CandidateSpec
    info: ServerInfo | None
    logit: LogitMetrics | None
    tasks: tuple[TaskMetrics, ...]
    agreement: AgreementMetrics | None
    perf: PerfMetrics | None
    preflight: tuple[PreflightFinding, ...]
    errors: tuple[str, ...] = ()
    outcomes: tuple[CaseOutcome, ...] = ()
    """Every scored case outcome (json, tools, code), so candidates can be compared to the
    reference case by case."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    suites: tuple[str, ...]
    top_k: int
    score_tokens: int
    allow_code_exec: bool
    seed: int
    prompts_file: str | None = None
    """Base name of the user's prompts file. Never a full path: cards are shared publicly."""
    max_size_bytes: int | None = None
    """The user's --max-size budget: the pick must be a download at most this large."""
    longest_prompt_tokens: int | None = None
    """Estimated length of the longest task or scoring prompt, to judge whether a context
    limit found by pre-flight could have affected these scores."""


@dataclass(frozen=True, slots=True)
class Report:
    quantdiff_version: str
    created_at: str
    """UTC timestamp in ISO 8601 format."""
    title: str
    settings: RunSettings
    reference: CandidateResult
    candidates: tuple[CandidateResult, ...]
    schema_version: int = 2
    notes: tuple[str, ...] = field(default_factory=tuple)


# Progress ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One step of a run, for progress displays.

    `completed` and `total` count weighted work units across the whole run, roughly one per
    server request, so a display can show an overall bar and an ETA. `total` is fixed for
    the run; `completed` only grows and reaches `total` on the final "done" event.
    """

    phase: ProgressPhase
    model: str
    """Label of the model being worked on; empty for run-level events."""
    completed: int
    total: int
    detail: str = ""
