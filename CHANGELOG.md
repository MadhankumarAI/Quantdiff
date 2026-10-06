# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public release.

### Added

- `quantdiff run`: compare one or more candidate models against a reference model.
- Backends: Ollama (`ollama:<tag>`), llama-server (`llamacpp:<base_url>`), and OpenAI-compatible
  servers (`openai:<base_url>#<model>`, with optional `@env:VAR` API key and `label=` prefix).
- Tier 1 logit metrics: teacher-forced top-1 agreement and partition KL divergence (a lower bound
  on full-vocabulary KL). Exact token ids on llama-server; text-based and labeled approximate on
  Ollama and OpenAI-compatible servers.
- Tier 2 task suites: `json`, `tools`, `code` (only with `--allow-code-exec`), and `chat`, plus
  tokens per second.
- User prompts via `--prompts` (full task cases or `{"prompt": ...}` shorthand) and raw scoring
  prompts via `--scoring-prompts`.
- Pre-flight checks: context truncation probe, embedded chat template versus upstream Hugging Face
  `tokenizer_config.json` (llama-server), tokenizer match, and logprob availability.
- Verdict line with a pooled two-proportion noise check (95%, at least 5 samples per side).
- Scorecards as terminal text, Markdown, and self-contained HTML (no JavaScript); `quantdiff card`
  re-renders from a saved run.
- Reference result cache in the user cache directory, keyed on the weights fingerprint, chat
  template and server URL.
- `card.png` rendered through an installed Chromium-based browser, and `quantdiff card --format png`.
- `quantdiff discover`: groups local Ollama downloads by model and prints a ready-to-run command.
- Progress display with an ETA, a connectivity check before any work starts, and actionable
  errors for unreachable servers and missing models.
- Reference row with signed deltas on every card; shared model prefixes shown once.
- Python API: `quantdiff.compare`, `quantdiff.verdict`, `quantdiff.render_html`.
- Zero runtime dependencies.

[0.1.0]: https://github.com/quantdiff/quantdiff/releases/tag/v0.1.0
