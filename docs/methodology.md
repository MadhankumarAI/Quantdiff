# Methodology

This document describes what quantdiff measures, how, and where the measurements are weak. If a
number on a scorecard surprises you, the answer is probably in the Limitations section.

quantdiff compares each candidate C against one reference R. It never compares a candidate against
"the true model"; the reference stands in for it. All decoding is greedy (temperature 0, fixed
seed) so that, as far as the server allows, the same input produces the same output.

There are two tiers of metrics and they answer different questions:

- **Tier 1 (logit)**: how far is C's next-token distribution from R's, one step at a time, on
  identical input?
- **Tier 2 (task)**: when C generates freely through its own chat template, does it still do the
  job?

## Tier 1: teacher-forced logit comparison

### Setup

1. Each scoring prompt is raw text. No chat template is applied, so every backend receives exactly
   the same string.
2. R greedily continues the prompt for `--score-tokens` tokens (default 32). At each step i it
   reports its chosen token y_i (its argmax) and the top-k log-probabilities of its next-token
   distribution P_i (`--top-k`, default 10; servers that speak the OpenAI or Ollama API return at
   most 20).
3. C is teacher-forced on the same sequence: for each position i it is given the prompt followed by
   y_1 ... y_(i-1), the reference's tokens, not its own, and reports the top-k of its distribution
   Q_i.
4. Each position i yields one comparison of P_i against Q_i.

Teacher forcing isolates the per-step distribution shift caused by quantization. Without it, the
first time C picks a different token the two sequences diverge and every later comparison is
between different contexts.

### Top-1 agreement

    top1 = (1 / N) * sum_i [ argmax Q_i == y_i ]

over all N scored positions across all prompts. It answers: given the same context, how often
would greedy decoding of C pick the same token as R? It ignores everything in the
distributions except the head, which is exactly what greedy decoding depends on.

### KL divergence on a partition, and why it is a lower bound

The quantity of interest is the full-vocabulary divergence at each position:

    KL(P || Q) = sum_{t in V} P(t) * log( P(t) / Q(t) )

Servers only return the top k tokens of each distribution, so this sum cannot be computed. Let
T_P and T_Q be the two top-k token sets at a position. Split the vocabulary into three groups:

- S = T_P intersect T_Q, tokens whose probability is known under both distributions;
- A = T_P \ T_Q, tokens R lists but C does not;
- B = everything else.

Partition the vocabulary into the singletons {t} for t in S plus the two cells A and B. On this
partition every mass is known except Q(A):

    P(A) = sum_{t in A} P(t)        P(B) = 1 - sum_{t in S} P(t) - P(A)
    U = A union B                   Q(U) = 1 - sum_{t in S} Q(t)

**Step 1: the partition KL is a lower bound.** Merging outcomes can only lose information, so for
the deterministic map that sends each token to its cell (data processing inequality, or the
log-sum inequality applied to each merged cell)

    KL_part(P || Q) = sum_{t in S} P(t) log( P(t) / Q(t) )
                    + P(A) log( P(A) / Q(A) ) + P(B) log( P(B) / Q(B) )  <=  KL(P || Q)

**Step 2: bound the unknown Q(A).** T_Q holds C's k most likely tokens, so no token outside T_Q can
be more likely than the least likely token inside it. And A lies entirely outside T_Q, so it can
hold at most the mass that all of T_Q leaves over, which is tighter than Q(U) whenever C lists
tokens R does not. Writing q_min = min_{t in T_Q} Q(t):

    0 <= Q(A) <= cap = min( |A| * q_min, 1 - sum_{t in T_Q} Q(t) )

Since T_Q \ S lies in B, this cap also keeps Q(B) = Q(U) - Q(A) at least the mass C reports
for those tokens.

**Step 3: minimize over what is not known.** With Q(B) = Q(U) - Q(A), the A and B terms form a
convex function of x = Q(A). Its unconstrained minimum is at x* = Q(U) * P(A) / (P(A) + P(B)), the
split proportional to P. quantdiff reports KL_part evaluated at

    Q(A) = min( x*, cap )

Nothing else the server reported constrains Q(A): the vocabulary size is unknown, so the tokens of
A may hold arbitrarily little mass. This is therefore the smallest partition KL consistent with
everything the server reported. It is at most KL_part at the true Q(A), and by step 1 at most the
true KL:

**Claim:** the reported value is a lower bound on KL(P || Q) at every position. []

When the cap does not bind, the A and B cells contribute exactly as if they were merged into one
remainder bucket, which is the plain two-cell bound. The cap binds when R puts substantial mass on
tokens that C did not even rank in its top k. Without it, two top-k lists that share no tokens
would score a KL of 0 however different they are. With it, a candidate that drops the reference's
likely tokens from its head is penalized, and the bound stays rigorous. The test suite checks the
bound against the exact full-vocabulary KL on randomly generated distributions.

What remains loose: probability mass that C places on tokens R does not list (C's head, R's tail)
is only seen through the B cell, so a candidate that is confidently wrong in that direction is
under-penalized by KL and caught by top-1 agreement instead. Read the two metrics together.

**End of sequence.** When C's most likely next token is end-of-sequence, Ollama and
OpenAI-compatible servers stop and return no distribution for that position. quantdiff counts the
position as a top-1 disagreement (R continued, C would have stopped) and leaves it out of the KL
statistics, which need a distribution. KL aggregates can therefore cover fewer positions than
top-1 agreement does.

Reported aggregates are the mean, 99th percentile (nearest rank), and maximum over positions.
Rounded logprobs can make a remainder mass slightly negative; masses are clamped to a small
positive floor before taking logs. C's remainder masses are also padded by the rounding error of
the sum they come from (one machine epsilon per listed probability). When the cap binds at a tiny
true mass, a few units of rounding error would otherwise be a large relative error and could
push the result above the exact KL. Extra mass for C can only lower the result.

The KL values are only as good as the logprobs the server reports. quantdiff assumes they are
log-softmax values over the full vocabulary. A server that reports probabilities renormalized
after its own top-k or top-p truncation would distort both P(B) and Q(B). quantdiff requests
greedy decoding without truncating samplers wherever the API allows it, but it cannot verify what
the server does internally.

### Exact versus approximate teacher forcing

**llama-server (exact).** quantdiff sends token ids: the reference tokenization of the prompt plus
each y_i's token id. C sees exactly the reference's token sequence. This requires that R and C
share a tokenizer, which the tokenizer pre-flight check verifies.

**Ollama and OpenAI-compatible servers (approximate).** These APIs accept text only. quantdiff
concatenates the prompt and the reference's token strings and sends the text; the server
re-tokenizes it. Usually that reproduces the same tokens. Sometimes it does not: the pieces " un"
and "able" emitted as two greedy steps may be re-tokenized as one token " unable", and the
positions after that point no longer line up one to one. The effect is mostly a small amount of
extra disagreement concentrated at merge points, but it is not bounded in a useful way, so the card
labels these results **approximate**. Compare approximate numbers with each other, not with exact
ones.

### What Tier 1 misses

Teacher-forced metrics measure one step at a time on the reference's text. They cannot see:

- **Generation drift.** In real use C conditions on its own outputs. A small per-step shift that
  flips one token early changes everything after it. Errors compound over long outputs in a way
  that per-position averages do not show.
- **Rare but decisive tokens.** A quant can be very close on average and still be wrong on the one
  token that matters: a closing brace, a tool name, a digit. Mean KL dilutes these among thousands
  of easy positions. The p99 and max help, but only if such tokens appear in the scoring prompts.
- **The chat template.** Scoring prompts are raw text, so a broken or stale template has no effect
  on Tier 1. It has a large effect on actual chat use.
- **Server configuration.** Truncated context, a different default system prompt, or other
  serving differences mostly affect the chat path.

Tier 2 exists to cover these.

## Tier 2: task suites

Each task case is a chat conversation sent through the server's own chat API (and so through the
server's chat template), generated greedily, and checked:

| Suite | Pass condition |
| --- | --- |
| json | Output parses as JSON and validates against the case's schema. |
| tools | The model calls the expected tool, its arguments parse and validate against the tool's parameter schema, and every key in `expected_arguments` is present with an equal value. |
| code | The generated function passes the case's hidden asserts. Runs only with `--allow-code-exec`, in a subprocess with a scrubbed environment, a timeout, and POSIX resource limits. Otherwise the cases are reported as skipped. |
| chat | No ground truth. Scored by agreement with the reference model's answer to the same prompt (exact match rate and mean similarity). |

json, tools, and code have deterministic pass conditions that do not depend on the reference, so
they also catch cases where the reference itself fails. chat measures closeness to the reference
only: it says nothing about whether either answer is correct.

Tier 2 works on every backend, needs no logprobs, and measures what users experience: free-running
generation through the real serving path. Its weakness is sample size. A suite of a few dozen cases
can only detect large differences; see how the verdict is decided, below.

Throughput (tokens per second) is reported alongside. It depends on hardware, server flags, and
concurrent load, and is only comparable between candidates measured on the same machine in the
same run.

## Pre-flight checks

Pre-flight checks exist to keep the comparison fair. A candidate that is served wrongly will score
badly for reasons that have nothing to do with its weights.

- **Context truncation probe.** A long prompt (default `--context-probe-tokens 6000`) with a
  needle at the start and a question about it at the end, plus a short control prompt with the
  same needle and question. Passing the control but failing the long prompt indicates the server
  dropped the start of the context (for example, a context window configured smaller than the
  prompt). Failing both suggests the model cannot do the task at all, and the probe is
  inconclusive.
- **Chat template (llama-server only).** The template embedded in the GGUF, as reported by the
  server, is compared with `chat_template` from the upstream repo's `tokenizer_config.json` on
  Hugging Face (`--hf-repo`). A mismatch is reported as a warning, since some differences are
  deliberate fixes by the uploader, with the diff summarized.
- **Tokenizer match.** Where backends can tokenize, a fixed set of strings is tokenized by R and C.
  Any difference means token positions are not comparable and Tier 1 is withheld for that
  candidate.
- **Logprob availability.** If a server returns no logprobs, Tier 1 is skipped for it.

## How the verdict is decided

The verdict answers one question: which download should I run? It is computed by fixed rules
from the report, so the same report always gives the same verdict. The rules live in
`quantdiff/verdict.py`; this section explains them.

Two principles shape the rules:

- **Every candidate is judged against the reference on its own**, never against the other
  candidates. Adding or removing a candidate never changes another candidate's status.
- **A recommendation needs positive evidence.** A candidate is only recommended when the data
  shows it is close to the reference. Failing to find a loss is not the same thing: with a few
  prompts and a dozen cases, nothing can be proven either way, and the verdict says so instead
  of picking the smallest file.

### Every comparison is paired

Every model answers the same task cases and is scored on the same prompts. quantdiff uses that:
it joins task outcomes by case id and logit results by prompt id, and compares the candidate
with the reference item by item. Only items scored on both sides count.

Pairing matters because cases differ far more from each other than quants do. A hard case is
hard for every quant. An unpaired test sees that case-to-case spread as noise and needs hundreds
of cases to see a difference; a paired test cancels it and looks only at the cases where the two
models disagree.

### The measurements

- **Per-suite task deltas.** For json, tools, and code, quantdiff reports the candidate's pass
  rate minus the reference's, in percentage points, with a 95% interval from Newcombe's hybrid
  score method for paired proportions (method 10 in Newcombe, Statistics in Medicine 17:2635,
  1998). The difference is **significant** when the exact McNemar test gives p < 0.05. McNemar
  looks only at discordant cases: b cases the reference passed and the candidate failed, c the
  other way round. Under no difference each discordant case is a coin flip, so the p-value is
  the two-sided binomial tail of min(b, c) in b + c trials.
- **Reliable suites.** A suite where the reference passes under half of the cases is shown but
  never used to judge a candidate: a reference that fails most cases says little about its
  quants.
- **Pooled task comparison.** To decide whether a candidate is close on tasks, the paired
  outcomes of every reliable suite are pooled into one comparison, with the same Newcombe
  interval. Pooling gives one interval over all cases instead of several wide ones.
- **KLD interval.** The mean of the per-prompt mean KLD, with a 95% percentile bootstrap
  interval: prompts are resampled with replacement 2000 times (fixed seed, so the interval is
  reproducible). Prompts, not positions, are the unit: positions inside one continuation are
  strongly correlated, and treating them as independent would make every interval far too
  narrow.

### Thresholds

| Name | Value | Used for |
| --- | --- | --- |
| Closeness bar (`CLOSE_KLD`) | 0.04 at top-10 | close on logits when the KLD interval ends below it; usable at best when the interval starts above it |
| Large KLD (`LARGE_KLD`) | 0.10 at top-10 | avoid when the mean KLD and the lower end of its interval are both at or above this |
| Minimum prompts (`MIN_PROMPTS`) | 8 | fewer paired prompts never make a candidate close, usable or avoid on logits |
| Minimum cases (`MIN_CASES`) | 20 | fewer pooled paired cases never prove a candidate close on tasks |
| Task margin (`TASK_MARGIN`) | 10 points | close on tasks when the pooled interval's lower end is at or above -10 points; a suite estimate below -10 that is not significant is an unresolved caveat |
| Near a bar (`NEAR_BAR`) | 10% | a KLD interval that straddles a bar or ends within 10% of it is flagged as near it |

The KLD bands shown on the card use the same edges (values for the default `--top-k 10`):

| Band | quantdiff KLD (top-10) | llama.cpp full-vocabulary KLD |
| --- | --- | --- |
| near-lossless | under 0.01 | under 0.015 |
| small | 0.01 to under 0.04 | 0.015 to under 0.06 |
| moderate | 0.04 to under 0.10 | 0.06 to under 0.15 |
| large | 0.10 and above | 0.15 and above |

The bars are set on llama.cpp's full-vocabulary scale, the numbers people know from
`llama-perplexity --kl-divergence`: 0.06 is about a Q4_K_M of a 7 to 8B model, and 0.15 is Q2_K
territory. quantdiff measures a top-k lower bound, which reads a stable fraction of the full
value. [docs/calibration.md](calibration.md) measured that fraction on identical positions
against llama.cpp: 0.24 at k=1, 0.57 at k=5, 0.67 at k=10 and 0.76 at k=20. The bars are the
full-vocabulary values times that fraction, so `--top-k 20` raises the closeness bar to 0.046,
and the card always prints the bars that applied to its run.

Two caveats remain. The fraction was measured on one small model family (Qwen2.5 0.5B) and may
shift somewhat on other models and text domains. And KLD is measured against your reference: if
that is Q8_0 rather than BF16, every candidate looks somewhat closer than it is to the original
weights.

### Status rules

Each candidate gets one status. The rules are applied in order. "The KLD interval" is the 95%
bootstrap interval of the mean per-prompt KLD.

1. **failed**: the candidate produced no metrics at all.
2. **avoid**, if any of these holds:
   - a reliable suite shows a significant regression (negative delta, McNemar p < 0.05);
   - the reliable suites pooled into one comparison show a significant regression;
   - with at least 8 paired prompts, the mean KLD is at or above the large bar (0.10 at
     top-10) and so is the lower end of its interval, for example "KLD is large (0.214, 95% CI
     0.179 to 0.251)".
3. **close**, only on positive evidence:
   - with logit metrics, closeness rests on them: at least 8 paired prompts, and the upper
     end of the KLD interval below the closeness bar (0.04 at top-10). Task suites then act
     as a breakage detector: rule 2 already turned any significant loss into avoid, and a
     wide but not significant pooled interval is reported ("no significant task loss vs q8_0
     on 24 cases (95% CI -21 to +12)") without counting against the candidate. The reason is
     statistical power: with a few dozen cases the pass-rate interval is 20 to 30 points
     wide, so requiring it to exclude a 10-point loss would make the verdict depend on suite
     size, not on the model. A mean per-token KL divergence shown below the bar is itself a
     strong statement about closeness: by Pinsker's inequality the per-token total variation
     distance is at most sqrt(KL / 2), about 0.17 at a full-vocabulary KLD of 0.06 and lower
     for typical Q4 values.
   - without logit metrics, closeness rests on tasks: at least 20 pooled paired cases from
     reliable suites, and the lower end of the pooled interval at or above -10 points.

   The headline says which evidence the call rests on ("close on logits (KLD 0.03, CI up to
   0.036) on 41 prompts", "close on task scores on 88 cases (no logit metrics)"). A candidate
   with neither kind (for example, only chat agreement) is never close.
4. **usable**: with at least 8 paired prompts, the lower end of the KLD interval is above the
   closeness bar and the mean is below the large bar, and no task suite shows a significant
   loss. This is a measured, moderate loss ("moderate loss: KLD 0.044 (95% CI 0.04 to
   0.049)"). The closeness bar sits about where a Q4_K_M of a 7 to 8B model lands, so the most
   popular quant can land here; usable says it is a sound choice when nothing closer fits.
5. **inconclusive**: everything else. That includes a KLD interval that straddles the
   closeness bar, a mean at or above the large bar whose interval reaches below it ("may be a
   large loss"), and a large-looking mean on fewer than 8 prompts ("looks large on 3 prompts,
   too few to call it").
6. **The pick.** With `--max-size`, only downloads at most that large count; a candidate whose
   size is unknown is never assumed to fit. Among the close candidates that fit (every close
   candidate without a budget), **recommended** is the smallest download, by size on disk
   when each of them reports one, otherwise the one with the lowest KLD. The other close
   candidates are **ok** ("also close to q8_0 but 17% larger", or "q5_K_M is close but needs
   7.1 GB" when it is over the budget). A candidate that is not close is never recommended or
   ok.

Two caveats sit next to a status without changing it:

- **Near a bar.** When the KLD interval straddles the closeness bar or the large bar, or its
  nearer end is within 10% of that bar, the candidate is flagged "near the closeness bar; a
  rerun could change this".
- **Unresolved task loss.** When a reliable suite's estimate is more than 10 points below the
  reference but the loss is not significant and its interval still reaches zero, the
  candidate carries "tools -17 unresolved (95% CI -42 to +6); rerun with --max-cases 60". The
  number is the case count at which the observed loss would become significant, with the
  observed outcome shares held fixed, rounded up to a multiple of 10; when the built-in suite
  is too small it says how many cases would settle it instead. The first such caveat (the
  headline candidate's own when it has one) is also a line of the verdict.

The headline follows from the statuses:

| Situation | Headline |
| --- | --- |
| A candidate is recommended | "Run q4_K_M: 25% smaller than q8_0, close on logits (KLD 0.03, CI up to 0.036) on 41 prompts." |
| Budget, no close candidate fits, a usable one does | "Best that fits 6 GB: q4_K_M, moderate loss (KLD 0.044)." The usable candidate keeps its status; the details name the close download that does not fit. |
| Budget, nothing close or usable fits | "Nothing that fits 4 GB is close to q8_0; q5_K_M is close but needs 7.1 GB." |
| No budget, nothing close, a usable candidate | "No download is close to q8_0; q4_K_M has the smallest loss among the smaller downloads (moderate, KLD 0.044)." It says "shown to be close" when an inconclusive candidate could still prove close. |
| Nothing close or usable, something inconclusive | "Keep q8_0 for now: no candidate is shown to be close on 12 prompts and 24 cases." |
| Every candidate is avoid | "Keep q8_0: every candidate shows a measured loss." |

When nothing is recommended there is always a next step, kept apart from the details. In
order: a rerun that would likely decide an inconclusive candidate ("Rerun with --max-cases 40
to decide."); confirming with llama-server when a text-forced candidate reads above the
closeness bar on mostly non-Latin prompts; passing `--max-size` when usable candidates exist
and no budget was given; rerunning with a `--max-size` large enough for a close download that
did not fit; confirming with llama-server when a text-forced candidate reads above the
closeness bar; otherwise what evidence is missing, or trying a larger quant.

Text forcing (Ollama and OpenAI-compatible servers) re-tokenizes the reference's text, which is
least faithful for scripts that byte-level tokenizers split into many pieces. When more than
half of the letters in the scoring prompts are outside the Latin script and a text-forced
candidate reads above the closeness bar, the details add "Most scoring prompts are not Latin
script; text forcing may read higher there. Confirm with llama-server (exact token ids) before
ruling out a download." The prompt texts are not stored in report.json, so `quantdiff card`
cannot add this note when it re-renders a saved run.

Candidates are ranked by status (recommended, ok, usable, inconclusive, avoid, failed), then by
mean KLD, then by size. Pass rate is not a ranking key: it is the noisiest number on the card,
and a one-case swing in a small suite should not reorder the ranking.

### What "inconclusive" means

Inconclusive means the run could not decide: the candidate is not shown close, usable or a
large loss. It does **not** mean the candidate is safe. The reasons say what is
missing, for example "KLD 0.031, but 3 prompts cannot bound it below 0.04", and estimate how
much more evidence would decide:

- **Prompts.** The KLD interval's upper half-width shrinks roughly with the square root of the
  prompt count. Keeping the observed mean and spread, the estimate is the prompt count n' at
  which mean + half-width * sqrt(n / n') falls below the closeness bar, and at least 8. When the
  mean is at or above the closeness bar, more prompts cannot make the candidate close; the
  estimate is then the count at which the interval's lower end would clear that bar (or the
  large bar, when the mean is at or above it), which decides between usable and avoid.
- **Cases.** Keeping the observed shares of the four paired outcomes (both pass, only the
  reference passes, only the candidate passes, neither), the estimate is the smallest pooled
  case count at which the Newcombe interval's lower end reaches -10 points, spread evenly over
  the reliable suites. When the observed difference is already 10 points or more below the
  reference, more cases cannot make the candidate close, and no estimate is given.

Both estimates are rounded up to a multiple of 10. When the built-in suites and scoring prompts
hold enough, the remedy is "Rerun with --max-cases N to decide." (`--max-cases` caps each suite
and the scoring prompts at N); otherwise it asks for your own prompts. Treat the number as an order of magnitude.

The case estimate only applies when a candidate has no logit metrics. With the default
settings (34 json and 32 tools cases), the pooled comparison has 66 cases, which cannot prove a
small task difference on its own; servers that return logprobs avoid that limit because KLD
carries the closeness proof.

### Never better than the reference

A quantized model is not expected to beat the weights it was made from. A positive delta that
is not significant is shown as "= reference (within noise)". A significant positive delta is
reported as a warning, not a win: it usually means the suite is too small, or that the
reference is itself quantized and lost something the candidate happened to keep.

### Caveats

- The cases are not a random sample of your workload. "Close" means "within these margins on
  these cases", not "will generalize".
- Each candidate is tested against the reference separately, with no correction for multiple
  comparisons. With many suites, expect an occasional false alarm at the 5% level, and add
  cases before acting on a marginal result.
- Servers are not always deterministic at temperature 0. Batched inference (llama-server with
  several slots, vLLM) can change floating point reduction order between requests. Run twice if
  a result is close.
- Reports written before schema version 2 have no per-case or per-prompt results. They still
  load and render, but nothing can be proven close from them, so no candidate is recommended.

## Choosing the reference

The ideal reference is the original BF16 (or F16) weights served by the same engine as the
candidates. When that does not fit in memory, Q8_0 is a common proxy: its own divergence from BF16
is small compared with 4-bit and lower quants.

Using Q8_0 has a bias you should know about. quantdiff then measures distance to Q8_0, not to BF16.

- Every candidate looks somewhat closer to the reference than it is to the original weights,
  because part of its error is shared with Q8_0 (both are rounded from the same weights, often
  with the same block layout).
- Differences between candidates that are smaller than Q8_0's own divergence from BF16 are not
  meaningful.
- A candidate that happens to be closer to BF16 in some direction than Q8_0 is can be scored as
  further away.

If you can run BF16 once, measure Q8_0 against it (with quantdiff or `llama-perplexity`, below)
to know the size of that floor.

The reference also must be the same model as the candidates: same base weights, same tokenizer.
Comparing different fine-tunes is outside what the metrics are designed for.

## Cross-checking with llama-perplexity

llama.cpp computes full-vocabulary KL divergence from saved base logits:

```
# 1. Save the reference's logits over a text file
llama-perplexity -m model-BF16.gguf -f sample.txt --kl-divergence-base base.kld

# 2. Score each candidate against them
llama-perplexity -m model-Q4_K_M.gguf --kl-divergence-base base.kld --kl-divergence
```

It reports mean KLD, KLD percentiles, and "Same top p" (top-1 agreement), over full context
windows of the text, scoring the second half of each chunk.

Expected relationships when you run both on comparable text:

- quantdiff's KLD should be at or below `llama-perplexity`'s, since it is a lower bound. If it is
  higher by more than noise, something is wrong (different references, mismatched tokenizers, or
  renormalized logprobs). Please open an issue with both outputs.
- Top-1 agreement should be close to "Same top p", though context lengths and text differ.
- The ranking of candidates should usually agree. Where it does not, the two tools are measuring
  different text, and quantdiff's Tier 2 results are the better guide to your workload.

`llama-perplexity` only covers GGUF in llama.cpp. quantdiff's value is extending a comparable
measurement to the servers people actually use, plus the task suites and pre-flight checks.

## Limitations

- KLD is a lower bound and can be very loose at positions where the top-k lists barely overlap.
- Logit metrics through Ollama and OpenAI-compatible servers are approximate because of
  re-tokenization.
- At most 20 logprobs per position are available from Ollama and OpenAI-compatible APIs.
- Everything is greedy. If you sample at temperature 0.7, the tail of the distribution matters
  more than these metrics reflect.
- Built-in suites are small. They detect broken quants and large regressions, not fine
  differences. Your own prompts, in quantity, are the real test.
- The chat suite measures agreement with the reference, not correctness.
- Results describe a model as served: the server, its version, its flags (context size, KV cache
  type, batch settings) are all part of what is measured. That is intended, but it means a result
  does not transfer to another server configuration.
- `--allow-code-exec` isolation is not a sandbox. See SECURITY.md.
