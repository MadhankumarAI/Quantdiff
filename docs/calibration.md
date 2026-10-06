# Calibration of quantdiff KLD against llama.cpp full-vocabulary KLD

This document measures how quantdiff's Tier 1 KLD (a lower bound computed from top-k
logprobs, teacher-forced through Ollama by text) relates to the full-vocabulary KLD that
`llama-perplexity --kl-divergence` reports on the same weights, and recommends values for the
verdict thresholds `NEAR_LOSSLESS_KLD`, `CLOSE_KLD` and `LARGE_KLD` in `src/quantdiff/verdict.py`.

Short version:

- On identical positions and identical distributions, quantdiff's top-10 bound is **0.67 to
  0.68 times** the full-vocabulary KLD, and the top-20 bound is **0.76 times**. The ratio was the
  same for Q4_K_M and Q2_K and for both llama.cpp evaluation paths, so it behaves like a constant
  scale factor rather than a model-dependent distortion.
- Compared with `llama-perplexity` run on the same prompt texts (not the same positions), quantdiff
  top-10 through Ollama reads 0.74 to 0.82 of the llama.cpp number for Q4_K_M and 0.46 to 0.47 for
  Q2_K. Part of that gap is that quantdiff scores the reference's greedy continuation, which is
  easier text than the prompts themselves.
- The self-comparison floor is about 0.00003 (Ollama, text forcing) and exactly 0 (llama-server,
  token ids). It is negligible next to Q4_K_M (about 0.03).
- Two noise sources that are not negligible were found: Ollama results depend on the runner's load
  history (about a 10% swing in Q4_K_M's mean KLD between otherwise identical runs), and text
  forcing badly inflates KLD on non-Latin text (Hindi prompts read about 3 times their exact value).
- Recommendation, in quantdiff's own scale at `--top-k 10`: `NEAR_LOSSLESS_KLD = 0.01` (keep),
  `CLOSE_KLD = 0.04` (was 0.05), `LARGE_KLD = 0.10` (was 0.15). These correspond to full-vocabulary
  KLD of about 0.015, 0.06 and 0.15. The reasoning is in the last section.

## Setup

| Item | Value |
| --- | --- |
| Models | `qwen2.5:0.5b-instruct-q8_0`, `-q4_K_M`, `-q2_K` from the Ollama library |
| GGUF files | the Ollama blobs, used directly by llama.cpp: q8_0 `sha256-48296496...`, q4_K_M `sha256-c5396e06...`, q2_K `sha256-b143160a...` |
| Base for KLD | q8_0 (no BF16 GGUF of this model was available locally) |
| Ollama | 0.35.1, GTX 1650 4 GB, all three models 100% GPU, num_ctx 4096 |
| llama.cpp | release b11425 (commit e117148a4), `llama-b11425-bin-win-cpu-x64.zip`, CPU only (AVX2) |
| Download check | SHA256 `61ad965cd5e17f5026a05b7258d346cb309e7752490c7e11fbedf332e2040db5`, equal to the `digest` published for the asset in the GitHub release API |
| quantdiff | 0.1.0 working tree as of 2026-10-05; `metrics/logit.py` and `backends/{ollama,llamacpp,_common}.py` identical to the measured snapshot |
| Python | 3.11.0 (project venv) |
| Prompts | the 41 built-in scoring prompts, `src/quantdiff/suites/data/scoring.jsonl` |

The working tree was being refactored while this study ran (`verdict.py` briefly imported names
that `stats.py` did not define yet), so the measurements drive quantdiff's own backend and metric
code directly from a copy of the tree instead of through `quantdiff run`. The calls are the same
ones `runner._reference_traces` and `runner._teacher_force` make: `generate_scored(prompt,
max_tokens=32, top_k=k)` on the reference, then `score_continuation(...)` on each candidate, then
`metrics.logit.logit_metrics`. The scripts are in the scratch directory listed at the end.

## Method

### 1. llama.cpp full-vocabulary KLD on the prompt texts

The 41 scoring prompts were joined with blank lines into one file (15,183 characters, about 4,300
tokens). Commands, with `<q8_0>` etc. standing for the blob paths:

```
llama-perplexity -m <q8_0> -f calib.txt -c 512 -t 8 --kl-divergence-base base_c512.kld
llama-perplexity -m <q4_K_M> -c 512 -t 8 --kl-divergence-base base_c512.kld --kl-divergence
llama-perplexity -m <q2_K>   -c 512 -t 8 --kl-divergence-base base_c512.kld --kl-divergence
llama-perplexity -m <q8_0>   -c 512 -t 8 --kl-divergence-base base_c512.kld --kl-divergence
```

and the same with `-c 256`. With `-c 512` this scores 8 chunks of 256 tokens (2,048 positions);
with `-c 256`, 16 chunks of 128 tokens (2,048 positions).

### 2. quantdiff through Ollama (approximate, text forcing)

Reference q8_0, candidates q8_0 (self), q4_K_M, q2_K, 32 scored tokens per prompt, 41 prompts,
1,263 positions. Run at `top_k=10` and at `top_k=20`. The equivalent CLI invocation is:

```
quantdiff run --ref ollama:qwen2.5:0.5b-instruct-q8_0 \
  --cand q8_0self=ollama:qwen2.5:0.5b-instruct-q8_0 \
  --cand ollama:qwen2.5:0.5b-instruct-q4_K_M --cand ollama:qwen2.5:0.5b-instruct-q2_K \
  --score-tokens 32 --top-k 10 --no-preflight --no-png --no-cache
```

The self-comparison was repeated four more times to measure its spread.

### 3. quantdiff through llama-server (exact, token ids)

Three `llama-server` processes from the same b11425 build, one per GGUF (`-c 2048 -np 1 -t 4`),
were scored with quantdiff's `llamacpp` backend, which teacher-forces by token id. The two Hindi
prompts (score-037, score-038) had to be left out; see Findings. 39 prompts, 1,219 positions.

### 4. Full-vocabulary KLD on exactly the positions quantdiff scores

To separate "top-k truncation" from "different text", every position of the exact traces from step
3 was sent again to all three servers with `n_probs = 2000`. For each position this gives the
full-vocabulary KLD (KL over the reference's top 2,000 tokens, which hold a median 99.97% of the
mass, computed with quantdiff's own `partition_kld` by token id) and, from the same responses
truncated to k, quantdiff's bound at k = 1, 2, 3, 5, 10, 15, 20. Every 16th position was also
requested with `n_probs = 151936` (the whole vocabulary) to check the top-2000 value: it was 98.4%
(Q4_K_M) and 99.0% (Q2_K) of the exact KLD on that subsample.

This was done twice: with `cache_prompt: true` (each position reuses the cached prefix and
evaluates one new token, which is what quantdiff does) and with `cache_prompt: false` (the whole
prefix is evaluated as one batch, which is what `llama-perplexity` does).

## Results

All KLD values in nats; base is q8_0.

### Main table

| Measurement | Positions | Q4_K_M mean | Q2_K mean | q8_0 vs itself |
| --- | --- | --- | --- | --- |
| llama-perplexity, prompt text, -c 512 | 2,048 | 0.0465 +/- 0.0018 | 0.2633 +/- 0.0083 | 0.000000 |
| llama-perplexity, prompt text, -c 256 | 2,048 | 0.0455 +/- 0.0015 | 0.2266 +/- 0.0069 | 0.000000 |
| Full vocab on quantdiff positions, cached (one-token) path | 1,219 | 0.0449 | 0.1747 | 0.000000 |
| Full vocab on quantdiff positions, full-batch path | 1,219 | 0.0447 | 0.1762 | 0.000000 |
| quantdiff top-10, llama-server exact | 1,219 | 0.0302 | 0.1187 | 0.0000 |
| quantdiff top-20, llama-server exact | 1,219 | 0.0340 | 0.1338 | 0.0000 |
| quantdiff top-10, Ollama, 41 prompts (warm runner) | 1,263 | 0.0379 | 0.1248 | 0.00003 |
| quantdiff top-10, Ollama, 41 prompts (fresh runner) | 1,263 | 0.0345 | 0.1201 | 0.00003 |
| quantdiff top-20, Ollama, 41 prompts (fresh runner) | 1,263 | 0.0401 | 0.1355 | 0.00004 |
| quantdiff top-10, Ollama, 39 prompts without Hindi | 1,216 | 0.0310 | 0.1148 | 0.0000 |
| quantdiff top-20, Ollama, 39 prompts without Hindi | 1,216 | 0.0344 | 0.1290 | 0.0000 |

Medians, tails and top-1:

| Measurement | Q4_K_M median / p99 / top-1 | Q2_K median / p99 / top-1 |
| --- | --- | --- |
| llama-perplexity -c 512 | 0.0266 / 0.300 / 89.7% | 0.1564 / 1.733 / 75.3% |
| llama-perplexity -c 256 | 0.0277 / 0.279 / 89.1% | 0.1378 / 1.526 / 78.3% |
| Full vocab on quantdiff positions (cached path) | 0.0307 / 0.268 / 86.9% | 0.1152 / 1.103 / 78.8% |
| quantdiff top-10, llama-server exact | 0.0170 / 0.219 / 87.5% | 0.0607 / 1.007 / 78.8% |
| quantdiff top-10, Ollama (warm, 41 prompts) | 0.0172 / 0.244 / 86.3% | 0.0567 / 1.062 / 78.2% |

### Ratio of the bound to the full-vocabulary KLD on identical positions

Mean of quantdiff's `partition_kld` at each k, divided by the mean full-vocabulary KLD, same
positions, same responses:

| k | Q4_K_M (cached) | Q2_K (cached) | Q4_K_M (batch) | Q2_K (batch) |
| --- | --- | --- | --- | --- |
| 1 | 0.235 | 0.250 | 0.237 | 0.251 |
| 2 | 0.378 | 0.390 | 0.373 | 0.394 |
| 3 | 0.469 | 0.484 | 0.462 | 0.487 |
| 5 | 0.569 | 0.578 | 0.569 | 0.577 |
| **10** | **0.671** | **0.677** | **0.668** | **0.675** |
| 15 | 0.722 | 0.729 | 0.720 | 0.724 |
| **20** | **0.757** | **0.763** | **0.755** | **0.763** |

The bound never exceeded the exact full-vocabulary KLD at any of the subsampled positions, so the
lower-bound claim holds empirically.

### Ratio of quantdiff to llama-perplexity on the prompt text

| quantdiff run | Q4_K_M / llama (-c 512) | Q2_K / llama (-c 512) |
| --- | --- | --- |
| top-10, Ollama, warm runner | 0.82 | 0.47 |
| top-10, Ollama, fresh runner | 0.74 | 0.46 |
| top-20, Ollama | 0.86 | 0.51 |
| top-10, llama-server exact | 0.65 | 0.45 |

This ratio is not stable across quants, unlike the one above. The reason is the text, not the
bound: on quantdiff's own positions (the reference's greedy continuation) Q4_K_M's full-vocabulary
KLD is about the same as on the prompt text (0.045 against 0.046), but Q2_K's is a third lower
(0.175 against 0.263). Greedy continuations are high-confidence text where even a damaged model
agrees more often. quantdiff therefore compresses large divergences more than small ones when
compared with `llama-perplexity` numbers people have seen.

### Self-comparison floor

| Run | Mean KLD (top-10) | p99 | Top-1 |
| --- | --- | --- | --- |
| Ollama, four of five passes | 0.00003 | 0.0009 | 98.65% |
| Ollama, one pass (reference trace generated right after a fresh model load) | 0.00110 | 0.0081 | 95.88% |
| llama-server, token ids | 0.00000 | 0.0000 | 100.00% |

In the normal Ollama passes, the floor comes almost entirely from the two Hindi prompts: 17
positions where the self-candidate, forced with the re-tokenized text, predicted end of sequence
where the reference had continued. Those 17 positions are the whole 1.35% top-1 shortfall.

Retokenization noise is about 1/1000 of Q4_K_M's KLD, and the worst observed floor (0.0011) is
about 1/30. On these English prompts, text forcing does not compete with Q4_K_M's signal.

### Bootstrap intervals the verdict would use

Mean per-prompt KLD and its 95% bootstrap interval (`stats.bootstrap_mean`, 2,000 resamples), at
`--top-k 10`:

| Run | Q4_K_M | Q2_K |
| --- | --- | --- |
| Ollama, 41 prompts, warm runner | 0.0444 [0.0290, 0.0649] | 0.1337 [0.1044, 0.1725] |
| Ollama, 41 prompts, fresh runner | 0.0383 [0.0281, 0.0538] | 0.1252 [0.1043, 0.1479] |
| Ollama, 39 prompts without Hindi | 0.0309 [0.0272, 0.0349] | 0.1143 [0.0996, 0.1296] |
| llama-server exact, 39 prompts | 0.0302 [0.0266, 0.0343] | 0.1183 [0.1036, 0.1332] |

Under the current thresholds (`CLOSE_KLD = 0.05`, `LARGE_KLD = 0.15`), Q2_K is avoid in every run,
because its interval starts above 0.05, but its band is only "moderate". Q4_K_M is inconclusive
on the full 41 prompts, because the Hindi prompts widen the interval past 0.05, and close on the
39 without them.

## Findings that matter beyond the thresholds

1. **Ollama numerics depend on the runner's load history.** Scoring Q4_K_M twice with identical
   requests gave two different, individually reproducible sets of distributions, depending on
   whether the model had just been loaded when the pass began ("fresh") or was already loaded
   ("warm"). Only 180 of 1,263 positions matched between the two states. Between the states, the
   same Q4_K_M diverges from itself by KLD 0.0048 (top-10 bound) with 94.9% top-1 agreement, and
   its mean KLD against the reference moves from 0.0345 to 0.0379 (10%). Q2_K moves by 0.0016;
   q8_0 by 0.0011. The same kind of effect appears in llama.cpp itself: evaluating a position as
   part of a multi-token batch or as a single cached token changes Q4_K_M's per-position KLD by
   0.008 on average, and top-1 agreement from 86.9% to 88.6%, although the means agree. Quantized
   matrix kernels differ by batch shape, and that difference is the same order of size as the
   paired-comparison margins quantdiff uses (`_NEGLIGIBLE_KLD_RATIO = 1.05` in the previous
   verdict). Mean KLD is stable enough for absolute bands, but per-position comparisons and
   small ratios between candidates are not.
2. **Text forcing inflates KLD on non-Latin scripts.** Byte-level BPE splits Devanagari characters
   across tokens. When the reference's token strings are joined and re-tokenized, some positions no
   longer correspond. On the two Hindi prompts Ollama reports Q4_K_M at 0.29 and 0.33 per prompt
   (median prompt 0.028). Scored by token id on llama-server, the positions that can be scored give
   0.08 to 0.12 full-vocabulary KLD (0.05 to 0.09 top-10 bound). The quant really is worse on
   Hindi, but text forcing roughly triples the reading. Those two prompts alone push Q4_K_M's
   interval upper end from 0.035 to between 0.054 and 0.065, depending on runner state.
3. **llama-server teacher forcing fails on partial UTF-8 tokens.** When the predicted token is an
   incomplete UTF-8 sequence, llama-server b11425 returns `"content": " \ufffd", "tokens": []` and
   no `completion_probabilities`. `LlamaCppBackend._complete` raises `CapabilityError` in that case,
   so a `llamacpp:` run of the built-in scoring prompts fails the logit tier at prompt score-037.
   This is a bug report, not something this study changed.

## Limitations

- One small model family (Qwen2.5 0.5B Instruct) and two quants. Small models are more sensitive
  to quantization than 7B to 8B models (Q4_K_M here is 0.045 full-vocab; llama.cpp's table gives
  0.031 for LLaMA 3 8B), so the absolute KLD values do not transfer. The bound-to-full ratio is
  more likely to transfer, since it was the same for a mild and a severe quant, but it depends on
  how much mass sits outside the top-k, which differs between models and between kinds of text.
- The base is q8_0, not BF16, so every candidate's KLD is measured against an already quantized
  model. On LLaMA 3 8B, q8_0 against FP16 is 0.0014, small next to the values here.
- Ollama teacher forcing is approximate (text, not token ids). On English prompts the result agreed
  with exact token-id forcing to within 4% for the means (Q4_K_M 0.0310 against 0.0302, Q2_K 0.1148 against 0.1187) once the Hindi prompts were removed.
- Calibration text and scoring positions are short (about 100 to 160 tokens of context), so long
  context behaviour is not covered.
- 41 prompts give per-quant intervals about 12% wide (half-width) on English text; the numbers
  above carry that uncertainty.
- Ollama ran on GPU and llama.cpp on CPU. The two produce slightly different numerics, which is part
  of what finding 1 measures.

## Reference points from llama.cpp

The llama.cpp perplexity README
(https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) lists mean
full-vocabulary KLD for LLaMA 3 8B against FP16 on Wikitext-2: q8_0 0.0014, q6_K 0.0055, q5_K_M
0.0108, q5_K_S 0.0166, q4_K_M 0.0313, q3_K_M 0.1019, q2_K 0.4451. These are the numbers experts
have in mind. Multiplying by about 0.67 gives what quantdiff would read at `--top-k 10` on the same
positions: q5_K_M about 0.007, q4_K_M about 0.021, q3_K_M about 0.068, q2_K about 0.30.

## Recommendation

Thresholds stay in quantdiff's own scale (the top-10 partition bound), because that is what the
scorecard prints. The full-vocabulary equivalent is the threshold divided by 0.67.

| Constant | Current | Recommended | Full-vocab equivalent of recommended | Full-vocab equivalent of current |
| --- | --- | --- | --- | --- |
| `NEAR_LOSSLESS_KLD` | 0.01 | 0.01 (keep) | about 0.015 | about 0.015 |
| `CLOSE_KLD` | 0.05 | **0.04** | about 0.06 | about 0.075 |
| `LARGE_KLD` | 0.15 | **0.10** | about 0.15 | about 0.22 |

Reasoning:

- **`CLOSE_KLD = 0.04`.** RUN requires the upper end of the 95% interval below this bar. On 39 to
  41 English prompts that interval's upper end sits about 12% above the point estimate, so a bar
  of 0.04 admits a point estimate of about 0.035, which is about 0.05 full-vocabulary KLD: Q4_K_M
  territory on a 7B to 8B model (0.031) with room to spare, and the Qwen2.5 0.5B Q4_K_M measured
  here (0.045) just inside it (its English-only interval ends at 0.035). The current 0.05 corresponds
  to about 0.075 full-vocabulary, which is three quarters of the way from Q4_K_M to Q3_K_M on LLaMA 3 8B.
  That is too lenient for a bar described as "close to the reference". Going lower than 0.04 (for
  example 0.035, a full-vocab 0.05 bar on the interval itself) would make the 10% load-history
  swing in finding 1 enough to flip a Q4_K_M verdict, and would leave most Q4 quants of small models
  inconclusive no matter how many prompts are added.
- **`LARGE_KLD = 0.10`.** "Large" should cover Q2_K-class damage. Here Q2_K reads 0.115 to 0.136
  (full-vocabulary 0.17 to 0.26), and with the current 0.15 it is only "moderate", even though top-1
  agreement is 78%. A bar of 0.10 (full-vocab about 0.15) labels it large while leaving Q3_K_M of
  an 8B model (about 0.068 in quantdiff's scale) as moderate. The verdict now calls a candidate
  avoid on KLD only when at least `MIN_PROMPTS` prompts put both its mean and the lower end of its
  95% interval at or above this bar; a moderate interval is "usable", and a large-looking mean on
  fewer prompts is inconclusive (see docs/methodology.md).
- **`NEAR_LOSSLESS_KLD = 0.01` (unchanged).** It corresponds to about 0.015 full-vocabulary, between
  q5_K_M and q5_K_S on LLaMA 3 8B, which matches the usual meaning of near-lossless. A lower value
  is not resolvable through Ollama: the load-history noise between two states of one model reached
  0.005, and the worst self-comparison floor 0.001.

Related recommendations, outside the three constants:

1. **Make the thresholds depend on `--top-k`.** The bound grows with k (ratio 0.57 at k = 5, 0.67 at
   10, 0.76 at 20), so a fixed bar is stricter at `--top-k 20` than at 10 and looser at 5. Either
   scale the bars by r(k) / r(10) using the table above, or document that the verdict is calibrated
   for `--top-k 10`.
2. **Do not use per-prompt KLD ratios near 1.05 as evidence on Ollama.** Load history alone moved a
   candidate's mean by 10%.
3. **Exclude or flag non-Latin prompts when teacher forcing by text.** Either drop score-037 and
   score-038 from the logit tier for Ollama and OpenAI-compatible backends, or report them
   separately. As they stand, they decide whether Q4_K_M is close or inconclusive.
4. **Handle missing `completion_probabilities` in the llama.cpp backend** (finding 3), for example by
   treating it like the Ollama empty case.
5. **Update the anchors in docs/methodology.md** ("Q4_K_M around 0.02 to 0.05") to state that they
   are full-vocabulary values, and that quantdiff reads about two thirds of them.

## Reproducing

The scripts and raw outputs are in the study's scratch directory, `calibration/`:
`collect.py` (reference traces and candidate top-k through quantdiff's backends), `analyze.py` and
`ci.py` (metrics and intervals through quantdiff's `logit_metrics` and `stats.bootstrap_mean`),
`fullvocab.py` and `fv_summary.py` (full-vocabulary comparison on identical positions),
`hindi_check.py`, `probe.py`, the `llama-perplexity` logs under `ppl/`, and the JSON dumps.

## Update: split characters fixed after this study

Two findings above came from one bug that has since been fixed. Ollama and llama-server hold
back a token that ends inside a UTF-8 character and report it together with the token that
completes the character, as one entry whose logprob and alternatives belong to the later token.
Comparing that entry with a teacher-forced candidate compared two different positions, which
produced per-position KLD of 3.7 to 4.3 nats on the Hindi prompts (finding 2), and the same
merge hid one token id per split character from exact forcing on llama-server (so the "exact"
Hindi figures above were measured on incomplete token sequences). quantdiff now marks merged
positions as unscored for every candidate, rebuilds text prefixes from token bytes, recovers the
distribution at a held-back token on llama-server, and skips positions that text cannot
reproduce when a candidate is forced by text.

Measured again after the fix (q8_0 reference, top-10 bound, 41 built-in prompts):

| Measurement | Before | After |
| --- | --- | --- |
| Ollama Q4_K_M, Hindi prompts score-037 / score-038 | 0.287 / 0.331 | 0.061 / 0.043 |
| Ollama Q4_K_M, 41-prompt mean | 0.0378 | 0.0315 |
| llama-server (exact) Q4_K_M, 41-prompt mean | tier failed at score-037 | 0.0315 |
| Ollama Q2_K, 41-prompt mean | 0.1259 | 0.1161 |
| Ollama q8_0 against itself | 0.0011, top-1 95.9% | 0.0000, top-1 100% |

Ollama and exact llama-server forcing now agree on the full prompt set. Finding 3 (the
llama-server partial-byte failure) is fixed, and recommendations about excluding non-Latin
prompts no longer apply. The thresholds recommended above are unchanged: they were derived from
the k-fraction measured on identical positions, which this bug did not affect.
