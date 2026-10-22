# Session Tracker

> **Superseded by `AGENTS.md` at the repo root for orientation.** `AGENTS.md`
> holds the ten-act story, the compute policy, the concept coverage checklist and
> the v2 conventions. This file remains the running session log.

## Active Ticket

**`[DONE]` Acts 0-10 COMPLETE. An Act 4 ladder rescue is RUNNING in the
background (4.1M and 9.0M re-trained with the canonical architecture on the
D = 20N line); the paper's Act 4 numbers are from the current four-rung clean
fit and the section will be regenerated when the rescue lands.** The report covers Acts 0-9 with Act 10's
artefacts in place. Act 8 is COMPLETE, Act 9 is COMPLETE (causal head ablation,
behavioural rollout, logit lens), and Act 10 is COMPLETE (claims audit table,
failure taxonomy, ethics at the floor, reproducibility script).** (hallucination, RAG controls, tool use). Acts 0–7 are
complete; Acts 0–6 are written up by the owner, Act 7 and Act 8 material are
ready for them to add.

Status of the suite: **357 tests, 0 failures, 1 skipped.**

Decisions locked with the owner on 2026-10-14:

- Primary machine is an **M4 / 16 GB**; the M3 / 8 GB is a fallback that must
  still run.
- Act 4 (scaling laws) uses an **adaptive, resumable, checkpointed sweep**,
  cheap end first, with overnight completion as the aspiration rather than the
  requirement.
- **RLHF / PPO / DPO is out of scope by owner decision** (it was Act 7).
  Replaced by Act 6: decoding sweep + best-of-n / self-consistency.
- Final deliverable is `report/final_report.tex` + tracked figures. **No local
  LaTeX build**; the owner compiles on Overleaf, so `report/` must stay
  self-contained.
- **The owner writes `report/final_report.tex` themselves** — requests for a
  finished report are answered with a summary of what is done and what remains.

v1 is complete and frozen.

### What happened this session (2026-10-14)

Course-planning session. Read all 18 lecture PDFs in `docs/lectures/` and built a
concept coverage checklist (now `AGENTS.md` §4). Three findings about the course
material:

- **Lecture 18 is empty** (title, 3 blank slides, thank-you).
- **Lecture 17 is truncated** — it covers hallucination motivation but RAG is
  never actually taught.
- **Unit 3/4 syllabus topics have no deck at all**: prompt engineering, chains /
  memory / agents, bias / toxicity, interpretability, multimodal, model editing,
  self-evolving LLMs, small LMs, BART/T5. Owner confirmed external sources are
  acceptable for these.

Repo hygiene work completed:

| # | Finding | Fix |
|---|---|---|
| H1 | `test_referenced_figures_exist_when_generated` **FAILED** — figures existed only under git-ignored `artifacts/tick-403/character-attention/` | Figures now tracked in `report/figures/`; `final_report.tex` `\graphicspath` repointed. The test now asserts against the tracked dir, which is what would have caught this |
| H2 | `.gitignore` ignored all of `docs/`, so the design record would not survive the M3→M4 machine change | Now ignores only `docs/lectures/` (165 MB of PDFs). `docs/*.md` is tracked |
| H3 | v1 metrics lived only in git-ignored `artifacts/` | Copied to tracked `results/v1/` as JSON plus a derived `comparison.csv` |
| H4 | Report had only 150 dpi PNGs loose in the `report/` root, one of them stale | Regenerated vector PDFs alongside PNGs in `report/figures/`; removed the stale root copies |

### Owner commands

```bash
cd ~/Desktop/ProjectWorkspace/LLM/LLM-Ascent

# Full suite — expect 100 tests, all passing
.venv/bin/python -m unittest discover -s tests -v 2>&1 | tail -n 20

# Optional: look at the interface, never yet verified in a browser
.venv/bin/python -m src.web.server --port 8000
```

### Browser checklist (TICK-702 / gate G5) — still outstanding

Pick a BPE model in the header unless stated:

1. **Header:** dropdown lists the checkpoints under `artifacts/` (baseline-smoke,
   tick-204/bpe, tick-204/character). Yellow "tiny research models" banner visible.
2. **Playground:** Generate with `ROMEO:`; chips show prompt/new tokens, seconds,
   tok/s, repetition. Try greedy, sample (seed 7 twice gives the same text) and
   beam. Type a capital-free prompt with a `#` or a digit and look for the
   `<unk>` notice.
3. **Chat + tools:** send `/calc (2+3)*4` → `20` (no model call; the prompt box
   says so). Send `hello` → word salad plus a "prompt is longer than the model's
   64-token context" notice. "New conversation" clears it.
4. **RAG:** `Which planet is red?` → source `knowledge.txt` score 0.75, a notice
   that the prompt is cropped, with-context and closed-book answers side by side.
5. **Attention:** "Show attention" → layer/head selectors, token chips, hover
   text, thumbnails you can click.
6. **Results:** the character-vs-BPE table (3.45 vs 3.09 bits/char) and the figures.
7. Optional: switch your OS to dark mode and reload; text should stay readable.

### What v2 is

Ten acts, each ending in a table with error bars, figures, and a stated
conclusion. Full map, compute policy and coverage checklist: **`AGENTS.md`**.

### Plan at a glance

| Stage | Contents | Status |
|---|---|---|
| A | Enablers: MPS, experiment framework, seed/CI harness, tracked results | **next** |
| B | Acts 0 + 1 (n-grams, smoothing, embeddings, RNN/LSTM, seq2seq ± attention) | todo |
| C | Acts 2 + 3 (Transformer ablations, tokenizers) | todo |
| D | Act 4 (scaling laws + MoE) — the centrepiece | todo |
| E | Acts 5 + 6 + 7 (KV cache, MQA/GQA, speculative, decoding sweep, SFT, PEFT) | todo |
| F | Acts 8 + 9 + 10 (hallucination, RAG, tools, interpretability, ethics) | todo |
| G | Report bundle, Results UI, final write-up | todo |

### Open decisions

All closed as of 2026-10-14 except the sweep size, which is adaptive by policy.

| # | Decision | Outcome |
|---|---|---|
| D1 | Compute target | **M4 / 16 GB primary**, M3 / 8 GB fallback |
| D2 | Version the results | yes — `results/` tracked, checkpoints ignored, `docs/*.md` tracked |
| D3 | Web stack | Python standard library only, one offline HTML file (done) |
| D4 | v1 files | Frozen; v2 outputs go to `artifacts/v2/`, `data/v2/`, `report/` |
| D5 | Alignment / RLHF | **Out of scope** by owner decision |
| D6 | LaTeX build | **None locally**; owner compiles on Overleaf. `report/` must be self-contained |

### Current Blocker/Focus

Nothing blocked. Next work is Stage A.

### Act 4 / Stage D — COMPLETE (2026-10-16)

Scaling law ladder ran to completion: 6 rungs, 12 runs (3+3+2+2+1+1 seeds),
2,776 surviving (N, D, loss) points after transient truncation. Joint nonlinear
least-squares fit `L(N,D) = E + A/N^α + B/D^β` gives α = 0.44 ± 0.32, β = 0.29 ±
0.16, E = 1.605 — consistent with Chinchilla (0.34, 0.28, 1.69) at 1/10⁴ the
scale. Leave-one-rung-out R² ≥ 0.985 on every held-out rung. Pre-registered
falsification condition NOT FALSIFIED. Full write-up in `docs/implementation.md`
Act 4 section. Artifacts in `results/v2/act4-scaling/` and `report/figures/`.

**Stage E done (Acts 5 + 6). Next: Stage F (Acts 8 + 9 + 10).**

### Act 7 — SFT, ICL vs SFT, PEFT head-to-head, LoRA rank sweep — **COMPLETE** (2026-10-19)

Base: pinned canonical 6.3M (val_loss 2.5152). 512 train / 128 held-out, 7-task
synthetic family, seeds 42/43/44, 24 epochs. Teacher-forced exact match with
Wilson 95% intervals. Tracked in `results/v2/act7/6.3M_sft_peft.json`.

- **7a** — zero-shot floor is **0.0000** on all 128 held-out items, so every
  number above zero is attributable to adaptation.
- **7b** — **ICL is 0.0000 at every set size and every seed.** Full fine-tune
  scales **0.0703 → 0.1667 → 0.3255 → 0.5651** at n = 16/64/256/512 with **no
  saturation**: still +0.24 from 256 → 512. At this scale LIMA's "less is more"
  is **not supported** — a few hundred examples is not enough.
- **7c** at matched 20,000-param budget — full 0.5755 · **adapters 0.2136** ·
  LoRA 0.1823 · prompt tuning 0.0859 · prefix tuning 0.0000. Adapters lead LoRA
  but the Wilson intervals overlap, so the ordering is suggestive only.
  `final_train_loss` distinguishes *did not fit* from *fit but scored zero*:
  prefix tuning sits at train_loss 3.2576, so its zero is a failure to fit, not
  an artefact.
- **7d** — LoRA rank sweep is **monotonic 0.1823 → 0.5078** across r = 1…64 with
  no plateau, tight seed spread (≤0.05). r=64 recovers 0.5078 vs full's 0.5755
  at 11.5% of the parameters. Rank is a smooth capacity dial, not a threshold.
- **Genuine negative retained:** `add` scores **0.9452 on train** (69/73) but
  **0.0000 on held-out** (0/18). Guesses are off-by-one (8+2→11, 7+7→15) —
  memorised sums, not learned addition.

**Two defects invalidated earlier Act 7 output and were fixed** (D11/D12 in the
ledger). D11: `InstructionExample` is `(instruction, response, input_text)` and
the payload was passed positionally, so **the input text never reached the
prompt** — six of seven tasks were unanswerable by construction, and the run
emitted zeros that looked like a real negative. Caught by measuring per-position
accuracy: position 0 was 0.070 while positions 1–2 were 0.93/0.90. D12: three
PEFT forwards applied a **second cross-entropy shift** and scoring sliced
`pred[-n:]`, which is off by one when virtual tokens are prepended — all three
methods read exactly 0.0000 while genuinely training. LoRA and full fine-tune
route through `MiniGPT.forward` and were never affected, so 7b/7d stand.

D4 (no full-batch SFT) and D5 (response terminator) fixed and tested.

Caveats: synthetic tasks not natural instructions; ICL is teacher-forced so not
comparable to free-running numbers; **the n=16 row is effectively n=1** (single
batch per epoch, so the seed controls nothing); seeds vary optimisation order
only, since base weights are pinned.

### Act 10 — Evaluation methodology and ethics — **COMPLETE** (2026-10-20)

- **10a** — `report/claims.csv` (81 rows) IS the audit table; the paper carries
  a generated 20-row headline table and references the artefact rather than
  duplicating it. `tools/check_tex.py` verifies the correspondence both ways.
- **10b** — the failure taxonomy, with the unifying principle stated: **every
  recorded measurement failure was caught by a sanity invariant or a positive
  control, and none by comparing two implementations.** The 5c double-feed is
  the sharpest case (both equivalence tests passed because both paths shared
  the bug; only the absolute acceptance invariant exposed it).
- **10c** — ethics at the floor, with the instruments and thresholds. The
  stereotype probe returns 1.0 in the associated families **and** in the
  negative control, so it detects a global attribute preference rather than an
  association, and is underpowered (only a rate above 0.8099 is detectable at
  n=10). Toxicity 0/40, Wilson [0, 0.0876]. **Calibration is a positive
  result: accuracy 0.5295 at mean confidence 0.5255, ECE 0.0233** — a
  well-calibrated next-token predictor that cannot follow instructions.
  The SFT comparison is impossible (no tracked fine-tuned checkpoint) and is
  reported as such.
- **10d** — `python -m tools.reproduce` re-derives six headline numbers from
  the tracked checkpoints, including one Act 4 fit and one Act 9 number. All
  six reproduce.

### Act 9 — Interpretability — **COMPLETE** (2026-10-20)

Base: canonical 6.3M, 8 layers x 8 heads = **64 heads**. Held-out: 64 sequences
of 128 tokens from the C4 shard at document offset 300,000; the intact baseline
(CE 2.8551) is measured in this run, so every delta is internal to it. Tracked
in `results/v2/act9/interpretability.json`.

- **9a** — every head ablated, against a **random-ablation null**. Exactly
  **1 of 64** heads has z >= 2: **layer 0, head 5** (dCE +0.832 nats, dPPL
  +22.5, z = +7.82, z vs null +42.8). The next head is z = +0.54; 59 of 64 are
  z <= 0. The **top-4 combination** costs +1.047 nats against **+0.071** for
  four random heads (~15x). The head that matters is **sharp** (attention
  entropy 0.379 nats) while the ones that do not are diffuse (2.480 nats).
- **9b** — attention rollout, scored **behaviourally** against the token
  holding the asserted number: **0/34 hits**, Wilson [0, 0.1015]. Raw
  attention is also **0/34**; uniform chance is 0.0078. A heatmap would have
  suggested the opposite of what the hit rate shows.
- **9c** — the logit lens (the model's own readout on each layer's residual)
  gives the median rank of the asserted number: **860, 721, 550, 439, 412,
  87, 227, 647** across layers 0-7. The number is most decodable at **layer 5**
  (rank 87 of 6,487) and is **lost before the output** (layer 7, rank 647). It
  is never the lens argmax on any item.

**This is the mechanistic explanation for Act 7b and Act 8b in one
measurement**: the asserted number is present in the residual stream and is
destroyed before it reaches the output. The model does not fail to *see* the
number; it fails to *carry* it — which is why grounding was at chance, why
free generation was right 0/40 times, and why the tool call was `CALC(s)`.

**Defect fixed on the way**: D18 — the held-out metric compared logits to the
token at the same position rather than the next one, returning CE 11.98
(perplexity 1.6e5) for a model whose real loss is ~2.5. The signature was
absurd on its face (uniform over 6,487 tokens is 8.78 nats). Pinned by a
constructed next-token oracle and a uniform model.

### Act 8 — Hallucination, RAG controls, tool use — **COMPLETE** (2026-10-20)

Base: pinned canonical 6.3M (val_loss 2.5152), 128-token context. Prompts from
the real C4 shard, never synthetic tokens. Tracked in
`results/v2/act8/hallucination.json` and `results/v2/act8/tool_use.json`.

- **8a** — one shared prompt builder for RAG and SFT (**D2** fixed), and the
  context budget denominated in **tokens** (**D1** fixed). D1's cost measured
  on the canonical window: 64 characters ≈ 32 tokens of a 302-token chunk,
  **9.2% of the passage survived**; the token budget admits 77 tokens (2.4×).
- **8b/8c** — the passage does **not** measurably change the answer. Oracle
  chose the item's answer 9/40 vs closed-book 11/40, **McNemar p = 0.625** —
  the model identifies the asserted number **at chance** (per-item chance
  0.1838, chance-adjusted +0.0412, n.s.), and the closed-book point estimate
  is *higher*. The counterfactual edit was followed in **0/40** items. A
  *measurable but weak* preference shift does exist (+0.204 nats, p = 0.030),
  but the symmetric test fails (+0.039, p = 0.839), so the model is **not**
  demonstrably reading which digit the passage states.
- **The retriever was broken, and 0/40 was a query-design failure.** The query
  was the instruction text, whose terms cannot come from a passage about a
  number, so ranking degenerated to a constant and the same chunk came back
  for all 40 items. The query is now an **anchor** — content words from a
  different sentence of the same document — so the retriever is answer-agnostic
  but has a handle: **39 distinct documents** returned (was 1), document hit
  rate **0.975**, retrieved chunk contains the asserted number **0.475**.
- **A new `nonnumeric` control** (same document, no number) scores **9/40 —
  identical to oracle's 9/40**. Swapping the asserting passage for a
  non-asserting one changes nothing: the model has a **context-number bias**,
  not an assertion-reading ability.
- **The tracked partner number is ABSENT from the passage** in 40/40 items, so
  the +0.583 lift is a generic context effect, not evidence about passage
  content. The only content-specific trace is the +0.204 nats residual.
- **The absolute hallucination rate is at a ceiling** (1.000 in every
  condition; 97/120 oracle generations contain no number at all) and is
  therefore **uninformative for ranking conditions**. It is reported and
  labelled, and not used for any comparison. The first design was replaced
  precisely because it confounded *cannot ground* with *cannot emit a label*.
- **8d** — tool use is a measured negative: **0/40 well-formed `CALC(...)`
  calls** (Wilson [0, 0.0876]), 0/120 sampled. The mechanism, verified by token
  id: **19/40 greedy generations emit the literal `CALC(` and in all 19 the
  argument is exactly `"s"`**. `CALC(` tokenizes to `[49, 4083, 49, 22]` and
  the argmax continuation is id 97 = `'s'` (p=0.0278) with the runner-up
  `'express'` — a fragment of **"expression" from the prompt's own template**.
  The model completes the instruction's template rather than writing an
  expression. Same shape as Act 7's ICL result: format acquired, operation
  absent.

**Denominator correction.** The unit is the **item**, not the generation: the
answer-copying instrument is deterministic, so n = 40 and a 0/40 Wilson upper
bound is **0.0876**, not 0.031. Free-generation repeats are averaged per item
before scoring; contrast intervals come from a percentile bootstrap that
**resamples items**.

**Act 7 write-up is unblocked.** The positive control settled 7a/7b: the
harness scores 1.0 on all three scoring routes, so **7a is assertable**. 7b is
a *rewrite, not a softening*: the demonstration is not ignored — on the 18
same-task addition items the first generated token is the **demonstration's**
answer on 18/18 versus 0/18 zero-shot. The model copied the response *format*
but not the operation. Both honesty notes stand: frozen model ⇒ ICL's three
seeds are bit-identical (n=1, not n=3), and shot selection is cost-driven so
110/128 items got a cross-task demonstration.

**Four defects found and fixed** (D14–D17 in the ledger). D14: the
answer-copying score absorbed the prompt's own log-probability, surfacing as a
passage-lift of **−246 nats**; D15: a class that was **unreachable by
construction** in closed-book, whose 0/40 read like a measurement; D16: a
grouped literal (`1,000`) crashed the C4 scan; D17: the BPE merges `35` into
one token but not `34` into three, so counterfactuals are now matched on
digit count **and** token count.

Caveats: the **retrieved condition is degenerate by construction** (the
retriever is queried with the instruction text, so retrieval success was 0/40
and it is indistinguishable from closed-book); the oracle and counterfactual
margins disagree and both are reported; numbers proxy for atomic claims.

### Act 6 — Decoding sweep, best-of-n, self-consistency — **COMPLETE** (2026-10-19)

6.3M canonical, 40 real C4 prompts, 32 new tokens, seeds 42/43/44 (120
generations per config), MPS. Tracked in `results/v2/act6/6.3M_decoding.json`.

- **6a sweep** — corr(distinct-2, BPC) = **+0.87**: diversity costs quality
  monotonically. Beam-4 beats greedy by **0.061 bits** (1.3760 vs 1.4366) but
  costs **4.26x** the time (corrected: beam now uses the KV cache and timings
  are taken after warm-up; the pre-correction figure was 4.9x). Beam-8 is worse
  than beam-4 at 2.2x the cost. Greedy is the quality/cost knee — nothing is
  both cheaper and better.
- **6b best-of-n** — reranking gains **+0.272 bits** over the mean sample at
  n=16, but is still **0.259 bits worse than greedy**. Structural, not a tuning
  failure: a self-referential scorer can only select text the model already finds
  likely, so best-of-n converges toward the model's own mode and never past it.
- **6c self-consistency (MNLI)** — **negative**. No gain at any n. Argmax
  accuracy is **exactly chance (0.3300)**; the model **never predicts "neutral"**
  (recall 0.000) and on the 216 answerable items argmax (0.4583) is *below* the
  0.5139 coin-flip baseline. Majority voting converges to argmax, so there is
  nothing to amplify. **Self-consistency cannot rescue absent task competence.**

Figures: `report/figures/act6_decoding_tradeoff.*`,
`act6_bestofn_selfconsistency.*`, generated by `src/eval/act6_figures.py`.

### Act 5 — Inference & efficiency (5a, 5b, 5c) — **COMPLETE** (2026-10-19)

5a KV cache, 5b MQA/GQA/SWA and 5c speculative decoding all measured, with
tracked results, three figures (PNG + vector PDF) and a stated conclusion.
Canonical pinned set 1.5M / 2.6M / 6.3M all trained and pinned.

**5b MQA quality cost at 2.6M is now SIGNIFICANT.** n=6 paired (seeds 42–47,
6/6 direction-consistent), mean **+0.0285 nats**, SD 0.0097, t=7.182 on df=5,
**two-sided p = 0.0008**. The earlier n=3 p=0.0811 was a small-sample artefact.
Excluding seed 42 (whose G=1 arm resumed in 0.73 s, provenance unverified):
n=5, p=0.0009 — conclusion unchanged. Tracked in
`results/v2/act5/mqa_quality_2.6m.json`.

**5c speculative decoding: a KV-cache bug invalidated the first 5c numbers.**
The draft and verification loops re-fed the whole sequence on top of a
prefilled cache, double-counting context, so `p != q` even for identical
weights — acceptance 15–40% at zero gap instead of 100%. Cached-vs-uncached
tests could not catch it because both paths shared the bug; the new
`test_zero_gap_acceptance_is_total` can. After the fix: acceptance 100% at every
K, throughput at K=3 up 10.7× (28.9 → 309.8 tok/s).

5c zero-gap (1.5M → 1.5M, cache both sides, honest cached baseline) is
**negative: 0.41–0.50× speedup**, exactly as expected since no target forward is
saved. It is a correctness proof, not a performance result.

**5c at a real gap is also negative: 0 of 50 cells beat plain decoding.** Best
cell 0.622× (K=1, 4.2× gap); median 0.502/0.441/0.399/0.346/0.300 at the 1.7×
gap for K=1/2/3/5/7. The cause is arithmetic: a round yields
(1−α^(K+1))/(1−α) tokens for 2K forwards, which is <1.0 for every K>1 even at
perfect acceptance. Acceptance itself is NOT near zero (0.86 at K=1) and is
*higher* for the larger gap, because 6.3M is the better-trained target — so the
negative result is about the forward-count budget, not about acceptance.

Canonical rungs: 1.5M 3.0571 / 2.6M 2.8111 / 6.3M 2.5152. The 2.6M canonical
reproduced the MHA seed-42 sweep loss exactly, an unplanned determinism check.

Figures: `report/figures/act5_kv_cache.*`, `act5_speculative.*`,
`act5_mqa_paired.*`, all generated by `src/eval/act5_figures.py` from the
tracked JSON so they cannot drift.

Test suite at Act 5 close: **230 tests, 0 failures, 2 skipped.**

**Act-crossing rule:** Act 4 (ALiBiTransformer, 4 heads, 6,487-token BPE) and Act 5
(MiniGPT, 8 heads) measure different models; losses are not comparable across acts.
Every metrics.json now records an `architecture` block.

**Act-crossing rule:** Act 4 (ALiBiTransformer, 4 heads, 6,487-token BPE) and Act 5
(MiniGPT, 8 heads) measure different models; losses are not comparable across acts.
Every metrics.json now records an `architecture` block.

**Act-crossing rule:** Act 4 (ALiBiTransformer, 4 heads, 6,487-token BPE) and Act 5
(MiniGPT, 8 heads) measure different models; losses are not comparable across acts.
Every metrics.json now records an `architecture` block.

Act 5 (KV cache, MQA/GQA/SWA, speculative decoding) done on M4. KV cache
byte-identical (max logit diff 1.64e-07), 18.36x speedup at context 1024, O(N*d)
memory confirmed. MQA/GQA: 8x cache reduction, −12.1% FLOPs, quality cost at
2.6M now significant (+0.0285 nats, p=0.0008, n=6). Speculative decoding:
distribution verified exact, but a 2–4x wall-clock LOSS at every configuration.

**Known hygiene issue:** running the test suite REWRITES tracked files under
`results/` (it regenerates an Act 0 metrics.json timing field). `results/` is
therefore not a pure append-only record. Noted, not yet fixed.

Next: Act 6 (decoding sweep + best-of-n). Act 5 is closed.

### Stage C Corrections (2026-10-15)

**(a) Act 1 — RNN vs LSTM vs GRU.** The original writeup claimed "std = 0.000
across seeds" and concluded "RNN beats LSTM." Both are wrong. The actual
`results/act1-rnn/summary.json` shows validation_loss_std = 0.215 (RNN),
0.285 (LSTM), 0.254 (GRU) over n=3 seeds. The RNN-LSTM gap is 0.152, INSIDE
one std. Corrected claim: **RNN, LSTM, and GRU are indistinguishable at n=3
on this corpus**, and the comparison was parameter-distorted (31K vs 58K vs
46K). Additionally, the gradient-norm demo does NOT show vanishing gradients:
RNN grad norms go 0.436 → 0.483 and LSTM 0.141 → 0.283 — both INCREASE, and
LSTM's are LOWER than RNN's. This is a null result; vanishing gradients were
NOT demonstrated at sequence length 20 on the copy task.

**(b) Act 0 — N-gram smoothing.** The original writeup reported n_seeds=3 with
byte-identical values across seeds. This is correct for count-based n-grams
(no stochasticity in training) but means there is NO error bar. Relabelled as
**deterministic, n=1 in substance**. Not presented as n≥3.

**(c) Act 2 — Transformer ablations.** The original writeup quoted 18 ablations
reporting "BPC" while the JSON field is `validation_loss`. The 18 individual
`metrics.json` files were all present; only the `summary.json` had been
overwritten by successive `--ablation` runs. **Rebuilt `summary.json` from all
18 metrics files.** Corrected claim: the numbers are validation loss (per-token
cross-entropy), NOT bits-per-character. BPC was computed in the writeup but not
stored in the metrics files. All 18 configs are reproducible from tracked
`metrics.json` files.

**(d) SQuAD dedup warning.** `data/v2/squad_v1.1_train.jsonl` has a 0.11%
duplicate rate but consecutive records legitimately share a paragraph (SQuAD
groups multiple questions per context). Do NOT dedup SQuAD by context. If
dedup is needed, key on (title, context, question).

### Owner Comments

_(Paste decisions, test output, or results here.)_

## Session Notes

- Initialized: 2026-10-04 (Asia/Kolkata). v1 results are plumbing and narrow-scope
  evidence only.
- **v1 final facts:** TICK-204 BPE 3.09 vs character 3.45 bits/char (single seed,
  500 steps, context-confounded); TICK-401 prompting 0/6 everywhere, few-shot/CoT
  prompts cropped at 64 tokens; TICK-402 `/calc` safe and memory persistent, model
  cannot emit `CALC` itself; TICK-403 four figures; TICK-404 report built.
- **Suite is now 100 tests, all passing** (was 99 passing + 1 failure).
- **Why v2 needs more than longer training:** code-level blockers, listed as
  D1–D10 in `AGENTS.md` §5 and in `docs/implementation.md` §1.
- **Pre-registered gates:** a claim is only made when its supporting comparison
  is actually run. Failed gates are reported as negative results, and anything
  run after seeing results is labelled exploratory.
- **Web interface facts for later tickets:** the registry only lists `checkpoint.pt`
  files (base models) under `artifacts/`; adapters (`adapter.pt`) are not loadable
  yet — that is defect D10. Generation is serialised by one lock (fine for one
  local user). The Results tab currently renders `artifacts/v2/summary.json` as
  raw JSON and still needs a README launch section and gate-status display.
- `report/` holds the `.tex`, the v1 PDF, and now tracked `figures/` (PNG + PDF)
  and `tables/`. `tests/test_report.py` expects `report/final_report.tex`; do not
  rename it.
- **2026-10-21 — Report revision (Act 4).** Four corrections and one new
  contribution, all sourced from tracked artefacts:
  1. **The "physically possible E" argument is WITHDRAWN.** The measured unigram
     entropy of the corpus under the ladder's 6,487-token vocabulary is
     **4.9946 nats** on the held-out region (`results/v2/act4-scaling/
     unigram_entropy.json`), and BOTH fitted E values lie below it (1.348 is
     0.27x it, 0.706 is 0.14x it). The ground does not discriminate between the
     two N conventions; the axis now rests on (i) the published convention and
     (ii) stability. E is an EXTRAPOLATED asymptote, not a measured entropy.
  2. **The central comparative claim is replaced**, not softened: the total-N
     alpha range [0.08, 0.54] contains 0.34 and 0.50, so the ladder is
     **compatible with published exponents across their entire range and
     discriminates between none of them**. The beta range [0.36, 0.56]
     **excludes** Chinchilla's 0.28, so no compatibility claim may be made for
     beta at all. The beta profile is now a figure panel and a ledger row.
  3. **The cause of the mid-campaign architecture change is known** and it is a
     PROCESS failure: `src/scaling/` was empty before commit `f1eaf77`, the
     committed ladder script builds the second family's architecture with the
     first family's shapes, and the script that trained the canonical MiniGPT
     rungs was never committed. It is the taxonomy's largest entry.
  4. **The N audit** finds three N conventions in one table, none of them
     flagged; the 0.8M rung's recorded N of 694,144 matches no known
     architecture at all, so it is a third, unidentifiable model carrying the
     act's worst hold-out fold.
  5. **The rescue plan and gate changed**: widen the N range (0.3M, 0.4M, 0.8M
     re-run, 4.1M, 9.0M at D=10N), and the pre-registered gate is now "does
     the alpha profile narrow materially from [0.08, 0.54]?". **No rescue number
     is reported**; the rescue is still in flight.
  - The 10c SFT checkpoint now exists (`results/v2/act10/sft_checkpoint.json`,
     `checkpoints/canonical/6.3M-sft/model.pt`): held-out instruction accuracy
     0.6016 (77/128) against the base model's 0.0000. Its **calibration is not
     measured**, so the SFT-versus-base calibration comparison is an open item,
     not a null.
  - `tools/check_tex.py` and its `--self-test` both pass; the ledger is 99 rows
     and the taxonomy 15 entries.
