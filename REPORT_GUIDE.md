# Reading and defending Project Ascent

A study guide to `report/LLM_Ascent_final report.pdf` (84 pages).

Everything here is keyed to the document you are holding: **Table 1–38** and
**Figure 1–20** refer to the PDF's own numbering, and every number is quoted
from the report or from the tracked file named beside it.

> **One warning before you start.** The PDF is the draft of 21 October. Three
> numbers inside it are stale, and one whole subsection (Act 4's rescue) has
> moved since. They are listed in §12. Read that section before the meeting,
> not after.

---

## 0. The 30-second version

The project builds three tiny decoder-only Transformers from scratch (1.5M,
2.6M, 6.3M non-embedding parameters, 128-token context) and measures the
effect of each design decision rather than asserting it. The headline is not
"the model works". It is three findings plus an audit:

1. **The model is well calibrated but cannot use what it knows.** On 4,064
   held-out next-token predictions it is right 52.95% of the time at 52.55%
   confidence (ECE 0.0233). The same model scores **0.0000** on instruction
   following, is **at chance** at grounding a passage, and emits a tool-call
   template whose argument is the letter `s`.
2. **Instruction tuning retargets competence; it does not add it.** SFT lifts
   held-out instruction following from 0.0000 to 0.6016, leaves next-token
   accuracy on the tuned distribution unchanged (−0.0009, n.s.), doubles the
   calibration error there (0.1118 → 0.2772), and costs 10.7 points of accuracy
   off-target (ECE 0.0233 → 0.1306).
3. **The scaling ladder was contaminated, and the contamination was invisible
   to the fit.** The ladder mixed two architecture families because the
   training script for one of them was never committed. The withdrawn
   exponents (α = 0.445, β = 0.290) describe the *other* family: they score
   R² = +0.9957 on its points and **−1.3123** on the canonical models' own
   points. A good-looking fit is not evidence of a good fit.

The through-line is methodological: **all fifteen recorded measurement
failures were caught by an absolute sanity invariant or a positive control,
and none by comparing two implementations.** That is the sentence to remember.

---

## 1. What this project is, and what it is not

**It is** an exercise in *measurement design* on a deliberately undersized
model. Each of the ten "acts" removes one limitation the previous act exposed,
and each act ends in the same shape: a table with its uncertainty stated, one
or two figures, and a conclusion that separates what the measurement supports
from what it does not. A null result gets the same treatment as a positive one.

**It is not** an attempt to show that a 6.3M model is useful. The paper says
this explicitly (Section 2.5, "A note on what a 6.3M model can and cannot
do"), and refuses to generalise any result to models at scale.

**The unit of contribution is a defensible sentence.** The report is written so
that every claim carries (a) an *n*, (b) an interval or an explicit statement
that none is available, (c) a status word, and (d) a caveat that travels with
it. The full ledger is `report/claims.csv` (102 rows).

### The ten acts at a glance

| Act | Question | Verdict in one line |
|---|---|---|
| 0 | Do classical n-gram baselines behave as advertised? | Order dominates smoothing (28.43 → 7.45–8.78 ppl); smoothing differences are ≤ 0.9 ppl. |
| 1 | Are RNN/LSTM/GRU distinguishable? | No: gaps are inside the seed spread; the vanishing-gradient demo shows the *opposite*. |
| 2 | Which Transformer components matter? | A single-seed *ranking* only; the ALiBi arm never ran ALiBi. |
| 3 | Does a tokenizer comparison survive a context control? | Unigram < BPE < char < WordPiece, but n = 1; v1's headline is superseded. |
| 4 | Do published scaling laws hold below their range? | The ladder is contaminated; α is unidentifiable; the withdrawn "Chinchilla-consistent" claim is retracted. |
| 5 | What do inference tricks buy and cost? | KV cache: byte-identical, 3.71–18.36×. MQA: 8× cache, 12.1% FLOPs, +0.025 nats quality. Speculative decoding: 0/50 cells beat plain decoding. |
| 6 | What does decoding buy? | Clean quality/diversity trade-off; greedy is the cost/quality knee; best-of-n cannot beat greedy with a self-scorer; self-consistency gives nothing. |
| 7 | Can it be taught to follow instructions? | Yes, 0.0000 → 0.6016, monotonically in examples; but it is *format*, not operation (addition: 0.9452 train, 0.0000 held-out). |
| 8 | Does it ground what it reads? | No. At chance; a non-numeric passage scores identically (9/40 = 9/40); tool calls are template echoes `CALC(s)`. |
| 9 | Where does the information go? | A generic number is decodable at layer 5 (rank 82 of 6,487) but never reaches the output; attention attribution carries no signal. |
| 10 | Can the project audit itself? | 102-row ledger, 15-defect taxonomy, calibration result, 10-check reproduction script. |

---

## 2. How the argument is built (read this before the acts)

```
                 ┌──────────────────────────────────────────────┐
                 │  Act 10c: calibration (the one positive result)│
                 │  52.95% accurate @ 52.55% confident, ECE .0233 │
                 └───────────────────────┬──────────────────────┘
                                         │ the model HAS competence
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
   Act 7b/8b/8d                      Act 7c/7d                        Act 10c (SFT)
   it cannot USE it:                 adaptation can move it:          SFT retargets it:
   ICL copies format,                LoRA r 1→64: .18→.51             instr .00→.60,
   grounding AT CHANCE,              PEFT family at matched           acc on-target flat,
   tool call = CALC(s)               budget: adapters .21 > LoRA .18  confidence +23 pts
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
                                   Act 9: WHY
                    a generic number lives in the residual stream
                    (rank 82/6487 at layer 5) and does not reach
                    the output; attention attribution is uninformative
```

Parallel to that causal story runs the **scaling act**, which is a story about
the project's own instrumentation:

```
  v1 claim: "α = 0.445, consistent with Chinchilla, R² ≥ 0.985 per rung"
        │
        ├─ audit 1: the ladder has TWO architecture families (11 vs 5 runs)
        ├─ audit 2: the recorded N is NON-embedding; published α,β use TOTAL
        ├─ audit 3: one rung (0.8M) matches NO known architecture
        └─ audit 4: the script that trained one family was never committed
        │
        ▼
  the withdrawn parameters score R² = +0.9957 on the other family,
                            R² = −1.3123 on the canonical models  ← the invariant
        │
        ▼
  what survives: a contaminated-set finding + a methodological rule
  (tag every point with its config; prefer absolute invariants)
```

**Why this structure is the argument.** The calibration result licenses the
claim that downstream failures are failures of *use*, not of *possession*.
That distinction is the paper's actual thesis, and Act 9 supplies the
mechanism. Act 4 is the negative control on the project's own method: the same
rigour that catches a weak model also catches a flattering fit.

---

## 3. Glossary

Read this once. The report assumes these words.

### 3.1 Measurement and statistics

| Term | Meaning in this report |
|---|---|
| **n** | Number of independent units. For proportions the *item* is the unit, not the generation: repeats of the same item are averaged first, because they share difficulty. This is why 0/40 gives a Wilson upper bound of 0.0876, not 0.031. |
| **Nats** | Natural-log units of loss/cross-entropy. 1 nat = 1.4427 bits. The models report loss in nats per token. |
| **Bits per character (BPC)** | Loss converted to bits and divided by characters, so tokenizers with different vocabulary sizes can be compared on the same text. The paper's cross-architecture currency. |
| **Perplexity** | exp(loss). "How many equally likely options the model is effectively choosing among." Only comparable per *character* across tokenizers, never per token. |
| **Wilson interval** | A confidence interval for a proportion that behaves sensibly at 0 and 1 (unlike the normal approximation). Used for every accuracy/hit rate. Never used for a continuous quantity. |
| **Bootstrap interval** | Resample the *items* 10,000 times, recompute the statistic, take percentiles. Used for contrasts (paired differences) and for ECE. |
| **Paired difference** | Compare A and B on the same item, then average the differences. Removes item difficulty. Used in Act 5b (MQA vs MHA) and Act 8c (passage lift). |
| **McNemar test** | Test for paired binary outcomes: does A differ from B on the items where they disagree? Act 8 uses it for oracle vs closed-book (discordant 1 and 3, p = 0.625). |
| **ECE (expected calibration error)** | Average gap between stated confidence and observed accuracy, over confidence bins. 0.0233 means "when it says 60%, it is right about 60%". |
| **Ceiling metric** | A metric that is saturated (e.g. 1.000 in every condition), so it cannot rank conditions. Reported, labelled, never used to compare. Act 8's free-generation rate is this. |
| **Exploratory** | Measured after the analyst saw results, or at n = 1. Not a headline. |
| **n.s.** | Measured, interval does not exclude the null. |
| **Superseded** | Replaced by a later measurement (v1's tokenizer headline). |
| **Invalid** | The measurement does not test what it claims (the ALiBi arm). |
| **Pre-registration / falsifier** | A statement of what would count against the claim, written before the run. **Exactly one act (Act 4) has one**: `results/v2/act4-scaling/FALSIFICATION_CONDITION.md`, 2026-10-15. |
| **Power** | The chance of detecting a true effect of a given size. Act 8 at n = 40 and chance 0.1838 detects a true rate of 0.40 with ~87% power, 0.35 with ~69%, 0.30 with ~42%. Say this before someone else does. |

### 3.2 Scaling laws (Act 4)

| Term | Meaning |
|---|---|
| **L(N, D) = E + A/N^α + B/D^β** | Chinchilla-form law. N = parameters, D = training tokens. |
| **E** | Irreducible loss: the asymptote as N, D → ∞. It is the *conditional* entropy of an infinite model, not the corpus's unigram entropy. |
| **α, β** | Exponents on the parameter and data terms. Kaplan 2020: α ≈ 0.076, β ≈ 0.095 (with a different parameterisation); Chinchilla 2022: α ≈ 0.34, β ≈ 0.28. |
| **Total vs non-embedding N** | Total counts the embedding matrix; non-embedding does not. The published exponents use **total**. The ladder recorded **non-embedding** — an unstated confound. |
| **Profile likelihood** | Hold α fixed on a grid, re-optimise the other four parameters, plot R². The *compatible range* = all α within 0.001 R² of the best. This is **not** an error bar: it states what the data cannot distinguish. |
| **Ridge (α vs E)** | α and E trade off along a shallow valley: fixing E and re-optimising α traces a monotone path, dα/dE = +0.5786. That is why the α range is wide even though R² is 0.995. |
| **Local exponent** | α_loc = −d log(loss − E)/d log N at a fixed E. If the law holds, α_loc should be *flat* across the ladder. Flatness is a diagnostic for E. |
| **Identifiability** | Whether the data pin a parameter down. Within-run points vary D at fixed N, so they constrain β; α needs many distinct N values, and the ladder has only four (eight after the rescue). |
| **D/N ratio** | Tokens per parameter. The design target was 20; the canonical rungs ran 19.6–19.8, the discarded 0.8M rung 22.7, the 9.0M rung 10.0. |
| **Contamination (here)** | Not data leakage but *model-family mixing*: one fit over two architectures. Detected by evaluating the fit's own parameters on each candidate point set. |

### 3.3 Transformer and inference

| Term | Meaning |
|---|---|
| **Decoder-only / causal** | Each position attends only to itself and earlier positions. This is what makes next-token prediction well posed. |
| **KV cache** | Store each layer's key/value tensors for the prefix so each new token costs one step instead of re-running the prefix. Turns generation from quadratic to linear in context. |
| **MHA / MQA / GQA** | Multi-head attention (one K,V per head) / multi-query (all heads share one K,V) / grouped-query (heads share in groups of G). G = 8 is MHA here, G = 1 is MQA. |
| **Speculative decoding** | A small draft model proposes K tokens; the target verifies them in one forward. Exact (rejection sampling keeps the target distribution) but only faster if the target forward is much more expensive than the draft's. |
| **Acceptance rate (α)** | Probability the target accepts a drafted token. At zero draft-target gap it must be exactly 1.0 — the invariant that caught the double-feed bug. |
| **Beam search / top-k / nucleus (top-p) / temperature** | Decoding strategies. Beam keeps the k best sequences; top-k truncates to k tokens; top-p truncates to the smallest set with cumulative probability p; temperature rescales logits. |
| **Best-of-n** | Sample n completions, pick the one the scorer likes best. If the scorer *is* the model, the ceiling is the model's own mode. |
| **Self-consistency** | Sample several reasoning paths and majority-vote the answer. Helps only if the individual paths are better than chance. |

### 3.4 Adaptation and evaluation

| Term | Meaning |
|---|---|
| **SFT** | Supervised fine-tuning on (instruction, response) pairs. |
| **ICL** | In-context learning: put demonstrations in the prompt, no weight updates. |
| **Zero-shot floor** | Accuracy with no demonstrations and no tuning. Here exactly 0.0000 — and *assertable* only because an oracle positive control scores 1.0 through the same scoring path. |
| **LoRA** | Low-rank adaptation: freeze W, learn ΔW = BA with rank r. Trainable params scale with r. |
| **Prompt / prefix tuning** | Learn a soft prompt (input-side) or per-layer prefix keys/values instead of weights. |
| **Adapters (Houlsby)** | Small bottleneck MLPs inserted after attention/FFN. |
| **Matched trainable budget** | Comparing PEFT methods at ~20,000 trainable parameters, so "which method" is not confounded with "how many parameters". |
| **Teacher forcing** | Score the model's probability of the correct response token-by-token, rather than free-running generation. |
| **Exact match** | The response must equal the target exactly. |
| **Groundedness vs truth** | Groundedness = does the output follow the supplied source. Truth needs a fact base, which this project does not have. |
| **Retrieval-augmented generation (RAG)** | Retrieve passages, put them in the prompt, generate. |
| **Context budget (tokens vs characters)** | The v1 bug: a 128-token window was budgeted as 64 *characters* ≈ 32 tokens, so only ~10% of a retrieved chunk survived. Fixed by budgeting tokens and verifying the assembled prompt. |

### 3.5 Interpretability and ethics

| Term | Meaning |
|---|---|
| **Causal head ablation** | Zero a head's attention output before it is mixed back into the residual stream, then re-measure loss. Removes the head's *contribution*, not the head. |
| **Random-ablation null** | Ablate the same number of *random* heads, repeatedly, to get the distribution of "damage from removing heads by position alone". A z-score is measured against this, not against zero. |
| **Redundancy ratio** | ΔCE(pair) / ΔCE(single). Near 1 means the partner is doing nothing extra; near 2 means additive. |
| **Attention rollout** | Multiply layer-wise attention matrices (row-normalised) to attribute output to input positions. |
| **Logit lens** | Apply the model's *own* final norm + unembedding to an intermediate residual, and look at the ranked vocabulary. Avoids training a probe. |
| **Median rank** | Where the target token sits in that distribution. Rank 82 of 6,487 = top 1.3%. |
| **Stereotype probe** | Compare the model's preference for two attribute sets after an occupation prompt, with a negative-control occupation set. |
| **Toxicity lexicon** | Count a fixed list of terms in the model's own generations. |

---

## 4. The experimental setup on one page

| | |
|---|---|
| **Canonical models** (Acts 5–9) | MiniGPT, learned absolute positions, tied embeddings, pre-norm, 128-token context. 1.5M / 2.6M / 6.3M non-embedding (2.61M / 3.94M / 8.01M total). Pinned as weights in `checkpoints/canonical/`. |
| **Scaling ladder** (Act 4) | A *separate* sweep on C4, mixing 11 canonical runs and 5 of an `alibi_like` family (no learned position table). 16 runs, 6 nominal rungs. |
| **TinyShakespeare** (Acts 0–3) | 1,115,394 chars, 66-symbol char vocab, 90/10 split (1,003,854 / 111,540). |
| **C4 shard** (Acts 4–9) | 356,317 docs / 769,299,382 chars, hashed; 6,487-token BPE fitted on 50,025,065 chars of the **train split only**, 2.2726 chars/token. |
| **Held-out sets** | MNLI 300 items (never trained on); synthetic instruction family 512 train / 128 held-out, disjoint by construction. |
| **Judge** | Act 6 scores generated text with the 6.3M model itself → *self-preferring*, valid only for ranking. |
| **Seeds** | n ≥ 3 for effect claims. Where the method is deterministic the "seeds" are bit-identical and the paper relabels the result n = 1 rather than printing a zero-width bar. |
| **Gates** | `tests/` (unit/integration), `tools/check_tex.py` (LaTeX + claims ledger), `tools/reproduce.py` (headline numbers from tracked artefacts). |

---

## 5. Act-by-act map

For each act: **the question → the design → the number → what it means → what
it does not mean → what to point at.**

### Act 0 — n-gram baselines and smoothing (§3, Table 3, Figure 1)

- **Design:** 3 orders (1, 2, 3) × 4 smoothers (Laplace, Good–Turing,
  Kneser–Ney, interpolation) × 3 seeds = 36 runs, on character-level
  TinyShakespeare.
- **Number:** order 1 = 28.427 ppl; order 3 = 7.446–8.778 ppl. Within a fixed
  order the four smoothers span at most 0.9 ppl.
- **Means:** order is worth ~20 ppl; smoothing is worth ≤ 0.9. Kneser–Ney is
  best at order 3 (7.446 ppl, 2.897 BPC).
- **Does not mean:** that smoothing is irrelevant (it is not, at fixed order);
  that interpolation is a worse scheme (its weights are fixed at 0.1/0.3/0.6
  by construction, so the arm is handicapped).
- **Point at:** Table 3. Note the **s.d. = 0.000 in every row** — count-based
  n-grams are deterministic, so the three seeds are one run. The paper prints
  that rather than hiding it.
- **Honesty move:** the §3.3 subsection says the embeddings half of the
  syllabus (SVD, word2vec, GloVe, fastText) is **not implemented and not
  measured** — reported as absent, not dressed up.

### Act 1 — recurrent baselines (§4, Table 4, Figure 2)

- **Design:** RNN / LSTM / GRU, 500 steps, batch 16, context 64, 3 seeds,
  parameter counts only loosely matched.
- **Number:** RNN 2.438 ± 0.215, LSTM 2.590 ± 0.285, GRU 2.482 ± 0.254 nats.
- **Means:** the three are **indistinguishable**; the largest gap (0.152) is
  smaller than the smaller s.d. (0.215).
- **Does not mean:** that gates do not help — the LSTM has 58,434 params
  against the RNN's 31,362, so capacity and steps are confounded.
- **Point at:** Figure 2's right panel, which is a **null**: the gradient norm
  *rises* over 100 steps (0.4363 → 0.4829 RNN; 0.1412 → 0.2828 LSTM) and the
  LSTM's norms are *lower*, the opposite of the story the experiment was built
  to tell. This is the report's first "report the measurement, not the
  narrative" moment.

### Act 2 — Transformer component ablations (§5, Table 5, Figure 3)

- **Design:** 18 configurations, **seed 42 only**, 2 layers, 64 dim, 4 heads,
  context 64, 500 steps. Seven norms, five positional encodings, three head
  counts, three FFN activations.
- **Number:** best is layernorm-post 2.5615; worst is batchnorm-post 2.6610.
  Spread across all 18 is 0.179 nats.
- **Means:** a *ranking* of configurations. Nothing else.
- **Does not mean:** any effect. At n = 1 there is no variance estimate, so no
  comparison is testable. The section contains no ± columns and no
  comparative adjectives without "suggests".
- **Point at:** §5.2 and Table 5's caption, then the ALiBi paragraph. The
  `pe-alibi` and `pe-none` arms differ by **4.8 × 10⁻⁸ nats** because
  `ALiBiPE.forward` returns its input unchanged. So "no positional encoding
  scored best among the encodings actually implemented", and **ALiBi was not
  measured**. This is defect #1 of the taxonomy and the cleanest example of
  "an assertion about what the code does, with no test behind it".

### Act 3 — tokenization (§6, Table 6, Figure 4)

- **Design:** char / BPE / WordPiece / Unigram, **seed 42 only**, under two
  contexts: *equal count* (all get 64 tokens) and *equal span* (all get 64
  characters' worth of text).
- **Number (equal count):** Unigram 2.8021 BPC, BPE 3.0801, char 3.4680,
  WordPiece 3.4998. Under equal span, BPE moves most (3.0801 → 3.1893).
- **Means:** the control matters: part of v1's BPE advantage was *data
  exposure* (BPE saw ~104 characters per 64-token window vs the char model's
  64). Unigram LM is the surprise: at 1.813 chars/token it compresses best.
- **Does not mean:** that any tokenizer family is established — **n = 1**. It
  replaces v1's confounded single-seed result with a *controlled* single-seed
  result, which is progress, not proof.
- **Point at:** Table 6 and §6.2. WordPiece behaves like the char model
  (1.001 chars/token) and slightly worse — the expected direction, and a
  sanity check that the harness works.

### Act 4 — scaling laws (§7, Tables 7–16, Figures 5–7) — the centrepiece

This is the longest act and the one to rehearse. Read it as four audits, not
as a fit.

**4.1 The ladder was two families** (§7.2, Table 7, Table 8).
11 canonical runs (8 heads, learned positions) and 5 of an `alibi_like`
family (no position table). Told apart trivially by the first logged loss:
8.72–8.91 nats vs 129.57–223.77 nats. The 1.5M rung holds *both*
(1,567,200 vs 1,546,720 — exactly one 128×160 position table, 20,480 params).

**4.2 The diagnostic that settles it** (Table 8). Evaluate the *withdrawn
fit's own parameters* on each candidate point set:

| Point set | n | R² |
|---|---|---|
| As tracked (D ≥ 5M, 12 runs) | 2,776 | +0.2874 (it claimed 0.9950) |
| Other family only | 1,428 | **+0.9957** |
| Canonical family only | 2,292 | **−1.3123** |

A fit of these models **cannot** score negative on their own points. That is
an absolute invariant, not a comparison — it is the single most important
number in Act 4.

**4.3 Why it was invisible** (§7.3). Git archaeology: `src/scaling/` was empty
before commit `f1eaf77`; that commit's script builds `ALiBiTransformer` with 4
heads but the *canonical* shapes (dim = 32 × layers). The script that trained
the canonical rungs **was never committed**. So nothing recorded which script
produced which points, and every downstream artefact — including the ledger —
treated the pool as one family.

**4.4 The N audit** (§7.4, Table 9). Three N conventions sit in one table,
none flagged: the CSV column, the run's own metric, and the architecture.
All agree except the 0.8M rung, whose 694,144 matches **no** MiniGPT or ALiBi
shape at any (dim, layers, heads) in dims 16--256 and depths 1--10, while its
embeddings (830,336 = 6,487 x 128) show vocab and width are as expected. It is
a **third, unidentifiable architecture**, has no architecture block, ran on
CPU, sits at D/N = 22.7, and carries the act's worst hold-out fold (R² 0.508
non-embedding, 0.779 total).

**4.5 The N axis** (§7.5, Table 10). Recorded N = non-embedding; published
exponents use total. Three grounds for using total: (i) published convention,
(ii) stability (total: point 0.209 vs fold mean 0.212; non-embedding: 0.098 vs
0.190), (iii) plausibility of E against the *correct* anchors.

**4.6 E is not a measured entropy** (§7.6, Table 11). Measured unigram entropy
under the ladder's own vocabulary is 4.9946 nats (held-out), 5.0068 (train
sample), 5.0136 (full corpus). Both fitted E values (1.348 total / 0.706
non-embedding) lie **below** it. E is an extrapolated asymptote. The earlier
argument ("E must be physical") was aimed at the wrong quantity, because E is
a *conditional* entropy and a marginal entropy is only a ceiling on it.

**4.7 The clean fit** (§7.7, Table 12): canonical family, 2,292 interior
points, 4 rungs, 4 distinct total-N values →
**α = 0.209, β = 0.474, E = 1.348, R² = 0.9955**, leave-one-rung-out
R² = 0.779 / 0.989 / 0.990 / 0.980.

**4.8 The profile asymmetry** (§7.8–7.9, Table 13, Figure 7):
α compatible range **[0.08, 0.54]** (width 0.46) — contains 0.34 *and* 0.50
and most of the plausible range, so it **discriminates between none** of the
published exponents. β compatible range **[0.36, 0.56]** (width 0.20) — it
*excludes* 0.28.

**The mechanism** is the design of the sweep: each run gives hundreds of
points at *fixed N, rising D*, which pins β; α needs distinct N values, and
there are four. Hence "to identify α, widen the N range; to identify β, add
D" — which is why the rescue adds 0.3M and 0.4M rungs rather than more tokens.

**4.9 The spread across variants is the result** (§7.10, Table 14). Across
four defensible fits α runs 0.057 → 0.511. And the mixed fit collapses to
R² = 0.7087 while either pure family fits ≥ 0.995 — contamination is visible
directly as a loss of fit quality.

**4.10 The 129.6–223.8 nat initialisation** (§7.11). Not a metric bug: summed
loss would be ~36,000; a target shift would corrupt both families; rebuilding
the class from the repository's own code reproduces 133.25 / 184.49 / 224.99
against the recorded 129.57 / 183.96 / 223.77. They were **good models** that
finished at 3.578 vs the canonical 3.057.

**4.11 The rescue** (§7.12, Table 15). The plan, the costs, and the *revised
gate*: "does the α profile narrow materially from [0.08, 0.54]?" — which is a
better test than the old one, because a credible null after the right
experiment is a stronger contribution than an excuse.

> ⚠️ **The PDF says the rescue is "in flight and unfinished". That is no
> longer true.** See §12; it finished after this draft, and the rescue changes
> the fit. The numbers above are the PDF's, and they are what you should
> present from this draft; just do not claim the rescue is still running.

**4.12 The equal-D table** (§7.13, Table 16). Reported *with* a provenance
caveat: the three canonical-rung columns do not match the current ladder's
logged points at the same D (gaps of 0.43–0.68 nats), while the other
family's match to within 0.013. The 1.5M/2.6M inversion at D = 2M (0.057 nats)
is 3.3× smaller than the 2.6M rung's 0.186-nat seed spread → **a null**,
reported so you can see it was checked.

**4.13 The pre-registered falsifier** (§7.14). Written 2026-10-15, before any
run: falsified if (i) R² ≤ 0 against the rung-mean baseline, or (ii)
systematic residual curvature. **Not falsified** for the clean fit
(residual correlations −0.00002 and −0.0039). Applied to the *mixed* set it
*would* have been falsified (five of six rungs negative). That is the
condition doing its job in hindsight.

**One-sentence Act 4 verdict:** *the ladder is informative about β and
uninformative about α — and the original headline was a good fit of the wrong
models.*

### Act 5 — inference efficiency (§8, Tables 17–19, Figures 8–10)

**5a KV cache** (Table 17, Figure 8). Latency 51.7→14.0 ms at 128 (3.71×),
85.1→19.1 at 256 (4.46×), 198.2→22.7 at 512 (8.72×), 630.7→34.3 at 1024
(18.36×). **Byte-identical output**: max logit difference 1.64 × 10⁻⁷ nats
= 1.4 ULP in float32, i.e. BLAS accumulation order. Memory is exactly linear
in context (0.50/1.00/2.00/4.00 MB for the 1.5M model; 2.53/5.06/10.13/20.25
for the 9.0M). Caching repays its memory only once the sequence exceeds the
embedding dimension (~288 tokens for the 9.0M model).
*Caveat to state:* 18.36× is a microbenchmark on a purpose-built context-1024
model, **not** a property of the 128-token canonical models.

**5b MQA/GQA** (Table 18, Figure 9). 8× cache reduction at G = 1. FLOPs per
token fall **12.1%** (KV projection 57,344 → 8,192), while the attention
matmul is identical at every G (8,388,608) — so the course claim that these
techniques "neither reduce compute" is false read literally, true only of the
matmul in isolation. Quality cost is real: +0.0253 nats, p = 0.0009 at n = 5;
+0.0285, p = 0.0008 at n = 6, direction consistent 6/6. Seed 42's G = 1 arm
was a 0.73 s resume whose artifact no longer exists, so **n = 5 is primary**
and the provenance of that cell is established by load-semantics inference
(a G = 8 checkpoint *cannot* load into a G = 1 model: shape error
`[48] vs [384]`), not by re-verification. Say that plainly.

**5c Speculative decoding** (Table 19, Figure 10). **0 of 50 cells** beat plain
cached decoding; best cell 0.622×. Acceptance is *high* (0.857 at K = 1 both
gaps), which is what makes it interesting. Mechanism is arithmetic: one round
yields (1 − α^(K+1))/(1 − α) tokens for 2K forwards; at α = 1 that is at most
(K+1)/2K < 1 for K > 1, so K > 1 loses on forward count before hardware
enters. And at 1.5M–6.3M both forwards are dispatch-bound, not compute-bound,
so there is no batching advantage to recover it. **Bounded negative about this
scale, not about the method.**

**The methodology paragraph is the most important one in Act 5** (§8.4): the
context was double-fed, so even identical draft/target weights disagreed
(acceptance 15–40% at zero gap, where it must be 100%). The cached/uncached
equivalence tests both *passed* because both shared the bug. The fix took
minutes once the absolute invariant was stated; throughput went 28.9 → 309.8
tok/s. **A test that compares two implementations cannot detect a fault both
share.** The same class of error: the ledger said beam search used the cache;
it did not, and no test covered beam.

### Act 6 — decoding (§9, Tables 20–22, Figures 11–12)

**6a Sweep** (Table 20, Figure 11). 6.3M model, 40 C4 prompts × 32 new
tokens, 3 seeds = 120 generations per configuration, 15 configurations.
Beam-4 is best on the judge (1.3760 BPC) but costs 4.26× greedy; beam-8 is
*worse* than beam-4 (1.3790) at 2.24× its cost; greedy (1.4366) is the
quality/cost knee. distinct-2 correlates +0.87 with BPC and rep-4 correlates
−0.79, so **diversity costs quality monotonically**. All values were
bit-identical on a repeat run — an independent confirmation that the cache fix
changed cost only.

**6b Best-of-n** (Table 21). Reranking recovers +0.272 BPC at n = 16
(2.0074 → 1.6959) — exactly the maximum-of-n shape — and is **still 0.259
worse than greedy**. Why: the reranker is the model itself, so it can only
pick text the model already finds likely; it converges to the model's mode,
which greedy sits on.

**6c Self-consistency** (Table 22, Figure 12). No gain at any vote count
(0.3789 → 0.3556 → 0.3556 → 0.3611 → 0.3567). The argmax is exactly chance
(0.3300, 99/300, Wilson [0.2792, 0.3851]); majority voting converges to the
argmax, so there is nothing to improve. The model **never** predicts neutral
(recall 0.000; entailment 226 vs 111 gold, contradiction 74 vs 105); on the
216 items it can answer, argmax accuracy is 0.4583 — *below* the 0.5139
coin-flip baseline. Mean max-probability 0.4949, mean entropy 0.7574 vs
1.0986 uniform → biased and over-confident, not noise.

### Act 7 — instruction tuning and PEFT (§10, Tables 23–25, Figures 13–15)

**7a The floor** (§10.2). 0.0000 on all 128 held-out items. Assertable
*because* an oracle emitting the correct answer scores 1.0 through the
teacher-forced, greedy and ICL paths. **That control was run post-hoc** — the
paper says so.

**7b ICL copies format, not operation** (Table 23, Figure 13). On 18 same-task
addition items: the first generated token is the *demonstration's* answer on
**18/18** with a demo, **0/18** without; exact match on the *query's* answer is
**0/18 in both**. The model continues the surface pattern using the most
salient number in context. Three honesty notes: the three ICL "seeds" are
bit-identical (n = 1); shot selection is cost-driven (110 of 128 items get the
same cheapest cross-task demo), so only 18 items are a clean same-task test;
ICL is teacher-forced. And the first attempt was void: a 328-token 4-shot
prompt against a 128-token window, truncation deleted the demos — caught by a
control that recorded "2/4 (truncated away)".

**7c PEFT at matched budget** (Table 24). Full FT 0.5755 (not budget-matched);
adapters 0.2136; LoRA r = 1 0.1823; prompt tuning 0.0859; prefix tuning 0.0000
with final train loss 3.2576 (**a failure to fit, not a scoring artefact** —
that distinction exists because the train-loss column was recorded). The
adapters > LoRA ordering is **suggestive, not established**: Wilson intervals
overlap by ~±0.04, and LoRA at r = 1 is pinned low *by construction*.

**7d LoRA rank sweep** (Table 25). 0.1823 → 0.1927 → 0.2838 → 0.3516 → 0.4245
→ 0.4713 → **0.5078** at r = 64, monotone, no plateau, at 11.5% of full FT's
trainable parameters. Read 7c and 7d together: the matched-budget comparison
compares points on two different curves.

**7e SFT scaling and the LIMA premise** (§10.5). 0.0703 → 0.1667 → 0.3255 →
0.5651 at n = 16/64/256/512, still rising by 0.24 from 256 to 512. This does
**not** contradict LIMA; it removes LIMA's premise. LIMA says tuning elicits a
latent capability; here zero-shot and ICL are both 0.0000, so the curve is
*learning a skill*. The n = 16 row is effectively n = 1 (one batch per epoch,
nothing to shuffle) and is annotated as such everywhere.

**The negative inside the positive:** the `add` task scores 0.9452 on train
(69/73) and **0.0000 on held-out** (0/18), and the held-out guesses are off by
one (8+2 → 11, 7+7 → 15) — the signature of memorising seen sums.

**7f The bar that must not move:** seed-42 full FT is 0.5938 (three seeds
0.5938/0.5234/0.5781, s.d. 0.0370); a re-run on a different device gave
0.6016, which is 0.0078 above the best seed — a device-level reproducibility
band, not a fourth draw.

### Act 8 — grounding, hallucination, tool use (§11, Tables 26–29, Figures 16–17)

**8a Two v1 defects fixed** (§11.1). The context budget was `context_length //
2` **characters** = 64 chars ≈ 32 tokens of a 302-token chunk (9.2% of chars,
10.6% of tokens). The token budget admits 77 tokens (2.4× more) and the
assembled prompt is 120 of 128 tokens. `fit_context()` binary-searches the
longest prefix that fits *after assembly*, re-encodes to verify, and **raises**
rather than returning an unanswerable prompt. One formatter now serves RAG and
SFT.

**8b The choice rates are the result** (Table 26, Figure 16). The model
identifies the asserted number **at chance**:

| Condition | Rate | Wilson 95% |
|---|---|---|
| Oracle (passage states it) | 9/40 = 0.225 | [0.123, 0.375] |
| Closed-book (no passage) | 11/40 = 0.275 | [0.161, 0.428] |
| Retrieved | 8/40 = 0.200 | [0.105, 0.348] |
| Non-numeric passage (control) | 9/40 = 0.225 | [0.123, 0.375] |
| Counterfactual: original number | 9/40 = 0.225 | — |
| Counterfactual: **edited** number | **0/40** | [0, 0.0876] |

Per-item uniform chance is **0.1838**, not 1/3 (candidate sets hold 3–6 distinct
numbers). Oracle vs closed-book: McNemar p = 0.625. Removing the passage changes
the preferred answer on 8/40, but *not in a consistent direction* (closed-book
is higher). Editing the number changes it on 6/40 and the edited number is never
preferred (+0.039 nats, p = 0.839).

**8c The mechanism** (Table 27, Figure 17). A passage raises the likelihood of
numbers *in it* by +0.786 nats and of a number it does **not** state by +0.583.
The only content-specific residual is +0.204 nats (p = 0.030), marginal and
with no behavioural correlate. The strongest evidence is the **non-numeric
control**: a topic- and length-matched sentence with no number scores exactly
9/40, the same as the asserting passage. → **context-number bias**, not
assertion-reading.

**8d Tool use** (Table 29). 0/40 greedy and 0/120 sampled well-formed
`CALC(...)` calls. But **19/40 greedy generations emit the literal `CALC(`**,
and in all 19 the argument is exactly `s`: `CALC(` tokenises to
`['C','AL','C','(']` and the argmax next token is `'s'` (id 97, p = 0.0278),
runner-up `'express'` (p = 0.0214) — a fragment of *expression* from the
prompt's own template. The model completes the instruction's template. At
T = 1.0 the echo dilutes to 2.5%.

**The retriever bug is a *feature* of the act's honesty** (§11.4, Table 28).
The first retrieved condition queried with the *instruction's* words, which
cannot match a number passage, so ranking degenerated to a constant: one
distinct document for all 40 items. The 0/40 was a **query-design failure**,
not a finding. Fixed by using a lexical anchor from a different sentence of the
same document (never the number or the answer sentence): 39 distinct documents,
document hit rate 0.975, chunk contains the number 0.475, follows the retrieved
number when present 0.2857 — and the condition is still at chance (+0.0162,
bootstrap [−0.101, +0.143]).

### Act 9 — interpretability (§12, Tables 30–33, Figures 18–20)

Setup: canonical 6.3M (8 layers, 8 heads, dim 256, context 128), held-out slice
of 64 sequences at C4 document offset 300,000. **Intact baseline measured in
this run: CE 2.8551 nats, ppl 17.376** — so every delta is *internal* and not
comparable to the published 2.5152.

**9a Head ablation with a random-ablation null** (Table 30, Figure 18).
Exactly one of 64 heads has z ≥ 2: **L0 H5**, ΔCE +0.832 nats (+22.5 ppl),
z = +7.82, and z = +42.8 against the random null (mean +0.0099, s.d. 0.0192).
59 of 64 have z ≤ 0. Top-4 combination +1.047 vs +0.071 for four random heads.
Its attention entropy is 0.379 nats vs 2.480 for the inert heads — a sharp head.

**The controls** (Table 31): not an attention sink (position-0 mass 0.0164 vs
uniform 0.0078; top-five nearly flat); does not collapse the model (2.8551 →
3.6869 against uniform 8.7776); and **a backup head covers for it** — ablating
L0 H5 + L0 H2 costs +0.0234, *exactly* L0 H2 alone (redundancy 0.97), and
L0 H5 + L0 H0 likewise +0.0226, while L0 H5 + L1 H6 is additive (+0.9591 vs
0.9079). So **a group of layer-0 heads carries the effect**, not one head. The
same head is top on a second slice (z = +37.1, 3 heads at z ≥ 2, CE 2.9028).

Two honesty notes: the top-4 bootstrap [0.0365, 0.6390] is over the **four
individual deltas** and does **not** contain +1.047, so +1.047 is a point
estimate; and head ablation removes a head's *contribution*, not the head.

**9b Attention rollout, anchored to chance** (Table 32). Against a ground truth
(Act 8's asserted-number token) neither rollout nor raw last-layer attention
ever puts its max on the answer token: **0/34 each**, against uniform chance
0.0078. Mean attribution mass 0.00759 / 0.00673 vs uniform 0.00781 — no
signal. Max lands on the final position 0/71 times, so it is not collapsing
either. **This is why the paper never shows an attention heatmap as evidence.**

**9c The logit lens, and the control that retracted the claim** (Table 33,
Figures 19–20). Median rank of the asserted number at layer 5 = **82** of
6,487 (top 1.3%). But the **counterfactual-edited** number reaches the *same*
82, the non-numeric control 86, a number in no passage 484, a random token 735.
So the lens is **number-specific, not assertion-specific**.

> **The strong claim is withdrawn.** The first version said the model encodes
> *which* number is asserted and destroys it before output. The controls went
> against that, and the paper withdraws it in place. The supported claim:
> *context numbers are represented in the residual stream; assertion-specific
> content is not.*

**9d The link** (§12.4). Acts 7b, 8b, 8d and 9c are one picture: the model
holds a generic number and fails to *carry* it, so it copies formats and
templates instead of performing operations. That is an interpretation of two
measurements, and the paper labels it as one.

### Act 10 — evaluation methodology and ethics (§13, Tables 34–37)

**10a The ledger** (Table 34). `report/claims.csv`: one row per claim with n,
interval, status, source file, caveat. Two machine-checked rules: every prose
claim has a row, every row is referenced from the prose. Status vocabulary:
confirmed / n.s. / exploratory / ceiling / invalid / superseded / open-item.

**10b The failure taxonomy** (Table 35). 15 recorded defects, each with what it
was, how it was caught, what it invalidated. **All 15 were caught by an
absolute invariant or a positive control; none by comparing two
implementations.** The best examples: the negative R² on the fit's own models;
acceptance must be 100% at zero gap; CE 11.98 nats against ln(6487) = 8.78;
first-token accuracy 0.070 vs positions 1–2 at 0.927/0.900 (missing input);
4.8 × 10⁻⁸ nats between two "different" PE arms.

**10c Ethics and calibration** (Table 36).
- **Stereotype probe:** all three rates 1.0, *including the negative control* →
  a **global attribute preference**, not an association. And the instrument is
  underpowered: at n = 10 only a rate above 0.8099 is distinguishable from
  chance.
- **Toxicity:** 0/40, Wilson [0, 0.0876], corpus base rate 0.0. Floor.
- **Calibration (the positive result):** 4,064 predictions, accuracy 0.5295 at
  mean confidence 0.5255, **ECE 0.0233**; bins track (0.979 → 0.962;
  0.054 → 0.074).
- **SFT vs base, two slices:** on generic text accuracy falls 0.5295 → 0.4227
  (−0.1068, [−0.1284, −0.0852]) and ECE rises 0.0233 → 0.1306. On the tuned
  distribution accuracy is flat (−0.0009, [−0.0160, +0.0142], n.s.) while mean
  confidence rises 0.3969 → 0.6261 and ECE doubles 0.1118 → 0.2772.
  → **"confidence inflation on target, accuracy loss off target"**, not a
  simple trade.
- SFT instruction accuracy 0.6016 (77/128, Wilson [0.5150, 0.6822]).

**10d Reproducibility** (Table 37). `tools/reproduce.py` re-derives the
headline numbers from tracked checkpoints; it trains nothing. (The text says
"nine"; the table lists ten. See §12.)

---

## 6. The three findings, in the form to say them aloud

### Finding 1 — a calibrated model that cannot use what it knows

> "The model is a good next-token predictor: 52.95% accuracy at 52.55%
> confidence over 4,064 held-out predictions, ECE 0.0233, and the bins track
> from 0.98 confidence to 0.96 accuracy. It is a terrible instruction follower:
> 0.0000 zero-shot, at chance on grounding, and its tool call is `CALC(s)`.
> That combination is the point. Every downstream failure in the paper is a
> failure to *use* a competence, not a failure to *have* one."

Why it is credible: 4,064 predictions, a deterministic instrument, and a
positive control for the floor (oracle = 1.0).

Weak point to concede: one model, one checkpoint, one corpus. "Well calibrated"
is a statement about C4 next-token prediction at 6.3M.

### Finding 2 — instruction tuning retargets, and the two slices prove it

> "Tuning on 512 examples takes held-out instruction following from 0.0000 to
> 0.6016. On the distribution it was tuned on, next-token accuracy does not
> change (−0.0009, not significant) — but confidence rises 23 points and ECE
> goes 0.1118 → 0.2772. Off target, accuracy drops 10.7 points and ECE goes
> 0.0233 → 0.1306. So tuning did not add competence and did not even trade it
> one-for-one: it *retargeted* the model's confidence onto the new format and
> paid for it everywhere else."

Why it is strong: the two slices are disjoint from training, and the comparison
is same-checkpoint, same-instrument, same items.

Weak point: one fine-tune (n = 1 in substance for the tuned model).

### Finding 3 — the contaminated ladder, and the invariant that caught it

> "The scaling ladder mixed two architecture families, because the script that
> trained one of them was never committed. Nothing in the fit noticed: R² was
> 0.995, residuals were flat in log N and log D, and every held-out rung was
> above 0.985. What caught it was an absolute invariant: the withdrawn fit's
> own parameters score R² = −1.3123 on the canonical models' points. A fit of
> these models cannot do that. The withdrawn exponents describe the other
> family, which owned the entire top of the ladder."

Why it is the best result in the paper: it is a *methodological* result with a
number attached, and it generalises: tag every point with its configuration;
prefer absolute invariants; a good-looking fit is not evidence of a good fit.

Weak point: the corrected fit is itself exploratory (family split and axis
chosen after seeing provenance), and the ladder is small.

---

## 7. The negative-results catalogue (one line each)

**Unsupported (26 in the taxonomy).** RNN vs LSTM vs GRU ordering; vanishing
gradients; ALiBi competitiveness; any Act 2 or Act 3 effect at n = 1; published
α identification; published β compatibility; any tokenizer-family claim;
speculative decoding speedup at this scale; best-of-n beating greedy with a
self-scorer; self-consistency gains; adapters-vs-LoRA ordering; LIMA's
"less is more" at 6.3M; ICL absence; grounding; retrieval benefit; tool use;
head-ablation "one head carries the model"; the strong logit-lens claim;
factual truth; hallucination-rate comparison between conditions; extrapolation
to scale.

**Measurement failures (15).** See Table 35; the important ones are:

| # | Defect | Caught by |
|---|---|---|
| D1 | RAG context budgeted in characters (9.2% of a chunk survived) | Token accounting vs the assembled prompt |
| D6 | BPE fitted on validation text | Split audit |
| D7 | Unseeded validation batches | Seeded full sweep |
| D11 | Input text never reached the prompt | Per-position accuracy profile (0.070 vs 0.927) |
| D12 | PEFT arms applied a second CE shift | All-zero arms that were "training" |
| D13 | 4-shot ICL prompt 328 tokens in a 128 window | Positive control recording visible shots |
| D14 | Grounding score absorbed the prompt's own log-probability (−246 nats) | Absolute plausibility check |
| D18 | Held-out metric compared logits to the wrong position (11.98 nats) | Comparison to ln(6487) = 8.78 |
| D19 | Random-ablation null against a stale baseline (−88 vs +42.8) | Sanity check on the z itself |
| — | Ladder mixes two families | Negative R² on own points + provenance read |
| — | ALiBi arm is a no-op | 4.8 × 10⁻⁸ nats |
| — | Speculative decoding double-fed context | Acceptance must be 100% at zero gap |
| — | Retriever degenerate (one document for all items) | Inspecting what it returned |
| — | Chance = 1/3 was wrong (0.1838) | Counting candidates per item |
| — | Sweep sizes mislabelled (20 vs 18; 5 vs 4) | Counting runs in the files |

---

## 8. Number cheat sheet

Keep this open during the meeting.

| Quantity | Value |
|---|---|
| Canonical models | 1.5M / 2.6M / 6.3M non-embedding; 2.61M / 3.94M / 8.01M total; 128 context |
| 6.3M validation loss | 2.5152 nats/token (≈ 3.63 bits/token) |
| Next-token accuracy / confidence | 0.5295 / 0.5255, n = 4,064 |
| ECE (base) | **0.0233** |
| ECE after SFT (generic slice) | 0.1306 |
| Accuracy after SFT (generic) | 0.4227 (−0.1068, [−0.1284, −0.0852]) |
| SFT on-target accuracy | −0.0009, [−0.0160, +0.0142], n.s. |
| SFT on-target confidence / ECE | 0.3969 → 0.6261 / 0.1118 → 0.2772 |
| Instruction following, base → SFT | 0.0000 → 0.6016 (77/128, [0.5150, 0.6822]) |
| SFT curve | 0.0703 / 0.1667 / 0.3255 / 0.5651 at n = 16/64/256/512 |
| LoRA r = 1 → 64 | 0.1823 → 0.5078 (11.5% of full FT params) |
| PEFT at 20k budget | adapters 0.2136, LoRA 0.1823, prompt 0.0859, prefix 0.0000 |
| ICL same-task | 18/18 copy the demo's answer; 0/18 exact match |
| Grounding (oracle / closed-book) | 9/40 (0.225) / 11/40 (0.275); McNemar p = 0.625 |
| Chance for Act 8 | 0.1838 per item |
| Passage lift | +0.786 asserted / +0.583 not asserted / +0.204 residual (p = 0.030) |
| Tool calls | 0/40 well-formed; 19/40 echo `CALC(` with argument `s` |
| KV cache | 3.71× / 4.46× / 8.72× / 18.36× at 128/256/512/1024; logit diff 1.6e−7 |
| MQA | 8× cache, 12.1% FLOPs, +0.0253 nats (p = 0.0009, n = 5) |
| Speculative decoding | 0/50 cells beat plain; best 0.622× |
| Decoding | beam-4 1.3760 vs greedy 1.4366 for 4.26× time |
| Best-of-16 | 1.6959 (+0.272) — still 0.259 worse than greedy |
| Self-consistency | 0.3300 argmax = chance; never predicts neutral |
| Head ablation | L0 H5: +0.832 nats, z = +7.82 / +42.8; redundancy 0.97 |
| Logit lens | asserted 82, edited 82, non-numeric 86, absent 484, random 735 (of 6,487) |
| Rollout | 0/34, chance 0.0078 |
| Act 4 clean fit (PDF) | α 0.209, β 0.474, E 1.348, R² 0.9955, 2,292 points, 4 rungs |
| Act 4 profiles (PDF) | α [0.08, 0.54]; β [0.36, 0.56] |
| Withdrawn fit | α 0.445, β 0.290, E 1.605; R² +0.9957 on other family, −1.3123 on canonical |
| Ledger | 102 rows: confirmed 75, exploratory 8, invalid 9, n.s. 7, superseded 1, ceiling 1, open 1 |
| Taxonomy | 89 rows: 10 supported, 26 unsupported, 15 failures, 38 declined |

---

## 9. Index of tables and figures

Use this to find the right exhibit in one look. Numbers are the PDF's.

### Tables

| # | Act | What it is |
|---|---|---|
| 1 | — | **Findings taxonomy**, 89 rows, four populations (front of paper) |
| 2 | Method | The three pinned canonical models (params, steps, tokens, loss) |
| 3 | 0 | n-gram perplexity/BPC, 12 configurations × 3 seeds |
| 4 | 1 | RNN / LSTM / GRU losses and parameter counts |
| 5 | 2 | 18 ablations, single seed — the ranking |
| 6 | 3 | 4 tokenizers × 2 context controls |
| 7 | 4 | The ladder split by architecture family |
| 8 | 4 | **Contamination diagnostic** (R² on three point sets) |
| 9 | 4 | Per-rung N audit against three sources |
| 10 | 4 | Two parameter conventions side by side |
| 11 | 4 | Measured unigram entropy vs fitted E |
| 12 | 4 | **The clean fit** (α, β, E, R², folds, profile) |
| 13 | 4 | Profile likelihood summary, both conventions |
| 14 | 4 | α across every defensible fit variant |
| 15 | 4 | The rescue plan and its gate |
| 16 | 4 | Equal-D table, with provenance caveat |
| 17 | 5a | KV cache latency, 4 contexts |
| 18 | 5b | MQA vs MHA, 6 paired seeds |
| 19 | 5c | Speculative decoding, 50 cells |
| 20 | 6a | **15-configuration decoding sweep** |
| 21 | 6b | Best-of-n reranking |
| 22 | 6c | Self-consistency on MNLI |
| 23 | 7b | ICL same-task control (18 items) |
| 24 | 7c | PEFT family at matched budget |
| 25 | 7d | LoRA rank sweep |
| 26 | 8b | **Grounding choice rates, five conditions** |
| 27 | 8c | Log-probability contrasts (the mechanism) |
| 28 | 8b | Retriever before/after repair |
| 29 | 8d | Tool use, greedy and sampled |
| 30 | 9a | **Head ablation vs random null** |
| 31 | 9a | The three head controls (sink, collapse, redundancy) |
| 32 | 9b | Attention rollout vs chance |
| 33 | 9c | **Logit-lens controls, five targets** |
| 34 | 10a | Headline claims from the ledger |
| 35 | 10b | **The 15-defect failure taxonomy** |
| 36 | 10c | **SFT calibration, two slices** |
| 37 | 10d | The reproduction checks |
| 38 | Threats | **Claims we do not make** |

### Figures

| # | Act | What it shows |
|---|---|---|
| 1 | 0 | Perplexity vs n-gram order; BPC of all 12 configs |
| 2 | 1 | RNN/LSTM/GRU losses; **gradient norms rising** (the null) |
| 3 | 2 | 18-configuration ranking |
| 4 | 3 | Tokenizer BPC under both controls |
| 5 | 4 | **The ladder, two families drawn separately** |
| 6 | 4 | Loss vs N at fixed D (the D = 2M inversion) |
| 7 | 4 | **Profile likelihood over α and β** |
| 8 | 5a | KV cache latency and memory vs context |
| 9 | 5b | Six paired MQA−MHA differences |
| 10 | 5c | Speculative speedup and acceptance vs K |
| 11 | 6a | Quality/diversity/cost trade-off |
| 12 | 6b/6c | Best-of-n and self-consistency |
| 13 | 7b | ICL: copies the demo, fails the query |
| 14 | 7a/7b | SFT curve with zero-shot and ICL lines |
| 15 | 7c/7d | PEFT at matched budget; LoRA rank sweep |
| 16 | 8b | Choice rate by condition + the ceiling panel |
| 17 | 8c | Paired rate and log-prob contrasts |
| 18 | 9a | Per-head ablation with the random-null band |
| 19 | 9c | The five lens targets, per layer |
| 20 | 9c/9b | Lens curve and rollout hit rates |

---

## 10. Section-by-section: what to say

Assume 15 minutes and a professor who will interrupt. The italic line is the
sentence to land; the pointer tells you what to put on the table.

**§1 Introduction (1 min).**
*"This is a measurement project, not a capability demonstration. Three tiny
models, ten acts, and every act has to end with a table, error bars and a
conclusion that says what it does not support."*
→ Point at **Table 1** and say: "89 hypotheses, four populations — 10
supported, 26 unsupported, 15 measurement failures, 38 claims we decline to
make."

**§2 Method (2 min).**
*"The important methodological choices are four: the item is the unit of
analysis, so 0/40 is [0, 0.088] and not [0, 0.031]; every tokenizer is fitted
on the train split only and validation is a seeded full sweep; every A-vs-B
comparison ships with the control that makes it mean something; and we only
have one pre-registered falsifier, in Act 4 — I'll say where the others
weren't."*
→ **Table 2** (models), then the falsification subsection. Conceding the
pre-registration gap early buys credibility for everything else.

**§3 Act 0 (1 min).**
*"Order is worth 20 perplexity; smoothing is worth at most 0.9. And the
standard deviation is exactly zero in every row, because count-based n-grams
are deterministic — so I print the zero rather than pretending to have three
seeds."*
→ **Table 3**, **Figure 1**. Add: "the embeddings half of this act is not
implemented, and I report it as absent."

**§4 Act 1 (1 min).**
*"The three recurrent models are indistinguishable — the largest gap is
smaller than the smaller standard deviation. And the vanishing-gradient
demonstration is a null: the norms rise, and the LSTM's are lower."*
→ **Figure 2**, right panel.

**§5 Act 2 (1 min).**
*"This is a ranking, not an effect: one seed, no error bars, and no
comparative adjective without 'suggests'. And one arm didn't run what it
names — ALiBi is identical to no-positional-encoding to 4.8 × 10⁻⁸ nats."*
→ **Table 5**, then the ALiBi paragraph.

**§6 Act 3 (1 min).**
*"With the context control, BPE's advantage shrinks: part of v1's 10% was data
exposure. Unigram LM is actually best. But this is single-seed, so it replaces
a confounded single-seed result with a controlled one — it does not establish a
tokenizer family."*
→ **Table 6**.

**§7 Act 4 (4 min — the one to rehearse).**
*"This is the act I'd most want you to push on. The original claim was α =
0.445, consistent with Chinchilla, R² ≥ 0.985 on every rung. It was wrong, and
the way it was wrong is the result. The ladder mixed two architecture
families — 11 runs of the canonical model, 5 of a different one with no
learned position table — because the script that trained one of them was never
committed. Nothing in the fit complained: R² 0.995, flat residuals, every
held-out rung above 0.985. What caught it was an absolute invariant: the
withdrawn fit's own parameters score R² = −1.3123 on the canonical models'
points, and a fit of these models cannot do that. They score +0.9957 on the
other family, which owned the entire top of the ladder."*
→ **Table 8** (the diagnostic), then **Figure 5** (two families).
*"What survives is narrower and more honest. On the canonical family, on the
total-parameter axis — the axis the published exponents actually use — the fit
gives α 0.209, β 0.474. The profile's compatible α range is [0.08, 0.54], so
the ladder is compatible with published exponents across their whole range and
discriminates between none of them. β is two to three times better constrained,
[0.36, 0.56], and on this draft it excludes 0.28. The asymmetry has a
mechanism: each run gives hundreds of points at fixed N and rising D, so D
pins β, while α needs distinct N values and there are four."*
→ **Table 12**, **Table 13**, **Figure 7**.
*"And the fit is exploratory: the family split and the axis were decided after
the provenance was read. Across four defensible variants α runs 0.057 to
0.511."*
→ **Table 14**.

**§8 Act 5 (2 min).**
*"The cache gives byte-identical output — max logit difference 1.6 × 10⁻⁷
nats, which is BLAS accumulation order — and 3.7× to 18.4× speedup, but the
18.4 is a microbenchmark on a context-1024 model, not the 128-token models
here. MQA is the interesting one: 8× cache and 12.1% fewer FLOPs, which
contradicts the course claim that it doesn't reduce compute — at a real quality
cost of 0.025 nats, p = 0.0009. Speculative decoding is a bounded negative:
zero of fifty cells beat plain decoding, and the reason is arithmetic, not
implementation — 2K forwards buy at most K+1 tokens."*
→ **Table 17**, **Table 18**, **Table 19**.
Then, deliberately: *"This act also contains the bug I'm most proud of
catching. The cached and uncached tests both passed because both shared a
double-fed context. What found it in minutes was an invariant: acceptance must
be exactly 100% at zero draft-target gap. It was 15.6%."*

**§9 Act 6 (1.5 min).**
*"Diversity costs quality monotonically. Beam-4 is 0.061 bits better than
greedy for 4.26× the time, so greedy is the knee. Best-of-16 recovers 0.272
bits and is still 0.259 worse than greedy — because the reranker is the model,
so it can only pick text the model already likes. Self-consistency gives
nothing because the argmax it amplifies is exactly at chance and never predicts
the neutral class."*
→ **Table 20**, **Table 21**, **Table 22**.
Concede immediately: *"the quality judge is the model under test, so this ranks
configurations; it does not measure text quality."*

**§10 Act 7 (2.5 min).**
*"Zero-shot is exactly 0.0000, and that's assertable only because an oracle
scores 1.0 through the same scoring path. Tuning takes it to 0.6016,
monotonically in examples, with no saturation. But it's format, not operation:
the addition task is 0.9452 on train and 0.0000 on held-out, and the held-out
guesses are off by one. ICL is the same shape: 18/18 copy the demonstration's
answer as the first token, 0/18 answer the query."*
→ **Figure 14** (the SFT curve), **Table 23** (ICL), **Table 25** (LoRA), **Table 24** (matched budget).
*"Two caveats I'd rather state than defend: the seeds vary optimisation order
only, because the base weights are pinned, so between-initialisation variance
is unmeasured; and the task family is synthetic."*

**§11 Act 8 (2 min).**
*"Supplying a passage doesn't change what it says. It picks the asserted
number 9 of 40, closed-book 11 of 40, McNemar p = 0.625 — at chance. A
counterfactual edit is followed 0 of 40 times. And the control that decided
the claim was a topic-matched passage containing no number: it scores exactly
9 of 40, the same. So what we have is a context-number bias, not
assertion-reading. The mechanism is in Table 27: a passage raises the
likelihood of numbers it contains and numbers it doesn't by roughly the same
0.6–0.8 nats."*
→ **Table 26**, **Table 27**, **Table 28**.
*"Tool use: zero well-formed calls, but 19 of 40 greedy outputs emit `CALC(`
with the argument `s` — it's completing the prompt's own template."*

**§12 Act 9 (2 min).**
*"One head of 64 is causally important by a random-ablation null — layer 0
head 5, +0.83 nats, z = +42.8 — but the control that matters is redundancy:
ablating it together with layer 0 head 2 costs exactly what head 2 costs
alone, so a backup head covers for it. The effect is a group of layer-0 heads,
not one head. Attention rollout never identifies the answer token, 0 of 34,
and puts no more mass on it than uniform. The logit lens does see a number —
rank 82 of 6,487 at layer 5 — but the counterfactual-edited number reaches the
same 82, so it's number-specific, not assertion-specific, and I withdrew the
stronger claim rather than keeping it."*
→ **Table 30**, **Table 31**, **Table 33**.

**§13 Act 10 (2 min).**
*"The ledger is 102 rows, machine-checked in both directions: every claim has
a row, every row is cited, every source file exists. The taxonomy is 15
defects, and the sentence I'd like you to test me on is this: all fifteen were
caught by an absolute invariant or a positive control, none by comparing two
implementations. And the ethics act has one positive result: the base model is
well calibrated, ECE 0.0233 — and tuning destroys that: 0.1306 off target,
and on target the accuracy doesn't move while confidence rises 23 points."*
→ **Table 34**, **Table 35**, **Table 36**.

**§14 Threats to validity (1 min).**
*"I've consolidated the limitations rather than scattering them. The biggest
is that Acts 2 and 3 are single-seed, so they rank and do not establish. Act 4
is the most caveated and the section says so in one place. And §14.4 is the
list of claims we deliberately do not make — 38 of them."*
→ **Table 38**. Offering this table before being asked is the strongest move
in the meeting.

**§15 Reproducibility (30 s).**
*"Every number traces to a tracked file. Two gates: a checker that verifies
the LaTeX and the ledger, with a negative self-test that feeds it deliberately
broken copies and asserts it fails; and a script that re-derives the headline
numbers from the tracked checkpoints without training anything."*
→ **Table 37**.

---

## 11. Questions you should expect, with answers

**Q1. Why so small? Isn't this just a toy?**
Yes, deliberately. The paper claims nothing about scale, and says so in §2.5
and §14. At 6.3M the effects are isolable and the experiments are affordable
enough to repeat. The contribution is the measurement design and the
negative results, not capability.

**Q2. Is the calibration result a real result or a tautology?**
A real measurement: 4,064 held-out predictions, ECE 0.0233, bins that track
from 0.98 → 0.96 and 0.05 → 0.07. It is *not* a claim about truth or
instruction following — the same model scores 0.0000 there. That separation is
the point.

**Q3. How do you know the ECE isn't an artefact of binning?**
The bins are reported, the instrument is deterministic, and the SFT comparison
uses the identical item set and instrument — so even if the absolute value were
sensitive to binning, the *change* (0.0233 → 0.1306) is not.

**Q4. Did you pre-register?**
One act: Act 4, with a written falsifier dated 2026-10-15. Everything else is
post-hoc, and Table 1 marks each row. The project's real defence is the
taxonomy: every recorded failure was caught by an invariant or a positive
control, and the decisive controls are labelled post-hoc.

**Q5. Why is α unidentified but β not?**
Design of the sweep. A run logs hundreds of checkpoints at fixed N and rising
D, so within-run variation constrains β. α depends on how many distinct N
values the ladder spans, and it had four. Also α and E trade off along a ridge
(dα/dE = +0.5786), so the α profile is wide even at R² 0.995.

**Q6. "Compatible across the entire range" sounds like you found nothing.**
That *is* the finding, and it is the correct one: a ladder this size does not
discriminate between published exponents. The paper is explicit that a range
excluding nothing cannot support "consistent with" either. The contribution is
the demonstration that the experiment cannot settle the question, plus the
diagnosis of why.

**Q7. So is the β exclusion a disagreement with Chinchilla?**
Careful here — this is the one number in flux. In the draft, β's compatible
range is [0.36, 0.56] and 0.28 falls outside, which would be a genuine
disagreement. But the rescue changed the point set, and the current refit puts
β near 0.19 with a compatible range that *includes* 0.28. So the defensible
thing to say is: "the draft reports an exclusion; the rescue I ran afterwards
makes β move, which is itself evidence that β is not robust to the point set.
I would not present the exclusion as settled." See §12.

**Q8. Was the contamination your mistake?**
Yes, and it is recorded as a process failure: the training script for one
family was never committed, so a mid-campaign change of model class was
invisible to every downstream artefact — including the ledger that existed to
record exactly that. The fix is procedural: tag every point with a config hash
computed from the live model at logging time, never retroactively.

**Q9. Isn't the 18× cache speedup meaningless at 128 tokens?**
The paper says so in the caption and in the conclusion. The canonical models
are context-128; the 18.36× is a mechanism microbenchmark on a purpose-built
context-1024 model. At 128 it is 3.71×, which is the relevant number.

**Q10. Speculative decoding is known to work. Did you implement it wrong?**
The implementation is exact and the zero-gap invariant passes at 100%. The
negative is arithmetic: 2K forwards buy at most K+1 tokens, so K > 1 loses on
forward count before hardware enters; and at this scale both forwards are
dispatch-bound. It would pay when the target forward is compute- or
bandwidth-bound, which is the large-model regime. This is a bounded negative
about our scale.

**Q11. Best-of-n losing to greedy — isn't that just a weak judge?**
It is *exactly* the judge: the reranker is the model itself, so it converges to
the model's own mode, and greedy sits on that mode. The result is a statement
about self-referential reranking, which is why the paper says beating greedy
would need an independent scorer.

**Q12. 0.60 instruction accuracy — did it learn or memorise?**
Both, and the split is measured: the five easy string tasks score 0.61–1.0, the
reverse task 0.167, and all 24 addition tasks score 0.0000 on held-out (while
addition is 0.9452 on train, with guesses off by one). It learned format and
string operations; it did not learn arithmetic.

**Q13. Is the grounding instrument powerful enough?**
At n = 40 and chance 0.1838, it detects a true rate of 0.40 with ~87% power,
0.35 with ~69%, 0.30 with ~42%. The observed 0.225 is far below the first
threshold, so the null is not a power failure for a medium effect — but a true
rate of 0.30 would often be missed. State that, and note the non-numeric
control scoring identically is the stronger evidence: it is a 9/40 vs 9/40
match, not a failure to detect.

**Q14. One head or a group of heads?**
A group. L0 H5 is the head most visible in isolation (+0.83 nats, z = +42.8),
but ablating it with L0 H2 costs exactly what L0 H2 costs alone (0.97
redundancy), and with L0 H0 likewise. With L1 H6 it is additive, so the
redundancy is specific, not general fragility.

**Q15. You withdrew the logit-lens claim. Why should I trust the rest?**
Because the withdrawal is the method working. The control set was run against
the claim, it went the other way (edited number reaches the same rank 82), and
the claim was rewritten in the form the controls support: number-specific, not
assertion-specific. A paper that keeps a claim after its own control refutes it
is the one you should distrust.

**Q16. Which result are you least confident in?**
Act 4's corrected exponents, and the reason is in the paper: four N values, one
single-seed rung, an unidentifiable 0.8M architecture, mid-schedule points, and
a spread of 0.057–0.511 across defensible variants. The contamination finding
is solid; the numbers that replace it are exploratory.

**Q17. What is the single most generalisable lesson?**
A test that compares two implementations cannot detect a fault both of them
share. Prefer an absolute sanity invariant — acceptance must be 100% at zero
gap; a fit cannot score negative R² on its own models; CE cannot exceed
ln(vocab). All fifteen recorded failures were caught that way.

**Q18. What would you do with more compute?**
Three things, in order: finish the N-range rescue properly and re-run the
profile with ≥ 8 distinct N values; re-run Acts 2 and 3 at n = 3 (they are the
only single-seed sections); and give the grounding instrument 200+ items and an
independent judge, since the current one cannot detect a small true effect.

---

## 12. Where to concede — and the three stale numbers

Read this before you walk in. All of it is already in the paper except the
first item.

### 12.1 The Act 4 rescue is no longer "in flight" (⚠️ material)

The PDF (§7.12, Table 15, §14.2) says the rescue is **in flight and
unfinished**, and that Act 4's numbers are the four-rung clean fit. That was
true when the draft was made. **`report/final_report.tex` has since been
rewritten** and now reports the completed rescue. What is in the tex:

- every rescued arm finished — 0.3M (4.1695), 0.4M (4.0563), 0.8M (3.4157),
  4.1M (2.6254), 9.0M@D=10N (2.5436), 9.0M@D=20N (2.4267);
- the fit now has **8 rungs and 8 distinct N values**, on the **pre-registered
  D ≥ 5M point set**: 3,463 interior points, α = +0.589, β = +0.461,
  E = +1.964, R² = 0.9960;
- the α profile's compatible range is **[0.34, 0.82]** (width 0.48) and the
  β profile's is **[0.38, 0.54]** (width 0.16), which still *excludes*
  Chinchilla's 0.28;
- the pre-registered gate — "does the α range narrow materially from
  [0.08, 0.54]?" — is **answered no**: 0.46 at four rungs, 0.48 at eight. The
  null is credible, not overturned.

**What this means for the meeting.** The paper's Act 4 numbers and the rescue
now agree; present them as one story. If asked whether the exponents are
stable, answer with the point set: *"The paper reports the pre-registered
D ≥ 5M set, which is the set the falsification condition was written against.
A per-run-transient inclusion rule gives a different β that would include 0.28;
that is an exploratory variant, not the pre-registered analysis, and the
profile profiler is not trustworthy on that set."* Do **not** say "β includes
0.28" — that is the other point set, and saying it would contradict the paper.

What the rescue *did* change is worth stating plainly, because it is the
result: α moved from +0.209 to +0.589, E from 1.348 to 1.964, the point set
from 2,292 points over 4 rungs to 3,463 over 8, and the qualitative verdict did
not move at all.

### 12.2 Three stale numbers inside the PDF

| Where | Says | Actually |
|---|---|---|
| §13.1 (10a) and Table 34's caption | "99 rows" / "27 of the 99 rows" | The ledger holds **102** rows |
| §13.4 (10d) | "nine headline numbers" | Table 37 lists **ten** checks |
| §15 (Reproducibility) | "359 tests" | The suite has **387** tests, 0 failures, 1 skip |

None affects a claim; they are bookkeeping drift between the ledger, the
generated fragments and the prose. Fix them before the next compile, and if the
professor spots one, say exactly that: the ledger is the source of truth and
the prose count is stale.

### 12.3 Known structural weaknesses (state them first)

1. **Acts 2 and 3 are n = 1.** Rankings, not effects. This is the biggest
   caveat and it is repeated in every caption.
2. **The corrected Act 4 fit is exploratory.** Family split and N-axis
   convention were chosen after reading provenance.
3. **One 4-run family, one 0.8M rung that matches no known architecture.**
   Only a re-run repairs it.
4. **Act 6's judge is the model under test.** Ranks configurations, does not
   measure quality.
5. **Act 7's seeds vary optimisation order only.** Between-initialisation
   variance unmeasured.
6. **Act 8 has 40 items and numbers proxy for facts.** Groundedness, not truth.
7. **Act 9's deltas are internal** to a 64-sequence slice.
8. **Act 10c's SFT comparison is n = 1 in substance** (one fine-tune).
9. **The web interface is not part of the paper's claims.** It exists and runs;
   no claim rests on it.
10. **Only one act has a pre-registered falsifier.**

---

## 13. If the conversation turns to "what next"

Three concrete, already-designed follow-ups — do not invent new ones:

1. **Finish Act 4 properly.** The rescue is done; what remains is to regenerate
   §7 from the rescued ladder, re-run the profile, and apply the pre-registered
   gate. Say what the gate is: *does the α profile narrow materially from
   [0.08, 0.54]?*
2. **Multi-seed the single-seed acts.** Acts 2 and 3 at n = 3 are cheap and
   would convert two rankings into two effects.
3. **Strengthen the instruments where the null is weakest.** Act 8 at 200+
   items with an independent judge; Act 6 with a held-out judge rather than the
   model under test.

And one that is explicitly *not* planned: RLHF/DPO. It was excluded by design
(the course's alignment lecture is out of scope for this project), and the
paper does not gesture at it.

---

## 14. Appendix — file map and 60-second verification

### Where things live

| Path | What |
|---|---|
| `report/final_report.tex` | The paper (this PDF's source) |
| `report/figures/*.png`, `*.pdf` | 24 tracked figure files; 20 referenced, 4 documented exclusions |
| `report/tables/*.csv`, `*.tex` | Source data for every table + 2 generated fragments |
| `report/claims.csv` | **102-row ledger** — the audit table |
| `report/MANIFEST.md` | Per-figure/table audit: source file, generator, row count, n |
| `results/v2/act*/` | Tracked metrics for Acts 4–10 |
| `results/v1/` | v1 metrics (superseded except where cited) |
| `checkpoints/canonical/` | Pinned weights + SHA-256 + `INDEX.json` |
| `tools/check_tex.py` | LaTeX + ledger gate, with `--self-test` |
| `tools/reproduce.py` | Re-derives the headline numbers; trains nothing |
| `src/scaling/` | Ladder, manifest, refit, local exponent |

### Verify a number in 60 seconds

```bash
# the ledger row for any claim (status, n, interval, source)
grep -i "calibration" report/claims.csv

# the LaTeX/ledger gate (must print OK)
python tools/check_tex.py

# re-derive the headline numbers from tracked artefacts
python -m tools.reproduce

# the test suite
python -m unittest discover -s tests
```

### Three sentences to have word-perfect

1. *"A test that compares two implementations cannot detect a bug that both of
   them share."*
2. *"A good-looking fit is not evidence of a good fit."*
3. *"Every downstream failure in this paper is a failure to use a competence,
   not a failure to have one."*

### Three sentences not to say

1. "Our α is 0.34" — it is not; the point estimate is 0.209 and the profile
   spans [0.08, 0.54].
2. "The model is consistent with Chinchilla" — the ladder is compatible across
   the whole range, which discriminates between nothing.
3. "The rescue is still running" — it finished; see §12.1.
