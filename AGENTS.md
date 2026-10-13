# AGENTS.md — LLM-Ascent (Project Ascent, DS 356)

**Read this file completely before doing anything in this repository.** It is the
full project context and it is designed to be self-sufficient: a fresh session on
a different machine, with no access to this conversation, should be able to pick
up the work from here alone.

If you are an AI agent: `docs/state.md` is the running session log, `docs/agent.md`
is the ticket board, and `docs/implementation.md` is the metrics ledger and v2
design record. This file is the summary and the map.

---

## 1. What this project is

An independent course project (DS 356) that builds a decoder-only Transformer
language model **from scratch**, in plain Python with no notebooks, and measures
how each design choice changes the result.

The deliverable is a **LaTeX report**: `report/final_report.tex` plus tracked
figures in `report/figures/`. The owner compiles it on **Overleaf**, so `report/`
must stay self-contained and must not depend on git-ignored paths.

**The thesis**: this is *not* "build an LLM and show it works". The models are
deliberately tiny and will remain too small to be useful. The point is to
**measure the effect of every technique** and report what the measurement says —
including negative results.

---

## 2. The story: ten acts

Each act removes a limitation the previous act exposed. Every act must end in
**the same shape**: a table with error bars, one or two figures, and a stated
conclusion. A negative or null result is a valid, publishable outcome. An act is
not done until it has all three.

| Act | Topic | Lectures | Cost | Status |
|---|---|---|---|---|
| 0 | n-grams, smoothing, embeddings (word2vec, GloVe, fastText) | L2, L3, L4 | cheap | todo |
| 1 | RNN / LSTM / GRU, seq2seq ± attention | L5, L6 | cheap | todo |
| 2 | Transformer anatomy + component ablations | L7, L8 | moderate | todo |
| 3 | Tokenizers: char, BPE, WordPiece, Unigram LM | L11 | cheap | todo |
| 4 | Scaling laws (fit your own α, β) + MoE | L12 | **expensive** | todo |
| 5 | KV cache, MQA / GQA / SWA, speculative decoding | L10, L16 | moderate | todo |
| 6 | Decoding sweep + best-of-n / self-consistency | L2, L6, L12 | cheap | todo |
| 7 | SFT, ICL-vs-SFT, PEFT family, LoRA rank sweep | L13, L15 | moderate | todo |
| 8 | Hallucination, RAG, tools, agents | L17 | moderate | todo |
| 9 | Interpretability: rollout, head ablation, logit lens | — | cheap | todo |
| 10 | Evaluation methodology + ethics metrics | L2, L12 | cheap | todo |

### Sequencing

```
Stage A  Enablers         MPS support, experiment framework, seed harness,
                          uniform metrics schema, tracked results       [NEXT]
Stage B  Cheap wins       Acts 0 + 1
Stage C  Core thesis      Acts 2 + 3
Stage D  Headline         Act 4  (scaling laws — the centrepiece)
Stage E  Depth            Acts 5 + 6 + 7
Stage F  Capability       Acts 8 + 9 + 10
Stage G  Showcase         Report bundle, Results UI, final write-up
```

Stages A–C are cheap and already make a presentable project. **Stage D is what
makes it exceptional.** Do not skip ahead to Stage D without Stage A, because the
scaling sweep is meaningless without the seed and metrics harness.

---

## 3. Compute: adaptive, overnight-capable

**Primary machine is an Apple M4 with 16 GB unified memory.** The project was
originally developed on an M3 with 8 GB; code must stay correct and runnable at
8 GB even though the M4 is the working machine. MPS support did not exist in v1
(`resolve_device` was CUDA-only — defect D8) and must be added in Stage A.

Approximate usable memory for torch: **~10–12 GB of the 16 GB**.

| | 8 GB (M3 fallback) | **16 GB (M4 primary)** |
|---|---|---|
| Comfortable training | 10–25M params | **50–150M params** |
| Largest single run | ~30M | ~500M with reduced batch/context |
| Scaling ladder | 0.5M – 30M | **1M – 500M** |

### Sweep policy — adaptive, incremental

Act 4 is the only compute-heavy act. Its policy is:

- The sweep is **resumable and checkpointed**. Every completed point is a usable
  data point, so stopping early still yields a valid (if partial) fit.
- Run the **cheap end first** (smallest models), scale up, and keep going until
  the owner says stop. Overnight completion is the aspiration, not the requirement.
- Never start a large run before a small run of the same family has succeeded.

**The method that makes this affordable**: Kaplan and Chinchilla fit
*intermediate checkpoints*, not just final models. **One training run yields
dozens of `(N, D, C, loss)` points** by logging loss every ~50 steps. Ten configs
give ~400 points and genuinely support fitting α and β. Do not treat a sweep as
"one point per model" — that is what makes it expensive.

Disk: checkpoints must be **evicted as a sweep progresses**. A multi-hour run
must not be able to fill the disk.

---

## 4. The concept coverage checklist

Derived from all 18 lecture PDFs. Use this as the definition of "done" for the
syllabus. ✅ = built and measured · 🔶 = built but never measured · ❌ = absent.

### Tokenization & representation (L2, L3, L4, L11)
- ❌ N-gram LM: unigram / bigram / trigram, Markov assumption
- ❌ Laplace smoothing; Good-Turing (`c* = c − 0.75`, Church & Gale 1991); Kneser-Ney; interpolation
- ❌ Perplexity; Shannon Game; intrinsic vs extrinsic evaluation
- ❌ One-hot vs distributional semantics; WordNet 3.0
- ❌ tf-idf; co-occurrence matrix; SVD low-rank reduction
- ❌ word2vec CBOW and Skip-gram, negative sampling, frequent-word subsampling
- ❌ GloVe (co-occurrence probability *ratios*, least-squares objective)
- ❌ fastText (character n-grams with `<` / `>` boundary symbols)
- 🔶 Character tokenizer
- 🔶 BPE — one merge count only, single seed, confounded
- ❌ WordPiece (`##` continuation, likelihood-based pair scoring)
- ❌ Unigram LM / SentencePiece (EM-style)

### Attention (L7, L10, L16)
- 🔶 Self-attention, scaled dot-product, multi-head, causal mask
- ❌ Cross-attention; encoder-decoder attention
- ❌ KV cache — **not implemented in v1**; `generate` recomputes the whole prefix each step
- ❌ Multi-Query Attention; Grouped-Query Attention; Sliding Window Attention; MHA→MQA uptraining
- ❌ Online softmax; tiled / block-wise attention (FlashAttention's IO story, without a GPU)
- ❌ Speculative decoding, speculative sampling, rejection sampling

### Architecture (L5, L6, L7, L9, L12)
- ❌ Fixed-window NNLM; RNN and BPTT; LSTM (forget/input/output gates); GRU
- ❌ Vanishing/exploding gradients — demonstrated, not asserted
- ❌ seq2seq encoder-decoder; the bottleneck problem; NMT
- 🔶 Transformer block, pre-norm, tied embeddings, position-wise FFN
- ❌ **Encoder-only model (BERT / ELMo)** — MLM with 15% masking and the 80/10/10 rule; NSP
- ❌ **Mixture of Experts**: router, noisy top-K gating, load-balancing loss `N Σ f_i P_i`, capacity factor
- ❌ Switch Transformer; expert parallelism

### Normalization (L8)
- 🔶 LayerNorm
- ❌ BatchNorm (and why it is structurally wrong for variable-length sequences)
- ❌ RMSNorm
- ❌ Pre-norm vs post-norm
- ❌ Internal covariate shift

### Positional encoding (L8)
- 🔶 Learned absolute (only option implemented)
- ❌ Sinusoidal `sin/cos(pos / 10000^(2i/d))`
- ❌ RoPE; relative PE; ALiBi; none (the ablation baseline)

### Training objectives (L9, L13)
- 🔶 Causal / autoregressive LM
- ❌ Teacher forcing; seq2seq loss `J = (1/T) Σ J_t`
- ❌ MLM; NSP; bidirectional vs unidirectional
- ❌ Data curation and filtering

### In-context learning & instruction tuning (L13)
- 🔶 Few-shot prompting — evaluated, scored 0/6
- ❌ **SFT quality measured on held-out instructions** (never done in v1)
- ❌ ICL vs SFT vs full fine-tune (LIMA: "less is more")
- ❌ Self-Instruct; Evol-Instruct; SuperNatural Instructions; Orca

### Decoding (L6, L13)
- 🔶 Greedy, top-k, nucleus, beam
- ❌ **Systematic sweep** across greedy / beam width / k / p / temperature
- ❌ Best-of-n and self-consistency
- ❌ Accuracy-vs-diversity tradeoff curve

### Alignment (L14) — **OUT OF SCOPE by owner decision**
- ❌ RLHF, PPO, reward models, DPO — **deliberately excluded.** Do not implement.

### Scaling laws (L12) — the centrepiece
- ❌ Fit own Kaplan/Chinchilla exponents: `L(N,D) = E + A/N^α + B/D^β`
- ❌ Compare fitted α, β against published values
- ❌ Emergence vs metric artifact (exact-match → edit distance; MCQ → Brier score)

### PEFT & compression (L15)
- 🔶 LoRA — attention-only targets, full-batch, no validation, no adapter loader
- ❌ Prompt tuning; prefix tuning; adapters (Houlsby)
- ❌ LoRA rank sweep r ∈ {1,2,4,8,16,32,64}
- ❌ PEFT family head-to-head at matched trainable-parameter budget
- ❌ Quantization (PTQ, int8, `s = (2^(b−1)−1)/α`); QLoRA
- ❌ Pruning: magnitude vs Wanda; unstructured vs structured
- ❌ Distillation; sequence-level distillation

### Evaluation (L2, L12)
- 🔶 Bits-per-character; repetition rate; exact/contains match
- ❌ **Error bars everywhere** — mean ± std, n ≥ 3, Wilson intervals
- ❌ Held-out evaluation with proper splits

### RAG / hallucination / agents (L17 + syllabus)
- 🔶 Lexical retriever, `/calc` command, conversation memory
- ❌ Hallucination **measured** as a rate (L17 is motivation only)
- ❌ RAG with oracle / closed-book / counterfactual controls
- ❌ Model-initiated tool use (never observed — v1 models cannot emit `CALC(...)`)
- ❌ Chains, ReAct-style traces

### Interpretability (syllabus; incidental in L6, L12)
- 🔶 Attention heatmaps (proves causal masking only)
- ❌ Attention rollout; logit lens; MoE routing visualisation
- ❌ **Causal head ablation** — disable head (ℓ,h), measure Δperplexity

### Not covered by any lecture deck
Bias and toxicity, prompt engineering, chains/memory/agents, multimodal,
model editing, self-evolving LLMs, small language models, BART/T5.
**Notes**: lecture 18 is empty; lecture 17 is truncated and never actually
teaches RAG. These topics come from the syllabus only — use standard sources.
The owner has confirmed external sources are acceptable here.

---

## 5. Known defects in v1 — do not build on these

Found in v1 and logged in `docs/implementation.md` §1. Any act that touches one
of these must fix it, or the measurement is invalid.

| # | Defect | Fix required |
|---|---|---|
| D1 | RAG context truncated to `context_length // 2` **characters** | token-based budget |
| D2 | RAG prompt format ≠ SFT format | one shared prompt builder |
| D3 | Vocabulary lacks digits, `#`, `(`, `)` → `<unk>` | byte-level / expanded vocab |
| D4 | SFT is full-batch on 3 examples, no validation | minibatching + held-out split |
| D5 | No response terminator | add EOS to responses |
| D6 | BPE fitted on the **validation** text too | fit on train split only |
| D7 | Validation uses unseeded random batches | seeded full sweep |
| D8 | CUDA-only device; no LR schedule; no dropout flag | MPS + schedule + flags |
| D9 | Context confound in the tokenization comparison | equal-character-span control |
| D10 | No adapter loader — `adapter.pt` unusable by the UI | base + adapter registry |

Also: **no KV cache** (recomputes the prefix every step), **no error bars anywhere**,
**single seed everywhere**, **no held-out instruction eval**, **no metrics schema**
(every script invents its own output shape, which is why results cannot be
aggregated into a report).

---

## 6. Repository layout

```
src/
  tokenizer/   character.py, bpe.py, checkpoint.py
  data/        text.py                      (Encoder, TextDataset, batch())
  model/       gpt.py, normalization.py     (MiniGPT, GPTConfig, RMSNorm)
  training/    train.py                     (TrainConfig, train(), CLI)
  generation/  decoding.py                  (greedy/sample/top-k/top-p/beam)
  adaptation/  instructions.py, lora.py, finetune.py, run_finetune.py
  rag/         retriever.py, query.py
  agent/       agent.py, calculator.py, memory.py, chat.py
  prompting/   prompts.py, evaluate.py
  eval/        metrics.py, compare_tokenizers.py, attention.py, figures.py
  web/         server.py, registry.py, handlers.py, static/index.html
tests/         11 modules, 100 tests
data/          tinyshakespeare.txt, knowledge.txt, *.jsonl
report/        final_report.tex, figures/ (tracked), tables/
results/       tracked metrics: results/v1/, results/v2/
docs/          state.md, agent.md, implementation.md, lectures/ (git-ignored)
artifacts/     checkpoints and raw runs — GIT-IGNORED, local only
AGENTS.md      this file
```

### v2 additions (Stage A)

```
src/bench/     experiment.py (schema + runner), seeds.py, stats.py (mean±std, Wilson)
src/ngram/     ngram.py, smoothing.py
src/embed/     svd.py, word2vec.py, glove.py, fasttext.py
src/rnn/       rnn.py, lstm.py, gru.py, seq2seq.py
src/encoders/  bert.py  (MLM + NSP)
src/tokenizer/ wordpiece.py, unigram.py, byte_level.py
src/moe/       router.py, sparse_moe.py
src/cache/     kv_cache.py
data/v2/       synthetic task families with held-out splits
results/v2/    tracked metrics, one subdir per act
```

---

## 7. Setup and commands

```bash
cd ~/Desktop/ProjectWorkspace/LLM/LLM-Ascent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
# Full suite — expect 100 tests, all passing
.venv/bin/python -m unittest discover -s tests -v

# Single module
.venv/bin/python -m unittest tests.test_web -v

# Regenerate v1 figures and the report bundle
.venv/bin/python -m src.eval.figures

# Web interface (stdlib only, no CDN) → http://127.0.0.1:8000
.venv/bin/python -m src.web.server --port 8000
```

**No `pdflatex` is required or installed.** The owner compiles on Overleaf.

---

## 8. Conventions for v2 work

These are not optional. They exist because v1 could not produce a defensible claim.

1. **Every experiment declares a config and emits uniform `metrics.json`.** One
   schema for all acts, so `results/` aggregates into the report automatically.
2. **Every claim needs n ≥ 3 seeds**, reported as mean ± std. Accuracy metrics get
   Wilson confidence intervals. A single-seed result is labelled exploratory.
3. **Controls before comparisons.** Every A-vs-B experiment ships with the control
   that makes the comparison mean something (equal-character-span for tokenizer
   work; oracle/closed-book/counterfactual for RAG; matched trainable-parameter
   budget for PEFT).
4. **Pre-register the claim before running.** Write down what would falsify it.
   Anything run after seeing results is labelled exploratory.
5. **Fix D6/D7 in every training path.** Tokenizer fit on train split only;
   validation from a seeded full sweep. These silently invalidate results.
6. **Negative results are results.** Report them with the same rigour. Do not
   quietly drop a configuration that did not work.
7. **One ticket at a time**, each with an executable assertion. Update
   `docs/state.md` at the end of every session.

---

## 9. Current state

- **v1 is frozen.** 100 tests passing. `report/final_report.tex` + tracked
  figures; owner compiles on Overleaf.
- **v1's only real comparison** is character vs BPE: 3.4525 vs 3.0946 bits per
  character, single seed, 500 steps, four self-declared confounds.
- **v1's headline numbers that are negative**: prompting scored 0/6 in every
  style and both tokenizers; the 500-step models cannot emit `CALC(...)` at all;
  RAG answer quality was never measured.
- **The web interface has never been opened in a browser.** Verify it early.
- **Stage A is the next work.**

### Environment note

`docs/lectures/` is 165 MB of course PDFs and is git-ignored on purpose.
Fetch it from the course share. Everything else in `docs/` is tracked.
