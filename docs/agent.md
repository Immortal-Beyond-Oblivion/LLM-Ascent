# Project Ascent: LLM — Master Plan & Kanban

## Project principles

- Core implementation is modular Python; notebooks are not required for operation.
- Every ticket has a focused assertion or sanity check before it can be marked done.
- Small CPU-friendly data and configurations are the default; all training entry points will accept CLI arguments and optional YAML configuration for Colab runs.
- `docs/state.md` is read before every new implementation action. Owner comments take priority over the currently planned ticket.
- Measured metrics, commands, configuration, and interpretation are recorded in `docs/implementation.md`.

## Architecture decisions

| Area | Initial decision | Evolution path |
|---|---|---|
| Model family | Decoder-only causal Transformer | Retain causal core through LoRA, RAG, and agent layer |
| Baseline tokens | Character vocabulary | Trainable BPE tokenizer in Phase 2 |
| Dataset | TinyShakespeare-compatible local text | Tiny instruction subset and local RAG documents |
| Runtime | PyTorch, CPU-first smoke configurations | CUDA/Colab selected only through configuration |
| Configuration | `argparse` plus optional YAML | Same interface across all phases |
| Evaluation | Reproducible JSON/CSV metrics and generated artifacts | Cross-phase charts and LaTeX report |

## Execution order and tickets

Status legend: `TODO`, `ACTIVE`, `BLOCKED`, `DONE`. `IMPLEMENTED (VALIDATION BLOCKED)` means source and tests exist, but a required runtime test has not yet executed.

## Implemented architecture snapshot (handoff)

### Phase 1 foundation

- `src/tokenizer/character.py`: deterministic Unicode character tokenizer; `<unk>` is ID 0 and corpus characters are sorted for reproducible IDs.
- `src/data/text.py`: UTF-8 loading and shifted causal batches shaped `[B, T]` with `torch.long` IDs.
- `src/model/gpt.py`: decoder-only MiniGPT with tied embeddings, learned positions, causal attention, residual blocks, and next-token cross-entropy.
- `src/training/train.py` and `train.py`: argparse/optional YAML configuration, CPU/CUDA selection, metrics, checkpointing, tokenizer persistence, and strict resume compatibility checks.
- `src/generation/decoding.py`: greedy, beam, top-k, and top-p decoding.

### Phase 2 upgrades

- `src/tokenizer/bpe.py`: deterministic character-initialized BPE with JSON serialization and exact training-text round trips.
- The transformer supports RMSNorm and PyTorch scaled-dot-product causal attention with a manual fallback.
- Checkpoints persist tokenizer type and BPE merge state, preventing a model/tokenizer mismatch after resume.

### Phase 3 adaptation and RAG

- `src/adaptation/instructions.py`: Alpaca-style formatting and response-only labels; truncation preserves response supervision.
- `src/adaptation/lora.py`: frozen-base low-rank adapters in attention `qkv` and output projection layers.
- `src/adaptation/run_finetune.py`: JSONL micro-instruction LoRA CLI and adapter-only artifact output.
- `src/rag/retriever.py` and `src/rag/query.py`: local document chunking, lexical retrieval, source provenance, bounded RAG prompts, and checkpoint-backed generation.

### Phase 1 — Vanilla Baseline

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 101 | `[TICK-101]` Implement character tokenizer | DONE | — | Encode/decode round-trip and token tensor type/range assertions pass |
| 102 | `[TICK-102]` Add TinyShakespeare-compatible data loader and batching | DONE | 101 | Batches have integer shape `[B, T]` and causal targets match the shifted inputs |
| 103 | `[TICK-103]` Implement decoder-only mini-GPT | DONE | 102 | Forward pass returns logits `[B, T, V]`; causal-mask test passes |
| 104 | `[TICK-104]` Add training loop, checkpoints, CLI/YAML configuration | DONE | 103 | CPU smoke training reduces loss and resumes from a checkpoint |
| 105 | `[TICK-105]` Add greedy generation and baseline evaluation | DONE | 104 | Generation runs with bounded output; loss, perplexity, and sample are logged |

### Phase 2 — Architectural Polish

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 201 | `[TICK-201]` Implement/train BPE tokenizer | DONE | 105 | Deterministic encode/decode and serialized vocabulary test pass |
| 202 | `[TICK-202]` Add RMSNorm and optimized attention variants | DONE | 201 | Shape, finite-gradient, and causal-attention tests pass |
| 203 | `[TICK-203]` Add beam, top-k, and top-p decoding | DONE | 202 | Each decoder respects token limits and deterministic seeded sampling works |
| 204 | `[TICK-204]` Run architectural comparison evaluation | DONE | 203 | Perplexity and repetition comparison are logged |

### Phase 3 — Adaptation & Fine-Tuning

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 301 | `[TICK-301]` Add micro-instruction data formatter | DONE | 204 | Prompt/target masking and batch-shape tests pass |
| 302 | `[TICK-302]` Implement LoRA adapters and PEFT trainer | DONE | 301 | Base weights remain frozen and adapter gradients update |
| 303 | `[TICK-303]` Implement local-document retrieval pipeline | DONE | 302 | Relevant chunks are retrieved and included within context budget |
| 304 | `[TICK-304]` Evaluate instruction following and RAG | DONE (SMOKE) | 303 | Metrics/samples and retrieval provenance are logged |

### Phase 4 — Agentic Layer, Prompting & Final Evaluation

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 401 | `[TICK-401]` Implement zero-shot, few-shot, and CoT prompt evaluation | DONE | 304 | Prompts and scored qualitative results are reproducibly logged |
| 402 | `[TICK-402]` Add prompt-chain memory and calculator tool | DONE | 401 | Multi-turn context persists and safe calculator calls pass tests |
| 403 | `[TICK-403]` Build evaluation charts and analysis figures | DONE | 402 | Loss/perplexity/parameter, attention, and latency figures render |
| 404 | `[TICK-404]` Generate LaTeX final report and final comparison | DONE | 403 | `final_report.tex` builds structurally with readable one-column-safe tables |

## v2 program (Phases 5–8): from "the pipeline works" to measured capability

**Goal.** Replace v1's plumbing evidence with measured, statistically honest results on four questions: (1) does the BPE advantage survive scale, multiple seeds and a context-matched control; (2) can the model follow held-out instructions after SFT/LoRA; (3) is RAG answering actually grounded in retrieved text; (4) can the model initiate tool calls itself. Plus a localhost web interface to interact with the models and browse the results.

**Freeze rule.** v1 is complete and frozen: `report/final_report.tex`, `report/LLM_Ascent_Project_Report v1.pdf`, `artifacts/tick-*`, and `tests/test_report.py` stay untouched. All v2 outputs go to `artifacts/v2/`, `data/v2/`, `report/final_report_v2.tex` (figures in `report/figures_v2/`). The v1 test suite (59 tests) must keep passing at every ticket.

**Design details, protocol, API spec, and risks are in `docs/implementation.md` → "v2 design record".** This section holds the ticket board, claim gates, and order of work.

### Claim gates (decided before any v2 run)

A claim may appear in the v2 report only if its gate is met; otherwise the report states the negative or inconclusive result as measured.

| Gate | Claim it unlocks | Required evidence |
|---|---|---|
| G1 | "BPE beats character tokenization at scale" | Lower mean validation bits/char for BPE in **all 3 seeds**, gap larger than the across-seed std, and the sign holds under the **equal-character-span context control** (removes v1's context confound). Full validation sweep, tokenizers fitted on the train split only. |
| G2 | "The model follows held-out instructions" | Held-out exact-match (n ≥ 200 per family) with 95% Wilson interval whose lower bound exceeds both the no-SFT baseline's upper bound and the majority/chance baseline, consistent across 3 seeds. Reported per family and per split (unseen operands / unseen words / unseen template). |
| G3 | "RAG answers are grounded" | On unseen-entity questions, oracle-context EM minus closed-book EM ≥ 0.20 with an interval excluding 0, **and** a counterfactual-swap context-following rate reported. Retrieval recall@k/MRR reported separately from generation. |
| G4 | "The model initiates tool calls" | Model-emitted valid `CALC(...)` rate and end-to-end accuracy with the tool vs without, on held-out operand ranges including longer numbers than seen in training. |
| G5 | "The interface works" | Backend tests pass; owner confirms each tab in a browser on `127.0.0.1`. |

### Phase 5 — v2 foundations (agent builds, owner runs a pilot)

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 501 | `[TICK-501]` Training infrastructure v2 + compute pilot | TODO | 404 | MPS/CUDA/CPU auto-selection, warmup+cosine LR, gradient clipping, dropout flag, seeded full-sweep validation, best-checkpoint saving, tokenizer fitted on train split only, multi-seed sweep runner; tests pass; owner pilot logs seconds/step for scale ladder S/M/L and a scale decision is recorded |
| 502 | `[TICK-502]` Synthetic world and instruction dataset generator | TODO | 501 | Deterministic (seeded) generator writes `data/v2/*.jsonl` + `world_docs.txt`; tests prove zero train/held-out overlap on entities, operand pairs, words and templates; per-family counts and vocabulary-coverage report logged |
| 503 | `[TICK-503]` Corpus mix, tokenizer v2, response terminator | TODO | 502 | Character and BPE tokenizers cover every character of the instruction/RAG/tool formats with 0 `<unk>`; chars/token reported; response end marker defined and tested; mix builder reproducible |

### Phase 6 — v2 training and evaluation (long owner-run compute)

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 601 | `[TICK-601]` Pretraining and tokenizer comparison at scale | TODO | 503 | 3 seeds × {character, BPE} plus equal-span control; mean ± std and per-seed values logged; **G1 decided** |
| 602 | `[TICK-602]` Instruction tuning and held-out evaluation | TODO | 601 | Full fine-tune vs LoRA (rank sweep, attention-only vs +MLP) vs no-SFT, minibatched with validation; per-family/per-split EM with Wilson intervals over 3 seeds; **G2 decided** |
| 603 | `[TICK-603]` RAG v2: retrieval and grounded-answer evaluation | TODO | 602 | RAG prompt uses the trained instruction format and fits the context; retrieval recall@k/MRR (overlap vs BM25); controls closed-book / oracle / retrieved / distractor-only / unanswerable / counterfactual swap; **G3 decided** |
| 604 | `[TICK-604]` Tool-use SFT, agent evaluation, prompting revisit | TODO | 602 | Model-initiated `CALC` rate and end-to-end accuracy with vs without tool incl. longer-number split; zero/few/CoT re-run on held-out arithmetic at the new context; **G4 decided** |

### Phase 7 — localhost interface (parallel track; needs only existing checkpoints to start)

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 701 | `[TICK-701]` Web backend: model registry and JSON API | IMPLEMENTED (OWNER RUN PENDING) | 404 | Stdlib server on `127.0.0.1`; `/api/{health,models,generate,chat,rag,attention,results}`; allow-listed model ids (no client paths), Host-header check, body/token caps; tests start the server on an ephemeral port with a tiny random checkpoint |
| 702 | `[TICK-702]` Web frontend single page | IMPLEMENTED (OWNER RUN PENDING) | 701 | One offline HTML file (no CDN): Playground, Chat+Tools, RAG side-by-side (closed-book vs with context), Attention heatmap, Results tabs; shows latency, token counts, and `<unk>` warnings; owner confirms in browser |
| 703 | `[TICK-703]` Results tab for v2 artifacts + run script + README section | TODO | 701, and 601–604 for real data | Results tab renders v2 summary JSON/figures when present and degrades cleanly when absent; one documented launch command; **G5 decided** |

### Phase 8 — v2 report

| Order | Ticket | Status | Depends on | Acceptance criterion |
|---:|---|---|---|---|
| 801 | `[TICK-801]` v2 figures | TODO | 601–604 | Error-bar loss/bpc curves, per-family held-out accuracy, retrieval-control chart, tool on/off chart, v1-vs-v2 comparison table render |
| 802 | `[TICK-802]` `final_report_v2.tex` + tests + PDF | TODO | 801, 703 | Builds structurally; every number cross-checked by test against the v2 summary JSON; every claim maps to a gate; negative results stated; v1 files untouched |

### Order of work and who does what

The agent has no shell, so each wave ends with owner-run commands; long compute is started early so it runs while the interface is built.

| Wave | Agent writes | Owner runs | Parallel note |
|---|---|---|---|
| 1 | 501, 502 | tests; pilot (timing/scale ladder) | — |
| 2 | 503 | tests; **start 601 compute** (background, hours) | — |
| 3 | 701, then 702 | tests; open browser, report issues | built while 601 runs |
| 4 | 602, then 603, then 604 | tests; SFT / eval runs | needs 601 checkpoints |
| 5 | 703, 801, 802 | tests; compile PDF | final assembly |

Total: **12 tickets** (3 + 4 + 3 + 2). Owner compute budget is a provisional assumption of a few hours in total; it is replaced by measured numbers after the TICK-501 pilot.

## Current handoff / next-agent instructions

1. Do not redo Phase 1–3 implementation: the owner has confirmed the expanded test suite passes (17 tests), and TICK-204 is closed with real metrics in `docs/implementation.md`.
2. v1 (TICK-101–404) is **complete and frozen**: 59 tests OK, `report/LLM_Ascent_Project_Report v1.pdf` built. Do not edit v1 files (see the freeze rule above).
3. TinyShakespeare has almost no digits (its 66-entry vocabulary includes `<unk>`), so prompts spell numbers out (e.g. "four"); the evaluator logs `unknown_prompt_tokens` per item to verify this.
4. The 8-step baseline and 6-step LoRA results are plumbing smoke tests, not language-quality evidence. Do not present them as final benchmark results.
5. v2 is planned (Phases 5–8 above; design in `docs/implementation.md` → "v2 design record"). TICK-701/702 (web interface) are code-complete with `tests/test_web.py` and await the owner's test run and browser check (see `docs/state.md`); the next agent ticket is `[TICK-501]`. Pre-existing v1 defects that v2 must fix are listed in the design record (§1).

## Known limitations / technical debt

- BPE training, checkpoint resume, and the TICK-204 comparison are done (single seed, 500 steps; see confounds in `docs/implementation.md`).
- Resume is unit/integration-tested; a longer resumed experiment belongs in final evaluation evidence.
- RAG is intentionally dependency-free lexical retrieval. Semantic retrieval is an optional extension, not a missing requirement.
- BPE is course-scale and deterministic, not production GPT byte-level BPE.
- NumPy and matplotlib are now installed in `.venv` (figures and report verified).

## Ticket transition rule

Before marking a ticket `DONE`, record its test command and result in `docs/implementation.md`, then update `docs/state.md` to name the next ticket and its focus. Do not begin a ticket whose dependencies are incomplete.
