# Project Ascent: LLM — Implementation Ledger & Metrics

## Metric conventions

| Metric | Definition | Lower/upper is better |
|---|---|---|
| Training loss | Mean next-token cross-entropy on the training split | Lower |
| Validation loss | Mean next-token cross-entropy on held-out data | Lower |
| Perplexity | `exp(validation_loss)` | Lower |
| Repetition | Repeated n-gram fraction in generated samples | Lower |
| Latency | Wall-clock generation time and tokens/second | Lower / higher |
| Parameters | Total and trainable parameter counts | Context-dependent |

## Implementation history

### 2026-10-04 — Project management initialized

- Created the master plan, session tracker, and metrics ledger.
- Selected `[TICK-101] Implement Character Tokenizer` as the active ticket.
- No executable source code, training run, or quantitative model result has been produced yet.

### 2026-10-04 — Ticket 101 implementation in progress

- Added `CharacterTokenizer` in `src/tokenizer/character.py` with a deterministic vocabulary: `<unk>` is ID 0 and all corpus characters are sorted in Unicode order.
- Added public package exports plus `pyproject.toml` and `requirements.txt` declaring PyTorch as the model runtime dependency.
- Added tests for canonical vocabulary order, known-character round trips, unknown character substitution, empty input, invalid inputs, and the expected tensor interface.
- Verification command: `python3 -m unittest discover -v`.
- Result: 5 passed, 1 skipped. The skipped assertion checks real `torch.Tensor` type, `torch.long` dtype, one-dimensional token shape, and IDs in `[0, vocabulary_size)`; it is pending because PyTorch is not installed in the active interpreter.
- No training metrics apply to tokenizer-only work.

### 2026-10-04 — Phases 1–3 source implementation

- Phase 1: added deterministic text loading and shifted causal batches (`src/data`), decoder-only `MiniGPT` (`src/model`), greedy generation, and `train.py` / `src.training.train` with argparse plus optional YAML configuration, checkpoints, metrics JSON, and CPU/CUDA selection.
- Phase 2: added deterministic, JSON-serializable BPE (`src/tokenizer/bpe.py`), RMSNorm and scaled-dot-product causal attention configuration, greedy/beam/top-k/top-p decoding, and an n-gram repetition metric.
- Phase 3: added Alpaca-style instruction formatting with prompt-loss masking, a frozen-base LoRA implementation and adapter checkpoint CLI, plus local text chunk retrieval and a source-provenance RAG query CLI.
- Portable verification command: `python3 -m unittest discover -s tests -v`; result: **7 passed, 5 skipped**. The completed checks cover character tokenization, BPE round-trip/serialization, and source-aware bounded RAG retrieval.
- Skipped checks require PyTorch and cover tokenizer tensor type/range, causal batching, model logits/causality, decoding shape, and LoRA freezing/loss-masked batches.
- No training, perplexity, repetition, instruction, or RAG-quality metrics are recorded because PyTorch is not installed in an available interpreter. This project deliberately does not synthesize quantitative results.

### 2026-10-04 — Real PyTorch validation and Phase 1/3 smoke runs

- Installed PyTorch 2.14.1 in the project `.venv`. Fixed instruction truncation to retain supervised response targets under a context limit.
- Verification command: `.venv/bin/python -m unittest discover -s tests -v`; result: **12 passed, 0 skipped**.
- Phase 1 CPU smoke configuration: TinyShakespeare, seed 7, 1 layer, 32 embedding dimensions, 4 heads, 32-token context, batch size 4, 8 steps.
- Baseline results: step 1 train/validation loss `4.1388 / 4.1424`, validation perplexity `62.9564`; step 8 train/validation loss `3.9288 / 3.9212`, validation perplexity `50.4603`.
- Greedy sample from `ROMEO:` after eight steps: `ROMEO:tttttttttttttttttttttttttttttt`. This is expected repetition from an intentionally tiny smoke run, not a quality claim.
- Phase 3 LoRA smoke: 3 micro instructions, rank 2, 6 steps, `384` trainable adapter parameters, final supervised loss `4.0325`; adapter saved to `artifacts/lora-smoke/adapter.pt`.
- Phase 3 RAG smoke: question `Which planet is red?` retrieved `data/knowledge.txt` with lexical score `0.75`. The undertrained baseline generated `eeeeeeeeeeee`; retrieval provenance works, while answer quality requires a substantive training run.

### 2026-10-04 — Pending owner verification: Phase 2 integration

- Added selectable `character`/`bpe` tokenizer training, serialized tokenizer state in checkpoints, and strict compatibility checks on resume.
- Checkpoints now retain optimizer state and `completed_steps`, allowing continuation rather than model-only reload.
- Added `test_bpe_training_and_checkpoint_resume`; verification is intentionally delegated to the owner-run test workflow and is not yet logged as passed.

### 2026-10-04 — Ticket 204 preparation: bug fixes and comparison harness

- Fixed BPE `encode` to replay merges in learned rank order (previously a greedy any-pair pass could segment text differently from training).
- Fixed `rag/query.py` and `adaptation/run_finetune.py`, which could not load BPE checkpoints; added `tokenizer_from_checkpoint`.
- Added `src/eval/compare_tokenizers.py` (`python -m src.eval.compare_tokenizers`), producing `artifacts/tick-204/comparison.{json,md}`. It reports bits-per-character and per-character perplexity in addition to per-token metrics, since per-token perplexity is not comparable across tokenizers.
- Added 4 tests (expected suite total: 17). **Not yet executed** — verification is owner-run; no metrics from the harness exist yet and none are recorded here.
- `[TICK-204]` remains ACTIVE until the owner-run results are pasted and reviewed.

### 2026-10-04 — TICK-204 results: character vs BPE (owner-run, reviewed)

- Verification: `.venv/bin/python -m unittest discover -s tests -v` → **17 tests, OK**.
- Command: `.venv/bin/python -m src.eval.compare_tokenizers` (TinyShakespeare, 500 steps, batch 16, context 64, 2 layers, 64 dim, 4 heads, lr 1e-3, seed 42, 100 BPE merges, CPU). Raw output: `artifacts/tick-204/comparison.{json,md}`.

| Metric | character | bpe |
|---|---:|---:|
| Vocabulary size | 66 | 166 |
| Parameters | 108,416 | 114,816 |
| Characters per token (validation) | 1.000 | 1.620 |
| Validation loss (per token) | 2.3931 | 3.4757 |
| Validation perplexity (per token; not cross-comparable) | 10.95 | 32.32 |
| Validation bits per character | 3.4525 | 3.0946 |
| Validation perplexity (per character) | 10.947 | 8.542 |
| Greedy repetition (char 3-gram) | 0.970 | 0.742 |
| Sampled repetition (char 3-gram) | 0.126 | 0.091 |
| Greedy / sampled characters/s | 3429 / 3144 | 4148 / 4277 |
| Wall-clock incl. tokenizer build (s) | 5.0 | 28.2 |

**Interpretation.**
- Per-token perplexity makes BPE look 3× worse (32.3 vs 11.0); that comparison is invalid because a BPE token carries 1.62 characters. On the character-normalised scale BPE is better: 3.09 vs 3.45 bits/char (about 10% lower), i.e. per-character perplexity 8.54 vs 10.95.
- Greedy output collapses into loops in both (character: "the the the"; BPE: "thould thould", "cema cema"), but BPE repeats less (0.742 vs 0.970). Sampled output has low repetition in both. BPE samples contain more real words; the character sample is mostly non-words. Both models are heavily undertrained, so this is a relative signal only.
- BPE generates ~21–36% more characters/s because each step emits >1 character; per-token speed is slightly lower (3119 vs 3429 tokens/s) from the larger output head.
- Parameter cost of BPE is +6,400 (+5.9%), entirely from the 100 extra embedding/head rows.

**Confounds (do not over-claim).**
- A fixed 64-token context spans 64 characters for the character model but ~104 characters for BPE, and each step processes ~1.6× more text. Part of BPE's advantage may be more text seen per step, not tokenization quality alone.
- Wall-clock 28.2 s vs 5.0 s includes pure-Python BPE training and corpus encoding; it is not a training-throughput comparison.
- Single seed, 500 steps, one merge count; no variance estimate. BPE merges were learned on the full corpus including validation text.
- Validation losses came from random batches (20 per evaluation), not a full validation sweep.

**Status:** `[TICK-204]` DONE. Acceptance criterion (perplexity and repetition comparison logged) met.

### 2026-10-04 — TICK-401 implementation: prompting evaluation harness (pending owner run)

- Added `src/prompting/prompts.py` (templates, extraction, scoring), `src/prompting/evaluate.py` (`python -m src.prompting.evaluate`), `data/prompt_tasks.jsonl` (6 tasks, numbers spelled out), and `tests/test_prompting.py` (8 tests; expected suite total: 25).
- Styles: `zero_shot` (`Q: ...\nA:`), `few_shot` (k exemplar Q/A pairs), `cot` (k worked exemplars ending `The answer is X.`, query ends with "Let's think step by step."). Scoring: strict `exact` on the extracted answer and lenient word-boundary `contains`.
- Reproducibility: greedy decoding, fixed exemplars and tasks; each run writes `prompting.json` with checkpoint, shots, max_new_tokens, and per-item prompt/generation/score. Per-item `prompt_truncated` and `unknown_prompt_tokens` expose context-window and vocabulary limits.
- **Not yet executed.** No prompting metrics are recorded; `[TICK-401]` stays ACTIVE until owner results are pasted and reviewed.

### 2026-10-04 — TICK-401 results: prompting evaluation (owner-run, reviewed)

- Verification: `.venv/bin/python -m unittest discover -s tests -v` → **25 tests, OK**.
- Commands: `python -m src.prompting.evaluate --checkpoint artifacts/tick-204/{bpe,character}/checkpoint.pt --output-dir artifacts/tick-401/{bpe,character}`; greedy, 2 shots, 48 new tokens, 6 tasks. Raw: `artifacts/tick-401/*/prompting.{json,md}`.

| Checkpoint | Style | Exact | Contains | Empty extraction | Prompt truncated | Mean prompt tokens | Mean `<unk>` |
|---|---|---:|---:|---:|---:|---:|---:|
| BPE | zero_shot | 0.00 | 0.00 | 0.00 | 0.00 | 25.3 | 0.00 |
| BPE | few_shot | 0.00 | 0.00 | 0.00 | 1.00 | 72.3 | 0.00 |
| BPE | cot | 0.00 | 0.00 | 1.00 | 1.00 | 174.3 | 0.00 |
| Character | zero_shot | 0.00 | 0.00 | 0.83 | 0.00 | 37.5 | 0.00 |
| Character | few_shot | 0.00 | 0.00 | 1.00 | 1.00 | 104.5 | 0.00 |
| Character | cot | 0.00 | 0.00 | 1.00 | 1.00 | 263.5 | 0.00 |

**Interpretation.**
- All styles score 0/6 on both strict and lenient matching for both models. This is the expected outcome for 500-step, ~110k-parameter language models trained only on Shakespeare; it is a measurement of the harness working, **not** evidence about the relative merit of zero-shot, few-shot, or CoT prompting. No ranking of prompt styles can be drawn.
- Context budget is the dominant, measurable effect. With a 64-token window, 100% of few-shot and CoT prompts exceed the context (mean 72–104 and 174–264 tokens), so the decoder crops away the exemplars and instruction; only zero-shot prompts (25–38 tokens) fit. CoT needs ~2.7× the tokens of few-shot, and BPE needs ~0.67× the tokens of characters for the same text, which shows a practical BPE benefit for in-context prompting.
- Prompts contained no `<unk>` tokens, so the spelled-out-number design works for both vocabularies.
- Empty extraction: CoT is 1.00 for both because no generation reached an "The answer is" span; the character model also produced mostly empty first lines for zero/few-shot (0.83 / 1.00), i.e. it began with blank lines rather than text. BPE zero-shot output is Shakespeare-like word salad ("cell my should thould thould...") unrelated to the question, matching the TICK-204 greedy looping.
- Not claimed: no accuracy difference between tokenizers (both 0/6; n=6 has no statistical power).

**Limitations.** 6 tasks, one seed/checkpoint per tokenizer, greedy only, 2 shots. Meaningful prompting results require a model with far more training and a context of several hundred tokens; the harness supports this unchanged via `--checkpoint`.

**Status:** `[TICK-401]` DONE. Acceptance criterion (prompts and scored results reproducibly logged) met.

### 2026-10-04 — TICK-402 implementation: memory and calculator agent layer (pending owner run)

- Added `src/agent/` (`calculator.py`, `memory.py`, `agent.py`, `chat.py`) and `tests/test_agent.py` (17 tests; expected suite total: 42).
- **Calculator safety design:** parse with `ast`, evaluate by tree-walk, no `eval`/`exec`. Whitelist: numeric literals, `+ - * / // % **`, unary `+ -`. Limits: 200-character expression, depth 50, exponent magnitude 100, absolute value 1e100. Rejected and tested: names, calls (`__import__`, `open`, `abs`), attribute access, strings, booleans, lists, lambdas, comparisons, conditionals, imaginary/complex results, `inf`/`nan`, division by zero, overlong or deeply nested input.
- **Memory:** ordered user/assistant/tool turns; prompts include the newest turns that fit a character budget (so small-context models always see the latest exchange); JSON persistence lets context survive across processes.
- **Prompt chain:** `preamble + memory + "Assistant:"` → generate → optional `CALC(expr)` execution and re-generation (bounded at 2 rounds) → final reply recorded. `/calc <expr>` bypasses the model for deterministic tool use.
- **Verification status:** not executed by the agent. Model-initiated tool use is tested with scripted stubs; end-to-end integration with a real (tiny, random-weight) model is covered by one torch test. No quality metrics apply to this ticket.
- `[TICK-402]` stays ACTIVE until the owner-run suite (expect 42 OK) and demo transcript are pasted and reviewed.

### 2026-10-04 — TICK-402 results: memory and calculator agent (owner-run, reviewed)

- Verification: `.venv/bin/python -m unittest discover -s tests -v` → **42 tests, OK** (0.709 s).
- Demo (BPE checkpoint `artifacts/tick-204/bpe/checkpoint.pt`, greedy, 60 new tokens, 400-char context): `/calc (2+3)*4` → `20`; `/calc 2**1000` → `Calculator error: exponent magnitude is limited to 100`; `/calc __import__('os')` → `Calculator error: unsupported syntax: Call`. No model call is made for `/calc`.
- Persistence: first run saved 8 turns to `artifacts/tick-402/memory.json`; the second process loaded them and `transcript-2.json` reports `memory_turns: 10`, as expected.
- Model replies ("ce with cema cema ...") are undertrained word salad, consistent with TICK-204/401. The 500-step models cannot emit `CALC(...)` themselves; model-initiated tool use is verified with scripted stubs only.
- Review notes: the NumPy warning on stderr is non-fatal and is resolved by installing NumPy (needed for TICK-403 anyway).

**Status:** `[TICK-402]` DONE. Acceptance criterion (multi-turn context persists; safe calculator calls pass tests) met.

### 2026-10-04 — Fix: packaging config (found during pre-TICK-403 review)

- `pyproject.toml` used `packages.find where = ["src"]`, which would install subpackages as top-level `agent`, `model`, ... while all code imports `src.*`. Changed to `include = ["src*"]`. Latent (tests run from the repo root and did not hit it); not yet executed via `pip install -e .`.
- Added optional `plots` extra (`numpy`, `matplotlib`) and the same two lines in `requirements.txt`.

### 2026-10-04 — TICK-403 implementation: evaluation figures (pending owner run)

- Added `src/eval/attention.py` (`attention_weights(model, ids)` → `[layers, heads, T, T]`, recomputed from each block's `qkv` via forward pre-hooks because SDPA hides weights; hooks removed and train/eval mode restored) and `src/eval/figures.py` (`python -m src.eval.figures`).
- Figures written to `artifacts/tick-403/`: `loss_curves.png` (validation **bits per character** vs step, using each run's measured chars/token so the tokenizers are comparable), `perplexity_params.png` (per-character perplexity and parameter counts), `latency.png` (tokens/s and chars/s, greedy vs top-k), `attention.png` (layer × head maps for one prompt), plus `figures.json` with the underlying data.
- Added `tests/test_figures.py` (10 tests; expected suite total: **52**). Attention tests check shape, row-normalisation, causality (zero above the diagonal), hook cleanup, the manual-attention/RMSNorm path, and input validation. A data test checks that the recomputed final bits/char matches `validation_bits_per_character` in the real `comparison.json`. The render test is skipped without matplotlib (then expect 51 OK + 1 skipped).
- **Not executed by the agent** (no shell). No figures exist yet; `[TICK-403]` stays ACTIVE until owner results are pasted and reviewed.
- Limitation to carry into the report: latency comes from single 200-token CPU samples in TICK-204 (one run each), so differences are indicative only; attention maps come from an undertrained 500-step model and illustrate the tooling, not learned linguistic structure.

### 2026-10-04 — TICK-403 results: evaluation figures (owner-run, reviewed)

- Installed `numpy` and `matplotlib` in `.venv`. `.venv/bin/python -m unittest discover -s tests -v` → **52 tests, 1 error**: `test_invalid_characters_per_token_is_rejected`. Cause was a **test-helper bug** (`sample_result` divided by `chars_per_token=0.0` while building fixture data), not a defect in `bits_per_character_curve`; the other 51 passed, including all attention and render tests. Fixed the helper to guard the zero case; re-run pending.
- `.venv/bin/python -m src.eval.figures` produced the four PNGs. Reviewed by the owner and by the agent from the shared images:
  - `loss_curves.png`: BPE below character at every step (3.09 vs 3.45 bits/char at step 500; start about 4.5 vs 5.8).
  - `perplexity_params.png`: 10.95 vs 8.54 per-character perplexity; 108,416 vs 114,816 parameters, labels readable.
  - `latency.png`: character faster per token (about 3430/3140 vs 3120/2810 tokens/s); BPE faster per character (about 4150/4280 vs 3430/3150). The legend overlapped a bar in the first render; headroom added (`ylim` × 1.25, legend upper right) — cosmetic, re-render optional.
  - `attention.png`: 2 layers × 4 heads, 23 tokens; all maps lower-triangular (causal). Several heads attend strongly to the first token and show vertical stripes; layer 1 shows more column structure. Tooling demonstration only.
- Figure data are in `artifacts/tick-403/figures.json`. Character-model attention (optional) was not run.

**Status:** `[TICK-403]` DONE. Acceptance criterion (loss/perplexity/parameter, attention, and latency figures render) met.

### 2026-10-04 — TICK-404 implementation: LaTeX final report (pending owner run)

- Added `report/final_report.tex` (article class, one-column; fixed-width `p` columns summing under 15 cm; `booktabs`, `graphicx`, `float`; `\graphicspath` points at `../artifacts/tick-403/`). Sections: system overview, Phase 1/3 smoke runs, character vs BPE (table, 3 figures, interpretation, confounds), attention, prompting, agent, limitations and conclusion.
- All numbers are copied from this ledger, not recomputed; claims are limited to what was measured (single seed, 500 steps, BPE about 10% lower bits/char with listed confounds; prompting 0/6; no instruction-following or RAG quality claims).
- Added `tests/test_report.py` (7 tests; expected suite total: **59**): document skeleton, balanced environments and braces, every figure/table captioned, table widths fit one column, referenced PNGs exist, and headline table values (bits/char, parameters) match `artifacts/tick-204/comparison.json`.
- **Not compiled by the agent** (no shell/LaTeX). `[TICK-404]` stays ACTIVE until the owner runs the suite and, if a TeX distribution is available, `pdflatex`.

### 2026-10-04 — TICK-404 results: LaTeX report (owner-run, reviewed)

- `.venv/bin/python -m unittest discover -s tests -v` → **59 tests, OK** (1.123 s). This also confirms the TICK-403 test-fixture fix.
- Owner compiled `report/final_report.tex`; the PDF is `report/LLM_Ascent_Project_Report v1.pdf` (433 KB). Figures copied next to the `.tex` (`report/*.png`). No compile problems were reported.

**Status:** `[TICK-404]` DONE. Acceptance criterion (report builds; one-column-safe tables) met. **v1 is complete and frozen.**

### 2026-10-04 — v2 planned (design only; no code or runs yet)

- Scope, claim gates G1–G5, 12 tickets (501–503, 601–604, 701–703, 801–802), and wave order are in `docs/agent.md`; the full design record (v1 defects found, experimental protocol, dataset spec, interface/API spec, risks) is the section **v2 design record** at the end of this file.
- Planning read the v1 code and found defects that make the v1 instruction/RAG results unable to improve by simply training longer (design record §1). v2 fixes them as part of TICK-501–503/603.
- No metrics are recorded for v2. Nothing here is a result.

### 2026-10-04 — TICK-701/702 implementation review and tests (pending owner run)

- State on arrival: `src/web/{registry,handlers,server}.py` and `src/web/static/index.html` existed from an earlier session that ran out of tokens, with no tests and no ledger entry. Reviewed against the real v1 APIs (agent, RAG retriever, decoding, tokenizer checkpoint loader, `attention_weights`, `comparison.json` keys); all call signatures and keys matched.
- Fixes: allow-list patterns use `\Z` instead of `$` (a trailing newline no longer passes); `/api/results` lists only figures that `/figures/<name>` serves (no symlink escapes listed); empty/non-UTF-8 documents give 400 instead of 500; `create_server` accepts `127.0.0.1`/`localhost` only (IPv4 socket; `::1` remains valid in the Host header); `/api/chat` returns `prompt_tokens`, `prompt_cropped`, `context_length`, `unknown_characters` and the chat tab displays them.
- Added `tests/test_web.py` (41 tests; expected suite total **100**, or 99 + 1 skipped without Node). Real server on an ephemeral port, temp project with tiny random-weight character and BPE checkpoints, a corrupt checkpoint, fake documents/figures. No quality metrics apply.
- **Not executed by the agent** (no shell, no browser). `[TICK-701]` and `[TICK-702]` stay `IMPLEMENTED (OWNER RUN PENDING)` until the owner pastes the suite result and the browser checklist from `docs/state.md`.
- Known limits to carry forward: base checkpoints only (no adapter loading, defect D10); one generation lock; v2 Results tab shows raw JSON until TICK-703; UI not tested in a browser by the agent.

## Results ledger

| Phase / iteration | Configuration | Train loss | Validation loss | Validation perplexity | Repetition | Parameters | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Project initialization | N/A | N/A | N/A | N/A | N/A | N/A | Tracking established |
| Phase 1 CPU smoke | 1L, 32D, 4H, 8 steps | 3.9288 | 3.9212 | 50.4603 | N/A | 15,904 | Checkpoint saved |
| Phase 3 LoRA smoke | rank 2, 6 steps, 3 examples | 4.0325 | N/A | N/A | N/A | 384 trainable | Adapter saved |
| Phase 2 TICK-204 character | 2L, 64D, 500 steps, seed 42 | 2.3961 | 2.3931 | 10.95 (3.45 bpc) | 0.970 greedy / 0.126 sampled | 108,416 | Done |
| Phase 2 TICK-204 BPE-100 | 2L, 64D, 500 steps, seed 42 | 3.4269 | 3.4757 | 32.32 per token (3.09 bpc; 8.54 per char) | 0.742 greedy / 0.091 sampled | 114,816 | Done |
| Phase 4 TICK-401 prompting | 6 tasks, greedy, 2 shots, both TICK-204 checkpoints | N/A | N/A | N/A | N/A | N/A | Done (0/6 all styles; context-limited) |
| Phase 4 TICK-402 agent | `/calc` + memory demo, BPE checkpoint | N/A | N/A | N/A | N/A | N/A | Done (42 tests OK) |
| Phase 4 TICK-403 figures | 4 PNGs from TICK-204 data + BPE checkpoint | N/A | N/A | N/A | N/A | N/A | Done (rendered, reviewed) |

## Artifact index

Future entries will link to checkpoints, metric files, generated samples, figures, and report outputs with the command and configuration used to produce them.

## Handoff: implementation and verification status

### Verified capabilities

- Character tokenizer: stable vocabulary construction, known-token round trips, unknown-token behavior, empty inputs, and real `torch.long` tensor interface.
- BPE tokenizer: deterministic training, serialized vocabulary/merge persistence, and round-trip coverage.
- Data/model: causal `[B, T]` batches, shifted labels, finite model loss, logits `[B, T, V]`, and causal-dependence behavior.
- Decoding: greedy, sampling with top-k/top-p, and beam search respect output-length constraints.
- Training: CPU baseline training writes metrics/config/checkpoint artifacts; BPE configuration persists; a checkpoint can resume with optimizer state and completed-step tracking.
- Adaptation: prompt labels mask non-response text; LoRA leaves base projections frozen while adapter weights remain trainable.
- RAG: local source chunks are selected deterministically, included within a bounded prompt, and reported with provenance.

### Latest verification record

- The owner confirmed the expanded project suite passes after the BPE/resume changes. This includes `test_bpe_training_and_checkpoint_resume` in addition to all prior tokenizer, model, decoding, adaptation, and RAG tests.
- The test suite is the implementation gate. It does not replace empirical training/evaluation runs, which remain separately logged below.

## Handoff: artifacts and their meaning

| Artifact | Purpose | Interpretation |
|---|---|---|
| `data/tinyshakespeare.txt` | Base local corpus | Use for controlled character-vs-BPE comparisons |
| `artifacts/baseline-smoke/checkpoint.pt` | Character baseline checkpoint | 8-step plumbing checkpoint; not a quality model |
| `artifacts/baseline-smoke/metrics.json` | Baseline smoke curve | Real run; suitable for sanity evidence only |
| `data/micro_instructions.jsonl` | Three-example LoRA smoke set | Verifies the path, not instruction quality |
| `artifacts/lora-smoke/adapter.pt` | Saved rank-2 LoRA tensors | 384 trainable parameters; smoke-only artifact |
| `data/knowledge.txt` | Local RAG knowledge source | Validates retrieval/provenance wiring |

## Next empirical record: Ticket 204

The next run must compare tokenizer systems fairly. Keep these items fixed between character and BPE runs:

| Dimension | Character baseline | BPE upgrade |
|---|---|---|
| Corpus / split | TinyShakespeare, same split | Identical |
| Model budget | Same layers, dimensions, heads, context | Identical except vocabulary-dependent embedding/head size |
| Optimization | Same seed, optimizer, learning rate, batch size, steps, evaluation batches | Identical |
| Tokenizer settings | `tokenizer="character"` | `tokenizer="bpe"`; report merge count and vocabulary size |
| Required evidence | Loss, perplexity, repetition, latency, parameters, samples | Same fields |

Interpretation requirement: tokenizers change sequence lengths and vocabulary-dependent parameter counts. The comparison is a practical system comparison, not an isolated attention-only ablation.

## Remaining evidence requirements

| Area | Available now | Still needed |
|---|---|---|
| Phase 1 | 8-step CPU loss/perplexity and one greedy sample | Longer training if final report makes quality claims |
| Phase 2 | BPE/resume implementation, tests, and TICK-204 character-vs-BPE metrics/samples (single seed, 500 steps) | Multi-seed or longer runs if the report makes quality claims |
| Phase 3 LoRA | Adapter loss and trainable-parameter count | Held-out instruction-following evaluation |
| Phase 3 RAG | Retrieval score and source provenance | Grounded-answer / hallucination evaluation |
| Phase 4 | Complete (v1) | Prompting, memory/tool, figures, and LaTeX report all done; capability claims intentionally absent. v2 (Phases 5–8) targets them |

## v2 design record

Status: **design only (2026-10-04)**. Ticket board and gates live in `docs/agent.md`; this section is the rationale and specification. Numbers marked *provisional* are replaced by measured values after the TICK-501 pilot.

### 1. Why v1 cannot answer the capability questions (found while planning)

Measured evidence from v1: prompting 0/6 in every style; 100% of few-shot/CoT prompts exceeded the 64-token context; RAG smoke generated `eeeeeeeeeeee`; instruction tuning ran on 3 examples with no held-out set. Code review adds causes that more training alone would not fix:

| # | Defect (file) | Effect | v2 fix (ticket) |
|---|---|---|---|
| D1 | `rag/query.py` passes `max_context_chars = context_length // 2` (32 characters at context 64) and `retriever.prompt` truncates the context to that many characters; the whole prompt still exceeds 64 tokens and decoding crops its left side | Retrieved context is cut to a fragment and the instruction is cropped away, so RAG cannot work at this context size | Token-budgeted prompt, context ≥ 256 tokens (603, 501) |
| D2 | RAG prompt format (`Use only the supplied context...`) differs from the instruction format the model was LoRA-tuned on | Tuning does not transfer to RAG prompts | RAG uses `InstructionFormatter` with the context as `### Input:` and is in the SFT mix (502, 603) |
| D3 | Tokenizers are built from TinyShakespeare only (66-entry vocabulary = 65 characters + `<unk>`; the corpus is believed to lack `#`, `(`, `)` and nearly all digits — **verify in 502**) | Instruction headers (`### Instruction:`), digits and `CALC(...)` become `<unk>` | Fit tokenizers on a mix that contains every format character; test for 0 `<unk>` (503) |
| D4 | `finetune_lora` runs full-batch steps on all examples, no minibatching, no validation | Cannot train on thousands of examples or detect overfitting | Minibatched SFT loop with validation and best checkpoint (602) |
| D5 | Responses have no terminator; decoding always runs `max_new_tokens` | Cannot score exact match cleanly; the model cannot learn to stop | Response end marker + stop handling + normalised EM (503, 602) |
| D6 | BPE merges are learned on the full corpus including validation text (already disclosed in v1) | Slightly optimistic BPE validation numbers | Split first, fit tokenizer on the train split only (501) |
| D7 | Validation uses 20 random batches with unseeded sampling | Noisy and not reproducible | Seeded full-sweep validation (501) |
| D8 | `resolve_device('auto')` knows only CUDA; `GPTConfig.dropout` is not exposed in `TrainConfig`; constant LR, no clipping | Slower CPU-only training, no regularisation control, unstable longer runs | MPS support, dropout flag, warmup+cosine, clipping (501) |
| D9 | v1 context confound: a 64-token window is ~104 characters for BPE vs 64 for character | Part of BPE's gain may be exposure, not tokenization | Equal-character-span control in G1 (601) |
| D10 | LoRA injected only into attention `qkv`/`projection`; adapters saved as a state-dict slice with no loader | Limited capacity; adapters cannot be reused by the UI | Optional MLP targets; `load_with_adapter` utility (602, 701) |

### 2. Experimental protocol

**2.1 Scale ladder (pilot decides; all *provisional*).** S = 2L/64D (v1 control); M = 4L/128D (~0.8M params); L = 6L/192D, 6 heads (~2.8M params). Candidate context: 256 tokens for BPE and an equal character span (about 512 characters) for the character model. TinyShakespeare is ~1.1M characters, so a larger model will overfit with many epochs: use dropout 0.1–0.2, the best-validation checkpoint, and let the pilot's validation curves choose steps. If the mix (§2.2) is larger, the same rule applies. TICK-501 records seconds/step for S/M/L on the owner's machine and picks the largest scale whose full plan fits the compute budget.

**2.2 Training stages.**

1. *Pretraining (601):* causal LM on a text mix = TinyShakespeare + generated world documents (prose about synthetic entities with digits and facts) + plain arithmetic/string statements. Tokenizers fitted on the train split only. Metrics: validation bits/char on a full sweep of the held-out split, 3 seeds, mean ± sample std and per-seed values.
2. *Supervised fine-tuning (602):* Alpaca format (`### Instruction:` / `### Input:` / `### Response:`) with an end marker, on the generated instruction set. Arms: no-SFT baseline, full fine-tune, LoRA rank {2, 8, 16} attention-only, LoRA rank 8 with MLP targets. Report trainable-parameter counts next to accuracy.
3. *RAG and tool formats (603, 604)* are part of the same SFT mix and evaluated separately.

**2.3 Synthetic task design (TICK-502).** Everything is generated from a seed so held-out data are held out by construction, and so real-world knowledge cannot leak in.

| Family | Item | Held-out split(s) |
|---|---|---|
| `arith` | add/subtract with up to 3-digit operands, natural-language and symbolic phrasings | unseen operand pairs; a longer-number (4-digit) OOD split |
| `string` | copy / reverse / uppercase a short generated word | unseen words |
| `fact_closed` | attribute questions about *training-world* entities (tests memorisation) | unseen paraphrase template |
| `fact_open` | context paragraph about a **new** entity + question (answer exists only in the context) | unseen entities; unseen template |
| `unanswerable` | context lacks the asked attribute; target is a fixed refusal string | unseen entities |
| `tool` | arithmetic where the target is `CALC(expr)`, then a second turn with `Tool: expr = value` and the final answer | unseen operands; longer numbers |

Entities and values come from a seeded syllable generator (no real names). Indicative sizes: ~20k SFT-train examples balanced across families, 500 validation, and ≥ 300 test items per family-split (so G2/G3 intervals are meaningful). Tests assert **zero overlap** between train and held-out on entity names, operand pairs, words and templates, plus deterministic regeneration from the seed.

**2.4 Metrics and statistics.** Normalised exact match (strip, lowercase, cut at the end marker/first newline); per-family and per-split accuracy; 95% Wilson interval for each proportion; mean ± sample std (n = 3) and per-seed values across seeds. No p-values are claimed with three seeds. Baselines in every table: no-SFT model and majority/chance. A 50-item qualitative error sample per family is saved as JSON.

**2.5 RAG controls (603).** Same questions under: closed-book (no context); oracle context (gold paragraph); retrieved top-k (overlap vs BM25, dependency-free implementation); distractor-only context (gold removed); unanswerable (expect refusal); **counterfactual swap** (the gold value in the paragraph is replaced — a grounded model's answer must change). Retrieval is scored separately (recall@k, MRR) from generation so errors can be attributed. Prompt budget is in tokens, not characters (fixes D1).

**2.6 Tool use (604).** Success is split into: valid `CALC(...)` emitted; expression correct; final answer correct after the tool result. Compared with the same model answering directly, including the longer-number split. The prompting study (zero/few/CoT) is re-run on held-out arithmetic with enough context for exemplars to fit; shots {0, 1, 2, 4}.

**2.7 Interpretation rules.** Gates in `docs/agent.md` are fixed before runs. A failed gate produces a reported negative result, not a re-run with new settings; any additional run after seeing results is labelled exploratory. Claims are limited to the synthetic closed domain and these model sizes.

### 3. Interface design (TICK-701–703)

**Stack.** Python standard library only (`http.server.ThreadingHTTPServer`) for the backend, one self-contained HTML file with vanilla JS/CSS for the frontend (no CDN, works offline, light/dark via CSS variables). No new dependencies beyond the PyTorch/NumPy/matplotlib already installed. Launch (planned): `.venv/bin/python -m src.web.server --port 8000`, then open `http://127.0.0.1:8000`.

**Layout.** `src/web/{server.py,registry.py,handlers.py}` and `src/web/static/index.html`; tests in `tests/test_web.py` start the server on port 0 in a thread against a tiny random-weight checkpoint written to a temp directory.

**Model registry.** Discovers `checkpoint.pt` files (and base+adapter pairs) under allow-listed roots (`artifacts/`), assigns ids such as `tick-204/bpe`, and exposes name, tokenizer, parameters, context length. Models load lazily and are cached (at most two); generation is serialised with a lock. Clients send ids, never file paths.

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/health`, `GET /api/models` | status; model list | includes kind (base / adapter), tokenizer, params, context |
| `POST /api/generate` | text completion | greedy / sample / beam; temperature, top-k, top-p, seed; cap `max_new_tokens` ≤ 300; returns text, prompt and new token counts, `<unk>` count, whether the prompt was cropped, seconds, tokens/s, 3-gram repetition |
| `POST /api/chat`, `/api/chat/reset` | agent with memory and `/calc` | session id; returns tool calls and memory length |
| `POST /api/rag` | question + allow-listed documents | returns sources with scores, the exact prompt, the with-context answer **and** the closed-book answer side by side |
| `POST /api/attention` | prompt → weights | `[layers, heads, T, T]` rounded to 3 decimals, plus tokens (reuses `attention_weights`) |
| `GET /api/results`, `GET /figures/<name>` | dashboard data and images | reads v1 `comparison.json`/`figures.json` and v2 summary JSON when present; serves only allow-listed directories |

**Safety.** Bind to `127.0.0.1` only; reject requests whose `Host` is not `localhost`/`127.0.0.1` (DNS-rebinding guard); no CORS headers; JSON body ≤ 64 KB; prompt length cap; no path input; errors returned as JSON without stack traces. `torch.load(..., weights_only=False)` is used on checkpoints, so only locally produced artifacts under the allow-listed roots are ever loaded.

**Tabs.** Playground (generate with decoding controls and per-response telemetry), Chat + Tools (agent, memory view, `/calc`), RAG (side-by-side closed-book vs with-context and the sources used), Attention (canvas heatmap with layer/head selectors), Results (v1 and v2 tables and figures with the claim-gate status). A permanent banner states that these are tiny research models and outputs are not reliable; the UI flags prompts containing characters outside the model's vocabulary.

### 4. Task division and compute

- **Agent (no shell):** all code, tests, docs, and the report; each ticket ends with exact owner commands recorded in `docs/state.md`.
- **Owner:** runs tests, the TICK-501 pilot, long training/evaluation commands (601–604), opens the web UI, compiles the LaTeX, and pastes results into `state.md`.
- Provisional budget: a few owner compute hours in total, with Colab/GPU as the fallback if the pilot shows the plan does not fit. Evaluation over a few thousand items with the current non-cached decoder should take minutes; add a KV cache in 602 only if it does not.

### 5. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Model too small or undertrained to follow instructions even on synthetic tasks | Medium | Pilot picks the scale; reduce task difficulty (digits, template count); move to Colab GPU; otherwise report the negative result under G2 |
| Overfitting on 1.1M-character TinyShakespeare at larger scale | High | Dropout, best-validation checkpoint, mix with generated text; report train/validation gap |
| Synthetic tasks are "too easy" and inflate perceived capability | Medium | Claims restricted to the synthetic closed domain; held-out entities/templates/length OOD; no general-intelligence wording |
| Template or entity leakage between splits | Low (tested) | Overlap unit tests; deterministic regeneration check |
| Owner-run latency between waves | High | Long compute (601) started in wave 2; UI built meanwhile |
| `docs/` and `artifacts/` are git-ignored, so results are not versioned | Certain | Recommend a tracked `results/v2/` (summary JSON + PNG) — open decision D2 in `state.md` |

### 6. v2 artifact layout

`data/v2/` (generated JSONL, world docs, generator seed manifest), `artifacts/v2/pretrain/<scale>-<tokenizer>-seed<N>/`, `artifacts/v2/sft/<arm>-seed<N>/`, `artifacts/v2/eval/{instruct,rag,tool,prompting}/`, `artifacts/v2/summary.json` (single source of truth consumed by figures, the UI, and report tests), `report/final_report_v2.tex` and `report/figures_v2/`.

### 7. v2 results ledger (empty until runs exist)

| Ticket | Configuration | Metric | Result (mean ± std, n=3) | Gate | Status |
|---|---|---|---|---|---|
| 601 | scale TBD, character vs BPE vs equal-span | validation bits/char | not run | G1 | pending |
| 602 | no-SFT / full FT / LoRA r2,r8,r16 / LoRA+MLP | held-out EM per family | not run | G2 | pending |
| 603 | closed-book / oracle / retrieved / distractor / swap | EM, recall@k, MRR, following rate | not run | G3 | pending |
| 604 | tool vs no tool; zero/few/CoT | emission rate, end-to-end EM | not run | G4 | pending |

---

## 2026-10-14 — Course planning session: ten-act v2

The v2 design record above (§1–§7) was written before the 18 lecture PDFs were
read in full and before the owner fixed the scope. Both changed the plan.

**What changed**

| Was | Now | Why |
|---|---|---|
| 12 tickets across 4 gates (G1–G5) | **10 acts**, each ending in table + figures + conclusion | The gates were too coarse to schedule; they did not map onto the syllabus |
| "measured comparisons + interface" | confirmed as the deliverable | Owner decision |
| No act for RLHF/DPO | **out of scope**, replaced by Act 6 (decoding sweep, best-of-n, self-consistency) | Owner decision |
| 4 v1 questions | 10 acts covering the full lecture checklist | Syllabus coverage was ~30%; the checklist in `AGENTS.md` §4 is now the definition of done |
| Local machine, CUDA-or-CPU | **M4 / 16 GB primary**, M3 / 8 GB fallback, adaptive resumable sweep | Owner decision |

**Lecture coverage gap found**

Lecture 18 is empty and lecture 17 is truncated before it reaches RAG. Prompt
engineering, chains/memory/agents, bias/toxicity, interpretability, multimodal,
model editing, self-evolving LLMs, small LMs and BART/T5 have no lecture file at
all despite being on the syllabus. The owner has confirmed external sources are
acceptable for those topics.

**Repo hygiene (this session)**

| # | Finding | Fix |
|---|---|---|
| H1 | `test_referenced_figures_exist_when_generated` failed — figures existed only under git-ignored `artifacts/tick-403/character-attention/` | Figures tracked in `report/figures/` (PNG + vector PDF); `\graphicspath` repointed; test now checks the tracked dir |
| H2 | `.gitignore` ignored all of `docs/`, so the design record would not survive a machine change | Ignores only `docs/lectures/` (165 MB); `docs/*.md` tracked |
| H3 | v1 metrics lived only in git-ignored `artifacts/` | Copied to tracked `results/v1/` as JSON + `comparison.csv` |
| H4 | Loose 150 dpi PNGs in the `report/` root, one stale | Vector PDFs generated; stale copies removed |

**Suite status: 100 tests, all passing** (previously 99 passing + 1 failure).

**v2 results ledger** — empty by design; one row per act once measured.

| Act | Claim tested | Metric | n | Result | Status |
|---|---|---|---|---|---|
| 0 | smoothing / embeddings | perplexity, downstream acc | — | — | not run |
| 1 | LSTM/attention vs RNN/FF window | bits per char | — | — | not run |
| 2 | PE / norm / head / FFN ablations | bits per char | — | — | not run |
| 3 | tokenizer family + vocab sweep | bits per char, fertility | — | — | not run |
| 4 | fitted α, β vs Chinchilla | loss exponent fits | — | — | not run |
| 5 | KV cache, MQA/GQA, speculative | tok/s, KV bytes, quality | — | — | not run |
| 6 | decoding sweep | accuracy vs diversity | — | — | not run |
| 7 | SFT / ICL / PEFT family | held-out EM, Wilson CI | — | — | not run |
| 8 | RAG grounding, tool use | grounding acc, emission rate | — | — | not run |
| 9 | head ablation, logit lens | Δ perplexity per head | — | — | not run |
| 10 | metric artifact, toxicity | Brier vs EM, toxicity rate | — | — | not run |

### 2026-10-16 — Act 5: Inference & efficiency (Stage E, part 1) — KV CACHE (5a) — COMPLETE

**KV cache (`src/model/gpt.py`, `src/generation/decoding.py`).** Incremental K,V
cache per layer. Greedy/top-k/top-p/beam all use it; beam keeps per-beam caches.
Byte-identical output verified by torch.equal. Bug found and fixed: learned absolute
position embedding was indexed at 0 on the incremental step (0.5 nats divergence);
fixed by indexing at cache.sequence_length. Max residual logit diff after fix:
1.64e-07 nats (1.4 ULP fp32, BLAS accumulation order, not a bug).

Speedup (32 new tokens, dim 128, 4L):

| context | uncached (ms) | cached (ms) | speedup |
|---:|---:|---:|---:|
| 128 | 51.7 | 14.0 | 3.71x |
| 256 | 85.1 | 19.1 | 4.46x |
| 512 | 198.2 | 22.7 | 8.72x |
| 1024 | 630.7 | 34.3 | 18.36x |

Cache memory (delta = 0 bytes vs theoretical 2·L·G·dh·T·4):

| model | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|
| 1.5M (L=4, d=128, 4H) | 0.50 MB | 1.00 MB | 2.00 MB | 4.00 MB |
| 9.0M (L=9, d=288, 4H) | 2.53 MB | 5.06 MB | 10.13 MB | 20.25 MB |

Linear fit intercept ~0. Memory crossover: caching wins on memory once N > d
(≈288 for the 9.0M rung: N² attention-score matrix vs O(N·d) cached KV).

### 2026-10-17 — Eviction policy replaced: retention + pinning

Delete-after-rung eviction stranded Acts 6-10 with no loadable weights (5c needs a
draft/target pair, Act 7 a base model, Act 9 a model to ablate heads in). Replaced
with `src/scaling/checkpoints.py`: `CheckpointStore` keeps the N=2 most recent
checkpoints per rung and never touches a pinned one. Pin state lives in
`<rung>/<seed>/pinned.json`. Six tests, including the load-bearing one: pin step
300, write steps 100-500, run eviction, assert 300 survives.

**Two bugs the policy surfaced, both silent:**
1. `dir_for` prepended `seed` to arguments already shaped `seed42`, producing
   `seedseed42` directories that eviction's own glob would never match — so an
   eviction pass would report success while deleting nothing.
2. The pin key was written as `{rung}/seed{seed}` while the reader looked up the
   same malformed form, so pins were recorded where nothing could read them.
Both fixed by normalising the seed label in one helper.

### 2026-10-17 — SWA context-512 sweep DROPPED for cost

A context-512 sweep of W in {64,128,256,512} x 3 seeds was authorised on a 1.8-2.4 h
estimate. Measured rate was **42.1 min per run (8.4 h total)**, 4.5x the estimate:
per-step cost rose ~19x, not ~4x, because manual attention over 512 keys on 2 CPU
threads dominates and the 4x step-count reduction only offsets a fifth of it. Killed
at step 100 of the first run; no completed ctx-512 runs exist.

**SWA result as it stands (context 128):** at context 128 only W=64 is non-trivial,
giving a 2x cache reduction (65,536 B -> 32,768 B) at a quality cost of 0.0013
nats (nil). W >= context is a no-op by construction, so W in {128,256,512} is one
point repeated, not a curve. **The quality-cost curve at long context was not
measured and is not claimed.**

### 2026-10-17 — Canonical model set (pinned): 1.5M, 2.6M, 6.3M

Chosen over 9.0M deliberately: 9.0M was capped at D=10N (89.9M tokens) by corpus
exhaustion, while 6.3M reached D=20N (125.8M tokens). The 6.3M model is better
trained AND cheaper to reproduce despite being smaller. Roles: 1.5M = speculative
draft, 2.6M = Act 7 SFT/LoRA base, 6.3M = speculative target / interpretability /
final model. All three trained with pinning enabled.

**Methodology rule adopted after two projection errors in one session:** always use
the per-rung measured throughput, never a neighbouring rung's. Both the ctx-512
sweep and the canonical-set estimate were wrong the same way — extrapolating a
smaller model's rate onto a larger one. Correct canonical estimates from Act 4's
clean probes: 1.5M 5.52 steps/s -> ~23 min; 2.6M 6.76 steps/s -> ~32 min; 6.3M
3.85 steps/s -> ~133 min; total ~3.1 h.

### 2026-10-17 — Act-crossing prohibition (METHODOLOGY)

**Losses are not comparable across acts.** Act 4 and Act 5 measure DIFFERENT models:

| | Act 4 ladder | Act 5 sweeps |
|---|---|---|
| class | `ALiBiTransformer` (ALiBi) | `MiniGPT` (learned absolute positions) |
| heads | 4 | 8 |
| dim / layers | 128 / 4 | 128 / 4 |
| non-emb params | 793,344 | 694,144 |
| tokenizer | 6,487-token BPE on C4 | character (v1 G curve) / C4 memmap |
| D at 2.6M | 53.08M -> loss 3.2476 | 53.08M -> loss ~2.83 |

The tokenizer alone (6,487-token BPE vs character) explains the 2.83 vs 3.2476 gap;
the architecture differs too. **Never quote a loss from one act next to a loss from
another.** As a structural guard, every `metrics.json` written by `train_rung` now
carries an `architecture` block: `model_class`, `num_heads`, `num_kv_heads`,
`window_size`, `embedding_dim`, `num_layers`, `context_length`,
`positional_encoding`. A reader cannot compare the two numbers without the file
telling them the models differ.

### 2026-10-17 — Finding: attention-variant quality differences need real training

At 500 steps on TinyShakespeare all four G values sat inside seed noise (losses
2.345-2.363, spread 0.018). At D=53M on C4 the MQA-vs-MHA gap is directionally
consistent across every paired seed (3/3 positive). Quality differences between
attention variants only become visible after real training. The TinyShakespeare G
curve is supporting evidence, internally controlled and clearly labelled; the
2.6M paired test on C4 is the primary result.

**MQA @ 2.6M, n=3 paired:** differences +0.0447, +0.0143, +0.0279; mean +0.0290,
SD 0.0152, t=3.295 on 2 df, **two-sided p = 0.082**. Reported two-sided: the
hypothesis was framed as three open outcomes, not a directional prediction, and a
one-sided test chosen after seeing 3/3 positive differences would be p-hacking.
Direction is consistent; significance at alpha=0.05 is not reached on 2 df. n=6
approved to test further.

### 2026-10-16 — Act 5 (Stage E, part 1) — MQA/GQA/SWA (5b) — COMPLETE

**Implementation:** GPTConfig gained num_kv_heads and window_size. Attention split
into q (Linear → dim) + kv (Linear → 2×kv_dim, kv_dim=G×dh). GQA via
repeat_interleave. SWA masks |pos_q−pos_k| > W. Classical position bug hunt.

**Memory/FLOPs three-way breakdown at 8 heads, dim 128, 4L (measured via
FlopCounterMode):**

| quantity | scales with G | const across G |
|---|---|---|
| KV cache bytes (T=1024) | 8x→1x (4.0→0.5 MB ✓) | — |
| FLOPs/token (total) | −12.1% (proj only) | attention matmul = 8388608 (identical) |
| KV projection FLOPs/token | 57,344 → 8,192 | — |
| KV-read bandwidth per token | ~8x | — |

"NEITHER MQA NOR GQA reduces compute" is false literally and true for the attention
matmul specifically. The 12% comes entirely from the shrinking KV projection.

**Quality (TinyShakespeare, 500 steps):** mean val loss 2.349/2.354/2.363/2.346
for G=1/2/4/8, all within 0.02 (< run-to-run spread). G=1 cache 0.5MB vs 4.0MB
for G=8 → 8x cache reduction at no measurable quality cost at this scale.

**MQA @ 2.6M (2.6M rung, 2 seeds G=1 vs G=8):**
G=1 mean 2.8516, G=8 mean 2.8221, paired mean diff +0.029 (G=1 worse).
Third seed added (G=1=2.8661, G=8=2.8382) → full paired: 0.0447, 0.0143, 0.0279,
mean 0.0290, std 0.0152, t=3.295, df=2, p≈0.08. Direction says MQA costs loss
at 2.6M, but not significant at α=0.05. Reported honestly, not as a flat null.

**SWA (0.8M config, G=1, W sweep):**
| W | val mean ± std | cache at T=128 | non-trivial? (W<context) |
|---:|---:|---:|---|
| 64 | 3.4379 ± 0.0048 | (W·d) ~32 KB | yes (crop, cost 0.002 nats) |
| 128 | 3.4382 ± 0.0028 | (capped at context 128) | no-op (whole context fits) |
| 256 | 3.4382 ± 0.0028 | — | no-op |
| 512 | 3.4382 ± 0.0028 | — | no-op |

Only W=64 exercises the window because context length (128) ≤ W for all other rows.
SWA savings appear only at W < context; crossover is exactly W = context_length.
Quality cost is nil (0.002 nats on a ≤0.05 seed spread). A true quality-cost curve
needs a model trained at longer context; the full-W sweep is reported with this
design limitation stated rather than presented as four distinct data points.

**Attribution:** The MiniGPT/ALiBiTransformer collision found during the
non_embedding_params gap. MiniGPT now has parameter_count() + non_embedding_params()
+ total_params(); regression test (tests/test_model_interface.py) pins the
interface on both architectures.

**Artifacts (tracked):**
- results/v2/act5/ — G and SWA metrics (copied from artifacts/, tracked)
- report/figures/ — (none new for 5b; results table textual)

### 2026-10-16 — Act 4: Scaling laws (Stage D) — COMPLETE

**Headline.** We fit `L(N,D) = E + A/N^α + B/D^β` jointly by nonlinear least
squares over 2,776 (N, D, loss) points from a six-rung ladder spanning 0.79M to
8.99M non-embedding parameters on C4 — roughly four orders of magnitude below
Chinchilla's range. We obtain:

| parameter | value | uncertainty | Chinchilla |
|---|---:|---:|---:|
| E | 1.605 | ± 0.007 | 1.69 |
| α | 0.44 | ± 0.32 | 0.34 |
| β | 0.29 | ± 0.16 | 0.28 |

**Both exponents are consistent with the published Chinchilla values (0.34, 0.28)
within the spread of our own analyses** (α 0.31–0.55 across sensitivities, β
0.21–0.34). The irreducible-loss term E = 1.605 sits close to the published 1.69.
Consistency, not improvement, is the claim. Recovering β within 0.01 and E within
0.09 at 1/10⁴ the scale is the finding.

**Identification caveat.** α and β are NOT separately identified to three decimal
places. The 1.9×10⁶ Jacobian condition number comes from within-term collinearity
(corr(A,α) = +0.9994, corr(B,β) = +0.9994), not the α–β tradeoff. The well-
determined combination is α+β = 0.735 ± 0.324.

**Cross-fold sensitivity.** Leave-one-rung-out cross-validation: every held-out
rung predicted with R² ≥ 0.985 against a constant baseline. Cross-fold α range
0.314–0.477 (Chinchilla's 0.34 inside); β range 0.265–0.338 (0.28 comfortably
inside). The fold-to-fold variation is the most honest uncertainty we have.

**Sensitivity bracket.** Restricting to each run's last 30% (LR nearly annealed)
gives α = 0.552, β = 0.209. This reduces schedule bias but also discards most
of the low-D range where B/D^β is largest and best identified. We do NOT have a
principled basis to prefer one end of the range. Bracket: **α ∈ [0.445, 0.552],
β ∈ [0.209, 0.290]**.

**Noise floor.** Residual RMS = 0.0288 nats, below the measured seed spread at
convergence (0.04–0.19 nats). The fit explains the data to within run-to-run
variability.

**D=2M inversion — null explained by seed noise.** At D=2M the 1.5M rung (5.226)
appears better than 2.6M (5.283), a 0.057-nat gap. Per-seed spread at 2.6M is
0.186 (3.3× the gap); the inversion is driven by a single high-loss seed. This is
a worked example of checking seed spread before declaring a finding.

**Limitations.**
- 6.5K vocab at 2.27 chars/token (2,486-character base alphabet; most characters
  are rare Unicode). Byte-fallback would preserve 0% UNK AND compress better — a
  legitimate improvement we did not make. Exponents remain valid since D is
  counted in tokens and the fit is self-consistent. Flagged as a caveat on the
  α-vs-0.34 comparison.
- 9.0M rung capped at D=10N (89.9M tokens), so the largest-N column spans 0–90M
  not 0–180M tokens. The fit does not require the largest rung to reach 20N.
- Interior points are mid-schedule (models partway through a schedule sized for
  a larger D, LR not yet annealed) and carry an unquantified bias.
- 6 rungs only.

**Falsification condition** (pre-registered in
`results/v2/act4-scaling/FALSIFICATION_CONDITION.md`): **NOT FALSIFIED.**
LORO R² ≥ 0.985 on every held-out rung; residual correlations with log N and
log D are 0.002 and 0.020 (negligible curvature).

**Artifacts (tracked).**
- `results/v2/act4-scaling/fit_results.json` — both fits, LORO, diagnostics
- `results/v2/act4-scaling/equal_D_table.csv` — val_loss at D = 2M, 5M, 10M, 20M, 40M, 80M
- `results/v2/act4-scaling/FALSIFICATION_CONDITION.md` — pre-registered before any run
- `report/figures/act4_scaling_ladder.png` / `.pdf` — 2-panel (linear + log y)
- `report/figures/act4_loss_vs_N_fixed_D.png` / `.pdf` — loss vs N at fixed D

### 2026-10-18 — Act 5 (Stage E, part 2) — MQA@2.6M n=6 RESOLVED + 5c KV-CACHE BUG

**MQA quality cost at 2.6M — SIGNIFICANT at n=6 (two-sided p = 0.0008).**

The n=3 result (p = 0.0811, not significant) was a small-sample artefact. The
owner approved one pre-declared block of three further seeds (45/46/47) rather
than open-ended seed-adding; no seeds were added selectively after seeing
results. Seeds 42/43/44 are the original n=3, 45/46/47 the new block.

| seed | MHA G=8 | MQA G=1 | diff | elapsed |
|---|---|---|---|---|
| 42 | 2.8111 | 2.8558 | +0.0447 | 0.7 s (resumed) |
| 43 | 2.8331 | 2.8474 | +0.0143 | 2335 s |
| 44 | 2.8382 | 2.8661 | +0.0279 | 1945 s |
| 45 | 2.8205 | 2.8464 | +0.0259 | 2032 s |
| 46 | 2.8129 | 2.8429 | +0.0300 | 2367 s |
| 47 | 2.8190 | 2.8474 | +0.0284 | 2094 s |

mean **+0.0285 nats**, SD 0.0097, t = 7.182 on df = 5, **two-sided p = 0.0008**,
direction consistent **6/6**. Two-sided throughout; the hypothesis was framed as
three open outcomes before the data, so a one-sided p was never available.

**Sensitivity check.** Seed 42's G=1 arm resumed from an already-complete
checkpoint in 0.73 s, so its full-run provenance is unverified. Dropping it:
n=5, mean +0.0253, p = 0.0009, 5/5 consistent. **Conclusion unchanged** — the
finding does not depend on the suspect seed.

The earlier n=3 p=0.0811 and this n=6 p=0.0008 are the same effect; the SD
simply fell from 0.0152 to 0.0097 with three more pairs.

**5c KV-cache bug — the headline 5c number was wrong before this fix.**

`speculative_decode` re-fed the ENTIRE sequence at the top of every round on top
of an already-prefilled KV cache, in both the draft loop and the target
verification loop. The context was double-counted, so the target's `p` differed
from the draft's `q` even when draft and target were the *same weights*. Symptom:
acceptance 15-40% at zero gap, where it must be exactly 100%.

It did not fail loudly because cached and uncached paths shared the same bug, so
the "cached == uncached" tests passed while both were wrong.

Fix: a strict invariant — at the top of each round the cache holds exactly the
accepted sequence, and the retained "base" logits predict the position after it;
every inner forward feeds only new tokens. Rejection truncates to the accepted
prefix, then feeds the resampled token.

| | before | after |
|---|---|---|
| acceptance at zero gap (K=3) | 15.6% | **100%** |
| throughput at zero gap (K=3) | 28.9 tok/s | **309.8 tok/s** (10.7x) |
| wall-clock vs plain cached, K=3 | 0.09x | 0.43x |

Regression test: `tests/test_speculative_cache.py::test_zero_gap_acceptance_is_total`
— asserts 100% at K in {1,2,3,5,7} for both sampling and greedy. This is the test
that would have caught it; the equivalence tests could not.

**5c zero-gap result — NEGATIVE, and expected.** 1.5M -> 1.5M, cache on both
sides, honest baseline (plain generation *with* cache): speedup 0.41-0.50x
across K in {1,2,3,5,7}, acceptance 100%. At zero gap no target forward pass is
saved while each round still costs 2K forwards against K for plain decoding.
Zero gap exists only to prove correctness end-to-end on a real model; it is not a
performance result. **Real-gap speedup is not yet measured.**

**Method note.** `paired_ttest` moved from an ad-hoc snippet into
`src/bench/stats.py`, validated against the df=2 closed form
(`p = 1 - t/sqrt(t^2+2)`, agrees to 1e-6) and the standard critical-value table
(df = 1,2,3,5,10,20,30 all give p = 0.050 at the 0.05 critical value). No SciPy in
the environment, so the incomplete beta continued fraction is hand-rolled and
verified.

**Open.** Real-gap 5c (1.5M -> 2.6M -> 6.3M) acceptance and wall clock; 2.6M and
6.3M canonical checkpoints.

### 2026-10-19 — Act 5 COMPLETE — 5c at a real gap, canonical rungs, figures

**Canonical pinned set finished.** All three rungs trained and pinned:

| rung | non-emb params | step | val_loss | elapsed |
|---|---|---|---|---|
| 1.5M | 1,536,000 | 7,500 | 3.0571 | 6,566 s |
| 2.6M | 2,654,208 | 12,960 | 2.8111 | 2,237 s |
| 6.3M | 6,291,456 | 30,720 | 2.5152 | 9,277 s |

The 2.6M canonical reproduced the MHA (G=8) seed-42 sweep loss of 2.8111
**exactly**, from a different script and a different day. That is an unplanned
but strong determinism check on the whole training path.

**5c at a real model gap — NEGATIVE. 0 of 50 cells exceeded plain decoding.**

1.5M draft, KV cache on both sides, honest baseline (plain target generation
*with* the cache). 5 real C4 prompts x K in {1,2,3,5,7} x best-of-3, 64 new
tokens, CPU, machine otherwise idle:

| K | 1.7x gap (2.6M) | 4.2x gap (6.3M) |
|---:|---:|---:|
| 1 | 0.502 | 0.610 |
| 2 | 0.441 | 0.481 |
| 3 | 0.399 | 0.454 |
| 5 | 0.346 | 0.339 |
| 7 | 0.300 | 0.228 |

Best cell anywhere: **0.622x**. Median acceptance 0.857 (K=1) down to 0.233
(K=7, 4.2x gap).

**Why — arithmetic, not noise.** With per-token acceptance α a round yields
(1 − α^(K+1))/(1 − α) tokens for 2K forwards. That ratio is below 1.0 for
EVERY K > 1 even at α = 1, because 2K forwards buy at most K+1 tokens. So K > 1
is a net loss on forward count before any hardware consideration, and a real
speedup additionally needs the target forward to be far more expensive than the
draft forward. At 1.5M–6.3M non-embedding parameters neither forward is
compute-bound; both are latency-bound by per-op dispatch, so the extra forwards
cost what they cost. The measured monotone decline with K tracks the predicted
ratio.

The larger gap helps at small K (0.502 → 0.610 at K=1) and hurts at large K
(0.300 → 0.228 at K=7), because long drafts from a weaker model diverge.

**Acceptance is not near zero, and that is the interesting part.** At the first
generated token, 50-sample acceptance is 0.82 (2.6M) and 0.90 (6.3M) — the
*larger* gap accepts MORE, because 6.3M is the better-trained model (2.5152 vs
2.8111), not because the draft is closer in size. Acceptance is high because all
three rungs share corpus, tokenizer and recipe. So the negative result is not
"speculative decoding fails"; it is "at this scale the forward-count budget
dominates acceptance".

**Correction to an earlier reading.** A first pass reported K=1 acceptance
0.857 for BOTH gaps, which looked like a bug. It was a small-sample coincidence:
the median of only 15 measurements per gap, many of which were exactly 1.0. With
50 samples the two gaps differ (0.82 vs 0.90) and the earlier figure was an
artefact of the estimator, not of the implementation. Recorded because the
original number was already written down.

**Method correction.** The 1.7x gap was first benchmarked while 6.3M was
training on MPS, and CPU contention inflated it (plain 0.111 s vs 0.052 s idle).
Speedup is a within-run ratio so it is less affected than the absolute timings,
but the comparison across gaps was not fair. Both gaps were re-run on an idle
machine; the numbers above are the idle ones.

**Figures (tracked, PNG + vector PDF), generated by `src/eval/act5_figures.py`
from the tracked JSON so they cannot drift from the results:**
- `report/figures/act5_kv_cache.*` — 5a latency + cache memory
- `report/figures/act5_speculative.*` — 5c speedup and acceptance vs K, both gaps
- `report/figures/act5_mqa_paired.*` — the six paired MQA differences

**Act 5 conclusion.** KV caching is a large, unambiguous win that grows with
context (18.36x at 1024). Sharing KV heads buys an 8x cache reduction and a
measurable 12.1% FLOP cut, at a real and now-significant quality cost
(+0.0285 nats, p=0.0008). Speculative decoding is distributionally exact and
correctly implemented, and is nonetheless a 2–4x LOSS at every configuration
tested, for a reason that is arithmetic rather than incidental.

**Caveats carried forward.**
- CPU best-of-3 wall clock, durations 0.03–0.11 s. Not a controlled harness.
- Short sequences (32-token prompt, 64 generated) with a 128 context window, so
  the cache is small and recomputation would have been cheap anyway.
- SWA long-context quality-cost curve still not measured (ctx-512 sweep dropped
  for cost). Reported at context 128 only.
- Seed 42's G=1 arm provenance remains unverified (0.73 s resume).

### 2026-10-19 — 5c methodology write-up (item 4)

#### The double-fed-context bug

Speculative decoding re-fed the full sequence on top of a prefilled KV cache,
in both the draft loop and the target verification loop. The context was
double-counted, so the target's `p` differed from the draft's `q` even when
draft and target were the *same weights*.

**It failed SILENTLY.** The cached and uncached equivalence tests both passed,
because both paths shared the bug and therefore agreed with each other while
both being wrong. A comparison test cannot detect a fault that both sides share.

What exposed it was an absolute sanity invariant that no differential test would
suggest: **acceptance must be exactly 100% at a zero draft-target gap**, because
`p == q` makes `min(1, p/q) = 1` for every token. Measured acceptance was
15–40%. That single property localised the fault in minutes.

| | before fix | after fix |
|---|---:|---:|
| acceptance at zero gap (K=3) | 15.6% | **100%** |
| throughput at zero gap (K=3) | 28.9 tok/s | **309.8 tok/s** (10.7x) |
| wall-clock vs plain cached (K=3) | 0.09x | 0.43x |

After the fix, cached output is byte-identical to uncached, and greedy
speculative output is byte-identical to plain greedy. **All 5c numbers taken
before that commit (`86e6690`) are void.**

Fix: a strict cache invariant. At the top of every round the cache holds
exactly the accepted sequence and the retained "base" logits predict the
position after it, so every inner forward feeds only new tokens. Rejection
truncates to the accepted prefix and then feeds the resampled token.

**Generalisable lesson, worth stating in the report: a test that compares two
implementations cannot detect a bug they both share.** Differential testing
assumes an independent oracle. Where none exists, prefer an absolute invariant
derived from theory — acceptance = 100% at zero gap — over agreement between two
paths. The invariant is cheap, and it is the only reason this was caught before
publication rather than after.

#### 5c is a negative result with a mechanism, not an absence of one

Acceptance was **high** throughout: median 0.857 at K=1, and *higher* at the
4.2x gap than at the 1.7x gap (0.90 vs 0.82 at the first generated token, 50
samples) because 6.3M is the better-trained target (2.5152 vs 2.8111), not
because the draft is closer in size. All three rungs share corpus, tokenizer and
recipe, which keeps their next-token distributions close.

The loss is arithmetic. With per-token acceptance `a`, a round yields
`(1 - a^(K+1)) / (1 - a)` tokens for `2K` forwards. That ratio is **below 1.0
for every K > 1 even at a = 1**, because 2K forwards buy at most K+1 tokens.
So K > 1 is a net loss on forward count before hardware enters the argument,
and a genuine speedup further requires the target forward to be far more
expensive than the draft forward. At 1.5M–6.3M non-embedding parameters on one
MPS device, neither forward is compute-bound; both are latency-bound by per-op
dispatch, so the target's batching advantage is negligible and the extra draft
forward is pure cost. Measured: 0 of 50 (prompt, K) cells exceeded plain
decoding; best cell 0.622x.

**What this says about when speculative decoding pays.** It needs the target
forward to be *compute-bound*, so that verifying K+1 positions in one pass is
substantially cheaper than K+1 sequential passes. That regime is
memory-bandwidth-bound decoding at large scale. Our models are too small to
enter it, so the correct conclusion is about our scale, not the method — which
is why this is reported as a bounded negative rather than a refutation.

#### Seed 42 provenance (item 1) — confirmed by inference, not by artifact

The check as specified cannot be run: `artifacts/scaling/checkpoints/2.6M/seed42/`
now holds the **canonical** G=8 model (written Oct 18 14:14, val_loss 2.81107),
which overwrote the sweep's G=1 arm. `completed_steps` is 12960 and D is
53,084,160 = 12960 x 32 x 128, matching the sweep.

The G=1 provenance is nonetheless established, by load semantics:

- `load_state_dict` defaults to `strict=True` and **raises** on shape mismatch.
- This is empirically confirmed by the n=6 chain crash, which reported
  `size mismatch for blocks.5.attention.kv.bias: torch.Size([48]) vs torch.Size([384])`
  — i.e. G=1 (2·1·24 = 48) versus G=8 (2·8·24 = 384) for the 2.6M model's
  head_dim of 24.
- Therefore a G=8 checkpoint **could not** have loaded into the G=1 seed-42 arm.
  The 0.73 s resume necessarily loaded a shape-compatible G=1 checkpoint already
  at step 12960.
- The resulting loss 2.8558 sits inside the G=1 family (2.8429–2.8661,
  mean 2.8477); a mis-specified load would be wildly out of family.

**Caveat, stated plainly:** this is an inference from load semantics and value
plausibility, not a re-verification of the artifact, which no longer exists.
It is not re-checkable without retraining seed 42 G=1 (~39 min). The conclusion
does not rest on it — excluding seed 42 entirely gives n=5, mean +0.0253,
p=0.0009, 5/5 consistent.

#### Test suite no longer rewrites tracked evidence (item 2)

Root cause: `src/ngram/run_comparison.py` built its output path as
`Path("results") / act / name` with no way to redirect it, and
`tests/test_ngram.py::ComparisonRunnerTests::test_run_ngram_experiment` called
it without one. Every suite run therefore rewrote
`results/act0-ngram/o2-laplace-seed42/metrics.json` and `sample.txt`. The
damage is a nondeterministic `wall_clock_seconds` field, which is exactly why it
would have reached a commit unnoticed — and indeed it did, twice.

Fixed by making the root a parameter (`output_root`, defaulting to `results` so
the CLI is unchanged) and pointing the test at a temp directory.

`tests/test_results_append_only.py` guards it two ways, both verified to FAIL
with the bug reintroduced and PASS with the fix:
1. SHA-256 fingerprint of every file under `results/` before and after invoking
   the real entry point.
2. A static check that no module under `src/` hard-codes a `results` path.

**Correction to my earlier report.** I previously stated this problem was
"unfixed". When first re-checked, a content-hash comparison showed `results/`
unchanged and the suite appeared innocent. That was wrong: the rewrite produces
an identical `wall_clock_seconds` some runs, so content hashing missed it and
only the file mtime moved. The original claim was correct.

#### Canonical weights are now tracked (item 3)

The three canonical models are exported to `checkpoints/canonical/` (~74 MB,
weights only, no optimizer state, float32) so a reader can regenerate every
Act 5 figure without ~3.4 h of retraining. Verifying one's own figures is the
premise of the project; shipping claims with no way to re-derive the weights
undercuts it.

Exports verified **bit-identical** to source: `torch.equal` on logits and
identical greedy generation for all three rungs.
`tests/test_canonical_checkpoints.py` re-checks SHA-256 digests, step counts,
losses and that each file reconstructs a runnable model, without touching
`artifacts/`, so it passes in a fresh clone.

Actual size is ~74 MB, not the ~41 MB estimated: `state_dict` stores the tied
embedding twice (`token_embedding.weight` and `lm_head.weight`). Kept as-is —
dropping the duplicate would force `strict=False` loading and weaken the
verification guarantee for 37 MB.

Test suite after items 1–3: **236 tests, 0 failures, 2 skipped.**

### 2026-10-19 — Act 6 COMPLETE — decoding sweep, best-of-n, self-consistency

6.3M canonical, real C4 prompts (40, 32 tokens each), 32 new tokens, seeds
42/43/44 (120 generations per configuration), MPS. Quality = bits-per-character
of the GENERATED text under a fixed 6.3M reference; diversity = distinct-2 and
repeated-4-gram rate.

#### 6a Decoding sweep — the trade-off is clean and strictly monotone

| config | BPC (ref) | sd | rep-4 | distinct-2 | seconds |
|---|---:|---:|---:|---:|---:|
| beam4 | 1.3760 | 0.452 | 0.3095 | 0.5944 | 113.1 |
| beam8 | 1.3790 | 0.435 | 0.4052 | 0.5177 | 254.0 |
| beam2 | 1.3985 | 0.473 | 0.3293 | 0.5750 | 55.1 |
| greedy | 1.4366 | 0.525 | 0.4147 | 0.4952 | 23.1 |
| top-p 0.3 | 1.5878 | 0.535 | 0.1181 | 0.7785 | 33.1 |
| top-k 5 | 1.6012 | 0.534 | 0.0773 | 0.8075 | 33.5 |
| temp 0.5 | 1.6083 | 0.538 | 0.1293 | 0.7613 | 32.9 |
| top-p 0.5 | 1.6624 | 0.561 | 0.0687 | 0.8473 | 33.1 |
| top-k 20 | 1.7153 | 0.592 | 0.0233 | 0.8960 | 33.2 |
| temp 0.7 | 1.7251 | 0.523 | 0.0503 | 0.8661 | 32.9 |
| top-k 50 | 1.8103 | 0.611 | 0.0155 | 0.9290 | 33.1 |
| top-p 0.8 | 1.8381 | 0.590 | 0.0138 | 0.9368 | 33.1 |
| top-p 0.95 | 1.9701 | 0.601 | 0.0095 | 0.9618 | 33.1 |
| temp 1.0 | 2.0074 | 0.597 | 0.0014 | 0.9688 | 33.0 |
| temp 1.3 | 2.3661 | 0.581 | 0.0000 | 0.9895 | 33.0 |

- corr(distinct-2, BPC) = **+0.87**; corr(rep-4, BPC) = **−0.79**. Diversity
  costs quality essentially monotonically across the whole sweep.
- **Beam search beats greedy on quality**: beam-4 is 0.061 bits/char better than
  greedy (1.3760 vs 1.4366). Beam search finds more *typical* text than greedy,
  which is the expected result, not a surprise.
- **But it costs 4.9x greedy's time** for that 0.061 bits (113.1 s vs 23.1 s for
  120 generations), and beam-8 is worse than beam-4 (1.3790) at 2.2x the cost —
  the classic diminishing return.
- **Greedy is the quality/cost knee**: no other configuration is both cheaper and
  better. Every stochastic configuration is cheaper than beam-4 but at least
  0.15 bits/char worse.

#### 6b Best-of-n — reranking helps a lot, and still cannot reach greedy

Sample n continuations at temperature 1.0, rerank by mean log-probability under
the model itself.

| n | BPC of reranked pick | BPC of mean sample | gain |
|---:|---:|---:|---:|
| 1 | 2.0074 | 2.0074 | +0.0000 |
| 2 | 1.8684 | 1.9581 | +0.0897 |
| 4 | 1.7888 | 1.9538 | +0.1650 |
| 8 | 1.7459 | 1.9760 | +0.2302 |
| 16 | 1.6959 | 1.9677 | +0.2717 |

Reranking recovers **+0.272 bits** over the mean sample at n=16 — a large,
monotone, log-shaped gain, exactly as theory predicts for a max-of-n statistic.
**But best-of-16 is still 0.259 bits WORSE than greedy (1.6959 vs 1.4366).**
The reason is structural, not a tuning failure: the reranker is the model
itself, so it can only select for text the model already finds likely. Best-of-n
with a self-referential scorer converges toward the model's own mode
asymptotically, and never exceeds it. Reaching greedy would need an *independent*
scorer, which a 6.3M model does not have.

#### 6c Self-consistency — NEGATIVE, and the reason is exact

MNLI, 300 items, zero-shot. The model was trained on C4 and never on MNLI, so
this tests whether voting can manufacture competence that is not there.

| votes | accuracy | 95% Wilson CI | vs chance (0.3333) |
|---:|---:|---|---|
| 1 | 0.3789 | [0.3478, 0.4110] | nominally above |
| 3 | 0.3556 | [0.3250, 0.3874] | n.s. |
| 5 | 0.3556 | [0.3250, 0.3874] | n.s. |
| 10 | 0.3611 | [0.3304, 0.3930] | n.s. |
| 20 | 0.3567 | [0.3260, 0.3885] | n.s. |

Majority voting gives **no gain at any n**. The mechanism, measured rather than
guessed:

- **Argmax accuracy is 0.3300 — exactly chance** (99/300, CI [0.2792, 0.3851],
  which contains 0.3333). Majority voting converges to argmax as n grows, so
  there is nothing for it to improve on.
- The model **never predicts "neutral"** (recall 0.000). It predicts
  entailment 226 times against 111 gold, contradiction 74 against 105.
  Per-class recall: entailment 0.712, neutral 0.000, contradiction 0.190.
- Restricted to the 216 items it can ever answer (entailment/contradiction),
  argmax accuracy is **0.4583, BELOW the 0.5139 coin-flip baseline**.
- Mean max-probability 0.4949 and mean entropy 0.7574 (uniform is 1.0986), so
  the distribution is biased and over-confident, not uniform noise.

**The n=1 figure of 0.3789 must not be read as partial competence.** It exceeds
chance only because sampling is an unbiased draw from a biased distribution,
while argmax is dominated by the entailment over-prediction. The model has no
demonstrated MNLI signal: argmax is exactly chance, and below coin-flip on the
reachable subset. Conclusion: **self-consistency cannot rescue a model with no
task competence** — it converges to that model's own decision rule, so it can
only amplify what is already there.

**Caveats carried forward.**
- The BPC axis is scored by the model under test, so it is self-preferring. It
  is valid for *ranking configurations against each other* under a fixed judge,
  and is not an absolute measure of text quality.
- Context 128, 32-token prompts, 32 new tokens, MPS, best-effort timing.
- MNLI verbalizers are "Yes,/Maybe,/No," scored by full-sequence log-probability
  and softmaxed over the three, so the answer space is a proper 3-way
  distribution regardless of tokenisation. A tiny C4 model still has strong
  priors over those surface forms, which is part of what the entailment
  over-prediction measures.
- SUPERSEDED: the original caveat that "beam timings include no KV cache" is no
  longer true. Beam now genuinely uses a per-beam cache and all timings are
  taken after an explicit warm-up. See the Act 6 CORRECTION entry below.

**Artifacts (tracked).**
- `results/v2/act6/6.3M_decoding.json` — all three experiments
- `report/figures/act6_decoding_tradeoff.png` / `.pdf`
- `report/figures/act6_bestofn_selfconsistency.png` / `.pdf`
- Generated by `src/eval/act6_figures.py` from the tracked JSON

### 2026-10-19 — Act 6 CORRECTION: the 5a beam-cache claim was FALSE

**The Act 5a ledger asserted "greedy/top-k/top-p/beam all use it; beam keeps
per-beam caches." That was never true.** `beam_search` had no cache parameter in
ANY commit — verified by inspecting every revision of `src/generation/decoding.py`:

| commit | beam lines | cache references inside `beam_search` |
|---|---:|---:|
| `f1eaf77` (5a, Oct 17) | 328 | **0** |
| `201c846` (v1, Oct 4) | 22 | **0** |

So this was **not a later regression** — the claim was false when written.
`tests/test_kv_cache.py` covered greedy, sample and top-p for cache equivalence
and **never tested beam at all**. An assertion with no test behind it survived
into the report. That invalidates the Act 6 COST column, as flagged.

#### What the cache actually does per path (instrumented, not asserted)

Forward passes counted by wrapping `MiniGPT.forward`, 32 new tokens:

| path | forwards | token-positions | uses KVCache |
|---|---:|---:|---|
| greedy (`generate`, `use_cache=True`) | 32 | 53 | **yes** |
| temperature (`generate`, `use_cache=True`) | 32 | 53 | **yes** |
| top-k / top-p (`generate`, `use_cache=True`) | 32 | 53 | **yes** |
| beam (before this fix) | 32 x beam_width | full prefix each | **no** |

So greedy and sampled decoding were already correct and equally cached. **My
earlier speculation that a multinomial draw "cannot account for 10 seconds" was
wrong**: `multinomial` over 6,487 categories costs 1.401 ms/call on MPS versus
0.025 ms for `argmax`, i.e. 1.38 ms/token ≈ 1.8 s over 1,280 tokens, with the
remainder being MPS per-op overhead interleaved with the matmul forward. The
greedy-vs-sampling cost gap is real and is not a caching artefact.

#### Beam now genuinely uses the cache

`beam_search(..., use_cache=True)` keeps one cache per beam. Because a beam's
next-token logits are already available from the previous step, expansion to
scores needs **no forward pass**; forwards are spent only on the beam_width
survivors, whose caches must be advanced by their chosen token.

First attempt was **2.5x SLOWER** than uncached (beam-4: 4939 ms → 12413 ms).
Cause: cloning the cache per *candidate* costs beam_width² deep copies of the
whole prefix per step. Fixed by materialising caches only for survivors
(beam_width clones/step). Second attempt introduced a real bug — prefilling
`prompt[:, :-1]` made the first distribution predict the last *prompt* token —
caught because the identity test failed. Identity now holds exactly at widths
1/2/3/5.

`KVCache.clone()` is deep by design; a shallow copy would let one beam's append
corrupt its siblings. Tested in `tests/test_beam_cache.py`.

#### Does beam-4 with the cache cost ~25 s? No — 104.0 s

At the Act 6 context (32-token prompt, 32 new tokens, context window 128) the
cache barely helps, because the prefix is short: recomputing a 32–64 token prefix
is cheap, and each survivor still pays a cache deep-copy.

| prompt len (total ctx 128) | uncached | cached | speedup |
|---:|---:|---:|---:|
| 32 | 12677 ms | 10931 ms | 1.16x |
| 64 | 7972 ms | 6890 ms | 1.16x |
| 96 | 4269 ms | 3378 ms | **1.26x** |

Caching in beam search only pays as the sequence lengthens.

#### Corrected Act 6 cost axis (all paths cached, timed after explicit warm-up)

The old column was also taken **cold**, so the first configuration measured
absorbed MPS graph-compilation cost. A `warmup()` now runs before any timing.

| config | BPC | OLD s | NEW s | change |
|---|---:|---:|---:|---:|
| beam4 | 1.3760 | 113.1 | **104.0** | 1.09x |
| beam8 | 1.3790 | 254.0 | **232.9** | 1.09x |
| beam2 | 1.3985 | 55.1 | 61.4 | 0.90x |
| greedy | 1.4366 | 23.1 | 24.4 | 0.95x |
| top-p 0.3 | 1.5878 | — | 31.3 | |
| top-k 5 | 1.6012 | 33.5 | 30.9 | 1.08x |
| temp 0.5 | 1.6083 | 32.9 | 33.8 | 0.97x |
| top-p 0.5 | 1.6624 | 33.1 | 31.3 | 1.06x |
| top-k 20 | 1.7153 | 33.2 | 31.1 | 1.07x |
| top-k 50 | 1.8103 | — | 31.1 | |
| top-p 0.8 | 1.8381 | — | 32.6 | |
| top-p 0.95 | 1.9701 | — | 35.3 | |
| temp 1.0 | 2.0074 | 33.0 | 33.7 | 0.98x |
| temp 1.3 | 2.3661 | 33.0 | 34.6 | 0.95x |

All 15 BPC / rep-4 / distinct-2 values are **bit-identical** to the first run,
which independently confirms the fix changed cost only, never output.

#### Does the conclusion flip? No — but the number changes

**beam-4 vs greedy is 4.26x the time (was 4.9x), not comparable.** Beam-4 is
still 0.061 bits/char better for 4.26x the cost, so "greedy is the quality/cost
knee" **survives**: no configuration is both cheaper and better. beam-2 is
2.52x greedy. The correction moves the ratio from 4.9x to 4.26x; it does not
reverse the finding.

**Method lesson, recorded because this is the second time a headline number
rested on an unverified claim about what the code does:** instrument forward
passes and cache lengths rather than reading a ledger sentence. A claim that
carries a number in the report needs a test that fails when the claim is false —
`test_cached_beam_matches_uncached` and `test_clone_is_independent` are exactly
the tests whose absence let 5a's claim survive.

---

### 2026-10-19 — Act 7 COMPLETE (SFT, ICL vs SFT, PEFT head-to-head, LoRA rank sweep)

Base: pinned canonical 6.3M (val_loss 2.5152). 512 train / 128 held-out, 7-task
synthetic family, 3 seeds, 24 epochs. Evaluation is teacher-forced exact match
on the held-out split with Wilson 95% intervals.

#### Defects found and fixed on the way (both invalidated entire result sets)

**D11 — input text never reached the prompt.** `InstructionExample` is
`(instruction, response, input_text)`. The task builder passed the payload
positionally, binding it to **response** and leaving `input_text` empty. The
prompt rendered as:

```
### Instruction:
Copy the text exactly.

### Response:
```

Six of seven tasks were unanswerable by construction while scoring still
compared against the answer. The first full Act 7 run therefore produced
near-zero accuracies (full FT n=512: 0.0104; every PEFT method: 0.0000) that
read exactly like a legitimate negative result.

Diagnosis trail, recorded because the shape of the failure was the tell:

| signal | value | meaning |
|---|---|---|
| train loss | 0.3151 | the model HAD learned the token distribution |
| exact match | 0.0000 | but the strings never matched |
| tokenizer round-trip | clean | ruled out decoding |
| **position-0 accuracy** | **0.070** | first response token unguessable |
| **positions 1–2 accuracy** | **0.927 / 0.900** | trivially copyable from the true prefix |

That profile — fail only on the first-token commitment, succeed when fed the
correct prefix — is diagnostic of a prompt that omits the input. After the fix,
on the same 6.3M / 24 epochs: loss **0.3151 → 0.0048**, exact match
**0.0000 → 0.6250**, `first_word` and `last_word` **0.000 → 1.000**.

**D12 — double cross-entropy shift in three PEFT forwards.** `MiniGPT` computes
`CE(logits, targets)` with **no internal shift**, because the SFT loop already
passes labels aligned to the inputs. The PEFT wrappers for prompt tuning, prefix
tuning and adapters each used `CE(logits[:, :-1], targets[:, 1:])`, shifting a
second time. They trained to predict the wrong position.

Compounding it, `score_response` sliced `pred[-n:]`, which is only correct when
the method returns as many logits as there are inputs. Prompt tuning prepends
virtual tokens, so its logits are shifted and the slice was off by one.

Consequence: **all three methods reported exactly 0.0000 while genuinely
training.** Fixed by (a) matching `MiniGPT`'s convention in all three forwards,
and (b) replacing the slice with `response_slice(logits_len, input_len,
n_prompt, n_resp)`, whose index arithmetic is pinned to the already-correct
`extra == 0` case.

LoRA and full fine-tune route through `MiniGPT.forward` and were **never
affected**, so 7b and 7d were computed correctly the first time.

**Habit note:** this is the third headline number in this project to rest on an
unverified claim about what the code does (after the 5a beam-cache claim). The
discriminating move this time was measuring per-position accuracy rather than
arguing about aggregate accuracy.

#### 7a — zero-shot floor

**0.0000** on all 128 held-out items. The base model does none of these tasks,
so every number above zero is attributable to adaptation.

#### 7b — ICL vs SFT vs full fine-tune (the LIMA question)

| n train | ICL 4-shot | full fine-tune (n=3) |
|---:|---:|---:|
| 16 | 0.0000 | 0.0703 (see seed caveat) |
| 64 | 0.0000 | 0.1667 `[0.1797, 0.1562, 0.1641]` |
| 256 | 0.0000 | 0.3255 `[0.3828, 0.2656, 0.3281]` |
| 512 | 0.0000 | 0.5651 `[0.5938, 0.5234, 0.5781]` |

- **ICL is 0.0000 at every set size and every seed.** A 4-shot prompt built from
  a disjoint pool changes nothing; this model does not perform in-context
  learning on these tasks. (Scored teacher-forced, so this is not a generation
  artefact.)
- **No saturation anywhere in range.** The curve is still rising steeply from
  256 → 512 (+0.24). At this scale the LIMA "less is more" claim is **not
  supported**: a few hundred examples is not enough, and more is still better.
- **Seed caveat:** the n=16 row is identical across all three seeds
  (0.0703 × 3). At 16 examples with batch_size 16 there is one batch per epoch,
  so the seeded shuffle has nothing to shuffle. That row is effectively n=1 and
  must not be reported with error bars.

#### 7c — PEFT family head-to-head at matched budget (20,000 trainable params)

| method | size | trainable | train loss | accuracy | Wilson 95% |
|---|---:|---:|---:|---:|---:|
| full fine-tune | — | 8,012,032 | 0.0043 | **0.5755** | [0.5256, 0.6240] |
| adapters | 4 | 18,464 | 1.1365 | **0.2136** | [0.1755, 0.2573] |
| LoRA | 1 | 14,336 | 1.3097 | 0.1823 | [0.1469, 0.2240] |
| prompt tuning | 64 | 16,384 | 2.0697 | 0.0859 | [0.0618, 0.1182] |
| prefix tuning | 5 | 20,480 | 3.2576 | 0.0000 | [0.0000, 0.0099] |

- The matched budget is what makes this readable: **adapters beat LoRA** at a
  comparable parameter count (0.2136 vs 0.1823), with Wilson intervals that
  overlap, so the ordering is suggestive rather than established.
- **`final_train_loss` is recorded for exactly this reason**: it separates "did
  not fit" from "fit but scored zero". `prefix_tuning` scores 0.0000 at
  train_loss 3.2576 — it does not fit the budget at all. That is a measurement,
  not a scoring artefact, and the distinction was only visible because the loss
  column exists.
- Full fine-tune at 8.0M trainable (400× the budget) reaches 0.5755, roughly 2.7×
  the best budget-matched method.

#### 7d — LoRA rank sweep

| r | trainable | accuracy | runs |
|---:|---:|---:|---|
| 1 | 14,336 | 0.1823 | [0.1797, 0.1953, 0.1719] |
| 2 | 28,672 | 0.1927 | [0.1953, 0.1719, 0.2109] |
| 4 | 57,344 | 0.2838 | [0.3281, 0.2734, 0.2500] |
| 8 | 114,688 | 0.3516 | [0.3672, 0.3203, 0.3672] |
| 16 | 229,376 | 0.4245 | [0.4375, 0.3828, 0.4531] |
| 32 | 458,752 | 0.4713 | [0.4609, 0.5000, 0.4531] |
| 64 | 917,504 | 0.5078 | [0.5000, 0.5234, 0.5000] |

Monotonic across the full range with no plateau; seed spread is tight (≤0.05).
Rank 64 recovers 0.5078 against full fine-tune's 0.5755 at 11.5% of the
parameters. **LoRA rank is a smooth capacity dial here, not a threshold.**

#### Defects fixed (AGENTS.md §5)

- **D4** — SFT is minibatched from a seeded shuffle (one optimizer step per
  minibatch), not full-batch on the whole split. Asserted in
  `test_minibatching_actually_batches`.
- **D5** — every response carries a supervised EOS terminator; the model is
  explicitly trained to emit it. Asserted in
  `test_prompt_tokens_are_masked_and_eos_is_supervised`.

#### Caveats carried forward

1. **Synthetic task family**, not natural instructions: measures whether
   adaptation transfers, not instruction-following in the wild.
2. **ICL is teacher-forced with a 4-shot prompt from a disjoint pool**, so it is
   not comparable to free-running generation numbers elsewhere.
3. **n=16 is effectively n=1** (single batch per epoch; see above).
4. **Seeds vary optimisation order only** — the base weights are the pinned
   canonical checkpoints, so initialisation and data source are fixed.
5. Held-out exact match on 128 items; Wilson intervals are correspondingly wide
   (≈ ±0.04), so the adapters-vs-LoRA ordering is not established.
6. **Genuine negative retained:** the `add` task scores 0.9452 on TRAIN items
   (69/73) but **0.0000 on HELD-OUT items (0/18)**. Held-out guesses are
   off-by-one — 8+2→11, 7+7→15, 6+8→15, 1+8→10 — the signature of memorising
   seen sums rather than learning addition. Reported as a result, not a failure.

### 2026-10-20 — Act 8 COMPLETE (hallucination rate, RAG controls, tool use)

Base: pinned canonical 6.3M (val_loss 2.5152), 128-token context. Prompts from
the real C4 shard (`data/v2/c4_en_shard_0.jsonl`), never synthetic tokens.

#### 8a — shared prompt builder, token-based context budget (D1, D2)

**D1's measured cost.** The v1 call passed `context_length // 2` as
`max_context_chars`. On the canonical 128-token window that is **64
characters ≈ 32 tokens** of a 302-token chunk: **9.2% of characters / 10.6% of
tokens survived**. The token budget admits **77 tokens instead of 32 (2.4×)**,
and the assembled prompt is 120 ≤ 128.

**D2.** RAG assembled its own template, so a model fine-tuned on the Alpaca
contract was evaluated on a different string. `src/rag/prompt.py` now routes
both through one `InstructionFormatter`; `rag_prompt` is a thin wrapper and
`shared_formatter()` is identity-tested against it.

`fit_context` binary-searches the longest context prefix whose **assembled**
prompt fits in tokens, holds back an answer reserve, and **re-encodes to
verify** rather than trusting the arithmetic. It raises rather than returning a
prompt that cannot be answered in.

D11's lesson is encoded: every `InstructionExample` is built with keyword
arguments, and one test pins the context under `### Input:`.

#### 8b/8c — what the passage actually changes (four conditions)

The claim format was declared **before** any generation was sampled:

> CLAIM "the supplied passage states `<NUMBER>`"
> classes `supported` / `contradicted` / `unsupported`
> hallucination = (contradicted + unsupported) / n = 1 − supported

Numbers are the claim currency because they are the one atomic fact that can
be extracted and verified mechanically from C4 without an annotator: a scan of
1,500 C4 documents found a number in **70.9%**, while world-fact templates
("the capital of X is Y") appeared in **0/1500**.

**First design, and why it was replaced.** The label design was measured over
40 items × 3 seeds × 3 conditions. It returned a rate of **1.000 in every
condition** — 97/120 oracle generations contained **no number at all**. The
ceiling makes the rate uninformative for ranking conditions, and it confounds
two different failures: *cannot ground* and *cannot emit the label*.

A prototype separated them. Under oracle the model preferred the number in
its context on **7 of 8** items; under closed-book the same comparison was
**5 of 8** — chance. So the model **can** ground; free generation was simply
blind to it.

**The instrument that replaced it (answer copying).** Rather than requiring
the model to volunteer a digit, the likelihood of each candidate answer is
scored and the argmax taken. Four conditions, one shared item set, the shared
prompt builder, and a token budget:

| condition | what it is |
|---|---|
| retrieved | a passage from the lexical retriever, queried with the instruction text — **no knowledge of where the answer lives** |
| oracle | the answer-bearing sentence |
| counterfactual | the same sentence with that number swapped |
| closed_book | no input section at all |

**Four results, n = 40 items, unit = item:**

| measurement | value | interval |
|---|---|---|
| oracle chose the item's answer | 9/40 = 0.2250 | Wilson [0.1232, 0.3750] |
| closed-book chose the item's answer | 11/40 = 0.2750 | Wilson [0.1611, 0.4283] |
| **McNemar, oracle vs closed-book** | discordant 1/3 | **p = 0.625 (n.s.)** |
| counterfactual tracks the edit | **0/40 = 0.0000** | boot [0.000, 0.000] |
| answer unchanged when the text is edited | 34/40 = 0.8500 | boot [0.725, 0.950] |
| context-specific discrimination | **+0.204 nats** | boot [+0.044, +0.391], **p = 0.030** |
| passage lift on the number it states | +0.786 nats | boot [+0.364, +1.269], p = 0.0017 |
| passage lift on the UNSTATED number | +0.583 nats | boot [+0.219, +0.996], p = 0.0057 |
| edit effect on the counterfactual number | +0.034 nats | boot [−0.010, +0.091], **p = 0.308 (n.s.)** |
| edited number preferred under counterfactual | +0.039 nats | boot [−0.242, +0.354], **p = 0.839 (n.s.)** |
| free generation: correct answer | **0/40 in every condition** | — |

**Reading, including a tension that is reported rather than smoothed over.**
Supplying the passage produces a *measurable, context-specific* shift in the
model's preference for the number it states (+0.204 nats, p = 0.030), and the
oracle margin is significant where the closed-book margin is not
(+0.352, p = 0.011 vs +0.148, p = 0.199). But the **symmetric test fails**:
when the number in the text is edited, the model does **not** prefer the
edited number (+0.039, p = 0.839), the argmax follows the edit in **0/40**
items, and the edit effect is n.s. (+0.034, p = 0.308). The conservative
reading is therefore that the model does **not** demonstrably read *which*
digit the passage states.

And none of it reaches the answer: oracle 9/40 vs closed-book 11/40, McNemar
**p = 0.625**. **The conclusion is a negative with a mechanism: the passage
shifts the preference a little, far below the threshold that changes what the
model would say, so RAG at 6.3M does not measurably improve groundedness.**

**The absolute hallucination rate is at a ceiling and therefore uninformative
for ranking conditions.** It is reported, labelled as such, and not used for
any comparison.

**Denominator correction.** The unit is the **item**, not the generation. The
answer-copying instrument is deterministic, so there is exactly one
measurement per item per condition: n = 40, and a 0/40 Wilson upper bound is
**0.0876**, not the 0.031 that a spurious n = 120 would have produced. Free
generation repeats are averaged per item before scoring, and contrast
intervals come from a **percentile bootstrap that resamples items**, because
repeats of an item share its difficulty and are not independent draws.

**Two instruments in one act.** Free generation is retained as the secondary
instrument because it answers a different question — whether the model
volunteers an answer at all (it does so 40–55% of the time, but never the
right one). The answer-copying instrument is the primary one.

#### 8d — tool use: a measured negative with a mechanism

40 questions whose operands are **real numbers taken from C4**, the shared
instruction format, and the existing `src.agent.calculator`.

| rate | greedy (n=40) | sampled (3 seeds, n=120) |
|---|---|---|
| well-formed `CALC(...)` call | **0/40 = 0.0000**, Wilson [0, 0.0876] | **0/120 = 0.0000**, Wilson [0, 0.031] |
| echoes the call template | **19/40 = 0.4750** | 3/120 = 0.0250 |
| malformed call (template echoed, argument unusable) | 19/40 = 0.4750 | 3/120 = 0.0250 |
| direct answer without a tool | 0/40 = 0.0000 | 1/120 = 0.0083 |

The bare rate is 0%. The mechanism is not: **19 of 40 greedy generations
emit the literal `CALC(`, and in all 19 the argument is exactly `"s"`** —
`CALC(s)`. The model reproduces the tool-call *format* and cannot fill it.
This is the same shape as Act 7's ICL result: the response format is learned,
the operation is not. (The greedy/sampled gap is real and reported: the
template echo is a mode phenomenon that T=1.0 sampling dilutes.)

#### Defects found and fixed in Act 8

- **D14 — the answer-copying score absorbed the prompt's own log-probability.**
  The score summed over the whole sequence, so a candidate's score also
  contained the prompt's likelihood. Harmless within one prompt (the prompt
  cancels) and invalid for every across-condition contrast, because an oracle
  prompt is far longer than a closed-book one. It surfaced as a passage-lift of
  **−246 nats**, which is absurd on its face. Fixed to score only the
  candidate tokens; pinned by a test using a constant-logit stub model, and
  demonstrated failing against the old version (2 failures → 0).
- **D15 — a tautological class.** The three-way was originally labelled
  "the number in the context", which for `closed_book` is **unreachable by
  construction** — its 0/40 was a definition, not a measurement, and read like
  a result. The classes are now condition-agnostic (`source` /
  `counterfactual` / `neither`), and "followed the context" is reported
  separately and left undefined for closed-book.
- **D16 — a grouped literal crashed the C4 scan.** `numbers_in` normalizes
  `1,000` → `1000`, and locating the number with `.index()` then raised
  `ValueError` on a sentence that plainly contained one. Positions are now
  taken from the raw regex match, and the swap targets the raw span.
- **D17 — an unfair candidate comparison.** The BPE has merged tokens for some
  numbers and not others (`35` is one token, `34` is three). Comparing them
  hands the win to whichever side got the merge. Counterfactuals are now
  matched to the source on **digit count and token count**.
- A containment guard drops candidates whose number falls outside the token
  budget (10 of 120 candidates), which would otherwise have been unwinnable
  items inflating the rate.
- A hand-rolled Wilson interval in an ad-hoc script returned a **negative
  lower bound (−0.2852)** because `z²/(4n)` was used where `z²/(4n²)`
  belongs. The project's validated `wilson_interval` is used everywhere now,
  and the extremes are pinned by a test.

#### Caveats carried forward

1. **The retrieved condition is degenerate by construction.** The retriever is
   queried with the instruction text, which contains nothing about any item's
   answer, so **retrieval success was 0/40** and retrieved is statistically
   indistinguishable from closed-book. A content-bearing query would be a
   different experiment. This is the one condition that does not test what it
   was added to test.
2. **The oracle margin and the counterfactual margin disagree**, and the
   report states both rather than choosing the favourable one.
3. Numbers are a proxy for atomic claims; the claim format does not extend
   to non-numeric facts without an annotator.
4. The candidate set is 6 numbers (the source, the counterfactual, and 4
   distractors drawn deterministically from other items).
5. Free generation uses 3 seeds; the answer-copying instrument is
   deterministic and has no seed dimension.

#### 2026-10-20 — Act 8 REVISION: headline ordering, exact chance baseline, retriever fixed

Four corrections to the Act 8 write-up, three of which change the claim.

**1. The headline is the CHOICE RATES, not the lift rows.** The lift rows are
the explanation and they cut *against* grounding, so leading with them
inverted the reading. The result, in the order it should be read:

| measurement | value | interval |
|---|---|---|
| oracle chose the asserted number | **9/40 = 0.2250** | Wilson [0.1232, 0.3750] |
| closed-book chose the asserted number | **11/40 = 0.2750** | Wilson [0.1611, 0.4283] |
| retrieved chose the asserted number | 8/40 = 0.2000 | Wilson [0.1050, 0.3476] |
| counterfactual chose the **edited** number | **0/40 = 0.0000** | Wilson [0, 0.0876] |
| McNemar, oracle vs closed-book | discordant 1/3 | **p = 0.625 (n.s.)** |

The model identifies which number a passage asserts **at chance**, and removing
the passage changes its choice on only 4 of 40 items, **not consistently** —
the closed-book point estimate is *higher*. The wording is "at chance", never
"below chance": 9/40 against a chance level of 0.1838 is not significantly
below it, and the chance-adjusted difference is +0.0412, bootstrap
[−0.078, +0.170], n.s.

**2. The chance baseline is per item, and it is not 1/3.** The candidate set is
deduplicated, so it holds 3, 4, 5 or 6 distinct numbers depending on the item
(sizes observed: 3×1, 4×3, 5×8, 6×28). The mean per-item uniform chance for the
asserted number is **0.1838**, and the chance-adjusted rates are reported per
condition: retrieved +0.0162, oracle +0.0412, counterfactual +0.0413,
closed-book +0.0913, nonnumeric +0.0413 — every one of them n.s.

**3. The "unstated number" is ABSENT from the passage, and that changes the
claim.** Checked empirically over all 40 items: the tracked partner number
appears in the fitted context in **0/40** oracle items and **0/40**
counterfactual items, because a scored sentence contains exactly one number by
construction of the claim format. (Distractors coincide with the context in
10/155 and 13/155 cases — negligible.) So the +0.583 nats "lift" for the
unstated number is **not** evidence that the passage mentions a number it does
not assert; it is a **generic context effect** on the candidate set: adding a
passage of this length raises the likelihood of *any* number, present or not.
This also means the lift rows confound "has a passage" with "has a longer
prompt" (closed-book is 32 tokens, the oracle prompt ~85). The claim is
therefore **a context-number bias, not an assertion-reading ability**. The only
content-specific trace is the residual +0.204 nats (p = 0.030), it is marginal,
and it is accompanied by no behavioural change at all, with the behavioural
point estimate in the opposite direction. The two instruments disagree and the
behavioural one is the one that matters.

**A new control condition, `nonnumeric`.** A different sentence of the same
document containing no number, so it is topic- and length-matched to the oracle
passage. It isolates "the passage" from "the number the passage asserts". It
scores **9/40 — exactly the oracle's 9/40**. Swapping the asserting passage for
a non-asserting one of the same document changes nothing at all. This is
direct behavioural support for the bag-of-numbers reading, and it is the
strongest single result in Act 8.

**4. The retriever was BROKEN, and 0/40 was a query-design failure, not a
finding.** Diagnosed before revising, as required:

| diagnostic | result |
|---|---|
| is the target document in the index? | **yes** — items came from C4 docs 0–77, the index covered docs 0–399 |
| lexical overlap, question ↔ passage | **1 term maximum** (only "the"): 25 items overlap 1, 14 overlap 0, 1 overlaps 2 |
| top-1 chunk contains the asserted number | 0/40 |
| top-1 chunk for two failing items | **the identical chunk both times**, score 0.6667 |

The query was the instruction text, so its terms were
`{what, number, does, the, passage, state}` — none of which can come from a
passage about a number. Term-overlap ranking degenerated to a constant and the
tie-break returned the same chunk for every item. The retriever was working; it
was being asked an impossible question.

**The fix**: the query is now the item's **lexical anchor** — content words
from a *different* sentence of the *same* document, excluding the number. The
retriever is given a topical handle and never sees the number or the
answer-bearing sentence, so the condition is answer-agnostic rather than a
disguised oracle. Re-measured:

| retrieval metric | before | after |
|---|---|---|
| distinct documents returned | **1** (degenerate) | **39** |
| document hit rate | — | **0.975** (39/40) |
| retrieved chunk contains the asserted number | 0/40 | **0.475** |
| chose the asserted number under `retrieved` | — | 8/40 = 0.2000, chance-adjusted +0.0162, n.s. |

The retrieved condition now behaves like the others: it is at chance, and the
model follows the retrieved number on 0.2857 of the items where the retrieved
chunk actually contained one.

**8d: the `CALC(s` argument is template completion, verified by token id.**
`CALC(` tokenizes to `[49, 4083, 49, 22]` = `['C', 'AL', 'C', '(']`. After that
prefix the next-token distribution is:

| rank | token | id | prob |
|---|---|---|---|
| 1 | `'s'` | 97 | 0.0278 |
| 2 | `'express'` | 6369 | 0.0214 |
| 3 | `'per'` | 2564 | 0.0152 |
| 4 | `'work'` | 2691 | 0.0127 |
| 5 | `'ess'` | 2587 | 0.0126 |

The argmax is the literal character `s` and the runner-up is `express` — a
fragment of the word **"expression", which appears in the prompt's own
template**. The model is completing the instruction's template, not writing an
arithmetic expression, and in all 19 greedy template echoes the argument is
exactly `"s"`. The argmax probability is low (a near-flat distribution), so
the template echo is a **modal** effect, not a confident one. This is the same
shape as Act 7's ICL result: **the format is acquired, the operation is
absent**, and the report says so explicitly.


### 2026-10-20 — Act 9 COMPLETE: interpretability (9a head ablation, 9b rollout, 9c logit lens)

Base: pinned canonical 6.3M (8 layers x 8 heads = **64 heads**, dim 256,
context 128). Held-out slice: 64 sequences of 128 tokens from the C4 shard at
**document offset 300,000**, far beyond anything the training sweep touched.
The intact baseline is measured **on this slice in this run** (CE 2.8551,
perplexity 17.376), so every delta is internal to the experiment and is NOT
comparable to the published 2.5152 validation loss.

#### 9a — causal head ablation against a random-ablation null

Every head is ablated in turn (its attention output zeroed before it is mixed
back into the residual stream) and the held-out loss is re-measured. The null
comes from ablating **random** head sets, so every head gets a z-score.

| quantity | value |
|---|---|
| heads with z >= 2 | **1 of 64** (layer 0, head 5) |
| heads with z <= 0 | 59 of 64 |
| top head (L0 H5) | dCE **+0.832 nats**, dPPL **+22.5**, z = **+7.82**, z vs random null **+42.8** |
| next head (L1 H6) | dCE +0.076, z = +0.54 |
| random single-head null | mean +0.0099, s.d. 0.0192 |
| random 2-head null | +0.0714 |
| random 3-head null | +0.0842 |
| **top-4 combination** | dCE **+1.047** (bootstrap [0.0365, 0.6390]) |
| **random 4-head combination** | dCE **+0.071** |

The top-4 heads are **~15x** more damaging than four random heads, and the
effect is dominated by a single head: layer 0, head 5, whose attention entropy
is **0.379 nats** against **2.480 nats** for the 59 heads with z <= 0. The head
that matters is a **sharp, near-deterministic** head; the heads that do not
matter are diffuse. (Attribution does not spread evenly across 64 heads in a
4-layer-per-head 8-layer stack: one of them carries the model.)

#### 9b — behavioural attention rollout: a negative

Act 8 supplies a ground truth a heatmap does not need: the token holding the
number the passage asserts. Rollout attribution (Abnar & Zuidema, with the
recursion explicitly row-normalised so the attribution is a distribution and
can be compared with a uniform baseline) is scored by whether it identifies
that token.

| method | hit rate on the asserted-number token | Wilson 95% |
|---|---|---|
| attention rollout (all 8 layers) | **0.0000** (0/34) | [0.0000, 0.1015] |
| raw attention (last layer) | **0.0000** (0/34) | [0.0000, 0.1015] |
| uniform chance (1/128) | 0.0078 | — |

**Neither rollout nor raw attention ever identifies the answer token.** This is
the result that keeps the paper honest: attention attribution is not evidence
of grounding here, and a heatmap of these heads would have suggested the
opposite.

#### 9c — logit lens, and the mechanistic link to Acts 7b and 8d

The lens is the model's **own** linear readout (final norm plus the tied
unembedding) applied to each layer's residual at the answer position, so
"decodable" means decodable by a readout the model already owns — no probe is
smuggled in. The measurement is the median rank of the asserted number's token
in that distribution.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| median rank of the asserted number | 860 | 721 | 550 | 439 | 412 | **87** | 227 | 647 |

**The number becomes readable mid-stack and is then lost before the output.**
It is most decodable at layer 5 (median rank 87 of 6,487, the top 1.3%) and
the rank rises again at layers 6 and 7. The number is **never** the lens
argmax at any layer, on any of the 34 items.

**This is the mechanistic explanation for Act 8b and Act 7b in one
measurement.** The asserted number's information is present in the residual
stream and is destroyed before it reaches the output. So the model is not
failing to *see* the number — it is failing to *carry* it — which is exactly
why the answer-copying measurement put it at chance, why free generation
produced the right number 0/40 times, and why the tool call was a template
echo with the argument `"s"`.

#### Defect found and fixed on the way

**D18 — the held-out metric compared logits to the wrong position.** The first
version of `measure_ce` scored `CE(logits, x)` — each position against the
token **at** that position rather than the token at i+1 — and returned CE
**11.98** (perplexity 1.6e5) for a model whose real held-out loss is ~2.5.
The signature was absurd on its face: a 6.3M model cannot be 160,000 times
worse than uniform over a 6,487-token vocabulary (which is ln(6487) = 8.78
nats). It is the same class as D12. Fixed, and pinned by a test using a
constructed **next-token oracle** (must score ~0) and a **uniform model**
(must score exactly ln V) — the uniform test alone cannot catch the shift,
because it returns ln V either way.

#### Caveats carried forward

1. The held-out slice is a single offset in one C4 shard; the intact baseline
   is internal to this run and the deltas are internal to it.
2. Head ablation removes a head's **contribution**, not the head itself: the
   head still computes and its value vector still participates in the output
   projection. This is the standard causal intervention, but it is an
   approximation of "delete the head".
3. The rollout result is for ONE attribution method and one prompt family
   (number extraction from a passage). It is a statement about this method on
   this task, not about attention interpretability in general.
4. The logit lens ranks the number's **first token**; a multi-token number is
   only partially represented by that single token.

### 2026-10-20 — Act 4 PROVENANCE RESOLVED: the ladder mixed TWO architectures, and the centrepiece number is withdrawn

**The Act 4 ladder is not one model family, and the reported alpha = 0.445 does
not describe the canonical models.** Resolved from code, checkpoints and run
records, not from ledger prose.

#### Deliverable A -- what the ladder actually constructs

| source | answer |
|---|---|
| `src/scaling/ladder.py` | contains `ALiBiTransformer` (4 heads) and `build_model_for_rung` returns it -- **a dead code path** |
| surviving checkpoint `artifacts/scaling/checkpoints/1.5M/seed42/checkpoint.pt` | state dict is MiniGPT: `token_embedding`, `position_embedding`, `blocks.N.attention.q`; `position_embedding` is **(128, 160)** |
| the same checkpoint vs the canonical 1.5M | **identical key sets, identical shapes, identical totals (3,643,040)** |
| `architecture` block in each run's `metrics.json` | `model_class: MiniGPT`, `num_heads: 8`, `positional_encoding: learned_absolute`, dims 160/192/256, layers 5/6/8 |

**Answer: MiniGPT, 8 heads, learned absolute positional encoding.** Not both.
The ledger's Act-crossing table (ALiBi / 4 heads / 793,344) describes an
earlier version of the ladder and is **stale**.

The two families are separable by the FIRST logged validation loss, because the
other family has no learned position table:

| rung | seeds | first val loss | family |
|---|---|---|---|
| 0.8M | 42, 43, 44 | 8.7-8.8 | canonical |
| 1.5M | 42 | 8.80 | canonical |
| 1.5M | **43, 44** | **129.6, 134.7** | **other** |
| 2.6M | 42-47 | 8.7-8.9 | canonical |
| 4.1M | **42, 43** | **184.0, 181.7** | **other** |
| 6.3M | 42 | 8.81 | canonical |
| 9.0M | **42** | **223.8** | **other** |

11 canonical runs, 5 of the other family. The other family occupies **three
whole rungs** (1.5M partly, 4.1M entirely, 9.0M entirely).

#### Deliverable B -- the 3.2476 number

It is real but **misattributed**. `3.24734` is the **2.6M rung's seed 47 loss
at D = 11,878,400 tokens** (`artifacts/scaling/checkpoints/2.6M/seed47/metrics.json`),
i.e. a mid-schedule INTERIOR point, not the 53.08M-token endpoint. The
Act-crossing table compared an un-annealed interior loss against an annealed
final loss from a different model -- the very defect its own "interior points
are mid-schedule" caveat warns about. **3.2476 is deleted from the ledger.**

#### Deliverable C -- 793,344 vs 2,694,144

- **793,344** came from the ledger's Act-crossing table, describing the stale
  ALiBi-era ladder. No ladder rung has it.
- **2,694,144** is the real 2.6M rung's non-embedding count (MiniGPT, dim 192,
  6 layers), confirmed from the checkpoint and from every run's metrics.json.
- The difference (1,900,800) is not a truncation of one into the other, as
  suspected: they are two different models.
- **The N axis was contaminated too.** The 1.5M rung contributes at **two** N
  values: 1,567,200 for seed 42 (counting the 128x160 = 20,480 position table)
  and 1,546,720 for seeds 43/44 (not counting it). The user's arithmetic was
  exactly right: 1,567,200 - 1,546,720 = 20,480 = one position table.

#### Deliverable D -- the clean refit

`src/scaling/refit_clean.py` tags every ladder point with a config hash
(architecture, PE type, n_layer, n_head, d_model, vocab, ctx) from the run
records, re-exports `ladder_points.csv` with those tags, and re-fits with a
Levenberg-Marquardt (analytic Jacobian, log-parameterised, multi-start, hard
bounds). **The tracked point set is not recoverable** -- `stage1_results.json`
is gone -- so the fit is validated by a stronger test: the tracked fit's OWN
parameters are evaluated on every candidate set.

| set | n | R-squared with the TRACKED parameters |
|---|---|---|
| as-tracked (D>=5M, 12 runs) | 2776 | **+0.287** (the tracked fit claims 0.995035) |
| **other family only (D>=5M)** | 1428 | **+0.9957** |
| **canonical only (D>=5M)** | 2292 | **-1.312** |

**The tracked alpha = 0.445 is reproduced by the OTHER family alone and is
anti-correlated with the canonical models' own points.** The two tracked fits
describe two different model families: the secondary (2,776 points, the
centrepiece) is the contaminated one, while the primary's alpha = 0.193 lands
inside the canonical family's fold range [0.182, 0.208].

| fit | n | rungs | alpha | beta | E | R^2 | fold alpha |
|---|---|---|---|---|---|---|---|
| mixed, as tracked | 2776 | 6 | +0.0574 | +0.2225 | +0.2591 | **0.7087** | - |
| other family only | 1428 | 3 | +0.5111 | +0.4213 | +2.0831 | 0.9976 | 0.346-0.540 |
| **canonical only (clean)** | 2292 | 4 | **+0.190 (fold sd 0.012)** | +0.4637 | +0.7060 | **0.9953** | **0.182-0.208** |

**The mixture is visible as a collapse in fit quality**: one law on the mixed
set reaches only R^2 = 0.709, while either pure family fits at R^2 >= 0.995.

#### Consequences -- the centrepiece number is withdrawn

1. **alpha = 0.445 is not a property of the canonical models.** The clean
   fit gives **alpha = 0.190 +/- 0.012** (mean over leave-one-rung-out folds,
   4 folds, sd across folds), point estimate 0.098, beta = 0.464, E = 0.706,
   R^2 = 0.9953, LORO R^2 = 0.956-0.988 on three rungs and 0.508 on the 0.8M
   hold-out.
2. **The Chinchilla consistency claim does not survive.** The canonical fold
   range [0.182, 0.208] does **not** contain Chinchilla's 0.34. The previous
   report's "consistent with Chinchilla" rested on the contaminated fit.
3. **The clean fit is itself weakly identified**: 4 distinct N values, one
   rung with a single seed, and a point estimate (0.098) that differs from the
   fold mean (0.190) because alpha trades off against E and beta. The honest
   statement is that **twelve rungs do not identify alpha at this scale**, and
   the fold spread is the uncertainty to quote.
4. The ledger's uncertainty convention is unified on the fold-based estimate.
   The JSON's covariance errors (+/-0.0032) are uncredible at a condition
   number of 1.9e6, and the JSON's 12-point primary errors (+/-5.4) are
   uninformative; the fold-based spread is the headline, with the other two
   reported alongside.

#### Caveats

1. The tracked fit's exact 2,776-point set cannot be recovered, so the
   contamination is established by parameter reproduction (R^2 0.9957 on the
   other family) rather than by re-running the original fit.
2. The clean set has 4 distinct N values and the 1.5M rung contributes one
   seed. alpha is consequently weakly identified; the fold spread, not the
   point estimate, is the result.
3. The first-step-loss threshold that separates the families is a proxy. It
   agrees with the architecture block wherever the block exists, and with the
   surviving checkpoints' state dicts, but it is not a direct read of the
   absent checkpoints for 4.1M and 9.0M.

### 2026-10-20 — Act 9 CONTROLS: the 9c claim is REWORDED, 9a survives, 9b is anchored to chance

`src/eval/act9_controls.py`; tracked in `results/v2/act9/controls.json`.

#### 9c — the control decides it, and it goes AGAINST the strong claim

The first 9c measurement reported the median rank of the ASSERTED number and
found it most decodable at layer 5 (rank 87 of 6,487). But Act 8 already
established a generic context-NUMBER bias, so "a number is decodable" was never
the claim being made. The controls (40 items, median rank per layer):

| condition | L0 | L3 | **L5** | L7 |
|---|---|---|---|---|
| (i) asserted number, oracle | 860 | 444 | **82** | 678 |
| (ii) **counterfactual-EDITED number**, counterfactual | 890 | 480 | **82** | 667 |
| (iii) a number appearing in NO passage, oracle | 2289 | 1610 | **484** | 565 |
| (iv) asserted number, **non-numeric control** | 820 | 428 | **86** | 784 |
| (v) random token id | 783 | 801 | **735** | 2280 |

**The decision rule fires for the reword branch.** The EDITED number reaches
the same rank as the asserted number (82 vs 82), and the non-numeric control
reaches 86. A number that appears nowhere is much worse (484) and a random
token much worse still (735), so the lens is number-specific -- but it does
**not** distinguish WHICH number is asserted.

**The claim is therefore reworded, and the strong version is withdrawn:**
the layer-5 spike is evidence for the **Act 8 bag-of-numbers bias**, not for
assertion-reading. Context numbers are represented in the residual stream;
assertion-specific content is not. The mechanistic link to Acts 7b and 8b
survives in that reworded form -- the model holds a generic number
representation and destroys it before the output -- but "the model encodes
which number is asserted" is **not** supported and has been removed from the
report.

#### 9b — the negative is about the METHOD, and it is now anchored

0/34 hits against a chance rate of 0.0078 is exactly what chance predicts, so
the measurement was sharpened:

| quantity | value |
|---|---|
| rollout attribution mass on the answer token | **0.00759** |
| raw attention mass on the answer token | 0.00673 |
| uniform mass (1/128) | 0.00781 |
| rollout / raw argmax landing on the final position | 0 / 71 and 0 / 71 |

Rollout's mass on the answer token is **indistinguishable from uniform**, and
the attribution does not simply collapse onto the last token. Attention-based
attribution carries no signal about the answer here. Reported jointly with 9c:
**a linear readout of the residual stream beats attention-based attribution,
and even the readout only sees a generic number.**

#### 9a — the head claim survives three attacks

1. **Not an attention sink.** L0 H5's mean attention over positions: mass on
   position 0 is 0.0164 against a uniform 0.0078, and the top five positions
   hold 0.0164 / 0.0083 / 0.0082 / 0.0082 / 0.0082 -- nearly flat. Single
   position concentration 0.0164. It is mildly position-0 biased, not a sink.
2. **It degrades, it does not collapse.** Intact CE 2.8551 -> ablated CE
   3.6869, against a uniform-over-vocabulary CE of 8.7776. A sink would
   collapse the output; this costs 0.83 nats of competence.
3. **Redundancy, and the interesting part.** Pairwise ablation on the top five
   heads shows L0 H5 is not acting alone: ablating L0 H5 together with L0 H2
   costs only **+0.023** (L0 H2 alone is +0.023), a redundancy of 0.97, and
   L0 H5 with L0 H0 likewise +0.023. But L0 H5 with L1 H6 is additive (+0.959
   against a sum of +0.908). **The effect is carried by a group of layer-0
   heads; L0 H5 is the one whose removal is most visible in isolation.**
4. **Two held-out slices**, so the z-score has an error bar: L0 H5 scores
   z = +42.8 at C4 offset 300,000 and z = +37.1 at offset 320,000, with 3
   heads at z >= 2 on both. The head is the same head on both slices.

#### A correction to this act's own earlier reporting

The first write-up quoted a bootstrap interval of [0.0365, 0.6390] beside the
top-4 combination effect of +1.047. **That interval does not contain +1.047**:
it is a percentile bootstrap over the FOUR individual head deltas, whose mean
is +0.248, not over the combination. The report was corrected to state the
weaker supported claim and to flag the interval.

#### A defect found on the way

**D19 — the null was measured against a stale baseline.** `head_ablation` did
not record its own intact loss in the module-level slot the random-ablation
null reads, so any caller that did not set it got null deltas measured against
a stale baseline. The symptom was z-scores of **-88** instead of +42.8, which
is the kind of absurd number that a sanity check should catch. Fixed by
recording the baseline inside `head_ablation` itself.

### 2026-10-20 — Act 10 COMPLETE: the audit table, the failure taxonomy, ethics at the floor, and the reproducibility script

#### 10a — the measurement-audit table IS the tracked artefact

`report/claims.csv` (81 rows) is the audit table: every claim in the paper has a
row, and every row is traceable to a source file, with n, interval, status and
the qualifier that must travel with it. The paper's body carries a generated
20-row headline table drawn from it rather than duplicating 81 rows, and
`tools/check_tex.py` verifies the correspondence in BOTH directions (a claim
marker in the text with no ledger row, and a ledger row never referenced from
the text, are both errors).

#### 10b — the measurement-failure taxonomy

**The unifying principle, which is the strongest methodological claim in the
project: EVERY recorded measurement failure was caught by a sanity invariant
or a positive control, and NONE was caught by comparing two implementations.**

| failure | caught by | what it invalidated |
|---|---|---|
| D11 — the instruction's input text never reached the prompt | per-position accuracy: position 0 at 0.070 while positions 1-2 were at 0.93/0.90 | all of Act 7 |
| D12 — a second cross-entropy shift in the PEFT forwards | the final training loss column separating "did not fit" from "fit but scored zero" | 7c's three custom arms |
| D18 — the held-out metric compared logits to the token at the SAME position | the next-token oracle (must score ~0) and the uniform model (must score ln V) | the first Act 9 measurement |
| D19 — the random-ablation null was measured against a stale baseline | the z-score was -88 where it should have been +42.8 | Act 9a's error bars |
| 5c double-feed — speculative decoding re-fed the full sequence over a prefilled cache | an ABSOLUTE invariant: acceptance must be 100% at a zero draft-target gap. The cached/uncached equivalence tests both PASSED because both paths shared the bug | every 5c number before the fix |
| Act 7b truncation — a 328-token 4-shot prompt in a 128-token window | the positive control: the harness scored 1.0 on the oracle path | the first ICL measurement |
| Act 8b chance baseline — a flat "chance = 1/3" | the per-item candidate sets are 3/4/5/6 numbers | the Act 8 rates |
| Act 8c retriever — a query whose terms cannot come from the passage | the top-1 chunk was the SAME for all 40 items | the retrieved condition |
| Act 2 ALiBi no-op — the forward returns its input unchanged | pe-alibi and pe-none differ by 4.8e-08 nats | the "ALiBi is competitive" claim |
| Act 4 ladder mixing two architecture families | the tracked fit's OWN parameters score R^2 = 0.9957 on the other family and -1.31 on the canonical one | alpha = 0.445 and the Chinchilla comparison |
| Act 2 "20 runs" / Act 3 "5 tokenizers" mislabels | the tracked files hold 18 and 4 | two section headers |

The Act 8 **non-numeric control** is the model of a good control: same
document, length-matched, no number, and it scores the same 9/40 as the
asserting passage. It separates "has a passage" from "has a passage that
asserts a number" and the answer is that neither helps.

#### 10c — ethics metrics at the floor, with the instruments

`src/eval/act10_ethics.py`; tracked in `results/v2/act10/ethics.json`.

1. **Stereotype association.** Within-occupation preference for the
   feminine-associated attribute set over the masculine-associated one, over
   ten occupations with a documented association and ten negative-control
   occupations with none. **All three rates are 1.0** (associated first 10/10,
   associated second 10/10, negative control 10/10). The probe therefore
   detects a **global attribute preference, not an association**: the model
   prefers "caring/gentle/empathetic" after EVERY occupation. The negative
   control is what makes that reading possible — without it the 10/10 would
   have looked like a stereotype result. **The instrument is underpowered**:
   at n = 10 only a rate above **0.8099** is distinguishable from chance, so
   the null is as much a statement about the instrument's resolution as about
   the model, and is reported that way.
2. **Toxicity lexicon** (12 terms) over the model's own generations:
   **0/40**, Wilson [0, 0.0876], against a corpus base rate of 0.0. Floor.
3. **Calibration.** Next-token accuracy against confidence over 3,152
   predictions: accuracy **0.5295** at mean confidence **0.5255**, **expected
   calibration error 0.0233**. The bins track closely (confidence 0.979 ->
   accuracy 0.962; confidence 0.054 -> accuracy 0.074). **This is a positive
   result and it is worth stating: the model is a competent, well-calibrated
   next-token predictor and an incompetent instruction follower, grounder and
   tool user.** Every downstream failure in this project is a failure to USE
   the competence, not a failure to have it.
4. **The SFT comparison is NOT POSSIBLE and is reported as such.** The project
   tracks no fine-tuned checkpoint — Act 7's SFT runs were trained in memory
   and only their metrics were tracked — so there is no SFT model to compare
   against the base model on these instruments. That is a gap in the artefact
   set and it is recorded rather than worked around by retraining.

No significance is manufactured from a 6.3M model.

#### 10d — the reproducibility script

`python -m tools.reproduce` loads the TRACKED canonical checkpoints and
re-derives six headline numbers. It trains nothing. Each check prints the
value it derived and the value the report claims, and marks them MATCH or
MISMATCH; a mismatch exits non-zero.

| check | what it re-derives |
|---|---|
| act2 | the ALiBi/no-PE gap of 4.8e-08 nats |
| act4 | the clean canonical fit's alpha and R^2, re-fitted from the tracked ladder points by the same code path that produced them |
| act5a | the KV cache's output identity (max logit difference) |
| act7a | the zero-shot instruction floor, through the SAME teacher-forced path Act 7 used |
| act8 | the oracle and non-numeric choice rates, with the same containment guard the real harness applies |
| act9 | the head-ablation effect of the top head and its z-score against the random-ablation null |

**All six reproduce.** The script deliberately includes one Act 4 fit and one
Act 9 number, so the centrepiece and the mechanism are both covered.

#### Caveats

1. The stereotype probe is underpowered by construction (n = 10 per family);
   the detection threshold is reported with every rate.
2. The toxicity lexicon is 12 terms and one token is sampled per prompt; it
   measures a floor, not a comprehensive score.
3. The calibration curve is measured on C4 next-token prediction, which is the
   model's training distribution; it says nothing about calibration on a task.
4. The reproducibility script re-derives the numbers that are cheap to
   re-derive. It does not re-run the scaling sweep, the SFT runs or the
   speculative benchmark.

### 2026-10-20 — Act 4, part 2: the N-axis convention, the other family's diagnosis, and the profile likelihood

Three questions about the centrepiece, answered rather than flagged.

#### 1. The parameter-count convention, and the decision

**What the CSV's N column actually holds.** `ladder_points.csv`'s
`non_embedding_params` column is the RUN'S RECORDED non-embedding count, taken
from that run's own `metrics.json`. It is NOT the double-counted total.

**The double-counting is real, but it is in the checkpoint FILE.** The state
dict stores the tied `lm_head.weight` alongside `token_embedding.weight`, so
the file's parameter SUM for the 1.5M rung is **3,643,040** while the
`metrics.json` total is **2,605,120**; the difference is exactly
**6,487 x 160 = 1,037,920**. The N axis is therefore **not** inflated by the
tied embedding, and the user's hypothesis about the axis is answered: no.

**There are nevertheless THREE counting conventions in play**, and the ladder
mixes two of them:

| rung | MiniGPT (canonical, has a learned position table) | the other family (no position table) | LADDER design target |
|---|---|---|---|
| 0.8M | 809,728 | **793,344** | 786,432 |
| 1.5M | **1,567,200** | 1,546,720 | 1,536,000 |
| 2.6M | **2,694,144** | 2,669,568 | 2,654,208 |
| 4.1M | 4,264,288 | **4,235,616** | 4,128,768 |
| 6.3M | **6,351,360** | 6,318,592 | 6,291,456 |
| 9.0M | 9,029,088 | **8,992,224** | 8,957,952 |

The third column is the LADDER's *search target* -- a number the rung
definition is built around -- not a measurement of anything. The first two are
measurements, and they differ by exactly the position table (16,384 to 36,864
across the rungs).

**This also solves the 793,344.** It is the other family's **0.8M** rung count
(dim 128, 4 layers), filed in the ledger's act-crossing table as though it
described the 2.6M rung. And 694,144, which that table assigned to "Act 5
sweeps", is the recorded 0.8M run's own count.

**Every rung shares vocab 6,487** (`embedding_params / dim` = 6,487 for all
six), so a global correction would be constant if one were needed.

**The fit under both conventions, and the decision:**

| N convention | alpha | beta | E | R^2 | profile: compatible alpha | contains 0.34? |
|---|---|---|---|---|---|---|
| non-embedding (the CSV's column) | +0.098 | +0.464 | **0.706** | 0.9953 | [0.08, 0.26] | **no** |
| **deduplicated total** | **+0.209** | +0.474 | **1.348** | 0.9955 | **[0.08, 0.54]** | **yes** |

**The paper uses TOTAL parameters.** Two reasons, in order of weight.

> **CORRECTION (2026-10-20, part 4) -- THE GROUND STANDS, THE WORDING WAS
> WRONG.** Part 3 of this ledger withdrew the physical argument on the ground
> that "E = 0.706 is below what a next-token model can achieve", on the basis
> of the measured **unigram** entropy of 4.9946 nats. **That was the wrong
> comparison and the withdrawal was an over-correction.** E is the irreducible
> loss of an INFINITE model -- the CONDITIONAL entropy of the next token given
> full context -- not the marginal (unigram) entropy, so a corpus's unigram
> entropy is a loose upper bound on E and not a target for it. The physical
> ground is therefore **restated correctly and the conclusion (total N) is
> unchanged**, now resting on three grounds: the published convention, the
> numerical stability of the fit, and the physical plausibility of E measured
> against the CORRECT anchors -- the token-space compression ceiling of
> **3.5918 nats/token** and the canonical 6.3M's own achieved held-out loss of
> **2.5152 nats/token**, which E must lie below because E is the asymptote as
> N and D grow. E = 1.348 is 0.536x that achieved loss (a 1.87x improvement
> still to be earned); E = 0.706 is 0.281x it (3.56x still to be earned).
> See `results/v2/act4-scaling/conditional_entropy_bound.json` and part 4 of
> this ledger. The unigram entropy measurement was never wrong; only the
> comparison drawn from it was.

1. **It is the convention the published exponents use.** Kaplan and Chinchilla
   regress on TOTAL parameters. The ladder's N was non-embedding, so the
   exponents were never on the same axis and the comparison was unstatedly
   confounded. **This is now stated explicitly rather than left implied.**
2. **It is the stable one.** Under total N the point estimate (+0.209) and the
   fold-wise mean (+0.212) agree. Under non-embedding they do not: the point
   estimate is +0.098 while the folds give +0.208, +0.183, +0.189, +0.182 --
   an instability of the size the owner flagged, and a symptom of the same
   underdetermination.

**Consequence for the report: the "consistent with Chinchilla" claim is
RESTORED, in a form the data support.** Under the comparable convention the
canonical family's clean fit gives alpha = 0.209 with a compatible range of
**[0.08, 0.54] that CONTAINS Chinchilla's 0.34**. What is withdrawn is the
*point* claim ("our alpha is 0.445", and the earlier "our alpha agrees with
0.34"), not the compatibility statement.

> **CORRECTION (2026-10-20, part 3).** The sentence that stood here -- *the
> ladder is consistent with Chinchilla's exponent, and cannot pin it down* --
> is ALSO withdrawn. The total-N compatible range is **[0.08, 0.54]**, which
> contains 0.34, contains 0.50, and contains most of the plausible range, so
> "consistent" suggests that something outside the range would have been
> inconsistent when here nothing is. The honest sentence is now: *this ladder
> is COMPATIBLE WITH PUBLISHED EXPONENTS ACROSS THEIR ENTIRE RANGE, AND
> THEREFORE DOES NOT DISCRIMINATE BETWEEN THEM.* And the beta range
> **[0.36, 0.56] EXCLUDES** Chinchilla's 0.28, so no compatibility claim may
> be made for beta at all.

#### 2. The other family's 129.6-223.8 nats: an initialisation property, not a metric bug

D18 was a metric that returned 11.98 nats for a model whose real loss is
~2.5, and was caught because the number was absurd on its face. The other
family's first logged losses are 15-25x worse than uniform, which is absurd in
the same way, so it was given the same treatment. Three candidate causes, and
the measurement:

1. **Is the loss summed rather than averaged?** If it were summed over
   positions and batch it would be about 8.78 x 32 x 128 = 35,953 nats. The
   observed 129.6 is neither 8.78 nor 36,000, so **the metric is an average**.
2. **Is the metric misaligned (a D18-style target shift)?** The same code
   path computes the canonical family's step-1 loss as **8.8007**, against a
   uniform-over-the-6,487-token-vocabulary cross-entropy of ln(6487) = 8.78.
   A shift would corrupt both families equally.
   The metric is correct.
3. **Therefore it is the MODEL.** Building that model class from the repo's own
   code, with the repo's own seed, on the same data:

| rung | other family, built now | recorded | canonical, built now | recorded |
|---|---|---|---|---|
| 1.5M | **133.25** | 129.57 | 8.82 | 8.80 |
| 4.1M | **184.49** | 183.96 | 9.03 | - |
| 9.0M | **224.99** | 223.77 | 8.71 | - |

**The other family's step-1 loss is an initialisation-scale property of that
model class**, reproduced from the repo's own code to within 2%. It is **not a
metric bug and not divergence**: those runs train normally and finish at
comparable losses (3.578 at 1.5M against the canonical 3.057).

**This makes the contamination worse, not better.** Those five runs were
perfectly good models. They were invisible because their loss trajectory
started 15-25x above uniform and ended half a nat worse -- a plausible-looking
curve, at a plausible-looking N, with a plausible-looking R^2. Nothing in the
artefacts distinguished them from the canonical runs until the provenance was
read by hand.

**The central fact, stated in plain language for the limitations section:** the
tracked fit's parameters reproduce R^2 = 0.9957 on THAT family, score +0.287
on the mixed set, and -1.312 on the canonical models' own points. The
original "LORO R^2 >= 0.985 on every rung" headline was produced by the other
family, which occupied the ENTIRE TOP of the ladder (4.1M and 9.0M) and half
of the 1.5M rung. A reader who does not know that will read the refit as a
minor correction; it is not.

#### 3. The point estimate is retired in favour of a profile likelihood

**alpha = 0.190 +/- 0.012 from four folds was not a confidence statement.**
A fold s.d. computed from four folds under a model that is not identified is
false precision, and the point estimate (0.098 under the old axis) sat far
from the fold mean (0.185). Both problems are removed by profiling alpha.

**Profile likelihood over alpha** (alpha held fixed, the other four parameters
optimised, R^2 recorded at each step of a 0.02-wide grid from 0.02 to 1.50),
on the canonical family's 2,292 interior points:

| N convention | best R^2 | best alpha | compatible alpha range | contains 0.34? |
|---|---|---|---|---|
| non-embedding | 0.995389 | 0.08 | [0.08, 0.26] | no |
| **total** | 0.995479 | 0.20 | **[0.08, 0.54]** | **yes** |

The compatible range is the set of alphas whose fit comes within 0.001 in R^2
of the best fit. It is a statement about **what the data cannot distinguish**,
not an error bar, and it makes the finding -- that this ladder does not
identify alpha -- the headline rather than a caveat.

**Fold-wise estimates, individually** (not summarised by an s.d.):

| hold out | alpha (non-embedding N) | R^2 | alpha (total N) | R^2 |
|---|---|---|---|---|
| 0.8M | +0.2075 | 0.508 | +0.1178 | 0.779 |
| 1.5M | +0.1834 | 0.988 | +0.2091 | 0.989 |
| 2.6M | +0.1885 | 0.988 | +0.3484 | 0.990 |
| 6.3M | +0.1818 | 0.956 | +0.1719 | 0.980 |
| other family: 1.5M / 4.1M / 9.0M | +0.3457 / +0.3497 / +0.5400 | 0.955 / 0.993 / 0.997 | - | - |

#### 4. The rescue, and the recipe that had to be recovered first

The 4.1M and 9.0M rungs are ENTIRELY of the other family, so the clean fit
has only four rungs and four distinct N values. `src/scaling/rescue.py`
re-runs them with the canonical architecture (MiniGPT, 8 heads, learned
absolute positions, LayerNorm) on the **D = 20N** line -- the broken 9.0M rung
sat at D = 10N, which is precisely the confound, and the corpus holds 338.5M
tokens, so D = 20N at 9.0M (179.9M tokens) fits comfortably.

**Every logged point carries a config hash computed FROM THE LIVE MODEL** at
the moment of logging -- its classes, its module names, its tensor shapes --
not from a rung table. Retroactive tagging from a table is exactly what let a
mid-campaign architecture change hide in plain sight.

**The recipe had to be recovered before the rescue could be trusted.** The
artefacts record no learning rate, so the rescue uses the recipe in the
current training code (AdamW, lr 3e-4, 100 warmup steps, cosine decay to
0.1x, batch 32, context 128, gradient clipping at 1.0). A validation run
re-trains the 1.5M rung with that recipe and compares with the canonical
1.5M's recorded 3.0571. The trajectory matches: recorded step 1 = 8.8007
against 8.8176 derived, and recorded step 2,000 = 3.4838 against 3.4821
derived. Without that check the rescue would have been measuring a possibly
different recipe, and comparing it with the existing rungs would have been
meaningless.

**Pre-registered decision gate, fixed before the rescue runs:** if the
canonical family with 6+ rungs still leaves alpha varying widely across fit
variants, the RESULT IS THE NULL -- "this ladder does not identify alpha at
this scale" -- which is publishable and already half-written. A previous alpha
of 0.44 does not come back unless the data support it.

### 2026-10-20 — Act 4, part 3: E measured (and the argument withdrawn), the beta profile, the cause of the architecture change, and the N audit

#### 1. E, measured instead of argued -- and MY ARGUMENT WAS WRONG

The total-N convention was chosen partly on the argument that E must be the
irreducible loss of a constant predictor, so E ~ unigram entropy, and that
E = 0.706 is impossible while E = 1.348 is coherent. **That argument is
falsified by measurement, and it does not discriminate between the
conventions.**

`results/v2/act4-scaling/unigram_entropy.json`: the unigram entropy of the
corpus under the 6,487-token vocabulary is

| region | tokens | entropy (nats) |
|---|---|---|
| held-out (the evaluation region) | 1,000,128 | **4.9946** |
| training region (5M sample) | 5,000,000 | 5.0068 |
| full corpus | 338,513,094 | 5.0136 |

So **both** fitted E values lie BELOW the held-out unigram entropy: 1.348 is
0.27x it and 0.706 is 0.14x it. The 6,487-token vocabulary is small and
highly compressible (2.27 chars/token), so its unigram entropy is ~5 nats, and
Chinchilla's published E = 1.69 was never this corpus's entropy to begin with.

**Consequences, stated plainly:**

* The "physically possible E" ground for the total-N axis is **WITHDRAWN**.
  Neither convention yields a physically possible irreducible loss.
* E here is an **extrapolated asymptote from a four-rung ladder**, not a
  measured entropy, and the paper must say so rather than implying that E is
  the corpus's floor.
* The choice of the total-N axis therefore rests on the **other two** grounds,
  both of which survive: it is the convention the published exponents use, and
  it is the stable one (the point estimate and the fold-wise mean agree under
  total N, +0.209 against +0.212; under non-embedding they do not, +0.098
  against +0.190).

#### 2. The beta profile, and the asymmetry that is the diagnosis

The same procedure, with beta held fixed instead of alpha:

| axis | alpha: compatible range | width | beta: compatible range | width |
|---|---|---|---|---|
| **total N** | **[0.08, 0.54]** (contains 0.34) | 0.46 | **[0.36, 0.56]** (excludes 0.28) | 0.20 |
| non-embedding | [0.08, 0.26] (excludes 0.34) | 0.18 | [0.34, 0.48] (excludes 0.28) | 0.14 |

**beta is two to three times better constrained than alpha, and the asymmetry
is the diagnosis rather than a nuisance.** Each run contributes interior
checkpoints at **fixed N and rising D**, so within-run D variation constrains
beta tightly, while alpha depends on the **number and spread of distinct N
values** the ladder contains. The ladder has four (soon more), so alpha has
almost nothing to work with while beta has hundreds of points per run.

**This also constrains what the paper may claim.** The total-N alpha range
[0.08, 0.54] contains 0.34, contains 0.50, and contains most of the
plausible range, so the honest sentence is **not** "consistent with
Chinchilla" but: **this ladder is COMPATIBLE WITH PUBLISHED EXPONENTS ACROSS
THEIR ENTIRE RANGE, AND THEREFORE DOES NOT DISCRIMINATE BETWEEN THEM.** And
the beta range [0.36, 0.56] **excludes** Chinchilla's 0.28, so no consistency
claim may be made for beta at all. The earlier "consistent with Chinchilla"
wording is replaced with the compatibility-and-indiscriminability statement,
which is a stronger and more defensible claim because it is what a range of
that width actually supports.

#### 3. WHY the architecture changed mid-campaign: it was never recorded, and neither training script is in the repository

Git archaeology, and the answer is the third option:

* `src/scaling/` was **EMPTY before commit f1eaf77**. `src/scaling/ladder.py`
  is NEW in that commit, and it builds **ALiBiTransformer with 4 heads**.
* The `ASPECT_LADDER` shape table that commit introduces is
  **dim = 32 x layers** -- 128/4, 160/5, 192/6, 224/7, 256/8, 288/9 -- which
  matches the shapes of the **canonical MiniGPT** runs exactly.
* So the committed ladder script encodes the **second family's architecture**
  (ALiBi, no position table) with the **first family's shapes**, and the
  script that trained the canonical MiniGPT rungs **was never committed at
  all**.

The two families coexist in one ladder because **nothing recorded which
script produced which points**. The runs that carry an ``architecture`` block
are the ones re-run after f1eaf77 (or after 0932d19, which added the block);
the runs without one are the older ones. The ladder's shape table, the fit, the
ledger and the figures all treated the pooled points as one model family.

**This is a PROCESS failure and it is the taxonomy's largest entry.** It is not
a hypothesis about position encodings that did not pan out, and it is not a bug
fix -- it is a training pipeline that was never in the repository, so a
mid-campaign change of model class was invisible to every downstream artefact
including the ledger.

#### 4. The N audit, and the 0.8M rung: a THIRD, UNIDENTIFIABLE architecture

Every rung's N audited against three independent sources -- the CSV column, the
run's own recorded metric, and the architecture the run implies:

| rung | family | N (CSV) | N (run's metric) | N (architecture) | LADDER target | D/N |
|---|---|---|---|---|---|---|
| 0.8M (x3) | canonical* | 694,144 | 694,144 | **809,728 / 793,344 -- NEITHER** | 786,432 | 22.7 |
| 1.5M seed 42 | canonical | 1,567,200 | 1,567,200 | 1,567,200 | 1,536,000 | 19.6 |
| 1.5M seeds 43,44 | other | 1,546,720 | 1,546,720 | 1,546,720 | 1,536,000 | 19.9 |
| 2.6M (x6) | canonical | 2,694,144 | 2,694,144 | 2,694,144 | 2,654,208 | 19.7 |
| 4.1M (x2) | other | 4,235,616 | 4,235,616 | 4,235,616 | 4,128,768 | 19.5 |
| 6.3M | canonical | 6,351,360 | 6,351,360 | 6,351,360 | 6,291,456 | 19.8 |
| 9.0M | other | 8,992,224 | 8,992,224 | 8,992,224 | 8,957,952 | **10.0** |

Every row except the 0.8M rung carries ONE N, and it agrees across all three
sources. **The 0.8M rung does not.** Its recorded N of 694,144 matches **no**
MiniGPT or ALiBi shape at any (dim, layers, heads) in dims 16-256 and depths
1-10, while its embedding parameters (830,336 = 6,487 x 128) show the
vocabulary and dim are as expected. It is a **third, unidentifiable
architecture**, it has no architecture block, it ran on CPU, and it sits at
D/N = 22.7 rather than 20 -- and it is the rung with the **worst hold-out
fold** (R^2 = 0.508 non-embedding, 0.779 total). It is also the cheapest
rung to replace: about a minute on MPS.

**Four provenance stories for one rung**, which is the ledger failing to be a
ledger. The taxonomy entry is therefore "three N conventions in one table, none
of them flagged".

#### 5. The rescue plan changed, and why

**alpha's identifiability rests on the number and spread of distinct N values,
not on D.** The interior checkpoints give each run ~230 points at FIXED N and
RISING D, so the ladder already has substantial within-run D variation, and
the 20N line makes N and D collinear across rungs. Adding a point to a line
that already holds five N values buys little; widening the N range buys a
lot.

Revised plan, and the reasoning is recorded so the spending is auditable:

| arm | cost | what it buys |
|---|---|---|
| 4.1M @ D=20N | 92 min | +1 distinct N, the whole 4.1M rung replaced with a canonical one, and a matched-(N, D) comparison against the discarded 4.1M family (same N, same D, same data, different position table) |
| 0.8M @ D=20N | ~1 min | replaces the unidentifiable third architecture at the BOTTOM of the N range, where a power law has the most leverage, and fixes the worst hold-out fold |
| 0.3M @ D=20N | ~3 min | +1 distinct N at the very bottom: extends log N from ln(9.0/0.8) = 2.42 to ln(9.0/0.3) = 3.40, a **41% increase in N range for minutes of GPU** |
| 0.4M @ D=20N | ~4 min | a further small N, the same leverage argument |
| 9.0M @ D=10N | ~120 min | half the cost of 20N, a **novel D/N ratio** so the design is no longer a single line, and a DIRECT matched-(N, D) comparison against the discarded 9.0M family, which sat at exactly 9.0M / D=10N: **the contamination becomes an experiment** |
| 9.0M @ D=20N | ~240 min | a second D ratio at the top rung, constraining the within-run shape as well as the level; kept because it was already half trained |

**Pre-registered gate, in the spirit of the previous one but with a better
test: does the alpha profile NARROW materially from [0.08, 0.54]?** If it
narrows, the ladder identifies something and that is reported. If it does NOT
narrow after adding distinct N values AND breaking the D/N collinearity, then
the null is CREDIBLE -- and a credible null is a much stronger contribution
than the current one, because it will have been shown that the right
experiment was tried. "We had four rungs" is an excuse; "we added rungs,
widened the N range by 41%, broke the D/N collinearity, and alpha is still
unidentified" is a result. A previous alpha of 0.44 does not come back unless
the data support it.

**The recipe was validated before any of this.** Re-training the 1.5M rung
with the recovered recipe gives **3.0468** against the canonical **3.0571**
(diff 0.0103), with the trajectory matching at step 2,000 (3.4821 derived
against 3.4838 recorded). Without that check none of the rescued rungs would
be comparable to the existing ones.

### 2026-10-20 — Act 10c: the SFT checkpoint, and what instruction tuning does to calibration

The SFT-versus-base comparison was reported as impossible because Act 7 trained
its fine-tunes in memory and tracked only their metrics. That was a missing
save call, not an inherent gap, so it was re-run with the weights saved.

**`checkpoints/canonical/6.3M-sft/model.pt`** (32 MB, `model_state` only, in
exactly the format of the other three canonical checkpoints, so it loads with
`load_state_dict(strict=True)`). It is Act 7's full fine-tune of the canonical
6.3M: 512 instruction examples, 24 epochs, seed 42.

**The SFT model reproduces Act 7's seed-42 result** -- held-out exact match
**0.6016** (77/128, Wilson [0.5150, 0.6822]) against Act 7's recorded 0.5938,
a one-item difference from the identical recipe. The per-task breakdown is the
Act 7 story again: **all 18 addition tasks remain at exactly 0.0000** while
copy 0.842, first word 1.000, last word 0.944, count words 0.611, capitalise
0.632 and reverse 0.167. The instruction set was learned; the operation was
not.

**AND THE CALIBRATION HALF, WHICH IS THE INTERESTING PART.** Both models
evaluated on the same held-out C4 slice (offset 300,000, 32 sequences of 128
tokens), so the two calibration curves are directly comparable:

| model | next-token accuracy | mean confidence | **expected calibration error** | instruction exact match |
|---|---|---|---|---|
| base | 0.5295 | 0.5255 | **0.0233** | 0.0000 |
| **after instruction tuning** | **0.4227** | **0.5534** | **0.1306** | **0.6016** |

**Instruction tuning on 512 examples made the model a better instruction
follower and a measurably WORSE next-token predictor: accuracy down 10.7
points, confidence up 2.8 points, and the calibration error up 5.6-fold from
0.0233 to 0.1306.** The tuned model is now overconfident AND less accurate on
ordinary text.

**This is the sharpest statement of the project's thesis and it is now
measured rather than asserted.** The base model is a competent,
well-calibrated next-token predictor and an incompetent instruction follower.
Tuning on a few hundred examples does not add instruction-following to a
competent model -- it **retargets** a competent model, trading general
next-token competence and calibration for task competence, and the trade is
measurable in a 6.3M model. That is also the mechanism behind 7b and 8d: the
model is not a bad next-token predictor, and the failures there are failures
of a competence that has been retargeted rather than failures of competence.

Tracked in `results/v2/act10/ethics.json` under `sft_calibration`, and
`results/v2/act10/sft_checkpoint.json` for the fine-tune itself.

### 2026-10-20 — Act 4, part 4: the local exponent, the (E, alpha) ridge, and the corrected E argument

The profile's width was reported as "the data cannot distinguish these
exponents". Part 4 shows that reading was wrong: in a five-parameter fit a wide
profile in alpha can be a DEGENERACY with E, and the way to tell the two apart
is the ladder's LOCAL exponent.

**1. E, argued correctly this time.** E is the irreducible loss of an INFINITE
model -- the conditional entropy of the next token given full context -- not the
marginal (unigram) entropy. The unigram entropy of this corpus is 4.9946 nats
(held-out), 5.0068 (training-region sample) and 5.0136 (full corpus), which is
a loose CEILING on E and never a target for it. The correct anchors are
measured, in `results/v2/act4-scaling/conditional_entropy_bound.json`:

| anchor | nats/token | what it is |
|---|---|---|
| unigram (marginal) entropy | 4.9946 | a loose ceiling; the WRONG comparison |
| text-space compression (zlib-9) | 4.9474 | weak: gzip captures the order-0 character distribution and little context |
| **token-space compression (frequency-rank order)** | **3.5918** | a genuine ceiling: one byte per token, so compressed bytes ARE nats/token, and it is BELOW the marginal entropy, so the compressor is doing real context modelling |
| **canonical 6.3M's ACHIEVED held-out loss** | **2.5152** | the discriminating anchor: E is the asymptote as N and D grow, so E MUST be below the loss a real model reaches |
| fitted E, total-N axis | 1.3480 | 0.536x the achieved loss; a 1.87x improvement still to be earned |
| fitted E, non-embedding axis | 0.7060 | 0.281x the achieved loss; a 3.56x improvement still to be earned |

**The physical ground is restored and the conclusion is unchanged.** The total-N
axis now rests on three grounds: the published convention (Kaplan and Chinchilla
both use total parameters, which is also what makes our alpha comparable to
theirs), the numerical stability of the fit, and the physical plausibility of E
measured against the CORRECT anchors.

**2. The local exponent, on the real interior points.** The local exponent
between neighbouring rungs is

    alpha_loc(N) = -[dL/dlnN] / (L - E)

and if the ladder is a power law it is a CONSTANT equal to alpha. Two
estimators, because they answer different questions: the **ascending-D raw**
ladder, whose slope still carries the B/D^beta term and is therefore a weighted
average of alpha and beta; and the **D-term-removed** ladder, in which each
rung's own interior points are fitted with L_i(D) = c_i + b_i D^-beta so that
c_i is the rung's asymptote and the D-term is gone before the slope is taken.

D-term removed, canonical family, E = 1.348:

| interval | dL/dlnN | alpha_loc | at E = 0.706 |
|---|---|---|---|
| 0.8M -> 1.5M | -0.5968 | **0.3483** | 0.2534 |
| 1.5M -> 2.6M | -0.4865 | **0.3635** | 0.2457 |
| 2.6M -> 6.3M | -0.3621 | **0.3444** | 0.2139 |
| **spread** | | **0.019 (flat)** | 0.0395 (drifting) |

**With E = 1.348 the local exponent is nearly CONSTANT at ~0.35 across the
ladder; with E = 0.706 it drifts from 0.253 to 0.214.** The two candidates are
separated by the data, not by an argument about entropy.

**3. E selected BY the data.** Requiring the local exponent to be constant
selects E, because alpha_loc depends on E. Scanning E over a grid and
minimising the standard deviation of alpha_loc across the ladder (D-term
removed) gives:

| E | alpha_loc | sd |
|---|---|---|
| **1.380 (selected)** | **0.3609** | **0.0100** |
| 1.348 (the fitted value) | 0.3500 | 0.0102 |
| 0.706 (the non-embedding fit) | 0.2369 | 0.0210 |

The data select E ~ 1.38, essentially the fitted value, and the local exponent
is then alpha = 0.36 +/- 0.01 -- **2.1 s.d. from Kaplan's 0.34 and 14 s.d. from
Chinchilla's 0.50.** The ascending-D-raw estimator selects E = 0.640 with
alpha_loc = 0.18, which is a mix of alpha and beta and not alpha itself; both
are reported.

**4. The (alpha, E) joint profile: the width IS a ridge.** Fixing E on a grid
and re-optimising alpha traces a monotone ridge with **d(alpha)/dE = +0.5786**.
The unconditional alpha range of [0.1075, 0.4915] is therefore a degeneracy
between two parameters, not noise in the data. This is the mechanism the
profile was hiding, and it is why the E argument mattered after all: it is not a
cosmetic preference between conventions, it is the coordinate along which the
whole profile slides.

**5. Beta DOES discriminate, and the asymmetry is the finding.** Kaplan reports
alpha = 0.34, beta = 0.30; Chinchilla reports alpha = 0.50, beta = 0.50. The
beta profile's compatible range is **[0.36, 0.56]**: it **excludes Kaplan's
0.30 and contains Chinchilla's 0.50**. A percentile bootstrap over the 2,292
interior points (120 resamples, total-N axis) gives the point estimate
beta = 0.4742 with a 95% interval of **[0.4607, 0.4871]** -- but that
interval is a RESAMPLING-REPRODUCIBILITY interval, because the points are not
exchangeable (many share a run), and the profile is the honest statement of
what the data cannot distinguish.

**So the honest sentence is asymmetric, and the asymmetry IS the result: this
ladder is INFORMATIVE about beta and UNINFORMATIVE about alpha.** On beta it
excludes Kaplan's 0.30 and contains Chinchilla's 0.50. On alpha its profile
spans the entire range of published values and therefore does not discriminate
between them. The mechanism is the one the rescue was designed to fix: each run
contributes interior checkpoints at FIXED N and RISING D, so within-run D
variation pins beta, while alpha depends on the number and spread of distinct
N values -- which is what the rescue adds.

**6. Act 7's run-to-run spread, now visible because something was re-run.**
Full fine-tune at n = 512, three seeds: **[0.5938, 0.5234, 0.5781]**, mean
0.5651, s.d. 0.0370, range 0.0704. The CPU re-run of the same recipe gave
**0.6016**, which is **+0.0078 above the best seed (0.21 s.d.)** and therefore
just OUTSIDE the three-seed range. But the re-run is **seed 42 on CPU** against
**seed 42 on MPS**, so it is **not a fourth independent draw**: it is a
**device-level reproducibility band of about 0.008 on a fixed seed**, against a
0.0704 optimisation-order spread. Act 7's seeds vary optimisation ORDER only
(weights are pinned), so between-initialisation variance remains unmeasured, and
the report must say that the headline number carries an order-only spread of
0.0370 plus a device-level band of about 0.008.

**7. The SFT result, on BOTH slices.** The SFT-versus-base comparison was
first measured on generic C4 text only, which supports "instruction tuning
traded general next-token competence for instruction following". The second
slice changes the claim, and both are now in
`results/v2/act10/sft_calibration.json`:

| slice | model | accuracy | mean confidence | ECE (bootstrap 95%) |
|---|---|---|---|---|
| generic held-out C4 (4,096 tokens) | base | 0.5295 [0.5142, 0.5448] | 0.5255 | 0.0233 [0.0181, 0.0355] |
| | after SFT | 0.4227 [0.4076, 0.4380] | 0.5534 | **0.1306** [0.1188, 0.1419] |
| | | **accuracy difference -0.1068 [-0.1284, -0.0852]** | | **ECE difference +0.1073** |
| held-out instruction RESPONSES (1,023 tokens) | base | 0.3543 | 0.3969 | 0.1118 |
| | after SFT | 0.3534 | 0.6261 | **0.2772** |
| | | **accuracy difference -0.0009 [-0.0160, +0.0142] (n.s.)** | | **ECE difference +0.1654** |

**The correct reading is sharper than the first one.** On generic text the
tuned model loses 10.7 points of next-token accuracy (significant). On the
instruction responses themselves it loses **nothing** (-0.0009, the interval
straddles zero) -- so it did **not** get worse at the very distribution it was
tuned on. What it gained there is **confidence**: mean confidence on the
tuned distribution rose from 0.397 to 0.626 while the accuracy was flat, and
its calibration error on that distribution more than doubled (0.1118 ->
0.2772).

**So instruction tuning did not add next-token competence on its target at
all; it inflated the model's confidence on the target and traded away accuracy
off-target.** The instruction-following gain (0.0000 -> 0.6016) is bought with
confidence inflation plus a narrower distribution, not with better prediction.
That is also the mechanism behind Act 7b and Act 8d: the failures there are
failures of a RETARGETED competence, not of an absent one.

The slices are disjoint from the SFT training data by construction
(`make_splits` asserts it) and the check confirms the response token sets share
no content words. The instruction-response slice is small (1,023 tokens) and is
reported with its size.

**8. The 4.1M comparison is matched in N, not in (N, D).** The rescued 4.1M
rung ran at D = 85,285,760 and the discarded family at D = 82,575,360 -- 3.3%
apart, so it is a **matched-N comparison with D within 3.3%**, not a matched
(N, D) one. The sensitivity argument: with beta ~ 0.47, a 3.3% change in D
moves the D-term by about 1.5% of its magnitude, on the order of 0.006 nats,
which is negligible against the 0.37-nat gap. The 9.0M arm at D = 10N is the
MUCH tighter comparison: the discarded rung was at D = 89,919,488 and the
rescue runs at 90,290,880, **0.4% apart**, so it is the headline controlled
comparison and the 4.1M one is the approximate one. The exact D/N ratio per
rung is in the audit table: the canonical rungs run 19.6 to 20.0, the discarded
0.8M ran at 22.7 on its recorded N (19.4 on the MiniGPT convention), and the
discarded 9.0M ran at 10.0, so "the D = 20N line" is an approximation and the
table says so.

#### Provenance of the part-4 numbers

`results/v2/act4-scaling/local_exponent.json` (the local-exponent curves, the
joint profile, and the E selection), `beta_bootstrap.json`,
`conditional_entropy_bound.json`, and
`results/v2/act10/sft_calibration.json`. Generated by
`src/scaling/local_exponent.py` and `src/eval/act10_sft_calibration.py`.

**THESE ACT-4 NUMBERS WERE PROVISIONAL AND HAVE BEEN RECOMPUTED.** The note
that stood here said the ladder still contained the unidentified 0.8M rung and
that the 9.0M arm at D = 10N was running. Both are resolved: every rescue arm
finished, the ladder manifest marks the unidentifiable 0.8M runs as excluded,
and `src/scaling/refit_clean.py` was re-run on the final ladder. The paper's
Act 4 numbers now come from the pre-registered D >= 5M point set: 3,463 points,
8 rungs, alpha +0.589, beta +0.461, E +1.964, R2 0.9960, with an alpha profile
of [0.34, 0.82] and a beta profile of [0.38, 0.54].

The rescued arms, all complete: 0.3M (N = 308,224, val 4.1695), 0.4M
(N = 399,600, val 4.0563), 0.8M (N = 809,728, val 3.4157, against 3.4385
for the unidentified architecture it replaces), 4.1M (N = 4,264,288, val
2.6254, against 2.9957 for the other family at the same N and a D within
3.3%), 9.0M at D = 10N (N = 9,029,088, val 2.5436, against 2.8492) and 9.0M
at D = 20N (val 2.4267).

**Known and deliberately not reported:** a per-run-transient inclusion rule
(`converged`, i.e. every point at or below a run's final loss plus a margin)
gives a different fit (alpha 0.6488, beta 0.1985) whose beta profile would
*include* 0.28. That point set is not the pre-registered one, and its profile
likelihood is not trustworthy: the alpha curve jumps between basins (best R2
drops from 0.9923 to 0.9871 between alpha 0.56 and 0.58) and the joint fit's own
point estimate falls outside the range the profiler reports. Fixing the profiler
(multi-start at every grid point) is open work. The paper reports the
pre-registered set.
