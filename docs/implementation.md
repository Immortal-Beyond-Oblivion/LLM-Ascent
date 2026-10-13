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
