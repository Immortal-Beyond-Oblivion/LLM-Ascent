# Session Tracker

> **Superseded by `AGENTS.md` at the repo root for orientation.** `AGENTS.md`
> holds the ten-act story, the compute policy, the concept coverage checklist and
> the v2 conventions. This file remains the running session log.

## Active Ticket

**`[STAGE-A]` v2 enablers** — MPS device support, experiment schema + runner,
seed/Wilson-CI harness, tracked `results/`, self-contained `report/` bundle.

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

v1 is complete and frozen. **Suite: 100 tests, all passing.**

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
