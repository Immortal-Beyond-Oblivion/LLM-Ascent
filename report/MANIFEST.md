# MANIFEST — the audit trail for `report/final_report.tex`

Every figure and every table in the report, with the exact file it was made
from, the script that made it, the number of rows, and the sample size. A
reader can re-derive any number in the paper from this table without running
anything but the generators.

Conventions:

* **Source** is the tracked file the numbers come from. Everything under
  `results/` is committed evidence; `docs/implementation.md` is the project
  ledger, which is committed and is the source of truth for caveats and for
  statistics that were not written into a JSON file (this is stated per row).
* **Generator** is the committed script that reads only tracked results and
  writes the figure (PNG + vector PDF) or the table CSV.
* **n** is the sample size of the plotted or tabulated quantity, as it is
  reported in the paper.
* No figure is drawn by hand. No table is typed by hand: every table's
  contents are written by a generator into `report/tables/*.csv`.

Reproduce everything with:

```bash
.venv/bin/python -m src.eval.act0_figures
.venv/bin/python -m src.eval.act1_figures
.venv/bin/python -m src.eval.act2_figures
.venv/bin/python -m src.eval.act3_figures
.venv/bin/python -m src.eval.act4_figures
.venv/bin/python -m src.eval.act5_figures
.venv/bin/python -m src.eval.act6_figures
.venv/bin/python -m src.eval.act7_figures
.venv/bin/python -m src.eval.act8_figures
.venv/bin/python -m src.eval.act9_figures
.venv/bin/python -m src.eval.act9_controls_figures
.venv/bin/python -m src.eval.report_tables
.venv/bin/python tools/build_claims_csv.py
.venv/bin/python tools/build_act10_tables.py
```

**The report bundle must contain the two generated LaTeX fragments.**
`report/tables/act10_headline_claims.tex` and
`report/tables/act10_failure_taxonomy.tex` are read into
`report/final_report.tex` with `\input`, so they are part of the document and
must travel with it. `tools/check_tex.py` now asserts that every `\input`
target exists inside `report/`, so a missing fragment fails the gate before it
fails the owner's compile.

---

## 1. Figures

The report references **20** figures. `report/figures/` holds **24** figure
stems: the 20 referenced ones plus the 4 v1 artefacts listed in section 1.1.
The four exclusions are also recorded, with their reasons, in
`FIGURE_EXCLUSIONS` in `tools/check_tex.py`, which asserts in both directions
that every referenced figure exists with a PDF sibling and that every file on
disk is either referenced or on that list.

| # | Figure (PNG + PDF) | Source | Generator | Rows | n |
|---|---|---|---|---|---|
| 1 | `act0_ngram_order_smoothing` | `results/act0-ngram/summary.json` | `src/eval/act0_figures.py` | 12 configurations | 3 seeds, deterministic (s.d. 0.000) |
| 2 | `act1_rnn_lstm_gru` | `results/act1-rnn/summary.json` | `src/eval/act1_figures.py` | 3 models + 2 gradient traces | 3 seeds; 1 trace of 100 steps per model |
| 3 | `act2_ablations` | `results/act2-ablations/summary.json` | `src/eval/act2_figures.py` | 18 configurations | 1 seed |
| 4 | `act3_tokenizer_comparison` | `results/act3-tokenizers/summary.json` + the per-run `tokenizer_stats.json` | `src/eval/act3_figures.py` | 8 configurations (4 tokenizers x 2 controls) | 1 seed |
| 5 | `act4_scaling_ladder` | `results/v2/act4-scaling/ladder_points.csv` (4,120 points, family-tagged) + `results/v2/act4-scaling/clean_refit.json` | `src/eval/act4_figures.py` | 16 runs; 4,120 logged points, 3,720 interior (D >= 5M): 2,292 canonical + 1,428 other | 11 canonical runs + 5 runs of the other family |
| 6 | `act4_loss_vs_N_fixed_D` | `results/v2/act4-scaling/equal_D_table.csv` | `src/eval/act4_figures.py` | 6 budgets x 6 rungs (17 values) | 1 value per rung per budget |
| 7 | `act4_profile_likelihood` | `results/v2/act4-scaling/clean_refit.json` (alpha and beta profiles) + `results/v2/act4-scaling/unigram_entropy.json` | `src/eval/act4_figures.py` | 4 profile curves (75 grid points each: alpha and beta, both conventions) + 4 leave-one-rung-out folds per convention | 2,292 interior points; 4 folds per convention |
| 8 | `act5_kv_cache` | `results/v2/act5/kv_cache.json` | `src/eval/act5_figures.py` | 4 latency cells + 2 models x 4 contexts | 3 timing repetitions per cell |
| 9 | `act5_mqa_paired` | `results/v2/act5/mqa_quality_2.6m.json` | `src/eval/act5_figures.py` | 6 paired seeds | 6 paired seeds (n = 5 is primary) |
| 10 | `act5_speculative` | `results/v2/act5/speculative_1.7x_multiprompt.json` + `speculative_4.2x_multiprompt.json` | `src/eval/act5_figures.py` | 2 gaps x 5 values of K (10 cells) | 5 prompts per cell, best-of-3 timings |
| 11 | `act6_decoding_tradeoff` | `results/v2/act6/6.3M_decoding.json` | `src/eval/act6_figures.py` | 15 configurations | 120 generations per configuration |
| 12 | `act6_bestofn_selfconsistency` | `results/v2/act6/6.3M_decoding.json` | `src/eval/act6_figures.py` | 5 + 5 rows | 120 generations per row; 300 items x 3 seeds |
| 13 | `act7_icl_vs_sft` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/act7_figures.py` | 4 training-set sizes + 2 reference lines | 3 seeds (the n = 16 point is effectively n = 1) |
| 14 | `act7_peft_matched_budget` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/act7_figures.py` | 5 methods + 7 ranks | 3 seeds, 128 held-out items |
| 15 | `act7_icl_copy_vs_compute` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/act8_figures.py` | 2 measures x 18 items | 18 items (the model is deterministic) |
| 16 | `act8_hallucination_by_condition` | `results/v2/act8/hallucination.json` | `src/eval/act8_figures.py` | 5 conditions x 2 instruments | 40 items per condition |
| 17 | `act8_contrast_distribution` | `results/v2/act8/hallucination.json` | `src/eval/act8_figures.py` | 3 rate contrasts + 5 log-probability contrasts | 40 items per contrast |
| 18 | `act9_head_ablation` | `results/v2/act9/interpretability.json` | `src/eval/act9_figures.py` | 64 heads (8 layers x 8 heads) | 64 held-out sequences of 128 tokens; 40 random null sets |
| 19 | `act9_logit_lens` | `results/v2/act9/interpretability.json` | `src/eval/act9_figures.py` | 8 layers for the lens + 2 attribution methods for the rollout | 34 items with a ground-truth answer token |
| 20 | `act9_lens_controls` | `results/v2/act9/controls.json` | `src/eval/act9_controls_figures.py` | 5 conditions x 8 layers | 40 items per condition |

Every figure exists in both formats: `report/figures/<name>.png` (for
Overleaf) and `report/figures/<name>.pdf` (vector). `tools/check_tex.py`
asserts that the `.pdf` sibling of every referenced `.png` exists.

### 1.1 Figures present in `report/figures/` but not referenced by the report

These are the v1 artefacts. They are kept because the v1 PDF
(`report/LLM_Ascent_Project_Report v1.pdf`) references them, and because the
v1 comparison they illustrate is discussed in Act 3 as **superseded context**.
Each one is on the checker's exclusion list with the same reason.

| Figure | Source | Generator | Status |
|---|---|---|---|
| `loss_curves` | `results/v1/comparison.json` | `src/eval/figures.py` | v1; superseded by Act 3 |
| `perplexity_params` | `results/v1/comparison.json` | `src/eval/figures.py` | v1; superseded by Act 3 |
| `latency` | `results/v1/comparison.json` | `src/eval/figures.py` | v1; superseded by Act 5a |
| `attention` | `results/v1/figures.json` | `src/eval/figures.py` | v1; not used as a result in v2, and Act 9b shows why a heatmap must not be read as evidence of grounding |

---

## 2. Tables

| # | Table (label) | Source | Generator | Rows | n |
|---|---|---|---|---|---|
| T1 | `tab:canonical` | `results/v2/act5/mqa_quality_2.6m.json` + `docs/implementation.md` | `src/eval/report_tables.py` -> `act5_canonical_models.csv` | 3 rungs | 1 seed per rung |
| T2 | `tab:act0` | `results/act0-ngram/summary.json` | `src/eval/act0_figures.py` -> `act0_ngram_perplexity.csv` | 12 configurations | 3 seeds, deterministic |
| T3 | `tab:act1` | `results/act1-rnn/summary.json` | `src/eval/act1_figures.py` -> `act1_recurrent_baselines.csv` | 3 models + 2 gradient rows | 3 seeds |
| T4 | `tab:act2` | `results/act2-ablations/summary.json` | `src/eval/act2_figures.py` -> `act2_ablation_summary.csv` | 18 configurations | 1 seed |
| T5 | `tab:act3` | `results/act3-tokenizers/summary.json` + per-run `tokenizer_stats.json` | `src/eval/act3_figures.py` -> `act3_tokenizer_comparison.csv` | 8 configurations | 1 seed |
| T6 | `tab:act4-rungs` | `results/v2/act4-scaling/clean_refit.json` | `src/eval/report_tables.py` -> `act4_ladder_rungs.csv` | 7 rows (6 rungs; the 1.5M rung is two families) | 16 runs (11 canonical + 5 other) |
| T7 | `tab:act4-contam` | `results/v2/act4-scaling/clean_refit.json` + `fit_results.json` | `src/eval/report_tables.py` -> `act4_contamination.csv` | 3 candidate point sets | 2,776 / 1,428 / 2,292 points |
| T8 | `tab:act4-axis` | `results/v2/act4-scaling/clean_refit.json` + `unigram_entropy.json` | `src/eval/report_tables.py` -> `act4_fit_results.csv`, `act4_profile_likelihood.csv`, `act4_unigram_entropy.csv` | 11 rows, the two parameter conventions side by side | 2,292 interior points per convention |
| T9 | `tab:act4-fit` | `results/v2/act4-scaling/clean_refit.json` | `src/eval/report_tables.py` -> `act4_fit_results.csv` | 16 rows | 2,292 interior points / 4 LORO folds |
| T10 | `tab:act4-profile` | `results/v2/act4-scaling/clean_refit.json` | `src/eval/report_tables.py` -> `act4_profile_likelihood.csv` | 4 rows (alpha and beta, one per parameter convention) | 2,292 interior points; 75-point profile grid |
| T11 | `tab:act4-variants` | `results/v2/act4-scaling/clean_refit.json` + `fit_results.json` | `src/eval/report_tables.py` -> `act4_fit_results.csv` | 4 fit variants + the withdrawn one | 2,292 / 1,428 / 2,776 points; 4 / 3 / 6 folds |
| T12 | `tab:act4-equalD` | `results/v2/act4-scaling/equal_D_table.csv` | `src/eval/act4_figures.py` -> `act4_equal_D_table.csv` | 6 budgets x 6 rungs | 1 value per rung per budget |
| T13 | `tab:act4-entropy` | `results/v2/act4-scaling/unigram_entropy.json` | `src/eval/report_tables.py` -> `act4_unigram_entropy.csv` | 7 rows (3 measured regions + 2 fitted E values + 2 ratios) | 1,000,128 held-out tokens; 5,000,000 training-region tokens; 338,513,094 full-corpus tokens |
| T14 | `tab:act4-naudit` | `results/v2/act4-scaling/ladder_points.csv` + `clean_refit.json` + `docs/implementation.md` | `src/eval/report_tables.py` -> `act4_n_audit.csv` | 6 rungs, audited against 3 independent sources | 16 runs; the first two columns re-derived from the tracked artefacts, the other two quoted from the ledger |
| T15 | `tab:act4-rescue` | `docs/implementation.md` | `src/eval/report_tables.py` -> `act4_rescue_plan.csv` | 6 arms | n/a; the rescue is in flight and no rescue result is reported |
| T16 | `tab:act5a` | `results/v2/act5/kv_cache.json` | `src/eval/report_tables.py` -> `act5_kv_cache.csv` | 4 context lengths | 3 timing repetitions per cell |
| T17 | `tab:act5b` | `results/v2/act5/mqa_quality_2.6m.json` | `src/eval/report_tables.py` -> `act5_mqa.csv` | 6 paired seeds | 6 paired seeds (n = 5 primary) |
| T18 | `tab:act5c` | `results/v2/act5/speculative_1.7x_multiprompt.json` + `speculative_4.2x_multiprompt.json` | `src/eval/report_tables.py` -> `act5_speculative.csv` | 5 values of K x 2 gaps | 5 prompts per cell |
| T19 | `tab:act6-sweep` | `results/v2/act6/6.3M_decoding.json` | `src/eval/report_tables.py` -> `act6_decoding_sweep.csv` | 15 configurations | 120 generations per configuration |
| T20 | `tab:act6-bon` | `results/v2/act6/6.3M_decoding.json` | `src/eval/report_tables.py` -> `act6_bestofn.csv` | 5 values of n | 120 generations per row |
| T21 | `tab:act6-sc` | `results/v2/act6/6.3M_decoding.json` | `src/eval/report_tables.py` -> `act6_selfconsistency.csv` | 5 vote counts | 300 items x 3 seeds |
| T22 | `tab:act7b` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/report_tables.py` -> `act7_icl_control.csv` | 6 measurements | 18 items (deterministic) |
| T23 | `tab:act7c` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/report_tables.py` -> `act7_peft_matched_budget.csv` | 5 methods | 128 held-out items x 3 seeds |
| T24 | `tab:act7d` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/report_tables.py` -> `act7_lora_rank_sweep.csv` | 7 ranks | 3 seeds per rank |
| T25 | `tab:act8b` | `results/v2/act8/hallucination.json` | `src/eval/report_tables.py` -> `act8_choice_rates.csv` | 6 rows (5 conditions + the edited-number row) | 40 items per condition |
| T26 | `tab:act8-retrieval` | `results/v2/act8/hallucination.json` | `src/eval/report_tables.py` -> `act8_retrieval.csv` | 5 retrieval metrics | 40 items |
| T27 | `tab:act8c` | `results/v2/act8/hallucination.json` | `src/eval/report_tables.py` -> `act8_logprob_contrasts.csv` | 8 statistics | 40 items (2 rows at n = 5) |
| T28 | `tab:act8d` | `results/v2/act8/tool_use.json` | `src/eval/report_tables.py` -> `act8_tool_use.csv` | 4 rates | 40 greedy items / 120 sampled generations |
| T29 | `tab:act9a` | `results/v2/act9/interpretability.json` | `src/eval/report_tables.py` -> `act9_head_ablation.csv` | 16 statistics (64 per-head rows in the CSV) | 64 heads; 64 held-out sequences; 40 random null sets |
| T30 | `tab:act9a-controls` | `results/v2/act9/controls.json` | `src/eval/report_tables.py` -> `act9_9a_controls.csv` | 9 checks (sink, collapse, redundancy, two slices) | 64 held-out sequences; 2 slices; 5 heads x 10 pairs |
| T31 | `tab:act9b` | `results/v2/act9/interpretability.json` + `results/v2/act9/controls.json` | `src/eval/report_tables.py` -> `act9_rollout.csv` and `act9_9b_attribution.csv` | 3 quantities (2 methods + the chance reference) | 34 items with a ground-truth answer token; 71 items with a located answer token |
| T32 | `tab:act9c` | `results/v2/act9/controls.json` | `src/eval/report_tables.py` -> `act9_9c_controls.csv` | 5 conditions x 8 layers | 40 items per condition |
| T33 | `tab:act10-headline` | `report/claims.csv` | `tools/build_act10_tables.py` -> `act10_headline_claims.csv` and `act10_headline_claims.tex` | 27 headline claims, selected from the 98-row ledger | per row |
| T34 | `tab:act10-taxonomy` | `docs/implementation.md` + the tracked result files named per row | `tools/build_act10_tables.py` -> `act10_failure_taxonomy.csv` and `act10_failure_taxonomy.tex` | 15 recorded measurement failures | n/a |
| T35 | `tab:reproduce` | `tools/reproduce.py` (re-derives from the tracked canonical checkpoints) | `tools/reproduce.py` | 10 checks | 10 checks, all MATCH |
| T36 | `tab:act10c-sft` | `results/v2/act10/ethics.json` (the `sft_calibration` block) + `results/v2/act10/sft_checkpoint.json` | `src/eval/act10_ethics.py` -> `results/v2/act10/ethics.json`; `src/eval/act10_sft_checkpoint.py` -> `results/v2/act10/sft_checkpoint.json` | 5 rows (base and tuned, accuracy / mean confidence / ECE) | 4,064 next-token predictions per model; 1 fine-tune, so n = 1 for the tuned model |
| T37 | `tab:dnm` | `results/v2/act7/6.3M_sft_peft.json` (caveat fields) + `results/v2/act8/*.json` + `results/v2/act9/interpretability.json` + `results/v2/act9/controls.json` + `results/v2/act4-scaling/clean_refit.json` + `results/v2/act10/ethics.json` | `src/eval/report_tables.py` -> `do_not_assert_claims.csv` and `act7_act8_caveats.csv` | 34 rows | n/a |

### Supporting CSVs that do not back a numbered table

These are written by the same generators and back figures or prose rather than
a table, and are listed so that the directory is complete.

| CSV | Backs | Source | Generator |
|---|---|---|---|
| `act7_sft_icl.csv` | the 7a/7b SFT curve in Figure `act7_icl_vs_sft` | `results/v2/act7/6.3M_sft_peft.json` | `src/eval/report_tables.py` |
| `act5_kv_cache_memory.csv` | the right panel of Figure `act5_kv_cache` | `results/v2/act5/kv_cache.json` | `src/eval/report_tables.py` |
| `act5_mqa_context_only.csv` | the v1-path G curve quoted as context only | `results/v2/act5/attention_variants.json` | `src/eval/report_tables.py` |
| `act7_act8_caveats.csv` | the caveat strings behind the do-not-assert table | `results/v2/act7/6.3M_sft_peft.json` + `results/v2/act8/*.json` | `src/eval/report_tables.py` |
| `act9_head_ablation.csv` | the per-head rows behind Table `tab:act9a` and Figure `act9_head_ablation` | `results/v2/act9/interpretability.json` | `src/eval/report_tables.py` |
| `act9_9c_controls.csv` | the five lens-control conditions behind Table `tab:act9c` and Figure `act9_lens_controls` | `results/v2/act9/controls.json` | `src/eval/report_tables.py` |
| `act9_9a_controls.csv` | the sink, redundancy and two-slice checks behind Table `tab:act9a-controls` | `results/v2/act9/controls.json` | `src/eval/report_tables.py` |
| `act9_9b_attribution.csv` | the attribution-mass and argmax rows behind Table `tab:act9b` | `results/v2/act9/controls.json` | `src/eval/report_tables.py` |
| `act9_logit_lens.csv` | the 34-item first lens run quoted in Section 9c and in Figure `act9_logit_lens` | `results/v2/act9/interpretability.json` | `src/eval/report_tables.py` |
| `act4_profile_likelihood.csv` | the profile-likelihood rows (alpha and beta, both conventions) behind Table `tab:act4-profile` and Figure `act4_profile_likelihood` | `results/v2/act4-scaling/clean_refit.json` | `src/eval/report_tables.py` |
| `act4_unigram_entropy.csv` | the measured unigram entropy behind Table `tab:act4-entropy` and the withdrawn ``physically possible E'' argument | `results/v2/act4-scaling/unigram_entropy.json` | `src/eval/report_tables.py` |
| `act4_n_audit.csv` | the per-rung N audit behind Table `tab:act4-naudit` | `results/v2/act4-scaling/ladder_points.csv` + `clean_refit.json` + `docs/implementation.md` | `src/eval/report_tables.py` |
| `act4_rescue_plan.csv` | the revised rescue plan behind Table `tab:act4-rescue` | `docs/implementation.md` | `src/eval/report_tables.py` |
| `act10_headline_claims.csv` | the CSV twin of the generated Table `tab:act10-headline` | `report/claims.csv` | `tools/build_act10_tables.py` |
| `act10_failure_taxonomy.csv` | the CSV twin of the generated Table `tab:act10-taxonomy`, with the per-row source | `docs/implementation.md` | `tools/build_act10_tables.py` |

---

## 3. The claims ledger

`report/claims.csv` is the machine-readable ledger: one row per quantitative
or qualitative claim in the paper, with its measured `n`, its `interval`, its
`status`, the exact `source_json` the number came from, and the `caveat` that
must travel with the claim. 99 rows.

`status` is one of `confirmed`, `n.s.`, `exploratory`, `ceiling`, `invalid`,
`superseded`, `open-item`.

The correspondence between the ledger and the prose is machine-checked: every
claim in the `.tex` carries an invisible `% claims: <id>` marker, and
`tools/check_tex.py` fails if a marker has no ledger row, if a ledger row has
no marker, if a row's `source_json` does not exist, or if a row's `status` is
not in the allowed set.

Regenerate the ledger with `.venv/bin/python tools/build_claims_csv.py`, and
the two generated tables of Section 10a and 10b with
`.venv/bin/python tools/build_act10_tables.py`. The 99-row ledger is **not**
reproduced in the body of the paper: the body carries the 27-row headline
summary (Table `tab:act10-headline`), which is generated from the ledger and
read in with `\input`, so the two cannot disagree.

---

## 4. Provenance notes and known audit gaps

These are the places where a reader should know that the audit trail is
incomplete. They are listed here rather than hidden.

| # | Note |
|---|---|
| P1 | **The Act 4 ladder's raw sweep output lived under the git-ignored `artifacts/` tree.** It has been exported into the tracked results directory as `results/v2/act4-scaling/ladder_points.csv` (4,120 points, 16 runs, now tagged with `family`, `config_hash`, `model_class` and `positional_encoding`) and `ladder_rungs.csv` (6 rungs), and the corpus/tokenizer manifest as `corpus_manifest.json`. |
| P2 | **The Act 4 fit's uncertainty brackets, the leave-one-rung-out protocol, the cross-fold exponent ranges, the residual RMS and the noise floor are recorded in `docs/implementation.md`, not in `fit_results.json`.** Every claim sourced that way says so in its `caveat` column. |
| P3 | **The Act 4 `primary` fit in `fit_results.json` (12 annealed endpoint points) is degenerate** --- it drives E to its lower bound and returns alpha = 0.193, beta = 0.169 with errors of +/-5.4 and +/-4.7. The `secondary` fit in that file (2,776 interior points, alpha = 0.445) is the one that has been **withdrawn**: it describes the other architecture family. The paper's fit is now the family-separated one in `clean_refit.json`. |
| P4 | **The Act 6 MNLI diagnostics (argmax accuracy 0.3300 = 99/300, the per-class recalls, the 0.4583 figure on the 216 answerable items, the mean max-probability and the mean entropy) are recorded in `docs/implementation.md`, not in the tracked decoding JSON.** The tracked JSON holds the vote-count table only. |
| P5 | **Act 8's 8a numbers (the measured cost of the character-based context budget) and the 40/40 verification that the tracked partner number is absent from the passage are recorded in `docs/implementation.md`.** The behavioural results are in the tracked Act 8 JSON. |
| P6 | **The 34-row "claims we do not make" list was derived by the report author from the caveat fields carried in the tracked Act 7, Act 8 and Act 9 result files, and extended in this revision with the Act 4 rows that the withdrawn `$E$` argument, the beta profile, the $N$ audit and the architecture cause force.** `report/tables/do_not_assert_claims.csv` records the caveat strings each line comes from, so the owner can substitute the final list without re-deriving it. |
| P7 | **Act 9 now has tracked results (`results/v2/act9/interpretability.json`) and a tracked control set (`results/v2/act9/controls.json`).** The control set decided the Act 9 claims: the logit lens is number-specific and not assertion-specific, so the strong version of the 9c claim is withdrawn, and the head effect is carried by a group of layer-0 heads rather than by one. Act 10's 10a and 10b are written from tracked artefacts (`report/claims.csv` and `docs/implementation.md`), and 10c (ethics metrics) and 10d (the reproducibility script) are now measured from `results/v2/act10/ethics.json` and `tools/reproduce.py`, so no act in the paper is a placeholder. |
| P8 | **The Act 4 equal-D table's provenance is not recorded and it does not match the current ladder for three of the six rungs.** At $D = 10$M the table's 0.8M, 2.6M and 6.3M columns are 0.43--0.68 nats worse than the ladder's logged points at $D = 10{,}035{,}200$, while the 1.5M, 4.1M and 9.0M columns match to within 0.013 nats. The table is kept because it is tracked, and the inversion it contains is reported as a null, but the paper states the caveat where the table appears. This was found while revising the report and is recorded in the ledger at claim `A4-10`; it is not yet recorded in `docs/implementation.md`. |
| P9 | **The Act 4 tracked fit's exact 2,776-point set cannot be recovered**, so the contamination is established by evaluating the tracked parameters on each candidate point set (`clean_refit.json`, `tracked_params_on_candidate_sets`) rather than by re-running the original fit. The reconstruction reproduces the point COUNT, not the point SET. |
| P10 | **The Act 9 control set is a second run over a different item set from the first.** `results/v2/act9/controls.json` measures the logit-lens controls over 40 items, while `results/v2/act9/interpretability.json` measured the asserted number over the 34 items with a located answer token. The layer-5 median rank of the asserted number is therefore 82 in the control run and 87 in the first run. The paper reports both and names the item sets; the difference is the item set, not the measurement. |
| P11 | **The Act 4 ladder's $N$ axis was a non-embedding count while the published exponents use total parameters.** The convention is now stated in the paper (Section `sec:act4-axis`) and the fit is reported under both conventions; the fit the paper reports is `canonical_only_D5M_total_N`, the canonical family's interior points on the total-parameter axis. The clean fit's source JSON holds the profile likelihood over $\alpha$ **and over $\beta$** under both conventions, and the fold-wise estimates are quoted individually: a fold standard deviation is not reported anywhere. **The physical-plausibility ground that was originally given for this choice is withdrawn** (see P14). |
| P12 | **The Act 4 rescue is IN FLIGHT and unfinished, under a revised plan and a revised gate.** `src/scaling/rescue.py` re-runs rungs with the canonical architecture, with a config hash computed from the live model at logging time. The revised plan (quoted in the paper and in `report/tables/act4_rescue_plan.csv`) adds a 0.3M and a 0.4M rung, replaces the 0.8M rung, re-runs 4.1M at $D = 20N$, and adds a 9.0M arm at $D = 10N$ as a novel D/N ratio and a matched-$(N, D)$ comparison against the discarded 9.0M family. The pre-registered gate is now: **does the $\alpha$ profile narrow materially from $[0.08, 0.54]$?** The recipe was validated first by re-training the 1.5M rung (3.0468 derived against 3.0571 canonical; step 2,000 loss 3.4821 derived against 3.4838 recorded). **No rescue result is reported in the paper**; the Act 4 numbers are the current four-rung clean fit, and the section will be regenerated from the rescued ladder. |
| P13 | **The other family's 129.57--223.77 nat step-1 losses are an initialisation property, not a metric bug.** The reproduction (133.25 / 184.49 / 224.99 against the recorded 129.57 / 183.96 / 223.77) and the reasoning are recorded in `docs/implementation.md`, not in a tracked result JSON; the ledger row `A4-14` cites the ledger as its source. |
| P14 | **The ``physically possible $E$'' argument for the total-parameter axis is WITHDRAWN.** It was falsified by measurement: `results/v2/act4-scaling/unigram_entropy.json` measures the corpus's unigram entropy under the 6,487-token vocabulary at 4.9946 nats on the held-out region (5.0068 on a 5M-token training-region sample, 5.0136 on the full 338,513,094-token corpus), and **both** fitted $E$ values lie below it (1.348 is 0.27x it, 0.706 is 0.14x it). Neither convention therefore yields a physically possible irreducible loss, so the ground does not discriminate between them. The paper states the two grounds that survive (the published convention, and the stability of the fit) and says plainly that $E$ is an extrapolated asymptote from a four-rung ladder, not a measured entropy. Ledger row `A4-17`. |
| P15 | **The cause of the Act 4 architecture change is a repository fact, not a tracked measurement.** `src/scaling/` was empty before commit `f1eaf77`; the ladder script committed in that commit builds `ALiBiTransformer` with 4 heads, and its `ASPECT_LADDER` shape table (`dim = 32 x layers`) matches the canonical MiniGPT runs' shapes exactly. The script that trained the canonical MiniGPT rungs was never committed. The finding is recorded in `docs/implementation.md` (section "Act 4, part 3", item 3) and cited as such by ledger rows `A4-19` and `A10-03`; it is verifiable with `git ls-tree f1eaf77^ src/scaling` and `git show f1eaf77:src/scaling/ladder.py`. |
| P16 | **The per-rung $N$ audit's architecture-implied and design-target columns are quoted from the project ledger, not from a result JSON.** The first two columns of `report/tables/act4_n_audit.csv` are re-derived from the tracked ladder points and the tracked run records; the third and fourth come from `docs/implementation.md` (section "Act 4, part 3", item 4), which is where the audit was recorded. Ledger rows `A4-20` and `A4-21`. |
| P17 | **The Act 10c SFT checkpoint now exists, AND the SFT-versus-base calibration comparison has since been MEASURED.** `src/eval/act10_sft_checkpoint.py` re-runs Act 7's full fine-tune (512 examples, 24 epochs, seed 42, CPU) with the weights saved; `results/v2/act10/sft_checkpoint.json` records its held-out exact-match accuracy of 0.6016 (77 of 128, Wilson 95% [0.5150, 0.6822]) against the base model's 0.0000 zero-shot floor, and `checkpoints/canonical/6.3M-sft/model.pt` holds the weights in the other canonical checkpoints' format. The calibration comparison has since been measured on the IDENTICAL held-out slice for both models and is in `results/v2/act10/ethics.json` under `sft_calibration`: the base model's expected calibration error is 0.0233 and the tuned model's is 0.1306, with next-token accuracy falling from 0.5295 to 0.4227 while mean confidence rises from 0.5255 to 0.5534. Instruction tuning made the model a better instruction follower and a measurably worse next-token predictor. Ledger row `A10C-04`; the instruction-accuracy result is `A10C-05`; a tenth reproducibility check (`act10sft`) re-derives the ECE pair. |

---

## 5. Verification

`tools/check_tex.py` is the correctness gate, because there is no `pdflatex` on
the machine that produces this report and the owner compiles on Overleaf. It
checks:

1. every `\includegraphics` target exists under `report/figures/` and has a
   vector PDF sibling;
2. every file in `report/figures/` is either referenced by the report or
   listed in `FIGURE_EXCLUSIONS` with a reason --- the check runs in both
   directions;
3. brace balance and environment balance;
4. every `\label` is unique, every `\ref`/`\eqref`/`\pageref` resolves, and
   every label is attached to a figure, a table, a longtable or a section;
5. every figure and table environment has a caption, and every figure
   environment contains an `\includegraphics`;
6. no undefined macro: every macro the file defines is defined before it is
   used, and no command outside the known-command list is used;
7. every package a used command needs is in the preamble;
8. no absolute path, no `\input`/`\include` outside `report/`, and every
   `\input` target present inside `report/` (so a generated fragment cannot go
   missing between the gate and the owner's compile);
9. the claims ledger: header, unique ids, allowed statuses, non-empty `n` and
   `interval`, an existing `source_json` for every row, and a two-way
   correspondence between the ledger and the `% claims:` markers in the text;
10. **inline math is balanced**: no math span crosses a blank line, no
   math-only command lands in text mode, and the file does not end in math
   mode. An unbalanced `$` is invisible without `pdflatex` and is not
   cosmetic when it happens: LaTeX reports `Missing $ inserted` and every
   later `$` in the document then toggles the wrong way. **This check caught
   a real defect in the previous revision of this report** --- the Act 10c
   calibration paragraph had one `$` too few, which inverted the math mode for
   the rest of the document. It is fixed, and the check is now permanent.

Run it with:

```bash
.venv/bin/python tools/check_tex.py
```

The checker also carries a **negative self-test**, which copies the report
into a scratch directory, breaks it eleven different ways (a missing figure, a
figure without its PDF sibling, an unreferenced figure on disk, an unbalanced
brace, a duplicate label, a dangling reference, an absolute path, a claim
marker with no ledger row, a ledger row nobody references, a missing
generated fragment, and an unbalanced inline-math span) and asserts that the
checker catches each one:

```bash
.venv/bin/python tools/check_tex.py --self-test
```
