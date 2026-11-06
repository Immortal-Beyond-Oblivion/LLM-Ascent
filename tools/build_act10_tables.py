#!/usr/bin/env python3
"""Generate the two Act 10 tables from the tracked artefacts.

  act10_headline_claims      10a.  The headline claims of the paper, taken
                             from report/claims.csv.  The CSV is the ledger;
                             this is the summary of it that the body carries.
  act10_failure_taxonomy     10b.  The measurement-failure taxonomy, built
                             from the defect records in docs/implementation.md.

Both tables are written twice: once as a CSV (the audit trail, the same
numbers a reader can diff against the tracked JSON) and once as a LaTeX
fragment that the report \\input's, so that the table in the paper cannot
drift from the ledger.

Usage:
    python tools/build_act10_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "report" / "claims.csv"
TABLES = REPO_ROOT / "report" / "tables"

# The headline claims, in the order the acts appear.  This is the short
# list the body carries; the full ledger is report/claims.csv.
HEADLINE_IDS = [
    "A0-01",
    "A1-01",
    "A2-02",
    "A3-01",
    "A4-01",
    "A4-06",
    "A4-08",
    "A4-09",
    "A4-12",
    "A4-17",
    "A4-18",
    "A4-20",
    "A5A-02",
    "A5B-03",
    "A5C-01",
    "A6-01",
    "A6-04",
    "A6-05",
    "A7-01",
    "A7-02",
    "A7-08",
    "A8B-01",
    "A8B-06",
    "A8D-02",
    "A9-02",
    "A9-09",
    "A10C-03",
]

# ── the taxonomy ────────────────────────────────────────────────────────────
#
# Every entry: what the defect was, how it was caught, and what it
# invalidated.  The catch column is the point of the table: every one of
# these was caught by an absolute check, and none by comparing two
# implementations.
TAXONOMY = [
    {
        "defect": "The architecture changed mid-campaign and nothing recorded it (Act 4)",
        "source": "docs/implementation.md",
        "what": "THE LARGEST ENTRY, and the one whose lesson generalises: "
                "A GOOD-LOOKING FIT IS NOT EVIDENCE OF A GOOD FIT. The cause "
                "of the ladder's architecture change is now known, and it is a "
                "PROCESS failure, not a hypothesis about position encodings "
                "that did not pan out and not a bug fix. src/scaling/ was "
                "EMPTY before commit f1eaf77. The ladder script committed in "
                "that commit builds ALiBiTransformer with 4 heads -- the "
                "SECOND family's architecture -- while its ASPECT_LADDER "
                "shape table is dim = 32 x layers (128/4, 160/5, 192/6, 224/7, "
                "256/8, 288/9), which matches the canonical MiniGPT runs' "
                "shapes EXACTLY. The script that trained the canonical MiniGPT "
                "rungs was NEVER COMMITTED AT ALL. A training pipeline that "
                "was never in the repository made a mid-campaign change of "
                "model class invisible to every downstream artefact, including "
                "the ledger.",
        "caught": "A PROVENANCE READ of the repository itself, on top of the "
                  "provenance read of the run records that separated the two "
                  "families. The two families coexist in one ladder because "
                  "NOTHING recorded which script produced which points: the "
                  "runs that carry an 'architecture' block are the ones "
                  "re-run after f1eaf77 (or after 0932d19, which added the "
                  "block), and the runs without one are the older ones. The "
                  "ladder's shape table, the fit, the ledger and the figures "
                  "all treated the pooled points as one model family. The "
                  "repair is a CONFIG HASH computed from the LIVE model at "
                  "the moment a point is logged, not retroactively from a "
                  "rung table, because retroactive tagging is exactly what "
                  "let the change hide in plain sight.",
        "invalidated": "The ladder's homogeneity, which was an assumption and "
                       "never a measurement, and with it every artefact that "
                       "pooled the two families: the fit, the figures and the "
                       "ledger's own record. It is the reason the paper's "
                       "generalised lesson is a procedure rather than an "
                       "admonition: you need the points to be one model "
                       "family, and the only way to know that is to tag every "
                       "point with its configuration and check.",
    },
    {
        "defect": "D11 (Act 7)",
        "source": "docs/implementation.md",
        "what": "The input text never reached the prompt. The task builder "
                "passed the payload positionally, binding it to 'response' and "
                "leaving 'input_text' empty, so six of seven tasks were "
                "unanswerable by construction while still being scored.",
        "caught": "A per-position accuracy profile: the first response token "
                  "was unguessable (0.070) while positions 1-2 were trivially "
                  "copyable from the true prefix (0.927 / 0.900), and the "
                  "training loss was 0.3151, so the token distribution had "
                  "been learned. That profile is diagnostic of a prompt that "
                  "omits the input, not of a weak model.",
        "invalidated": "The whole first Act 7 result set (full fine-tuning "
                       "0.0104, every PEFT method 0.0000), which read exactly "
                       "like a legitimate negative result. After the fix, the "
                       "same recipe scores 0.6250.",
    },
    {
        "defect": "D12 (Act 7)",
        "source": "docs/implementation.md",
        "what": "A second cross-entropy shift in three PEFT forwards. "
                "MiniGPT computes the loss with no internal shift, but the "
                "prompt-tuning, prefix-tuning and adapter wrappers shifted a "
                "second time, and the response scorer sliced the logits off "
                "by one where a method prepends virtual tokens.",
        "caught": "All three methods reported exactly 0.0000 while genuinely "
                  "training, which is impossible for a method that is "
                  "fitting; the off-by-one slice was then pinned by "
                  "constructing the already-correct case.",
        "invalidated": "The prompt-tuning, prefix-tuning and adapter arms of "
                       "7c. LoRA and full fine-tuning route through the "
                       "MiniGPT forward and were never affected, so 7b and 7d "
                       "were correct the first time.",
    },
    {
        "defect": "D18 (Act 9)",
        "source": "docs/implementation.md",
        "what": "The held-out metric compared logits to the wrong position: "
                "each position was scored against the token at that position "
                "rather than the token one ahead of it.",
        "caught": "An absolute sanity check: cross-entropy 11.98 nats "
                  "(perplexity 1.6e5) for a model whose real held-out loss is "
                  "about 2.5, which is 160,000 times worse than uniform over "
                  "a 6,487-token vocabulary (ln 6487 = 8.78 nats). The "
                  "uniform test alone cannot catch the shift, because it "
                  "returns ln V either way; the next-token oracle is the "
                  "test that does.",
        "invalidated": "Every Act 9 ablation delta. The metric is now pinned "
                       "by a constructed next-token oracle (must score about "
                       "0) and a uniform model (must score exactly ln V).",
    },
    {
        "defect": "5c double feed (Act 5)",
        "source": "docs/implementation.md; results/v2/act5/speculative_zero_gap.json",
        "what": "Speculative decoding re-fed the whole sequence at the top "
                "of every round, on top of an already-prefilled KV cache, so "
                "the target's distribution p differed from the draft's q even "
                "when the two were the same weights.",
        "caught": "An absolute invariant: acceptance must be exactly 100% at "
                  "a zero draft-target gap, because p = q makes "
                  "min(1, p/q) = 1 for every token. The cached-versus-"
                  "uncached equivalence tests both PASSED, because both paths "
                  "shared the bug and agreed with each other while both were "
                  "wrong.",
        "invalidated": "Every 5c number taken before the fix: zero-gap "
                       "acceptance 15.6% at K = 3 and 28.9 tokens/s. After the "
                       "fix, acceptance is 100% and throughput is 309.8 "
                       "tokens/s, a 10.7x improvement.",
    },
    {
        "defect": "Truncated demonstrations (Act 7b)",
        "source": "docs/implementation.md; results/v2/act7/6.3M_sft_peft.json",
        "what": "The conventional 4-shot ICL prompt was 328 tokens against a "
                "128-token window, so truncation deleted the demonstrations "
                "before the model ever saw them.",
        "caught": "A positive control that recorded the original prompt's "
                  "token count and the number of visible shots "
                  "('2/4 (truncated away)'). The metric was sound; the "
                  "prompt was not.",
        "invalidated": "The first ICL measurement, which reported 0.0000 and "
                       "read as a negative result about in-context learning.",
    },
    {
        "defect": "Flat chance baseline (Act 8b)",
        "source": "docs/implementation.md; results/v2/act8/hallucination.json",
        "what": "A flat 'chance = 1/3' baseline, which is wrong because the "
                "candidate set is deduplicated and heterogeneous: it holds 3, "
                "4, 5 or 6 distinct numbers (1, 3, 8 and 28 items "
                "respectively).",
        "caught": "Counting the candidates per item, which gives the per-item "
                  "uniform chance of 0.1838 and forces every chance-adjusted "
                  "rate to be computed per condition.",
        "invalidated": "The direction of the comparison. Against a false 1/3 "
                       "baseline every condition looks below chance; against "
                       "the true 0.1838 baseline every chance-adjusted rate is "
                       "not significant.",
    },
    {
        "defect": "Retriever query design (Act 8c)",
        "source": "docs/implementation.md; results/v2/act8/hallucination.json",
        "what": "The retriever was queried with the instruction text, whose "
                "terms (what, number, does, the, passage, state) cannot come "
                "from a passage about a number, so term-overlap ranking "
                "degenerated to a constant and the tie-break returned the "
                "SAME chunk for all 40 items.",
        "caught": "Inspecting what the retriever actually returned: one "
                  "distinct document for all 40 items, and the identical top "
                  "chunk (score 0.6667) for two different failing items.",
        "invalidated": "The retrieved condition's earlier 0/40, which read "
                       "like a finding. The retriever was working; it was "
                       "being asked an impossible question. After the fix: 39 "
                       "distinct documents, a 0.975 document hit rate, and a "
                       "condition measured at 8/40, at chance.",
    },
    {
        "defect": "ALiBi no-op (Act 2)",
        "source": "results/act2-ablations/summary.json; docs/implementation.md",
        "what": "The 'pe-alibi' arm ran the no-positional-encoding model, "
                "because the ALiBi positional-encoding module returns its "
                "input unchanged and the forward pass carries the comment "
                "'ALiBi: apply attention bias (not implemented, skip)'.",
        "caught": "The two rows differ by 4.8e-08 nats (2.4850885868 against "
                  "2.4850885391), which is float32 round-off, and the two "
                  "loss trajectories agree to the same precision at every "
                  "logged step. No real effect is that small.",
        "invalidated": "The claim that ALiBi is competitive with learned "
                       "absolute positions. The honest reading of the table "
                       "is that no positional encoding scored best among "
                       "those actually implemented, and that ALiBi was not "
                       "measured at all.",
    },
    {
        "defect": "Two architecture families in one ladder (Act 4)",
        "source": "docs/implementation.md; results/v2/act4-scaling/clean_refit.json",
        "what": "THE LARGEST ENTRY, and the one whose lesson generalises: "
                "A GOOD-LOOKING FIT IS NOT EVIDENCE OF A GOOD FIT. At the time of "
                "the audit 11 of the 16 ladder runs were the canonical MiniGPT "
                "and 5 were a "
                "different family with no learned position table, and the "
                "tracked fit consumed points from both, so 51.4% of the "
                "fitted set came from a different model class. The fit was "
                "GOOD: R^2 = 0.9950 in sample, a plausible irreducible loss "
                "of 1.605 nats, plausible residuals, and a leave-one-rung-out "
                "headline of R^2 >= 0.985 on every held-out rung. It was "
                "fitting the wrong models. The other family occupied the "
                "ENTIRE TOP of the ladder (the 4.1M and 9.0M rungs) and two "
                "of the three runs at the 1.5M rung, so the fit's good "
                "looking was the other family's good looking.",
        "caught": "Three things, and the first is the one that generalises. "
                  "(i) An ABSOLUTE INVARIANT: the fit's own parameters score "
                  "R^2 = +0.9957 on the other family's 1428 points, +0.2874 "
                  "on the reconstructed mixed set, and R^2 = -0.6661 on the "
                  "canonical models' own 3463 points. No fit of these models "
                  "can score a negative R^2 on their own points, so that "
                  "single number is a violation of an absolute, not a "
                  "comparison. A residual plot could not have shown it: the "
                  "mixed fit's residuals are still flat in log N and log D. "
                  "(ii) A PROVENANCE READ of the run records, the surviving "
                  "checkpoints' state dicts and the architecture block that "
                  "the training code writes into every metrics.json, which "
                  "is what identified WHICH runs were contaminated. The "
                  "repair is a CONFIG HASH computed from the LIVE model at "
                  "the moment a point is logged, not retroactively from a "
                  "rung table, because retroactive tagging is exactly what "
                  "let a mid-campaign architecture change hide in plain "
                  "sight. (iii) A REPRODUCTION FROM THE REPOSITORY'S OWN "
                  "CODE: the other family's 129.57-223.77 nat step-1 losses "
                  "are an INITIALISATION property of that model class, not a "
                  "metric bug and not divergence -- building that class with "
                  "the repository's own seed on the same data reproduces "
                  "133.25 / 184.49 / 224.99 against the recorded 129.57 / "
                  "183.96 / 223.77, the metric is an average rather than a "
                  "sum (a summed metric would be about 36,000 nats) and it "
                  "is not misaligned (the canonical family's step-1 loss is "
                  "8.80 against a uniform-over-6,487-vocabulary "
                  "cross-entropy of ln 6487 = 8.78). Those five runs were "
                  "GOOD models that trained normally and finished at 3.578 "
                  "nats at the 1.5M rung against the canonical 3.0468; they "
                  "were invisible because their trajectory started 15-25x "
                  "above uniform, so a plausible-looking curve sat at a "
                  "plausible-looking N with a plausible-looking R^2 and "
                  "nothing in the artefacts distinguished it.",
        "invalidated": "The centrepiece number. alpha = 0.445, beta = 0.290 "
                       "and E = 1.605 are withdrawn; the leave-one-rung-out "
                       "'R^2 >= 0.985 on every rung' headline is withdrawn, "
                       "because it was produced by the other family; and the "
                       "comparison with Chinchilla that rested on that fit is "
                       "withdrawn with it, and has to be re-established on "
                       "the surviving fit under the correct parameter "
                       "convention rather than assumed. What the refit is "
                       "NOT is a minor correction: a reader who does not know "
                       "that the other family owned the top of the ladder "
                       "will read the refit as a small adjustment, and it is "
                       "not one. The CAUSE of the contamination is now known "
                       "too, and it is the entry above this one.",
    },
    {
        "defect": "An unstated N-axis convention (Act 4)",
        "source": "results/v2/act4-scaling/clean_refit.json; "
                  "results/v2/act4-scaling/unigram_entropy.json; "
                  "docs/implementation.md",
        "what": "The ladder's recorded N was a NON-EMBEDDING parameter "
                "count, while the published exponents (Kaplan, Chinchilla) "
                "are fitted on TOTAL parameters. The two were therefore never "
                "on the same axis, and the confound was unstated: every "
                "comparison of our exponent with a published one was a "
                "comparison of two different quantities.",
        "caught": "One ground that survives, and two that were withdrawn. "
                  "(i) The convention is the PUBLISHED one: Kaplan and "
                  "Chinchilla regress on total parameters, so the total axis "
                  "is the only axis on which a comparison is like-for-like. "
                  "This is the ground the paper still uses. (ii) It is the "
                  "STABLE one: under total N the point estimate and the "
                  "fold-wise mean agree (+0.589 against +0.587) -- but they "
                  "now agree under non-embedding too (+0.407 against +0.407), "
                  "so on the widened ladder stability no longer separates the "
                  "two axes and the argument is RETIRED. (iii) An ABSOLUTE "
                  "PLAUSIBILITY CHECK on the fitted irreducible loss, which "
                  "put the two axes on opposite sides of what a next-token "
                  "model can reach, was FALSIFIED BY MEASUREMENT and is "
                  "withdrawn: the corpus's measured unigram entropy under the "
                  "6,487-token vocabulary is 4.9946 nats, BOTH fitted E "
                  "values lie below it (1.964 is 0.39x it and 1.911 is "
                  "0.38x it), and against the 2.5152 nats the best model "
                  "actually reaches the two axes are 0.78x and 0.76x. The "
                  "physical-plausibility ground therefore does not "
                  "discriminate.",
        "invalidated": "Every comparison of our exponents with a published "
                       "value, in both directions. The result reverses: the "
                       "profile likelihood's compatible alpha range is "
                       "[0.34, 0.82] on the total-parameter axis and "
                       "[0.20, 0.60] on the non-embedding axis, so the "
                       "alpha comparison is compatible on both axes and "
                       "discriminates between neither. The convention is now "
                       "stated in the paper rather than left implied, and the "
                       "paper reports one ground for it rather than three.",
    },
    {
        "defect": "Three N conventions in one table, none of them flagged (Act 4)",
        "source": "results/v2/act4-scaling/ladder_points.csv; "
                  "results/v2/act4-scaling/clean_refit.json; "
                  "docs/implementation.md",
        "what": "A per-rung audit of every N against three independent "
                "sources -- the ladder CSV's column, the run's own recorded "
                "metric, and the architecture the run implies -- finds that "
                "the table mixes three N conventions and flags none of them. "
                "Every row agrees except the 0.8M rung, whose recorded N of "
                "694,144 matches NO MiniGPT or ALiBi shape at any (dim, "
                "layers, heads) in dims 16-256 and depths 1-10, while its "
                "embedding parameters (830,336 = 6,487 x 128) show the "
                "vocabulary and dim are as expected. The 0.8M rung is "
                "therefore a THIRD, unidentifiable architecture: it has no "
                "architecture block, it ran on CPU, it sits at D/N = 22.7 "
                "rather than 20, and it carries FOUR provenance stories "
                "(694,144 recorded, 809,728 the MiniGPT shape, 793,344 the "
                "other family's shape, 786,432 the LADDER design target).",
        "caught": "AN EXACT INTEGER AUDIT against three independent sources, "
                  "one of them a re-derivation rather than a quotation: the "
                  "CSV column and the run's recorded metric are re-derived "
                  "from the tracked ladder points and the tracked run "
                  "records, and the architecture-implied and design-target "
                  "columns are quoted from the project ledger. The 0.8M rung "
                  "is then confirmed to be unidentifiable by a search over "
                  "every (dim, layers, heads) in dims 16-256 and depths 1-10.",
        "invalidated": "The 0.8M rung's standing as a canonical model, and "
                       "with it the worst hold-out fold in the act (R^2 = "
                       "0.508 non-embedding, 0.779 total), which is the only "
                       "fold whose N falls outside the fitted range once it "
                       "is removed. The audit cannot repair the rung -- only "
                       "a re-run can -- and the re-run is in the rescue plan, "
                       "which is in flight and unfinished, so no repair "
                       "number is reported in the paper.",
    },
    {
        "defect": "Mislabelled sweep sizes (Act 2, Act 3)",
        "source": "report/REPORT_SPEC.md; results/act2-ablations/summary.json; results/act3-tokenizers/summary.json",
        "what": "The report specification says Act 2 ran 20 runs and Act 3 "
                "ran 5 tokenizers.",
        "caught": "Counting the runs in the tracked result files: 18 "
                  "configurations in the Act 2 summary and 8 runs (4 "
                  "tokenizer families x 2 controls) in the Act 3 summary.",
        "invalidated": "No measurement: both sweeps were correct, and the "
                       "report's own text says 18 configurations and four "
                       "tokenizer families. What it invalidated was the "
                       "description of the sample, which is the first thing a "
                       "reader checks.",
    },
    {
        "defect": "D19 (Act 9)",
        "source": "docs/implementation.md",
        "what": "The random-ablation null was measured against a stale "
                "baseline, because the head-ablation function did not record "
                "its own intact loss in the module-level slot the null reads.",
        "caught": "A sanity check on the z-scores themselves: -88 instead of "
                  "+42.8. A negative z of that size is a sign error, not a "
                  "measurement.",
        "invalidated": "The random-ablation null itself, and therefore every "
                       "z-score in the act. The fix is to record the baseline "
                       "inside the head-ablation function.",
    },
    {
        "defect": "D14 (Act 8)",
        "source": "docs/implementation.md",
        "what": "The answer-copying score absorbed the prompt's own "
                "log-probability, because it summed over the whole sequence. "
                "Harmless within one prompt, invalid for every "
                "across-condition contrast, since the oracle prompt is far "
                "longer than the closed-book one.",
        "caught": "An absolute plausibility check: a passage-lift of -246 "
                  "nats, which is absurd on its face. The fix is pinned by a "
                  "test with a constant-logit stub model, demonstrated "
                  "failing against the old version (2 failures to 0).",
        "invalidated": "Every across-condition lift contrast in the first "
                       "Act 8 write-up, and with them the interpretation of "
                       "the passage-lift rows.",
    },
]


# ── LaTeX escaping ──────────────────────────────────────────────────────────

def tex_escape(text: str) -> str:
    """Escape a plain-text field for LaTeX text mode."""
    out = []
    i = 0
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "\\": r"\textbackslash{}",
    }
    while i < len(text):
        ch = text[i]
        if ch == "^":
            # a superscript: take the next token (a digit or a brace group)
            j = i + 1
            if j < len(text) and text[j] == "{":
                k = text.find("}", j)
                group = text[j:k + 1] if k != -1 else "{}"
                out.append(r"\textsuperscript{" + group + "}")
                i = k + 1
                continue
            out.append(r"\textsuperscript{" + text[j] + "}")
            i = j + 1
            continue
        out.append(replacements.get(ch, ch))
        i += 1
    return "".join(out)


def write_csv(name: str, header: list[str], rows: list[list[str]]) -> None:
    path = TABLES / name
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def write_tex(name: str, body: str) -> None:
    path = TABLES / name
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def headline_table() -> None:
    with CLAIMS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r["claim_id"]: r for r in rows}

    missing = [i for i in HEADLINE_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"claims.csv is missing headline ids: {missing}")

    selected = [by_id[i] for i in HEADLINE_IDS]

    write_csv("act10_headline_claims.csv",
              ["claim_id", "claim", "n", "interval", "status"],
              [[r["claim_id"], r["claim"], r["n"], r["interval"], r["status"]]
               for r in selected])

    tex_rows = []
    for r in selected:
        tex_rows.append(
            " & ".join([
                tex_escape(r["claim"]),
                tex_escape(r["n"]),
                tex_escape(r["interval"]),
                tex_escape(r["status"]),
            ]) + r" \\")

    n_ledger = len(rows)
    body = f"""% GENERATED by tools/build_act10_tables.py -- do not edit by hand.
% Source: report/claims.csv ({n_ledger} rows) and the ids listed in that
% script.  Regenerate with: python tools/build_act10_tables.py
{{\\small
\\begin{{longtable}}{{L{{6.5cm}}L{{1.6cm}}L{{4.3cm}}L{{1.9cm}}}}
\\caption{{The headline claims of this paper: {len(selected)} of the
{n_ledger} rows of the tracked claims ledger
(\\texttt{{report/claims.csv}}), with the sample size, the interval and the
status of each. \\textbf{{What is measured:}} the claim as it is stated in
the body, its measured $n$, its interval and its status. \\textbf{{Interval
type:}} per row, as recorded in the ledger; the ledger is the complete
artefact and this table is its summary.}}\\\\
\\toprule
\\textbf{{Claim}} & \\textbf{{$n$}} & \\textbf{{Interval}} & \\textbf{{Status}} \\\\
\\midrule
\\endfirsthead
\\caption[]{{The headline claims (continued).}}\\\\
\\toprule
\\textbf{{Claim}} & \\textbf{{$n$}} & \\textbf{{Interval}} & \\textbf{{Status}} \\\\
\\midrule
\\endhead
""" + "\n".join(tex_rows) + """
\\bottomrule
\\end{longtable}
}
"""
    write_tex("act10_headline_claims.tex", body)


def failure_taxonomy() -> None:
    write_csv("act10_failure_taxonomy.csv",
              ["defect", "what it was", "how it was caught",
               "what it invalidated", "source"],
              [[t["defect"], t["what"], t["caught"], t["invalidated"],
                t["source"]] for t in TAXONOMY])

    tex_rows = []
    for t in TAXONOMY:
        tex_rows.append(
            " & ".join([
                tex_escape(t["defect"]),
                tex_escape(t["what"]),
                tex_escape(t["caught"]),
                tex_escape(t["invalidated"]),
            ]) + r" \\")

    body = f"""% GENERATED by tools/build_act10_tables.py -- do not edit by hand.
% Source: the defect records in docs/implementation.md, quoted in that file.
% Regenerate with: python tools/build_act10_tables.py
{{\\small
\\begin{{longtable}}{{L{{2.3cm}}L{{4.3cm}}L{{4.4cm}}L{{3.2cm}}}}
\\caption{{The measurement-failure taxonomy: {len(TAXONOMY)} recorded defects,
each with what it was, how it was caught, and what it invalidated.
\\textbf{{What is measured:}} the defect records of
\\texttt{{docs/implementation.md}}; no new measurement is made here.
\\textbf{{Interval type:}} none; this is a taxonomy, not an estimate.}}\\\\
\\toprule
\\textbf{{Defect}} & \\textbf{{What it was}} & \\textbf{{How it was caught}} &
\\textbf{{What it invalidated}} \\\\
\\midrule
\\endfirsthead
\\caption[]{{The measurement-failure taxonomy (continued).}}\\\\
\\toprule
\\textbf{{Defect}} & \\textbf{{What it was}} & \\textbf{{How it was caught}} &
\\textbf{{What it invalidated}} \\\\
\\midrule
\\endhead
""" + "\n".join(tex_rows) + """
\\bottomrule
\\end{longtable}
}
"""
    write_tex("act10_failure_taxonomy.tex", body)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    headline_table()
    failure_taxonomy()


if __name__ == "__main__":
    main()
