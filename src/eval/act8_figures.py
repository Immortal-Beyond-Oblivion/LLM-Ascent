"""Act 8 figures. Reads tracked results only -- no re-measurement.

Three figures:
  act8_hallucination_by_condition  the free-generation rate (at a ceiling) and
                                   the answer-copying rate, with intervals
  act8_contrast_distribution       the paired per-item contrasts that are the
                                   headline, with bootstrap intervals
  act7_icl_copy_vs_compute         Act 7's 18/18 copy vs 0/18 compute result
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "v2"
FIGURES = ROOT / "report" / "figures"

FLOOR = "#8c8c8c"
SFT = "#1f77b4"
PEFT = "#d62728"
OK = "#2ca02c"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


def hallucination_by_condition() -> None:
    d = json.loads((RESULTS / "act8" / "hallucination.json").read_text())
    conds = list(d["conditions"]["copy_instrument"].keys())
    free = d["conditions"]["free_instrument"]
    copy = d["conditions"]["copy_instrument"]
    head = d["headline"]["choice_rates"]
    chance = head["mean_uniform_chance"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"wspace": 0.42})

    # Left: THE RESULT. Choice rates, with the per-item chance line.
    ax = axes[0]
    rates = [copy[c]["rates"]["source"] for c in conds]
    los = [copy[c]["chose_source_wilson95"][0] for c in conds]
    his = [copy[c]["chose_source_wilson95"][1] for c in conds]
    x = list(range(len(conds)))
    colors = [SFT if c == "oracle" else FLOOR for c in conds]
    ax.bar(x, rates, color=colors, width=0.6)
    ax.errorbar(x, rates, yerr=[[r - l for r, l in zip(rates, los)],
                                [h - r for r, h in zip(rates, his)]],
                fmt="none", ecolor="black", capsize=4, lw=1.2)
    # The per-item chance level, which is NOT 1/3: candidate sets are
    # deduplicated and vary between 3 and 6 members.
    ax.axhline(chance, color=PEFT, lw=1.6, ls="--",
               label=f"per-item chance ({chance:.4f})")
    for i, c in enumerate(conds):
        ax.text(i, rates[i] + 0.02, f"{rates[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8.5)
    ax.set_ylabel("share of items")
    ax.set_ylim(0, max(his) + 0.14)
    ax.set_title("THE RESULT: the model identifies the asserted number\n"
                 "AT CHANCE (n = 40 items, Wilson 95%)", fontsize=9.5)
    ax.annotate("not significantly below chance;\n"
                "closed-book (no passage) is HIGHER",
                xy=(0.02, 0.86), xycoords="axes fraction", fontsize=8.5,
                color=PEFT, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    # Right: the ceiling, which is never used as a comparison.
    ax = axes[1]
    vrates = [free[c]["volunteers_a_number_rate"] for c in conds]
    ax.bar(x, vrates, color=[PEFT] * len(conds), alpha=0.85, width=0.6)
    for i, c in enumerate(conds):
        ax.text(i, vrates[i] + 0.02, f"{vrates[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8.5)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Free generation: volunteers ANY number\n"
                 "(never the right one -- 0/40 in every condition)", fontsize=9.5)
    ax.annotate("CEILING -- uninformative for ranking conditions",
                xy=(0.5, 0.62), xycoords="axes fraction", ha="center",
                fontsize=9, color=PEFT, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    m = head["mcnemar_oracle_vs_closed_book"]
    fig.suptitle("Act 8: supplying the passage does not change the answer "
                 "(McNemar p = %.3f)" % m["mcnemar_p_value"],
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "act8_hallucination_by_condition")


def contrast_distribution() -> None:
    d = json.loads((RESULTS / "act8" / "hallucination.json").read_text())
    con = d["contrasts"]
    log = d["logprob_contrasts"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8),
                             gridspec_kw={"wspace": 0.55})

    # Left: paired per-item contrasts (rates).
    ax = axes[0]
    keys = [("oracle_vs_closed_book_answer_changes",
             "answer changes:\noracle vs closed-book"),
            ("counterfactual_tracks_the_edit",
             "counterfactual\ntracks the edit"),
            ("counterfactual_answer_unchanged_from_oracle",
             "answer UNCHANGED\nwhen the text is edited")]
    vals = [con[key]["rate"] for key, _ in keys]
    los = [con[key]["bootstrap95"]["ci_low"] for key, _ in keys]
    his = [con[key]["bootstrap95"]["ci_high"] for key, _ in keys]
    ys = range(len(keys))
    ax.errorbar(vals, ys,
                xerr=[[v - l for v, l in zip(vals, los)],
                      [h - v for v, h in zip(vals, his)]],
                fmt="o", color=SFT, capsize=5, lw=1.6, ms=7)
    for i, (v, (key, _)) in enumerate(zip(vals, keys)):
        text = f"{v:.3f}  ({con[key]['count']}/{con[key]['n']})"
        # High values go ABOVE the marker: placed to the side they would sit
        # on top of the error bar.
        if v > 0.6:
            ax.text(v, i + 0.26, text, va="bottom", ha="center", fontsize=8.5)
        else:
            ax.text(v + 0.03, i, text, va="center", ha="left", fontsize=8.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for _, lab in keys], fontsize=8.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("share of items (95% bootstrap CI over items)")
    ax.set_title("Paired per-item contrasts", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    # Right: paired log-probability contrasts.
    ax = axes[1]
    lkeys = [("context_specific_discrimination",
              "context-specific\ndiscrimination"),
             ("passage_lift_on_the_number_it_states",
              "passage lift on\nthe number it states"),
             ("passage_lift_on_the_counterfactual_number",
              "passage lift on the\nUNSTATED number"),
             ("edit_effect_on_the_counterfactual_number",
              "effect of editing\nthe number"),
             ("counterfactual_margin_edited_minus_original",
              "edited number preferred\nunder counterfactual")]
    vals = [log[key]["mean"] for key, _ in lkeys]
    los = [log[key]["bootstrap95"]["ci_low"] for key, _ in lkeys]
    his = [log[key]["bootstrap95"]["ci_high"] for key, _ in lkeys]
    ys = range(len(lkeys))
    colors = [OK if (log[key].get("p_value") or 1) < 0.05 else FLOOR
              for key, _ in lkeys]
    ax.errorbar(vals, ys,
                xerr=[[v - l for v, l in zip(vals, los)],
                      [h - v for v, h in zip(vals, his)]],
                fmt="o", color=FLOOR, ecolor=FLOOR, capsize=5, lw=1.6, ms=7)
    for i, (v, (key, _), c) in enumerate(zip(vals, lkeys, colors)):
        ax.plot([v], [i], "o", color=c, ms=7, zorder=3)
        p = log[key].get("p_value")
        ax.text(v + 0.03, i, f"{v:+.3f}  p={p:.3f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for _, lab in lkeys], fontsize=8.5)
    ax.set_xlabel("mean paired difference in log-probability (nats)")
    ax.set_title("Paired log-probability contrasts\n"
                 "(green: p < 0.05; grey: n.s.)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Act 8: what the passage actually changes", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "act8_contrast_distribution")


def icl_copy_vs_compute() -> None:
    d = json.loads((RESULTS / "act7" / "6.3M_sft_peft.json").read_text())
    ctl = d["icl_control"]
    n = ctl["with_demo"]["n"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["first generated token is\nthe DEMONSTRATION's answer",
              "exact match on the\nquery's answer"]
    with_demo = [ctl["first_token_is_demonstration_answer_with_demo"], n]
    without = [ctl["first_token_is_demonstration_answer_zero_shot"], n]
    x = [0, 1]
    ax.bar([i - 0.18 for i in x], with_demo, width=0.34, color=OK,
           label="with demonstration in context")
    ax.bar([i + 0.18 for i in x], without, width=0.34, color=FLOOR,
           label="zero-shot control")
    for i, (wd, wo) in enumerate(zip(with_demo, without)):
        ax.text(i - 0.18, wd + 0.4, f"{wd}/{n}", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(i + 0.18, wo + 0.4, f"{wo}/{n}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("items (of %d)" % n)
    ax.set_ylim(0, n * 1.45)
    ax.set_title("Act 7: the demonstration is copied, the operation is not\n"
                 "18 held-out addition items, unseen operand pairs", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "act7_icl_copy_vs_compute")


def main() -> None:
    hallucination_by_condition()
    contrast_distribution()
    icl_copy_vs_compute()


if __name__ == "__main__":
    main()
