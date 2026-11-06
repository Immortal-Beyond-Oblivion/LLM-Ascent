"""Table 1: the findings taxonomy, generated from the tracked artefact.

WHAT THIS READS
    ``results/v2/findings_taxonomy.json`` -- the tracked, curated taxonomy.
    One row per pre-registered hypothesis, plus one row per recorded
    measurement failure, plus one row per claim the paper declines to make.
    The four status values partition the rows:
    ``supported`` / ``unsupported`` / ``measurement_failure`` /
    ``declined_claim``.

WHAT THIS WRITES
    ``report/tables/findings_taxonomy.csv``  -- the CSV twin, one line per row
    ``report/tables/findings_table.tex``    -- Table 1, the longtable fragment
    ``report/tables/do_not_make_claims.tex`` -- the "claims we do not make"
                                                longtable fragment

WHY A GENERATOR
    The table is the paper's front door, and a hand-maintained front door is a
    front door that drifts.  The taxonomy lives in ONE tracked JSON file; the
    table, the CSV and the do-not-make list are all generated from it, so the
    three cannot disagree.  The counts in Table 1's caption are computed here
    from the rows, so a count in the paper is a count in the data.

Usage:
    python -m src.eval.findings_table
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = REPO_ROOT / "results" / "v2" / "findings_taxonomy.json"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

STATUS_VOCABULARY = ("supported", "unsupported", "measurement_failure",
                     "declined_claim")

CSV_HEADER = ["id", "act", "hypothesis", "test", "result", "status",
              "preregistered", "preregistration_note", "do_not_make",
              "source_json", "caveat"]

# The four populations, in the order the caption quotes them.
POPULATION_ORDER = ("supported", "unsupported", "measurement_failure",
                     "declined_claim")

# LaTeX characters that must be escaped inside a p{} cell.
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """Escape a plain-text cell for LaTeX."""
    out = []
    for ch in str(text):
        out.append(_ESCAPES.get(ch, ch))
    return "".join(out)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    seen = set()
    for row in rows:
        for field in CSV_HEADER:
            if field not in row:
                raise ValueError(f"{path}: row {row.get('id')!r} has no "
                                 f"{field!r} field")
        if row["status"] not in STATUS_VOCABULARY:
            raise ValueError(f"{path}: row {row['id']!r} has status "
                             f"{row['status']!r}, which is not one of "
                             f"{list(STATUS_VOCABULARY)}")
        if not isinstance(row["preregistered"], bool):
            raise ValueError(f"{path}: row {row['id']!r} has a non-boolean "
                             f"preregistered field")
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate row id {row['id']!r}")
        seen.add(row["id"])
        for one in row["source_json"].split(";"):
            one = one.strip()
            if not one:
                continue
            if not (REPO_ROOT / one).exists():
                raise ValueError(f"{path}: row {row['id']!r} points at a "
                                 f"source file that does not exist: {one}")
    return rows


def write_csv(rows: list[dict], tables: Path) -> None:
    path = tables / "findings_taxonomy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(CSV_HEADER)
        for row in rows:
            w.writerow([row[f] for f in CSV_HEADER])


def counts(rows: list[dict]) -> dict[str, int]:
    c = Counter(r["status"] for r in rows)
    return {s: c.get(s, 0) for s in POPULATION_ORDER}


def write_findings_table(rows: list[dict], tables: Path) -> None:
    """The findings taxonomy, as a longtable."""
    c = counts(rows)
    n = len(rows)
    caption = (
        "The findings taxonomy: one row per pre-registered "
        "hypothesis, per recorded measurement failure and per declined claim. "
        f"\\textbf{{Rows:}} {n} --- {c['supported']} \\emph{{supported}}, "
        f"{c['unsupported']} \\emph{{unsupported}}, {c['measurement_failure']} "
        f"\\emph{{measurement failures}} and {c['declined_claim']} "
        f"\\emph{{declined claims}}, and every row is in exactly one of those "
        "four populations. \\textbf{What is measured:} the hypothesis each act "
        "put at risk, the test it was put at risk by, and the outcome, read "
        "from the tracked result files named in the CSV twin "
        "(\\texttt{report/tables/findings\\_taxonomy.csv}). "
        "\\textbf{Interval type:} per row, as stated in the paper's own "
        "tables; this table is a taxonomy, not an estimate. "
        "\\textbf{Pre-registration:} only Act~4's row "
        "(\\texttt{FALSIFICATION\\_CONDITION.md} in the act's result "
        "directory) was falsifier-first; every other row is marked post-hoc in the "
        "\\emph{preregistered} column of the CSV, and the paper says so."
    )
    lines = [
        "{\\small",
        "\\begin{longtable}{L{0.9cm}L{2.8cm}L{2.7cm}L{3.0cm}L{1.9cm}L{2.1cm}}",
        "\\caption{%s}\\label{tab:findings}\\\\" % caption,
        "\\toprule",
        "\\textbf{Act} & \\textbf{Hypothesis} & \\textbf{Test} & "
        "\\textbf{Result} & \\textbf{Status} & \\textbf{Do-not-make note} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{6}{c}{\\emph{Table~\\thetable{} (continued)}}\\\\",
        "\\toprule",
        "\\textbf{Act} & \\textbf{Hypothesis} & \\textbf{Test} & "
        "\\textbf{Result} & \\textbf{Status} & \\textbf{Do-not-make note} \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for row in rows:
        declined = row["status"] == "declined_claim"
        # A declined claim's reason is published once, in the do-not-make list
        # (Section "Claims we do not make"); the table points at it rather than
        # printing it twice.
        note = ("published in Section~\\ref{sec:dnm}" if declined
                else tex_escape(row["do_not_make"]))
        lines.append(
            "%s & %s & %s & %s & %s & %s \\\\" % (
                tex_escape(row["act"]),
                tex_escape(row["hypothesis"]),
                tex_escape(row["test"]),
                tex_escape(row["result"]),
                tex_escape(row["status"].replace("_", " ")),
                note,
            )
        )
    lines.append("\\end{longtable}")
    lines.append("}")
    (tables / "findings_table.tex").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")


def write_do_not_make_table(rows: list[dict], tables: Path) -> None:
    """The "claims we do not make" longtable, generated from the same rows."""
    declined = [r for r in rows if r["status"] == "declined_claim"]
    caption = (
        "The claims we do not make, with the reason for each and the act it "
        "belongs to. \\textbf{What is measured:} nothing --- these are the "
        "boundaries of the paper's own evidence, and each one is a row of "
        "the tracked taxonomy (\\texttt{results/v2/findings_taxonomy.json}) "
        "and of Table~\\ref{tab:findings}. \\textbf{Interval type:} not "
        "applicable. \\textbf{Rows:} %d."
    ) % len(declined)
    lines = [
        "\\begin{longtable}{L{7.6cm}L{5.4cm}L{1.6cm}}",
        "\\caption{%s}\\label{tab:dnm}\\\\" % caption,
        "\\toprule",
        "\\textbf{We do not claim} & \\textbf{Because} & \\textbf{Act} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\caption[]{The claims we do not make (continued).}\\\\",
        "\\toprule",
        "\\textbf{We do not claim} & \\textbf{Because} & \\textbf{Act} \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for row in declined:
        lines.append("%s & %s & %s \\\\" % (
            tex_escape(row["hypothesis"]),
            tex_escape(row["do_not_make"]),
            tex_escape(row["act"]),
        ))
    lines.append("\\end{longtable}")
    (tables / "do_not_make_claims.tex").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()

    rows = load_rows(args.taxonomy)
    write_csv(rows, args.tables_dir)
    write_findings_table(rows, args.tables_dir)
    write_do_not_make_table(rows, args.tables_dir)
    c = counts(rows)
    print(f"wrote {args.tables_dir / 'findings_taxonomy.csv'} ({len(rows)} rows)")
    print(f"wrote {args.tables_dir / 'findings_table.tex'} (Table 1)")
    print(f"wrote {args.tables_dir / 'do_not_make_claims.tex'} "
          f"({c['declined_claim']} declined claims)")
    print("populations: " + ", ".join(f"{k}={c[k]}" for k in POPULATION_ORDER))


if __name__ == "__main__":
    main()
