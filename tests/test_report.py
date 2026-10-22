"""Structural tests for report/final_report.tex.

The report is compiled by the owner on Overleaf, and there is no LaTeX
toolchain on the machine that writes it, so these tests plus
``tools/check_tex.py`` are the only correctness gate. This module covers the
invariants that a reader of the report would notice; ``tools/check_tex.py``
covers the mechanical ones (paths, references, the claims ledger).

History: this file was written for the v1 report, whose only headline
comparison was character versus BPE. It was updated when the report was
rewritten to cover Acts 0-8, and the v1-specific assertions were replaced with
the v2 ones rather than being kept alongside.
"""

from __future__ import annotations

import csv
import json
import re
import unittest
from collections import Counter
from pathlib import Path

REPORT = Path("report/final_report.tex")
FIGURES = Path("report/figures")
TABLES = Path("report/tables")
CLAIMS = Path("report/claims.csv")
MANIFEST = Path("report/MANIFEST.md")
ACT3 = Path("results/act3-tokenizers/summary.json")


class ReportStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not REPORT.exists():
            raise unittest.SkipTest("report/final_report.tex not present")
        cls.text = REPORT.read_text(encoding="utf-8")

    def test_document_skeleton(self) -> None:
        self.assertIn("\\documentclass", self.text)
        self.assertEqual(self.text.count("\\begin{document}"), 1)
        self.assertEqual(self.text.count("\\end{document}"), 1)

    def test_environments_are_balanced(self) -> None:
        begins = Counter(re.findall(r"\\begin\{([^}]+)\}", self.text))
        ends = Counter(re.findall(r"\\end\{([^}]+)\}", self.text))
        self.assertEqual(begins, ends)

    def test_braces_are_balanced(self) -> None:
        stripped = re.sub(r"\\[{}]", "", self.text)
        stripped = "\n".join(line.split("%", 1)[0] if "\\%" not in line else line for line in stripped.splitlines())
        self.assertEqual(stripped.count("{"), stripped.count("}"))

    def test_every_figure_and_table_has_a_caption(self) -> None:
        for kind in ("figure", "table", "longtable"):
            blocks = re.findall(rf"\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}", self.text, flags=re.S)
            self.assertTrue(blocks, kind)
            for block in blocks:
                self.assertIn("\\caption{", block)

    def test_tables_fit_one_column(self) -> None:
        # Sum of fixed p-column widths (cm) must stay under the 16 cm text width of A4 with 2.5 cm margins.
        # The generated LaTeX fragments that the report reads in with \input are
        # checked too: a table that overflows the page is as broken there as
        # it is in the main file.
        texts = {"report/final_report.tex": self.text}
        for target in re.findall(r"\\input\{([^}]+)\}", self.text):
            path = REPORT.parent / target
            if path.exists():
                texts[target] = path.read_text(encoding="utf-8")
        for name, text in texts.items():
            for spec in re.findall(r"\\begin\{tabular\}\{([^\n]+)\}", text):
                widths = [float(w) for w in re.findall(r"[LR]\{([\d.]+)cm\}", spec)]
                if widths:
                    self.assertLess(sum(widths), 15.0, f"{name}: {spec}")

    def test_every_input_target_is_in_the_bundle(self) -> None:
        """A generated fragment that the report reads in must exist.

        The owner compiles the report on Overleaf, so a fragment that is
        missing from the bundle is a compile error at the worst possible
        moment.
        """
        targets = re.findall(r"\\input\{([^}]+)\}", self.text)
        self.assertTrue(targets, "the report reads in no generated fragment")
        for target in targets:
            self.assertFalse(target.startswith("/"), target)
            self.assertNotIn("..", target)
            self.assertTrue((REPORT.parent / target).exists(),
                            f"\\input{{{target}}} is missing from the bundle")

    def test_referenced_figures_exist_with_a_pdf_sibling(self) -> None:
        # Every \includegraphics target must resolve from report/figures/ alone,
        # and every PNG must have the vector PDF alongside it. This is the check
        # that catches a figure existing only under the git-ignored artifacts/
        # tree, which is invisible to Overleaf and to a fresh clone.
        names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", self.text)
        self.assertTrue(names, "report references no figures")
        if not FIGURES.exists():
            self.skipTest("report/figures not present")
        for name in names:
            self.assertTrue((FIGURES / name).exists(), f"{name} missing from report/figures/")
            self.assertTrue(
                (FIGURES / name).with_suffix(".pdf").exists(),
                f"{name} has no vector PDF sibling in report/figures/")

    def test_graphicspath_stays_inside_the_report_bundle(self) -> None:
        """`report/` must be self-contained so it compiles when unzipped on Overleaf.

        The owner compiles the report on Overleaf, so a figure living only under
        the git-ignored `artifacts/` tree would be invisible there. This is the
        guard for that class of mistake.
        """
        paths = re.findall(r"\\graphicspath\{(.+?)\}", self.text)
        self.assertTrue(paths, "report declares no \\graphicspath")
        for spec in paths:
            # The macro argument is itself brace-delimited: \graphicspath{{figures/}}
            # so the captured text is "{figures/}" and the inner braces must come
            # off before the entries are usable as paths.
            for entry in spec.strip().strip("{}").split(","):
                entry = entry.strip().strip("{}")
                if not entry:
                    continue
                # Resolve against report/ and check it does not escape the bundle
                # via `..`; a figure that only exists under the git-ignored
                # artifacts/ tree would be invisible once unzipped on Overleaf.
                resolved = (REPORT.parent / entry.rstrip("/")).resolve()
                self.assertTrue(
                    resolved.is_relative_to(REPORT.parent.resolve()),
                    f"\\graphicspath entry {entry!r} escapes report/ and will not "
                    f"exist on Overleaf",
                )

    def test_every_act_section_is_present(self) -> None:
        """Acts 0-10 are all written; no act is left as a placeholder.

        Every act in the paper now has measured results, so the document
        must carry no PENDING marker at all. A placeholder that outlives its
        measurement is the same category of error as a metric that outlives
        the thing it measured.
        """
        for act in range(0, 11):
            self.assertIn(f"\\section{{Act {act}", self.text,
                          f"Act {act} section missing")
        self.assertNotIn("[PENDING - not yet measured]", self.text,
                         "a placeholder marker survived its measurement")

    def test_acts_9_and_10_are_measured(self) -> None:
        """Acts 9 and 10 have been measured; neither keeps a placeholder.

        Act 9 was a placeholder, and Act 10 was too until the ethics
        instruments and the reproducibility script were measured. Both must
        now carry their results and no PENDING marker.
        """
        for label, want_pending in (("sec:act9", False), ("sec:act10", False)):
            m = re.search(rf"\\label\{{{label}\}}\n(.*?)(?=\\section{{|\Z)",
                          self.text, flags=re.S)
            self.assertIsNotNone(m, f"missing {label}")
            body = m.group(1)
            if want_pending:
                self.assertIn("PENDING", body,
                              f"{label} must carry a visible PENDING marker")
            else:
                self.assertNotIn("PENDING", body,
                                 f"{label} is measured and must not carry a "
                                 f"PENDING marker")

    def test_act_9_headline_numbers_match_the_tracked_results(self) -> None:
        """The Act 9 numbers must agree with results/v2/act9/interpretability.json."""
        act9 = Path("results/v2/act9/interpretability.json")
        if not act9.exists():
            self.skipTest("act9 results not present")
        d = json.loads(act9.read_text(encoding="utf-8"))
        a = d["9a_head_ablation"]
        self.assertEqual(a["n_heads_z_ge_2"], 1)
        self.assertEqual(a["n_heads_z_le_0"], 59)
        self.assertEqual(d["9b_rollout"]["rollout_hit_rate"], 0.0)
        self.assertEqual(d["9b_rollout"]["raw_attention_hit_rate"], 0.0)
        for text in ("2.8551", "0.832", "7.82", "42.8", "1.047", "0.379",
                     "2.480", "0.1015", "0.0078"):
            self.assertIn(text, self.text,
                          f"Act 9 value {text} is not in the report")

    def test_act3_headline_numbers_match_the_tracked_results(self) -> None:
        """The Act 3 table must agree with results/act3-tokenizers/summary.json."""
        if not ACT3.exists():
            self.skipTest("act3 results not present")
        summary = json.loads(ACT3.read_text(encoding="utf-8"))
        for tokenizer, key, digits in (
            ("character", "character-equal_char_span", "3.4680"),
            ("bpe", "bpe-equal_char_span", "3.1893"),
            ("wordpiece", "wordpiece-equal_char_span", "3.5116"),
            ("unigram", "unigram-equal_char_span", "2.8341"),
        ):
            self.assertAlmostEqual(summary[key]["bpc_mean"], float(digits), places=4)
            self.assertIn(digits, self.text,
                          f"Act 3 value {digits} for {tokenizer} not in the report")

    def test_claims_ledger_is_present_and_consistent(self) -> None:
        if not CLAIMS.exists():
            self.skipTest("report/claims.csv not present")
        with CLAIMS.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        self.assertEqual(
            list(rows[0].keys()),
            ["claim_id", "act", "claim", "n", "interval", "status",
             "source_json", "caveat"])
        ids = [r["claim_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)), "duplicate claim_id")
        allowed = {"confirmed", "n.s.", "exploratory", "ceiling", "invalid",
                   "superseded", "open-item"}
        for row in rows:
            self.assertIn(row["status"], allowed, row["claim_id"])
            self.assertTrue(row["n"].strip(), row["claim_id"])
            self.assertTrue(row["interval"].strip(), row["claim_id"])
            self.assertTrue(row["source_json"].strip(), row["claim_id"])
            for one in row["source_json"].split(";"):
                one = one.strip()
                if one:
                    self.assertTrue(Path(one).exists(), f"{row['claim_id']}: {one}")

    def test_manifest_lists_every_figure_and_table(self) -> None:
        if not MANIFEST.exists():
            self.skipTest("report/MANIFEST.md not present")
        manifest = MANIFEST.read_text(encoding="utf-8")
        for name in re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", self.text):
            # The manifest documents each figure by its stem, because the
            # figure exists as a PNG and a vector PDF.
            stem = name.rsplit(".", 1)[0]
            self.assertIn(stem, manifest,
                          f"figure {name} is missing from MANIFEST.md")
        for label in re.findall(r"\\label\{(tab:[^}]+)\}", self.text):
            self.assertIn(label, manifest,
                          f"table {label} is missing from MANIFEST.md")

    def test_table_source_data_exists(self) -> None:
        if not TABLES.exists():
            self.skipTest("report/tables not present")
        self.assertTrue(list(TABLES.glob("*.csv")),
                        "report/tables holds no source CSVs")


if __name__ == "__main__":
    unittest.main()
