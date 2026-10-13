from __future__ import annotations

import json
import re
import unittest
from collections import Counter
from pathlib import Path

REPORT = Path("report/final_report.tex")
COMPARISON = Path("artifacts/tick-204/comparison.json")
# Figures are tracked in report/figures/ so the report stays self-contained and
# compiles on Overleaf. artifacts/ is git-ignored and may legitimately be absent.
FIGURES = Path("report/figures")


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
        for kind in ("figure", "table"):
            blocks = re.findall(rf"\\begin\{{{kind}\}}.*?\\end\{{{kind}\}}", self.text, flags=re.S)
            self.assertTrue(blocks, kind)
            for block in blocks:
                self.assertIn("\\caption{", block)

    def test_tables_fit_one_column(self) -> None:
        # Sum of fixed p-column widths (cm) must stay under the 16 cm text width of A4 with 2.5 cm margins.
        for spec in re.findall(r"\\begin\{tabular\}\{([^\n]+)\}", self.text):
            widths = [float(w) for w in re.findall(r"[LR]\{([\d.]+)cm\}", spec)]
            if widths:
                self.assertLess(sum(widths), 15.0, spec)

    def test_referenced_figures_exist_when_generated(self) -> None:
        # Every \includegraphics target must resolve from report/figures/ alone.
        # This is the check that catches a figure existing only under the
        # git-ignored artifacts/ tree, which is invisible to Overleaf and to a
        # fresh clone.
        names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", self.text)
        self.assertTrue(names, "report references no figures")
        self.assertEqual(
            set(names),
            {"loss_curves.png", "perplexity_params.png", "latency.png", "attention.png"},
        )
        if not FIGURES.exists():
            self.skipTest("report/figures not present")
        for name in names:
            self.assertTrue((FIGURES / name).exists(), f"{name} missing from report/figures/")

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

    def test_headline_numbers_match_comparison_json(self) -> None:
        if not COMPARISON.exists():
            self.skipTest("comparison.json not present")
        results = {r["tokenizer"]: r for r in json.loads(COMPARISON.read_text(encoding="utf-8"))}
        for name, column in (("character", 0), ("bpe", 1)):
            result = results[name]
            row = next(line for line in self.text.splitlines() if line.startswith("Validation bits per character"))
            cells = [c.strip(" \\") for c in row.split("&")[1:]]
            self.assertEqual(cells[column], f"{result['validation_bits_per_character']:.4f}")
            row = next(line for line in self.text.splitlines() if line.startswith("Parameters &"))
            cells = [c.strip(" \\") for c in row.split("&")[1:]]
            self.assertEqual(cells[column], f"{result['parameters']:,}")


if __name__ == "__main__":
    unittest.main()
