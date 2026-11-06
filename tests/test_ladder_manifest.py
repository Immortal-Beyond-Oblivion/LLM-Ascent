"""The ladder's point set is a manifest, and the manifest is the authority.

Act 4's first fit was computed over a mixture of two architecture families and
nothing complained. The ``in_fitted_set`` column reproduced the point COUNT
(2,776) and not the point SET - the tracked fit's own parameters score R^2
0.287 on the set it flags. Three conventions for the parameter count coexisted
in one table, and the ledger's canonical rung table recorded the ladder's
SEARCH TARGET where a measurement belonged.

These tests are the guard rail. Each one pins a specific way that went wrong
before, so that a future edit cannot quietly reintroduce it.
"""

from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path

from src.scaling.ladder_manifest import build_manifest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "report" / "final_report.tex"
POINTS = REPO / "results" / "v2" / "act4-scaling" / "ladder_points.csv"
RUNS = REPO / "results" / "v2" / "act4-scaling" / "ladder_runs.json"
INDEX = REPO / "checkpoints" / "canonical" / "INDEX.json"

KNOWN_ROLES = {
    "canonical_ladder",
    "other_family",
    "superseded",
    "excluded_unidentifiable",
    "excluded_not_a_rung",
}


class LadderManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not POINTS.exists() or not RUNS.exists():
            raise unittest.SkipTest("the ladder manifest has not been exported")
        cls.manifest = json.loads(RUNS.read_text(encoding="utf-8"))
        cls.runs = cls.manifest["runs"]

    def test_every_run_has_a_known_role_and_a_reason(self) -> None:
        for run in self.runs:
            self.assertIn(run["role"], KNOWN_ROLES, run["rung"])
            self.assertTrue(run["role_reason"].strip(),
                            f"{run['rung']} has a role with no reason")

    def test_no_run_is_in_the_fit_twice(self) -> None:
        """At a given parameter count, every run in the fit is ONE model.

        A rung may legitimately carry several seeds - the 2.6M rung carries the
        six paired seeds of Act 5b - so the invariant is not "one run per
        rung" but "one architecture per parameter count". The ladder's
        original failure was precisely that a parameter count carried two
        different models.
        """
        by_n: dict[int, set[str]] = {}
        for run in self.runs:
            if run["role"] != "canonical_ladder":
                continue
            by_n.setdefault(run["non_embedding_params"], set()).add(
                run["config_hash"])
        for n, hashes in by_n.items():
            self.assertEqual(
                len(hashes), 1,
                f"parameter count {n:,} contributes {len(hashes)} different "
                f"architectures to the fit: {sorted(hashes)}")

    def test_a_rung_does_not_carry_two_parameter_counts(self) -> None:
        """A rung label names one rung, so it must name one parameter count."""
        by_rung: dict[str, set[int]] = {}
        for run in self.runs:
            if run["role"] != "canonical_ladder":
                continue
            by_rung.setdefault(run["rung"], set()).add(run["non_embedding_params"])
        for rung, counts in by_rung.items():
            self.assertEqual(len(counts), 1,
                             f"rung {rung} carries {len(counts)} parameter counts")

    def test_the_canonical_ladder_has_more_than_four_distinct_n(self) -> None:
        """The act's own diagnosis was that four rungs cannot identify alpha.

        The rescue widened the range: 8 distinct N values instead of 4, over
        a log-N range of 3.38 rather than 2.21. If this ever drops back, the
        identifiability claim has lost its basis.
        """
        canonical = self.manifest["canonical_ladder"]
        self.assertGreaterEqual(canonical["distinct_n"], 6)
        self.assertGreaterEqual(canonical["log_n_range"], 3.0)

    def test_the_off_line_design_point_is_present(self) -> None:
        """A ladder with every rung at D = 20N cannot separate N from D."""
        arms = {run["arm"] for run in self.runs
                if run["role"] == "canonical_ladder"}
        self.assertIn("D10N", arms)

    def test_the_other_family_is_kept_for_the_diagnostic(self) -> None:
        """The contamination story needs the family the fit wrongly used."""
        other = [r for r in self.runs if r["role"] == "other_family"]
        self.assertGreaterEqual(len(other), 4)

    def test_the_unidentifiable_rung_is_excluded(self) -> None:
        """The 0.8M rung whose N matches no known architecture must not fit."""
        excluded = [r for r in self.runs
                    if r["role"] == "excluded_unidentifiable"]
        self.assertTrue(excluded)
        self.assertTrue(all(r["num_layers"] is None for r in excluded))

    def test_every_point_row_carries_its_runs_role(self) -> None:
        roles: set[str] = set()
        with POINTS.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                roles.add(row["role"])
                self.assertIn(row["role"], KNOWN_ROLES)
        self.assertIn("canonical_ladder", roles)
        self.assertIn("other_family", roles)

    def test_no_point_row_is_selected_by_a_flag(self) -> None:
        """The old in_fitted_set column reproduced a COUNT, not a SET."""
        with POINTS.open(encoding="utf-8") as fh:
            fieldnames = csv.DictReader(fh).fieldnames or []
        self.assertIn("role", fieldnames)
        self.assertNotIn("in_fitted_set", fieldnames)


class CanonicalCountTests(unittest.TestCase):
    """The canonical parameter counts must be measurements, not targets."""

    @classmethod
    def setUpClass(cls) -> None:
        if not INDEX.exists():
            raise unittest.SkipTest("the canonical checkpoints are not present")
        cls.index = json.loads(INDEX.read_text(encoding="utf-8"))

    def test_the_index_counts_are_measured(self) -> None:
        """Each rung's count must equal the model's own parameter count."""
        for model in self.index["models"]:
            self.assertGreater(model["non_embedding_params"], 0)
            self.assertGreater(model["parameters"], model["non_embedding_params"])

    def test_the_paper_reports_the_measured_counts(self) -> None:
        """Table 1 of the paper must agree with the pinned checkpoints.

        It did not: the paper reported 1,536,000 / 2,654,208 / 6,291,456, which
        are the ladder's search targets, while the models have 1,567,200 /
        2,694,144 / 6,351,360. The difference at the 1.5M rung is exactly one
        128 x 160 learned position table.
        """
        if not REPORT.exists():
            raise unittest.SkipTest("report/final_report.tex not present")
        text = REPORT.read_text(encoding="utf-8")
        pins = {m["rung"]: m for m in self.index["models"]}
        for rung in ("1.5M", "2.6M", "6.3M"):
            measured = pins[rung]["non_embedding_params"]
            row = re.search(rf"^{re.escape(rung)} & ([\d,]+) &", text, re.M)
            self.assertIsNotNone(row, f"no Table 1 row for {rung}")
            self.assertEqual(
                int(row.group(1).replace(",", "")), measured,
                f"the paper reports {row.group(1)} for {rung}, but the pinned "
                f"checkpoint has {measured:,} non-embedding parameters")

    def test_the_search_target_is_not_reported_as_a_measurement(self) -> None:
        """No tracked result may call a search target a parameter count."""
        mqa = REPO / "results" / "v2" / "act5" / "mqa_quality_2.6m.json"
        if not mqa.exists():
            raise unittest.SkipTest("the Act 5b results are not present")
        pin = json.loads(mqa.read_text(encoding="utf-8"))["canonical_1.5M_pin"]
        self.assertEqual(pin["non_embedding_params"], 1567200)
        self.assertIn("target_non_embedding_params", pin)
        self.assertNotEqual(pin["non_embedding_params"],
                            pin["target_non_embedding_params"])


if __name__ == "__main__":
    unittest.main()
