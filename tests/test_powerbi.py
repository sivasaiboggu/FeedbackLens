"""Check exported values and theme locally; not a Power BI rendering test."""
import hashlib
import json
from pathlib import Path
import unittest
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

class PowerBIKitTests(unittest.TestCase):
    def test_csv_manifest_matches_delivered_exports(self):
        manifest = json.loads((ROOT / "powerbi/exports_manifest.json").read_text())
        for name, value in manifest["exports"].items():
            path = ROOT / "powerbi" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), value["sha256"])
            self.assertEqual(len(pd.read_csv(path)), value["rows"])

    def test_theme_and_kpis_match_frozen_evidence(self):
        theme = json.loads((ROOT / "powerbi/graphite-sage-theme.json").read_text())
        self.assertEqual(theme["background"], "#151D21")
        self.assertEqual(theme["dataColors"][0], "#A8C4A2")
        result = json.loads((ROOT / "artifacts/results.json").read_text())
        kpi = pd.read_csv(ROOT / "powerbi/kpi_summary.csv").iloc[0]
        measured = result["models"][result["selected_model"]]["holdout"]
        self.assertAlmostEqual(kpi.holdout_sentiment_accuracy, measured["accuracy"])
        self.assertEqual(kpi.new_human_issue_labels, 0)
        self.assertEqual(pd.read_csv(ROOT / "powerbi/confusion_matrix.csv").rows.sum(), 200)
