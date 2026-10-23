import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from feedbacklens.pipeline import clean_data, split_data, issue_evidence, ROOT
from feedbacklens.review import evaluate_reviews


class PipelineTests(unittest.TestCase):
    def test_sql_dedup_and_split(self):
        d = clean_data()
        a, b, c = split_data(d)
        self.assertEqual(len(c), 200)
        self.assertEqual(len(b), 200)
        self.assertEqual(len(set(a) | set(b) | set(c)), len(d))
        self.assertFalse(set(a) & set(c))
        self.assertFalse(d.text.str.lower().str.strip().duplicated().any())

    def test_issue_abstention(self):
        self.assertEqual(issue_evidence("Battery failed")["category"], "battery_power")
        self.assertEqual(
            issue_evidence("Battery and sound both fail")["category"], "needs_review"
        )
        self.assertEqual(
            issue_evidence("Unclear experience")["category"], "needs_review"
        )
        self.assertFalse(issue_evidence("sound")["validated"])

    def test_missing_review_labels_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_reviews(ROOT / "artifacts/blind_review_template.csv")

    def test_reference_labels_are_not_claimed_as_new_human_work(self):
        r = json.loads((ROOT / "artifacts/results.json").read_text())
        self.assertEqual(r["human_issue_labels_collected"], 0)
        self.assertIsNone(r["issue_accuracy"])
        self.assertIn("publisher", r["reference_labels"])

    def test_duplicate_review_ids_rejected(self):
        with tempfile.TemporaryDirectory() as p:
            f = Path(p) / "reviews.csv"
            pd.DataFrame(
                [
                    {
                        "review_id": 1,
                        "sentiment_label": 0,
                        "issue_label": "other",
                        "reviewer": "synthetic test fixture",
                        "reviewed_at": "2026-10-23T00:00:00Z",
                    }
                ]
                * 200
            ).to_csv(f, index=False)
            with self.assertRaises(ValueError):
                evaluate_reviews(f)

    def test_valid_synthetic_review_is_evaluated_with_self_reported_attribution(self):
        # Fixture copies predictions intentionally to test mechanics, not accuracy.
        # These labels are never added to the project's measured results.
        queue = json.loads((ROOT / "artifacts/queue.json").read_text())
        rows = [
            {
                "review_id": x["review_id"],
                "sentiment_label": int(x["predicted_sentiment"] == "positive"),
                "issue_label": (
                    x["category"] if x["category"] != "needs_review" else "other"
                ),
                "reviewer": "SYNTHETIC TEST FIXTURE NOT A PERSON",
                "reviewed_at": "2026-10-23T00:00:00Z",
            }
            for x in queue
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = evaluate_reviews(path)
            self.assertEqual(result["rows"], 200)
            self.assertIn("not performed", result["identity_verification"])
            self.assertEqual(result["sentiment_accuracy"], 1.0)

    def test_review_policy_uses_validation(self):
        r = json.loads((ROOT / "artifacts/results.json").read_text())
        self.assertIn("validation only", r["review_policy"]["threshold_source"])
        self.assertEqual(r["splits"]["holdout"], 200)
