"""Evaluate independently supplied labels; never manufacture human provenance."""

import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .pipeline import ROOT, TAXONOMY


def evaluate_reviews(path, root=ROOT):
    labels = pd.read_csv(path, keep_default_na=False)
    required = {
        "review_id",
        "sentiment_label",
        "issue_label",
        "reviewer",
        "reviewed_at",
    }
    if not required <= set(labels):
        raise ValueError("Missing review columns")
    if labels.review_id.duplicated().any():
        raise ValueError("Duplicate review IDs")
    if len(labels) != 200:
        raise ValueError("Expected all 200 blind review rows")
    if not labels.sentiment_label.astype(str).isin(["0", "1"]).all():
        raise ValueError("Sentiment must be 0 or 1")
    if not labels.issue_label.isin([*TAXONOMY, "other", "multiple"]).all():
        raise ValueError("Invalid issue label")
    if (labels.reviewer.str.strip() == "").any() or (
        labels.reviewed_at.str.strip() == ""
    ).any():
        raise ValueError("Reviewer and timestamp required")
    if pd.to_datetime(labels.reviewed_at, errors="coerce", utc=True).isna().any():
        raise ValueError("Invalid review timestamp")
    queue = pd.DataFrame(json.loads((root / "artifacts/queue.json").read_text()))
    if set(labels.review_id) != set(queue.review_id):
        raise ValueError("IDs do not match the frozen holdout")
    merged = queue.merge(labels, on="review_id", validate="one_to_one")
    predictions = (merged.predicted_sentiment == "positive").astype(int)
    return {
        "rows": len(merged),
        "reviewers_as_submitted": sorted(labels.reviewer.unique().tolist()),
        "identity_verification": "not performed; reviewer attribution is self-reported",
        "sentiment_accuracy": float(
            accuracy_score(merged.sentiment_label.astype(int), predictions)
        ),
        "sentiment_macro_f1": float(
            f1_score(merged.sentiment_label.astype(int), predictions, average="macro")
        ),
        "issue_accuracy": float(accuracy_score(merged.issue_label, merged.category)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    args = parser.parse_args()
    print(json.dumps(evaluate_reviews(args.labels), indent=2))
