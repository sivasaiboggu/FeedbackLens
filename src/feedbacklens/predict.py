"""Local inference; model artifacts must be built locally, never load untrusted pickle."""

import argparse
import json
import joblib
from .pipeline import ROOT, issue_evidence


def predict(text):
    if not text.strip() or len(text) > 10000:
        raise ValueError("Text must contain 1-10000 characters")
    model = joblib.load(ROOT / "artifacts/sentiment.joblib")
    p = float(model.predict_proba([text])[0, 1])
    threshold = json.loads((ROOT / "artifacts/results.json").read_text())[
        "review_policy"
    ]["threshold"]
    return {
        "sentiment": "positive" if p >= 0.5 else "negative",
        "confidence": max(p, 1 - p),
        "sentiment_needs_review": max(p, 1 - p) < threshold,
        **issue_evidence(text),
        "warning": "Binary sentiment trained on curated Amazon sentences; novel inputs are out of distribution.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(predict(args.text), indent=2))
