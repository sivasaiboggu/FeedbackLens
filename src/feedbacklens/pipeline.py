"""Offline feedback intelligence with explicit evidence and annotation provenance."""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = {
    "battery_power": r"\bbatter(?:y|ies)\b|\bcharg(?:e|er|ing)\b|\bpower\b",
    "audio_calls": r"\bsound\b|\bvolume\b|\bhear\b|\bspeaker\b|\bcall\b|\bmicrophone\b|\baudio\b",
    "fit_usability": r"\bfit\b|\bcomfort\w*\b|\bdesign\b|\bbutton\b|\bear\b|\bwear\b",
    "durability_quality": r"\bbrok\w*\b|\bquality\b|\bdurab\w*\b|\bplastic\b|\bdefect\w*\b|\bstopped\b",
    "delivery_service": r"\bship\w*\b|\bdeliver\w*\b|\bservice\b|\brefund\b|\breturn\w*\b",
    "price_value": r"\bprice\b|\bcheap\b|\bworth\b|\bmoney\b|\bexpensive\b",
}


def issue_evidence(text):
    matches = [
        (name, re.search(pattern, text, re.I)) for name, pattern in TAXONOMY.items()
    ]
    found = [(name, m.group()) for name, m in matches if m]
    return {
        "category": found[0][0] if len(found) == 1 else "needs_review",
        "candidate_categories": [n for n, _ in found],
        "evidence_spans": [s for _, s in found],
        "method": "transparent_keyword_rules_not_llm",
        "validated": False,
    }


def clean_data(root=ROOT):
    rows = []
    for index, line in enumerate((root / "data/amazon.tsv").read_text().splitlines()):
        text, label = line.rsplit("\t", 1)
        rows.append({"review_id": index + 1, "text": text, "sentiment": int(label)})
    source = pd.DataFrame(rows)
    conflicts = source.groupby(source.text.str.lower().str.strip()).sentiment.nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting sentiment labels for duplicate text")
    db = sqlite3.connect(root / "data/feedback.db")
    pd.DataFrame(rows).to_sql("raw", db, if_exists="replace", index=False)
    db.execute("DROP TABLE IF EXISTS reviews")
    db.executescript((root / "sql/clean.sql").read_text())
    data = pd.read_sql("SELECT * FROM reviews ORDER BY review_id", db)
    db.close()
    return data


def split_data(data):
    indices = np.arange(len(data))
    trainval, test = train_test_split(
        indices, test_size=200, stratify=data.sentiment, random_state=42
    )
    train, val = train_test_split(
        trainval, test_size=200, stratify=data.iloc[trainval].sentiment, random_state=43
    )
    return train, val, test


def score(y, p):
    predicted = p >= 0.5
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def run(root=ROOT):
    data = clean_data(root)
    train, val, test = split_data(data)
    y = data.sentiment
    candidates = {
        "tfidf_logistic": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=2, max_iter=1500, random_state=42),
        ),
        "tfidf_naive_bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            MultinomialNB(alpha=0.5),
        ),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(data.iloc[train].text, y.iloc[train])
        scores[name] = {
            "validation": score(
                y.iloc[val], model.predict_proba(data.iloc[val].text)[:, 1]
            )
        }
    winner = max(scores, key=lambda n: scores[n]["validation"]["macro_f1"])
    model = candidates[winner]
    pv = model.predict_proba(data.iloc[val].text).max(axis=1)
    vpred = model.predict(data.iloc[val].text)
    # Choose a review threshold using validation only. Require 40 accepted rows
    # so a tiny perfect subset cannot masquerade as reliable automation.
    viable = []
    for threshold in np.linspace(0.5, 0.95, 46):
        mask = pv >= threshold
        if (
            mask.sum() >= 40
            and accuracy_score(y.iloc[val].to_numpy()[mask], vpred[mask]) >= 0.9
        ):
            viable.append(float(threshold))
    threshold = min(viable) if viable else 1.0
    for name, candidate in candidates.items():
        scores[name]["holdout"] = score(
            y.iloc[test], candidate.predict_proba(data.iloc[test].text)[:, 1]
        )
    prob = model.predict_proba(data.iloc[test].text)[:, 1]
    confidence = np.maximum(prob, 1 - prob)
    accepted = confidence >= threshold
    selected_accuracy = (
        float(
            accuracy_score(y.iloc[test].to_numpy()[accepted], (prob >= 0.5)[accepted])
        )
        if accepted.any()
        else None
    )
    # Topic vocabulary and factors fit only training text. No topic quality score
    # is fabricated; topic labels are keyword lists for analyst interpretation.
    vector = TfidfVectorizer(
        stop_words="english", min_df=3, max_df=0.9, max_features=1500
    )
    matrix = vector.fit_transform(data.iloc[train].text)
    nmf = NMF(n_components=6, init="nndsvda", random_state=42, max_iter=500).fit(matrix)
    names = vector.get_feature_names_out()
    topics = [
        {"id": i, "keywords": names[np.argsort(component)[-7:][::-1]].tolist()}
        for i, component in enumerate(nmf.components_)
    ]
    weights = nmf.transform(vector.transform(data.iloc[test].text))
    queue = []
    for j, (_, row) in enumerate(data.iloc[test].iterrows()):
        issue = issue_evidence(row.text)
        queue.append(
            {
                "review_id": int(row.review_id),
                "text": row.text,
                "predicted_sentiment": "positive" if prob[j] >= 0.5 else "negative",
                "confidence": float(confidence[j]),
                "sentiment_needs_review": bool(not accepted[j]),
                "topic": int(weights[j].argmax()) if weights[j].sum() > 0 else None,
                **issue,
                "review_status": "unreviewed",
                "human_label": None,
                "reviewer": None,
                "priority": "review_negative" if prob[j] < 0.5 else "standard",
                "priority_method": "predicted sentiment, not validated severity or financial risk",
            }
        )
    result = {
        "dataset": "UCI Amazon cells subset only",
        "source_rows": 1000,
        "clean_rows": len(data),
        "duplicates_removed": 1000 - len(data),
        "raw_sha256": hashlib.sha256(
            (root / "data/amazon.tsv").read_bytes()
        ).hexdigest(),
        "splits": {"train": len(train), "validation": len(val), "holdout": len(test)},
        "selected_model": winner,
        "selection_metric": "validation macro F1",
        "models": scores,
        "reference_labels": "publisher-provided manual sentiment labels, not new labels from Sivasai or an expert panel",
        "human_issue_labels_collected": 0,
        "issue_accuracy": None,
        "review_policy": {
            "threshold": threshold,
            "threshold_source": "validation only; target >=0.90 accuracy with >=40 accepted validation examples",
            "holdout_accepted": int(accepted.sum()),
            "holdout_coverage": float(accepted.mean()),
            "holdout_accepted_accuracy": selected_accuracy,
        },
        "topics": topics,
        "limitations": [
            "No timestamps: no temporal trend",
            "No revenue/exposure data: no money or risk estimate",
            "Binary curated sentiment only; no neutral benchmark",
            "No new expert validation; issue rules are not an LLM",
            "Review selection is not a guarantee of accuracy",
        ],
        "baseline_holdout_accuracy": float(
            accuracy_score(
                y.iloc[test], np.full(len(test), int(y.iloc[train].mean() >= 0.5))
            )
        ),
    }
    (root / "artifacts").mkdir(exist_ok=True)
    for name, value in [("results", result), ("queue", queue)]:
        (root / f"artifacts/{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    data.iloc[test][["review_id", "text"]].to_json(
        root / "artifacts/blind_rows.json", orient="records", indent=2
    )
    labels = data.iloc[test][["review_id", "sentiment"]].copy()
    labels["label_source"] = "publisher_manual_sentiment"
    labels.to_csv(root / "artifacts/reference_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "review_id": q["review_id"],
                "text": q["text"],
                "sentiment_label": "",
                "issue_label": "",
                "reviewer": "",
                "reviewed_at": "",
            }
            for q in queue
        ]
    ).to_csv(root / "artifacts/blind_review_template.csv", index=False)
    (root / "artifacts/splits.json").write_text(
        json.dumps(
            {
                n: data.iloc[i].review_id.tolist()
                for n, i in [("train", train), ("validation", val), ("holdout", test)]
            }
        )
    )
    joblib.dump(model, root / "artifacts/sentiment.joblib")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
