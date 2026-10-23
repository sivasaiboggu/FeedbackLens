# Export data dictionary

These are disconnected analysis tables, not a relational operational model.
Fractions stay 0-1 and should be formatted as percentages, not multiplied.

## confusion_matrix.csv

Rows: 4. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| reference | object | Text / do not summarize |
| prediction | object | Text / do not summarize |
| rows | int64 | Whole Number; Max for one-row summary, Sum only for disjoint counts |

## issue_counts.csv

Rows: 7. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| issue_proposal | object | Text / do not summarize |
| rows | int64 | Whole Number; Max for one-row summary, Sum only for disjoint counts |

## kpi_summary.csv

Rows: 1. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| unique_sentences | int64 | Whole Number; Max for one-row summary, Sum only for disjoint counts |
| holdout_sentiment_accuracy | float64 | Decimal Number / Max for precomputed metrics |
| holdout_macro_f1 | float64 | Decimal Number / Max for precomputed metrics |
| new_human_issue_labels | int64 | Whole Number; Max for one-row summary, Sum only for disjoint counts |

## model_comparison.csv

Rows: 2. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| model | object | Text / do not summarize |
| selected_on_validation | bool | True/False |
| accuracy | float64 | Decimal Number / Max for precomputed metrics |
| macro_f1 | float64 | Decimal Number / Max for precomputed metrics |
| brier | float64 | Decimal Number / Max for precomputed metrics |

## review_queue.csv

Rows: 200. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| review_id | int64 | Whole Number / categorical, do not summarize |
| text | object | Text / do not summarize |
| predicted_sentiment | object | Text / do not summarize |
| confidence | float64 | Decimal Number / Max for precomputed metrics |
| sentiment_needs_review | bool | True/False |
| category | object | Text / do not summarize |
| topic | float64 | Whole Number / categorical, do not summarize |
| review_status | object | Text / do not summarize |

## review_routing.csv

Rows: 2. UTF-8 with header.

| Column | CSV-inferred type | Power BI treatment |
|---|---|---|
| decision | object | Text / do not summarize |
| rows | int64 | Whole Number; Max for one-row summary, Sum only for disjoint counts |

## Interpretation limits

IDs are dataset row identifiers, not identities to contact. Risk/confidence are model outputs.
Summary tables report frozen evaluation, not live performance. Any scenario value is hypothetical.
Feedback issue categories and topics are unvalidated proposals, not human labels.
