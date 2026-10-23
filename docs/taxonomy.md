# One issue taxonomy, explicit abstention

One primary issue is allowed: battery_power, audio_calls, fit_usability,
durability_quality, delivery_service, price_value. Annotators choose `other`
when none apply and `multiple` when equally important issues coexist. Read the
sentence literally. Do not infer a product defect from negative sentiment
alone. Negated praise and sarcasm may require review.

The initial system proposes keyword evidence, not an LLM extraction. No match
or multiple matching categories maps to `needs_review`. A single match is still
unvalidated and remains available for human review. Proposed severity is NOT
measured; the queue prioritizes predicted negative sentiment only. Topic IDs
come from training-only NMF and are not this taxonomy or validated categories.

## Blind 200-row review protocol

1. Export `artifacts/blind_review_template.csv`. It contains holdout texts and
   IDs, without predictions or publisher reference labels.
2. An actual reviewer records sentiment (0 negative, 1 positive), one issue,
   their name/role and an ISO timestamp. Do not consult model outputs first.
3. A second reviewer should independently adjudicate ambiguous examples.
4. Run `python -m feedbacklens.review path/to/review.csv` after completing all
   rows. It checks schema, ID coverage and attribution before reporting scores.
5. Preserve the review file as evidence only after its author agrees to share
   it. The script cannot authenticate identities or prove expert qualifications.

No new human/expert issue labels have been collected. Publisher manual sentiment
labels provide an independent sentiment benchmark, not validation of our issue
taxonomy or an expert-in-the-loop outcome. AI annotations must be named as AI
annotations, never entered under a fictional human reviewer's identity.
