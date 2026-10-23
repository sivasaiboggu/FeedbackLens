# Reviewer guide

## System boundary

This repository is a local research workbench, not a live customer system.
The Streamlit page reads frozen evidence and computes only explicitly labeled
local views. It does not retrain on startup, contact customers, send feedback,
write to a remote account or require an API key.

```text
data/ + sql/       -> src/feedbacklens/pipeline.py -> artifacts/
                                                   |
                         manifest verification + population checks
                                                   |
                                Streamlit + Plotly / local CSV export
```

- `src/feedbacklens/pipeline.py`: source cleaning, split, fitting and evaluation.
- `src/feedbacklens/evidence.py`: artifact integrity and dashboard read boundary.
- `streamlit_app.py`: presentation, controls and filtered downloads.
- `dashboard.css`: readable original UI styling; font data lives in `assets/fonts/fonts.css`.
- `scripts/verify_artifacts.py`: no-training evidence check.
- `tests/`: model-boundary, workflow and Streamlit AppTest regressions.
- `docs/`: source attribution, protocols and review decisions.

## Decisions worth challenging

| Decision | Reason | Cost / unresolved risk |
|---|---|---|
| Amazon subset only | One clearly scoped, attributable starting domain | Old, small, curated binary sentiment is not representative of live feedback |
| TF-IDF logistic/NB baselines | CPU reproducibility and a transparent comparison | No transformer or LLM comparison has been performed |
| Train-only NMF | Keep exploratory topics out of holdout fitting | Topics are not validated issue labels or stable personas |
| Explicit keyword spans and abstention | Let a reviewer inspect why a rule fired | Negation, sarcasm and multi-issue text remain weak points |
| Separate blind text/ID bundle | Avoid exposing predictions or references to annotators | Self-reported names/timestamps do not authenticate a reviewer |
| Confidence routes sentiment review | Measurable accepted coverage and accuracy | Holdout accepted accuracy misses the 90% validation target; all issue proposals still need review |

## What to verify first

1. Deduplication precedes the fixed 587/200/200 split; vectorizers fit train only.
2. The two model rows and validation-based selection match the artifact.
3. The 140 accepted examples and 124 correct do not become a 90% claim.
4. `blind_rows.json` has only IDs/texts, never predictions or references.
5. Zero new human issue labels means no issue-accuracy or expert-approval claim.

## Reproduction contract

The manifest is a versioned checksum map for the displayed evidence bundle.
Checks run before a bundle enters Streamlit's immutable cache. Every rerun
compares the current bundle fingerprint with that cache key; changed files are
rechecked. Hashes detect accidental corruption or mixed outputs, not malicious
authorship. A person able to replace the files and manifest can replace both.

Use Python 3.10 and the exact pinned dependencies. Training writes generated
artifacts and local SQLite files, so rerun in a disposable clone when reviewing.
Compare the generated artifacts and split IDs to the committed checkpoint.
Never tune to remove an unfavorable holdout result. The dashboard SHA-256 files are an integrity checkpoint, not digital signatures.
A dashboard cache invalidates on artifact byte changes; no periodic source polling
is involved.

Optional browser tests
require local Chrome and a separate Playwright workspace; Python tests do not.

## Before calling this production-ready

External data validation, genuine independent review, privacy/security design,
monitoring, access control, operational ownership and deployment verification
remain separate work. Passing synthetic and AppTest checks does not establish
those properties. No production release is part of this checkpoint.
