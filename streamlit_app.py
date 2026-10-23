"""Read-only feedback evidence dashboard from frozen evaluation artifacts."""

from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
NAVY, ORANGE, WHITE = "#0B1F33", "#FF5A1F", "#FFFFFF"
st.set_page_config(
    page_title="FeedbackLens | Evidence workbench", page_icon="📋", layout="wide"
)


st.html(
    """<style>
[data-testid="stMetric"]{background:#13304D;border-top:3px solid #FF5A1F;border-radius:8px;padding:16px}
[data-testid="stAlert"]{background:#13304D;color:#FFFFFF;border-left:3px solid #FF5A1F}
[data-testid="stAlert"] p,[data-testid="stAlert"] svg{color:#FFFFFF!important;fill:#FFFFFF}
</style>"""
)


def chart(fig):
    fig.update_layout(
        template="plotly_dark",
        title_font_size=20,
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font_color=WHITE,
        colorway=[ORANGE, WHITE],
        margin=dict(l=12, r=12, t=66, b=24),
        height=360,
    )
    st.plotly_chart(
        fig, use_container_width=True, theme=None, config={"displayModeBar": False}
    )


@st.cache_data
def artifacts():
    return (
        json.loads((ROOT / "artifacts/results.json").read_text()),
        pd.DataFrame(json.loads((ROOT / "artifacts/queue.json").read_text())),
    )


st.caption("FEEDBACKLENS / EVIDENCE BEFORE AUTOMATION")
st.title("Listen at scale. Keep judgment visible.")
st.write("One Amazon sentence corpus. One issue taxonomy. Explicit abstention.")
st.write(
    "Research prototype. Sentiment is measured against publisher-provided manual labels. New human issue labels: zero. Issue proposals are unvalidated keyword rules, not LLM extraction."
)
try:
    result, queue = artifacts()
except (OSError, ValueError):
    st.error("Artifacts unavailable. Run the documented pipeline first.")
    st.stop()
measured = result["models"][result["selected_model"]]["holdout"]
policy = result["review_policy"]
cols = st.columns(4)
for col, title, value in zip(
    cols,
    [
        "Unique sentences",
        "Holdout sentiment accuracy",
        "Holdout macro-F1",
        "New human issue labels",
    ],
    [
        f"{result['clean_rows']:,}",
        f"{100 * measured['accuracy']:.1f}%",
        f"{measured['macro_f1']:.3f}",
        "0",
    ],
):
    col.metric(title, value)
st.caption(
    "Frozen 200-row holdout: 159 correct sentiment predictions. Selected by validation macro-F1, not holdout. Curated binary sentiment has no neutral benchmark."
)
left, right = st.columns(2)
with left:
    comparison = pd.DataFrame(
        [
            {
                "Model": name.replace("_", " "),
                "Holdout accuracy (%)": values["holdout"]["accuracy"] * 100,
            }
            for name, values in result["models"].items()
        ]
    )
    chart(
        px.bar(
            comparison,
            x="Model",
            y="Holdout accuracy (%)",
            range_y=[0, 100],
            title="Holdout sentiment<br>accuracy: 79.5%",
            color_discrete_sequence=[ORANGE],
        )
    )
with right:
    cm = measured["confusion_matrix"]
    chart(
        px.imshow(
            cm,
            x=["Predicted negative", "Predicted positive"],
            y=["Reference negative", "Reference positive"],
            text_auto=True,
            title="Errors remain in<br>both directions",
            color_continuous_scale=[NAVY, ORANGE],
            aspect="auto",
        )
    )

left, right = st.columns(2)
with left:
    review = pd.DataFrame(
        {
            "Decision": ["Accepted by sentiment threshold", "Sent to sentiment review"],
            "Holdout rows": [
                policy["holdout_accepted"],
                200 - policy["holdout_accepted"],
            ],
        }
    )
    chart(
        px.bar(
            review,
            x="Holdout rows",
            y="Decision",
            orientation="h",
            title="Confidence rule accepts<br>70% of holdout",
            color_discrete_sequence=[ORANGE],
            range_x=[0, 200],
        )
    )
    st.caption(
        f"Threshold {policy['threshold']:.2f}, chosen on validation. Accepted holdout accuracy {100*policy['holdout_accepted_accuracy']:.2f}%, below the 90% validation target. Confidence is not a guarantee. All issue proposals still need review."
    )
with right:
    counts = (
        queue.category.value_counts()
        .rename_axis("Issue proposal")
        .reset_index(name="Holdout rows")
    )
    chart(
        px.bar(
            counts,
            x="Holdout rows",
            y="Issue proposal",
            orientation="h",
            title="Issue proposals need<br>independent review",
            color_discrete_sequence=[ORANGE],
        )
    )
    st.caption(
        "These are keyword-rule outputs, not validated issue prevalence, severity or financial risk."
    )

st.subheader("Review queue")
a, b, c = st.columns(3)
with a:
    sentiment = st.selectbox("Predicted sentiment", ["All", "negative", "positive"])
with b:
    topic = st.selectbox("Topic", ["All"] + [str(x["id"]) for x in result["topics"]])
with c:
    flagged = st.checkbox("Only sentiment flagged for review", value=True)
filtered = queue.copy()
if sentiment != "All":
    filtered = filtered[filtered.predicted_sentiment == sentiment]
if topic != "All":
    filtered = filtered[filtered.topic == int(topic)]
if flagged:
    filtered = filtered[filtered.sentiment_needs_review]
filtered = filtered.sort_values("confidence")
st.caption(
    f"{len(filtered)} filtered rows. Lowest confidence first. No predictions are labeled as human-approved."
)
st.dataframe(
    filtered[
        [
            "review_id",
            "text",
            "predicted_sentiment",
            "confidence",
            "category",
            "topic",
            "review_status",
        ]
    ],
    hide_index=True,
    width="stretch",
)
st.download_button(
    "Download filtered prediction queue",
    filtered[
        [
            "review_id",
            "text",
            "predicted_sentiment",
            "confidence",
            "category",
            "topic",
            "review_status",
        ]
    ].to_csv(index=False),
    "feedback-predictions.csv",
    "text/csv",
)
with st.expander("Topic keywords: exploratory, not validated taxonomy"):
    for item in result["topics"]:
        st.write(f"Topic {item['id']}: {', '.join(item['keywords'])}")
with st.expander("Independent human-review workflow and limits"):
    st.write(
        "The existing blind HTML annotation page is served separately with python -m http.server. It loads only holdout IDs/texts, not predictions or reference labels. An actual reviewer can export labels with self-reported attribution. The evaluation script checks completeness but cannot authenticate experts. No new human issue labels have been collected."
    )
    st.download_button(
        "Download blind 200-row review template",
        (ROOT / "artifacts/blind_review_template.csv").read_bytes(),
        "blind-review-template.csv",
        "text/csv",
    )
    st.write(
        "No timestamps or revenue exist: no trend, monetary risk or savings claim. TF-IDF sentiment, NMF and keyword rules run locally on CPU. No BERTopic or LLM was used."
    )
    st.link_button(
        "UCI source and citation",
        "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
    )
st.caption(
    "Streamlit + Plotly Express / one-page local dashboard / no public deployment"
)
