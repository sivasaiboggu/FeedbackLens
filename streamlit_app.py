"""Read-only feedback evidence dashboard from frozen evaluation artifacts."""

from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
import altair as alt
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from feedbacklens.evidence import load_dashboard, bundle_fingerprint
CANVAS, ACCENT, TEXT = "#151D21", "#A8C4A2", "#F3F0E7"
st.set_page_config(
    page_title="FeedbackLens | Evidence workbench", page_icon="📋", layout="wide"
)


st.html("<style>" + (ROOT / "assets/fonts/fonts.css").read_text()
        + (ROOT / "dashboard.css").read_text() + "</style>")


def chart(fig):
    """Use the same neutral/accent system across all Plotly views."""
    fig.update_layout(
        template="plotly_dark",
        title_font_size=20,
        paper_bgcolor="#202B30",
        plot_bgcolor="#202B30",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=11),
        title=dict(font=dict(family="DM Sans, sans-serif", size=20)),
        colorway=[ACCENT, TEXT],
        margin=dict(l=12, r=12, t=66, b=24),
        height=340,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    st.plotly_chart(
        fig, use_container_width=True, theme=None, config={"displayModeBar": False}
    )


@st.cache_data
def artifacts(fingerprint):
    """A changed bundle cannot reuse a previously checked cache entry."""
    return load_dashboard(ROOT)


st.html('<div class="masthead"><div class="wordmark"><span class="brandmark">f</span>FeedbackLens</div><div class="meta">Research workbench / 02</div></div>\n<div class="hero"><div><div class="eyebrow">Voice of customer / Amazon sentences</div><h1>More signal.<br><em>Human judgment intact.</em></h1><p>A closer read of customer feedback. Sentiment, emerging themes and a review queue that makes uncertainty visible.</p></div><div class="hero-aside"><div class="meta">The principle</div><strong>Review first.</strong><p>Model confidence is a routing signal, never a substitute for independent judgment.</p></div></div>\n<div class="boundary"><b>Evidence before automation.</b><span>Publisher-manual sentiment labels. Zero new human issue labels. Issue proposals use unvalidated keyword rules, not LLM extraction.</span></div>')
try:
    result, queue = artifacts(bundle_fingerprint(ROOT))
except (OSError, ValueError, KeyError, TypeError):
    st.error("Evidence bundle is missing or inconsistent. See the README verification steps.")
    st.stop()
measured = result["models"][result["selected_model"]]["holdout"]
policy = result["review_policy"]
with st.container(key="kpi-strip"):
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

style_metric_cards(background_color="#202B30", border_color="#35413F",
                   border_left_color="#A8C4A2", border_radius_px=10, box_shadow=False)

st.caption(
    "Frozen 200-row holdout: 159 correct sentiment predictions. Selected by validation macro-F1, not holdout. Curated binary sentiment has no neutral benchmark."
)
st.html('<div class="section-label"><span class="section-num">01 /</span><span>Read the evidence</span></div>')
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
            title="Sentiment accuracy",
            color_discrete_sequence=[ACCENT],
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
            title="Where predictions miss",
            color_continuous_scale=[CANVAS, ACCENT],
            aspect="auto",
        )
    )

left, right = st.columns(2)
with left:
    review = pd.DataFrame(
        {
            "Decision": ["Accepted", "Needs review"],
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
            title="What goes to review",
            color_discrete_sequence=[ACCENT],
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
    issue_chart = alt.Chart(counts).mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
        x=alt.X("Holdout rows:Q", title="Holdout rows"),
        y=alt.Y("Issue proposal:N", title=None, sort="-x"),
        tooltip=["Issue proposal", "Holdout rows"],
    ).properties(title="Proposed issue signals", height=290, padding={"left": 16, "right": 22, "top": 18, "bottom": 14}).configure(
        background="#202B30", font="DM Sans"
    ).configure_axis(labelColor=TEXT, titleColor=TEXT, gridColor="#35413F", domain=False).configure_title(
        color=TEXT, fontSize=20, anchor="start", offset=24
    ).configure_view(strokeOpacity=0)
    st.altair_chart(issue_chart, use_container_width=True, theme=None)

    st.caption(
        "These are keyword-rule outputs, not validated issue prevalence, severity or financial risk."
    )

st.subheader("Review queue")
queue_order = option_menu(None, ["Lowest confidence", "Review ID"],
    icons=["sort-numeric-down", "table"], orientation="horizontal", key="queue-order",
    styles={"container":{"background-color":"#202B30","padding":"4px","border-radius":"8px"},
            "icon":{"color":"#A8C4A2"}, "nav-link":{"font-size":"13px","color":"#F3F0E7","--hover-color":"#35413F"},
            "nav-link-selected":{"background-color":"#35413F","color":"#F3F0E7"}})

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
filtered = filtered.sort_values("review_id" if queue_order == "Review ID" else "confidence")
st.caption(
    f"{len(filtered)} filtered rows. Queue order: {'review ID' if queue_order == 'Review ID' else 'lowest confidence'}. No predictions are labeled as human-approved."
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
st.html('<div class="footer"><span>FeedbackLens / Research edition</span><span>Frozen evidence · Local workspace · No deployment</span></div>')
