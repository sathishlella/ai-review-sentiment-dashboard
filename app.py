
import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple
import plotly.express as px

# Lightweight default: NLTK VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Optional: Transformers (SST-2)
USE_TRANSFORMERS = st.secrets.get("USE_TRANSFORMERS", os.getenv("USE_TRANSFORMERS", "0")) == "1"
TRANSFORMERS_AVAILABLE = False
if USE_TRANSFORMERS:
    try:
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
    except Exception:
        TRANSFORMERS_AVAILABLE = False

# Ensure nltk resources
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

st.set_page_config(
    page_title="AI Review Sentiment Dashboard",
    page_icon="🧠",
    layout="wide",
)

# -------------------- Helpers --------------------
def load_demo_data() -> pd.DataFrame:
    path = "sample_reviews.csv"
    df = pd.read_csv(path)
    # Coerce date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def validate_df(df: pd.DataFrame) -> Tuple[bool, str]:
    cols = [c.lower().strip() for c in df.columns]
    if "review_text" not in cols:
        return False, "CSV must include a 'review_text' column."
    # Normalize column names
    mapping = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return True, ""

def apply_vader(texts: pd.Series) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = texts.fillna("").astype(str).apply(sia.polarity_scores)
    out = pd.DataFrame(scores.tolist())
    # Label
    def to_label(c):
        if c >= 0.05: return "positive"
        if c <= -0.05: return "negative"
        return "neutral"
    out["label"] = out["compound"].apply(to_label)
    return out

def apply_transformers(texts: pd.Series) -> pd.DataFrame:
    # Use a small, fast model
    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = nlp(texts.fillna("").astype(str).tolist(), truncation=True, batch_size=16)
    df = pd.DataFrame(results)
    # Map to consistent columns
    df["compound"] = df["score"].where(df["label"]=="POSITIVE", -df["score"])
    df["neg"] = np.where(df["label"]=="NEGATIVE", df["score"], 0.0)
    df["pos"] = np.where(df["label"]=="POSITIVE", df["score"], 0.0)
    df["neu"] = 1 - (df["pos"] + df["neg"])
    df["label"] = df["label"].str.lower()
    return df[["neg","neu","pos","compound","label"]]

def compute_top_words(df: pd.DataFrame, text_col: str, label_filter: Optional[str]=None, top_n: int=20):
    import re
    from collections import Counter
    sw = set(stopwords.words("english"))
    texts = df[text_col].fillna("").astype(str)
    if label_filter:
        texts = df.loc[df["sent_label"]==label_filter, text_col].fillna("").astype(str)
    words = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower())
        words.extend([w for w in t.split() if len(w)>2 and w not in sw])
    return Counter(words).most_common(top_n)

st.sidebar.title("⚙️ Settings")
model_choice = "Transformers (SST-2)" if (USE_TRANSFORMERS and TRANSFORMERS_AVAILABLE) else "NLTK VADER"
st.sidebar.write(f"Model: **{model_choice}**")
st.sidebar.caption("Set secret USE_TRANSFORMERS=1 to enable Transformers on Streamlit Cloud.")

st.title("🧠 AI Review Sentiment Dashboard")
st.caption("Upload a CSV of reviews, run NLP sentiment analysis, and explore interactive charts.")

with st.expander("📄 CSV format — click to see required columns"):
    st.markdown("""
**Required column**: `review_text`  
**Optional columns**: `date` (YYYY-MM-DD), `rating` (1-5), `product` or `category`  
Example rows are included in the demo data.
    """)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Using demo data.")
    df = load_demo_data()

ok, msg = validate_df(df)
if not ok:
    st.error(msg)
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
text_col = "review_text"
if "rating" in df.columns:
    try:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    except Exception:
        pass

with st.container():
    st.subheader("🔎 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("🧪 Run Sentiment Analysis")
run_btn = st.button("Run Analysis", type="primary")
if run_btn:
    with st.spinner("Running sentiment analysis..."):
        if USE_TRANSFORMERS and TRANSFORMERS_AVAILABLE:
            sent = apply_transformers(df[text_col])
        else:
            sent = apply_vader(df[text_col])
        df_out = df.copy()
        df_out[["sent_neg","sent_neu","sent_pos","sent_compound","sent_label"]] = sent[["neg","neu","pos","compound","label"]]
        st.session_state["results"] = df_out

results = st.session_state.get("results")
if results is not None:
    st.success(f"Analyzed {len(results)} reviews.")
    c1, c2, c3, c4 = st.columns(4)
    pct_pos = (results["sent_label"]=="positive").mean()
    pct_neu = (results["sent_label"]=="neutral").mean()
    pct_neg = (results["sent_label"]=="negative").mean()
    avg_score = results["sent_compound"].mean()
    c1.metric("Positive", f"{pct_pos*100:.1f}%")
    c2.metric("Neutral", f"{pct_neu*100:.1f}%")
    c3.metric("Negative", f"{pct_neg*100:.1f}%")
    c4.metric("Avg Sentiment", f"{avg_score:+.3f}")

    fig_dist = px.histogram(results, x="sent_compound", nbins=30, title="Sentiment Score Distribution (compound)")
    st.plotly_chart(fig_dist, use_container_width=True)

    fig_pie = px.pie(results, names="sent_label", title="Sentiment Labels")
    st.plotly_chart(fig_pie, use_container_width=True)

    if "date" in results.columns and results["date"].notna().any():
        tmp = results.dropna(subset=["date"]).copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        time_series = tmp.groupby(pd.Grouper(key="date", freq="W"))["sent_compound"].mean().reset_index()
        fig_time = px.line(time_series, x="date", y="sent_compound", title="Average Sentiment Over Time (Weekly)")
        st.plotly_chart(fig_time, use_container_width=True)

    if "rating" in results.columns and results["rating"].notna().any():
        fig_rating = px.box(results, x="rating", y="sent_compound", points="all", title="Sentiment by Star Rating")
        st.plotly_chart(fig_rating, use_container_width=True)

    st.subheader("🔤 Top Keywords")
    def compute_top_words(df: pd.DataFrame, text_col: str, label_filter: Optional[str]=None, top_n: int=20):
        import re
        from collections import Counter
        sw = set(stopwords.words("english"))
        texts = df[text_col].fillna("").astype(str)
        if label_filter:
            texts = df.loc[df["sent_label"]==label_filter, text_col].fillna("").astype(str)
        words = []
        for t in texts:
            t = re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower())
            words.extend([w for w in t.split() if len(w)>2 and w not in sw])
        return Counter(words).most_common(top_n)

    kw_col = st.selectbox("View keywords for:", ["all","positive","neutral","negative"], index=0)
    label_filter = None if kw_col=="all" else kw_col
    top_words = compute_top_words(results, text_col=text_col, label_filter=label_filter, top_n=20)
    kw_df = pd.DataFrame(top_words, columns=["word","count"])
    fig_kw = px.bar(kw_df, x="word", y="count", title=f"Top Words ({kw_col})")
    st.plotly_chart(fig_kw, use_container_width=True)

    st.subheader("🧭 Review Explorer")
    filt_label = st.multiselect("Filter by sentiment", ["positive","neutral","negative"], default=[])
    view = results.copy()
    if filt_label:
        view = view[view["sent_label"].isin(filt_label)]
    if "product" in view.columns:
        by_prod = st.multiselect("Filter by product", sorted(view["product"].dropna().unique().tolist()), default=[])
        if by_prod:
            view = view[view["product"].isin(by_prod)]
    st.dataframe(view.head(500), use_container_width=True)

    st.subheader("📥 Download Results")
    @st.cache_data
    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")
    csv_bytes = _to_csv_bytes(results)
    st.download_button("Download CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

else:
    st.info("Click **Run Analysis** to compute sentiment on your dataset.")

st.markdown("---")
st.caption("Built with Streamlit + NLTK (VADER) and optional Transformers (SST-2).")
