import os
import re
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from lime.lime_text import LimeTextExplainer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM PRESENTATION STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hinglish Fake News AI | LIME XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #312e81 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        color: #f8fafc;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin: 0 0 0.9rem 0;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .tech-badge {
        background: rgba(99, 102, 241, 0.22);
        border: 1px solid rgba(129, 140, 248, 0.45);
        color: #e0e7ff;
        padding: 0.24rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Verdict Cards */
    .verdict-fake {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        color: #fef2f2;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(127, 29, 29, 0.22);
    }
    .verdict-real {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border-left: 6px solid #22c55e;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        color: #f0fdf4;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(20, 83, 45, 0.22);
    }
    .verdict-label {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-desc {
        font-size: 0.92rem;
        opacity: 0.92;
        margin: 0.3rem 0 0 0;
    }

    /* Metric Stat Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0;
    }
    .stat-lbl {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Highlighted Token Box */
    .highlight-box {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.15rem 1.3rem;
        line-height: 2.05;
        font-size: 1.02rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    .tok-fake {
        background-color: #fee2e2;
        color: #991b1b;
        border-bottom: 2px solid #ef4444;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 700;
        margin: 0 2px;
    }
    .tok-real {
        background-color: #dcfce7;
        color: #166534;
        border-bottom: 2px solid #22c55e;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 700;
        margin: 0 2px;
    }
    .tok-neutral {
        color: #1e293b;
        margin: 0 2px;
    }

    /* White Wrapper so LIME's built-in HTML is always high-contrast on projectors */
    .lime-frame-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. TEXT CLEANING, STOPWORDS & MODEL LOADING
# -----------------------------------------------------------------------------
HINGLISH_FILLERS = set(ENGLISH_STOP_WORDS).union({
    "hai", "hain", "ho", "tha", "thi", "the", "ke", "ka", "ki", "ko",
    "mein", "me", "se", "par", "aur", "bhi", "hi", "ne", "ye", "yeh",
    "wo", "woh", "jo", "jab", "tab", "ab", "ek", "liye", "raha", "rahi",
    "rahe", "gaya", "gayi", "gaye", "kar", "karne", "kiya", "kya", "hua",
    "urltoken",
})

def clean_hinglish_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model():
    candidates = [
        "hinglish_fake_news_v2_model.joblib",
        "hinglish_fake_news_model.joblib",
        "hinglish_ensemble_lime.joblib",
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

@st.cache_data
def load_sample_dataset():
    csv_candidates = [
        "hinglish-fake-news-dataset-final-cleaned-v2_2.csv",
        "hinglish-fake-news-dataset-final-cleaned-v2.csv",
        "hinglish-fake-news-dataset-final-cleaned.csv",
    ]
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            text_col = "clean_text" if "clean_text" in df.columns else "text"
            return df[[text_col, "label"]].rename(columns={text_col: "clean_text"}), path
    return None, None

model, model_path = load_model()
dataset_df, dataset_path = load_sample_dataset()

# -----------------------------------------------------------------------------
# 3. HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🛡️ Hinglish Fake News Detection & LIME Explainability</div>
        <div class="hero-sub">
            Explainable AI (XAI) system for detecting misinformation in Romanized Hindi-English code-mixed news & social forwards.
        </div>
        <div class="badge-row">
            <span class="tech-badge">Word (1–3) + Char (3–5) TF-IDF</span>
            <span class="tech-badge">Calibrated Soft-Voting Ensemble</span>
            <span class="tech-badge">LIME Local Surrogate XAI</span>
            <span class="tech-badge">85.72% Test Accuracy • 0.924 ROC-AUC</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model file not found! Place `hinglish_fake_news_v2_model.joblib` in the same folder as `app.py` and refresh."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Demo & XAI Controls")
    st.success(f"Model Ready:\n`{model_path}`")
    if dataset_path:
        st.info(f"Dataset Linked:\n`{dataset_path}` ({len(dataset_df):,} rows)")

    st.markdown("---")
    num_samples = st.slider(
        "LIME Perturbation Samples",
        min_value=300,
        max_value=1500,
        value=900,
        step=100,
        help="Higher values produce smoother, more stable word weights.",
    )
    top_k = st.slider(
        "Max Highlighted Keywords",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
    )
    hide_fillers = st.toggle(
        "Filter Hinglish Filler Stopwords",
        value=True,
        help="Hides grammatical connectors (hai, ke, ki, mein, the) from the top keyword chart.",
    )

    st.markdown("---")
    st.markdown("**Viva Quick Reference**")
    st.caption(
        "• **0 = Real News** (Green)\n"
        "• **1 = Fake News** (Red)\n"
        "• **LIME** perturbs input words locally and fits a weighted linear surrogate model to explain `predict_proba()`."
    )

# -----------------------------------------------------------------------------
# 5. HELPER FUNCTIONS FOR LIME & VISUALIZATION
# -----------------------------------------------------------------------------
explainer = LimeTextExplainer(
    class_names=["Real News", "Fake News"],
    split_expression=r"\s+",
    bow=True,
    random_state=42,
)

def predict_proba_for_lime(texts):
    cleaned_batch = [clean_hinglish_text(t) for t in texts]
    return model.predict_proba(cleaned_batch)

def get_submodel_breakdown(cleaned_text: str):
    """Extracts individual estimator probabilities from the VotingClassifier."""
    try:
        features = model.named_steps["features"].transform([cleaned_text])
        clf = model.named_steps["clf"]
        name_map = {
            "lr": "Logistic Regression",
            "svm": "Calibrated LinearSVC",
            "cnb": "Complement Naive Bayes",
        }
        rows = []
        for key, est in clf.named_estimators_.items():
            p = est.predict_proba(features)[0]
            rows.append({
                "Ensemble Estimator": name_map.get(key, key.upper()),
                "Real (%)": round(p[0] * 100, 2),
                "Fake (%)": round(p[1] * 100, 2),
                "Vote": "Fake News" if p[1] >= 0.5 else "Real News",
            })
        return pd.DataFrame(rows)
    except Exception:
        return None

def render_highlighted_sentence(cleaned_text: str, weight_map: dict) -> str:
    tokens = cleaned_text.split()
    rendered = []
    for tok in tokens:
        norm = re.sub(r"[^a-z0-9]", "", tok.lower())
        w = weight_map.get(tok, weight_map.get(norm, 0.0))
        safe_tok = html.escape(tok)
        if w > 0.005:
            rendered.append(
                f'<span class="tok-fake" title="Fake weight: {w:+.4f}">{safe_tok}</span>'
            )
        elif w < -0.005:
            rendered.append(
                f'<span class="tok-real" title="Real weight: {w:+.4f}">{safe_tok}</span>'
            )
        else:
            rendered.append(f'<span class="tok-neutral">{safe_tok}</span>')
    return f'<div class="highlight-box">{" ".join(rendered)}</div>'

# -----------------------------------------------------------------------------
# 6. MAIN TABS (LIVE DEMO | BATCH EVALUATOR | PRESENTATION SLIDES)
# -----------------------------------------------------------------------------
tab_live, tab_batch, tab_slides = st.tabs([
    "🔍 Live Detector & LIME XAI",
    "📂 Batch CSV Evaluator",
    "📊 Architecture & Viva Overview",
])

# === TAB 1: LIVE DETECTOR ===
with tab_live:
    PRESETS = {
        "🚨 WhatsApp RBI Hoax": (
            "breaking news rbi ne aaj raat se sabhi 500 ke note band karne ka aadesh jari kiya hai "
            "is message ko turant sabhi groups mein forward karein"
        ),
        "🎁 Free Laptop Clickbait": (
            "sarkar ne sabhi college students ko free laptop dene ki nayi yojana shuru ki hai "
            "abhi niche diye gaye link par click karke apna naam register karein"
        ),
        "🏏 Verified Sports News": (
            "bharat aur australia ke beech test series ka agla mukabla guruwar se khela jayega "
            "jiske liye team india ne apni taiyari poori kar li hai"
        ),
        "🏛️ Policy & Economy Report": (
            "kendriya cabinet ne aaj railway aur highway infrastructure ke vikas ke liye nayi pariyojanaon ko "
            "manzoori de di hai jisse rozgar ke avsar badhenge"
        ),
    }

    if "news_input" not in st.session_state:
        st.session_state["news_input"] = PRESETS["🚨 WhatsApp RBI Hoax"]
        st.session_state["ground_truth"] = None

    st.markdown("**Quick-Load Demo Samples (Click any preset for live presentation):**")
    btn_cols = st.columns(5)
    for idx, (label, sample_txt) in enumerate(PRESETS.items()):
        if btn_cols[idx].button(label, use_container_width=True):
            st.session_state["news_input"] = sample_txt
            st.session_state["ground_truth"] = None

    if dataset_df is not None:
        if btn_cols[4].button("🎲 Random Dataset Row", use_container_width=True):
            row = dataset_df.sample(1).iloc[0]
            st.session_state["news_input"] = row["clean_text"]
            st.session_state["ground_truth"] = int(row["label"])

    user_text = st.text_area(
        "Hinglish News Headline or Full Article:",
        key="news_input",
        height=125,
        placeholder="Type or paste Romanized Hinglish news text here...",
    )

    # Live input telemetry strip
    words_now = user_text.strip().split()
    info_c1, info_c2, _ = st.columns([2, 2, 4])
    info_c1.caption(f"📝 **Words:** {len(words_now)} | **Characters:** {len(user_text)}")
    if st.session_state.get("ground_truth") is not None:
        gt_lbl = "Fake News (1)" if st.session_state["ground_truth"] == 1 else "Real News (0)"
        info_c2.caption(f"🏷️ **Dataset Ground Truth:** `{gt_lbl}`")

    analyze_clicked = st.button("🚀 Analyze & Generate LIME Explanation", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter a Hinglish headline or article first.")
        else:
            cleaned = clean_hinglish_text(user_text)
            probs = predict_proba_for_lime([cleaned])[0]
            pred_idx = int(np.argmax(probs))
            is_fake = pred_idx == 1
            confidence = probs[pred_idx] * 100

            # 1. Verdict Banner
            if is_fake:
                st.markdown(
                    f"""
                    <div class="verdict-fake">
                        <div class="verdict-label">🚨 FAKE NEWS DETECTED ({confidence:.2f}% Confidence)</div>
                        <div class="verdict-desc">
                            The ensemble detected linguistic and subword patterns strongly associated with misinformation or sensational clickbait.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-real">
                        <div class="verdict-label">✅ LIKELY REAL NEWS ({confidence:.2f}% Confidence)</div>
                        <div class="verdict-desc">
                            The ensemble found phrasing and structural patterns consistent with factual reporting.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 2. Probability Metrics & Sub-Model Agreement
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Verdict", "Fake News (1)" if is_fake else "Real News (0)")
            m2.metric("Real News Probability", f"{probs[0] * 100:.2f}%")
            m3.metric("Fake News Probability", f"{probs[1] * 100:.2f}%")

            sub_df = get_submodel_breakdown(cleaned)
            if sub_df is not None:
                with st.expander("🔎 View Individual Ensemble Model Votes (Soft-Voting Breakdown)", expanded=False):
                    st.dataframe(sub_df, use_container_width=True, hide_index=True)

            # 3. Run LIME Explainer
            with st.spinner("Running LIME local perturbations & computing word attributions..."):
                exp = explainer.explain_instance(
                    cleaned,
                    predict_proba_for_lime,
                    num_features=max(top_k + 8, 18),
                    num_samples=num_samples,
                )

                raw_weights = exp.as_list()
                filtered_weights = []
                for w, wt in raw_weights:
                    norm_w = re.sub(r"[^a-z]", "", w.lower())
                    if hide_fillers and norm_w in HINGLISH_FILLERS:
                        continue
                    filtered_weights.append((w, wt))
                filtered_weights = filtered_weights[:top_k]

                weight_lookup = {w: wt for w, wt in filtered_weights}

                # 4. Inline Annotated Text View
                st.subheader("🖍️ Inline Word-Level Attribution (Hover over highlighted words)")
                st.caption("Red = pushes prediction toward Fake News | Green = pushes prediction toward Real News")
                st.markdown(render_highlighted_sentence(cleaned, weight_lookup), unsafe_allow_html=True)

                st.markdown("")

                # 5. Side-by-Side Bar Chart + Interactive LIME HTML Report
                left_col, right_col = st.columns([1.05, 1.35])

                with left_col:
                    st.subheader("📊 Top Influential Tokens")
                    if filtered_weights:
                        chart_df = pd.DataFrame(filtered_weights, columns=["Token", "LIME Weight"])
                        chart_df["Direction"] = chart_df["LIME Weight"].apply(
                            lambda x: "Fake News (+)" if x > 0 else "Real News (-)"
                        )
                        st.bar_chart(
                            chart_df.set_index("Token")["LIME Weight"],
                            color="#ef4444" if is_fake else "#22c55e",
                            use_container_width=True,
                        )
                        st.dataframe(
                            chart_df.style.format({"LIME Weight": "{:+.4f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No non-filler keywords exceeded threshold; toggle off stopword filtering in the sidebar.")

                with right_col:
                    st.subheader("🔬 Official LIME Interactive Report")
                    lime_html = exp.as_html()
                    st.markdown('<div class="lime-frame-card">', unsafe_allow_html=True)
                    components.html(lime_html, height=430, scrolling=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.download_button(
                        label="⬇️ Download LIME HTML Explanation (For Presentation / Report)",
                        data=lime_html,
                        file_name="hinglish_lime_explanation.html",
                        mime="text/html",
                        use_container_width=True,
                    )

# === TAB 2: BATCH EVALUATOR ===
with tab_batch:
    st.subheader("📂 Batch Hinglish News Evaluation")
    st.write(
        "Upload any CSV containing a `clean_text` or `text` column to run batch predictions, or test a random slice of your dataset."
    )

    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])
    batch_source = None

    if uploaded_csv is not None:
        batch_source = pd.read_csv(uploaded_csv)
    elif dataset_df is not None:
        if st.button("Load 25 Random Samples from Project Dataset"):
            batch_source = dataset_df.sample(25, random_state=None).reset_index(drop=True)

    if batch_source is not None:
        col_name = "clean_text" if "clean_text" in batch_source.columns else (
            "text" if "text" in batch_source.columns else batch_source.columns[0]
        )
        texts = batch_source[col_name].astype(str).apply(clean_hinglish_text).tolist()
        probs = model.predict_proba(texts)
        preds = np.argmax(probs, axis=1)

        res_df = batch_source.copy()
        res_df["Predicted_Label"] = ["Fake News (1)" if p == 1 else "Real News (0)" for p in preds]
        res_df["Fake_Prob (%)"] = np.round(probs[:, 1] * 100, 2)
        res_df["Real_Prob (%)"] = np.round(probs[:, 0] * 100, 2)

        b1, b2, b3 = st.columns(3)
        b1.metric("Total Articles Analyzed", len(res_df))
        b2.metric("Predicted Fake News", int((preds == 1).sum()))
        b3.metric("Predicted Real News", int((preds == 0).sum()))

        st.dataframe(res_df, use_container_width=True)

# === TAB 3: PRESENTATION SLIDES OVERVIEW ===
with tab_slides:
    st.subheader("📊 Project Architecture & Benchmark Summary (Viva Ready)")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown('<div class="stat-card"><div class="stat-val">16,843</div><div class="stat-lbl">Deduplicated Rows</div></div>', unsafe_allow_html=True)
    k2.markdown('<div class="stat-card"><div class="stat-val">85.72%</div><div class="stat-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
    k3.markdown('<div class="stat-card"><div class="stat-val">0.8514</div><div class="stat-lbl">Macro F1-Score</div></div>', unsafe_allow_html=True)
    k4.markdown('<div class="stat-card"><div class="stat-val">0.9244</div><div class="stat-lbl">ROC-AUC Score</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### 1. Why Hybrid TF-IDF for Hinglish?")
        st.markdown(
            """
            - **Spelling Inconsistency:** Romanized Hindi lacks standardized spelling (e.g., *sarkar* vs. *sarkaar*, *kyonki* vs. *kyonk*).
            - **Word N-Grams (`1–3` words, 50k features):** Captures multi-word clickbait phrases (*free laptop*, *link par click*, *note band*).
            - **Subword Character N-Grams (`3–5` chars, 65k features):** Captures shared phonetic character roots across spelling variants.
            """
        )

        st.markdown("### 2. Why Soft-Voting Ensemble?")
        st.markdown(
            """
            - Combines **Logistic Regression** (weight 2), **Calibrated LinearSVC** (weight 3), and **Complement Naive Bayes** (weight 1).
            - Produces **smooth, well-calibrated probabilities** via `predict_proba()`, which prevents erratic LIME surrogate fits.
            """
        )

    with c_right:
        st.markdown("### 3. Classification Report (Held-Out 20% Test Set)")
        metrics_table = pd.DataFrame({
            "Class": ["Real News (0)", "Fake News (1)", "Macro Average", "Weighted Average"],
            "Precision": [0.86, 0.86, 0.86, 0.86],
            "Recall": [0.90, 0.79, 0.85, 0.86],
            "F1-Score": [0.88, 0.82, 0.85, 0.86],
            "Support": [1965, 1404, 3369, 3369],
        })
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        st.markdown("### 4. How LIME Explains Predictions")
        st.markdown(
            """
            1. **Local Perturbation:** Randomly drops words from the input Hinglish sentence (`N = 900` variations).
            2. **Black-Box Querying:** Queries the ensemble's `predict_proba()` on every perturbed sentence.
            3. **Surrogate Linear Fit:** Fits a local Ridge Regression weighted by cosine proximity to the original sentence, assigning positive weights to Fake indicators and negative weights to Real indicators.
            """
        )