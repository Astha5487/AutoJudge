import streamlit as st
import numpy as np
import re
import os
import joblib
from scipy.sparse import hstack

# ----- Page config -----
st.set_page_config(page_title="AutoJudge", page_icon="⚖️", layout="wide")

# ----- Custom CSS -----
st.markdown("""
<style>
.main { padding: 2rem 3rem; }
.app-title { font-size: 2.2rem; font-weight: 700; }
.metric-card { 
    background: linear-gradient(135deg, #f8f9fb 0%, #e5e7eb 100%);
    padding: 1.5rem; 
    border-radius: 1rem; 
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
}
.result-badge { 
    padding: 0.5rem 1rem; 
    border-radius: 50px; 
    font-weight: 600; 
    font-size: 1.1rem;
}
.easy { background: #dcfce7; color: #166534; }
.medium { background: #fef3c7; color: #92400e; }
.hard { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ----- Load models (joblib) -----
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "logreg_classifier.pkl"))
reg = joblib.load(os.path.join(MODEL_DIR, "gb_regressor.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ----- Text preprocessing -----
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----- Numeric features -----
def extract_numeric_features(text: str) -> dict:
    text_lower = text.lower()
    algo_groups = {
        "dp": ["dp", "dynamic programming", "knapsack", "bitmask"],
        "graph": ["graph", "bfs", "dfs", "dijkstra", "shortest path"],
        "ds": ["segment tree", "fenwick", "heap", "union find"],
        "math": ["modulo", "prime", "gcd", "combinatorics"],
        "geometry": ["geometry", "convex hull"],
        "string": ["string", "palindrome", "kmp", "suffix"],
        "greedy": ["greedy", "two pointers", "sliding window"],
    }
    group_counts = {
        f"{group}_count": np.log1p(sum(text_lower.count(k) for k in keywords))
        for group, keywords in algo_groups.items()
    }

    math_symbols = "+-*/^=<>(){}[]|&!%"
    math_count = sum(text.count(sym) for sym in math_symbols)

    features = {
        "text_length": np.log1p(len(text)),
        "math_symbol_count": np.log1p(math_count),
        "has_constraints": int("≤" in text or "<=" in text_lower or "constraints" in text_lower),
        "has_big_n": int("10^" in text),
        "has_time_limit": int("time limit" in text_lower or "seconds" in text_lower),
        **group_counts,
    }
    return features

# ----- UI -----
st.markdown('<h1 style="text-align: center;">⚖️ AutoJudge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #6b7280;">Predict programming problem difficulty from text</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])

with col1:
    desc = st.text_area("Problem Description", height=200)
    inp_desc = st.text_area("Input Description", height=120)
    out_desc = st.text_area("Output Description", height=120)

    predict_clicked = st.button("🔍 Predict", type="primary")

if predict_clicked and desc:
    combined = f"{desc} {inp_desc} {out_desc}"
    cleaned = clean_text(combined)

    X_tfidf = tfidf.transform([cleaned])
    num_feats = extract_numeric_features(cleaned)
    num_array = np.array([list(num_feats.values())])

    X_final = hstack([X_tfidf, num_array])

    class_pred_enc = clf.predict(X_final)[0]
    class_pred = le.inverse_transform([class_pred_enc])[0]
    score_pred = reg.predict(X_final)[0]

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Difficulty Class**")
        badge_class = f'<span class="result-badge {class_pred.lower()}">{class_pred.capitalize()}</span>'
        st.markdown(badge_class, unsafe_allow_html=True)
        st.markdown("**Score**")
        st.markdown(f'<span class="result-badge">{score_pred:.2f}/10</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "*AutoJudge: TF-IDF + Numeric Features + Logistic Regression + Gradient Boosting*"
)
