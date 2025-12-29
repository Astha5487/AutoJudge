import streamlit as st
import pickle
import re

# ---------- Load Models ----------
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("models/regressor.pkl", "rb") as f:
    regressor = pickle.load(f)

# ---------- Page Config ----------
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="centered"
)

# ---------- UI ----------
st.title("⚖️ AutoJudge")
st.subheader("Programming Problem Difficulty Predictor")

st.markdown(
    "Paste the problem details below. The system will predict "
    "**difficulty level** and **difficulty score** using NLP."
)

st.markdown("### 📝 Problem Details")

problem_desc = st.text_area(
    "Problem Description",
    height=180,
    placeholder="Describe the problem statement..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Describe the output format..."
)

# ---------- Preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\-\*\=<>]", " ", text)  # keep math symbols
    return text

# ---------- Prediction ----------
if st.button("🔍 Predict Difficulty"):
    if not problem_desc.strip():
        st.warning("Please enter the problem description.")
    else:
        full_text = f"{problem_desc} {input_desc} {output_desc}"
        full_text = preprocess_text(full_text)

        # Transform using TF-IDF
        X = tfidf.transform([full_text])

        # Predict
        difficulty = classifier.predict(X)[0]
        score = regressor.predict(X)[0]

        st.markdown("---")
        st.success(f"📘 **Difficulty Level:** {difficulty.capitalize()}")
        st.info(f"📊 **Difficulty Score:** {round(score, 2)}")
