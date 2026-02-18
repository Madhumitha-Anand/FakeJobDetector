import streamlit as st
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Job Detector")
st.write("This app predicts whether a job posting is **Fake** or **Real** using Machine Learning.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    # model.pkl contains (model, accuracy)
    model, accuracy = pickle.load(open("model.pkl", "rb"))
    return model, accuracy

model, accuracy = load_model()

st.info(f"📊 Model Accuracy: **{round(accuracy * 100, 2)}%**")

# -----------------------------
# User input
# -----------------------------
st.subheader("📄 Paste Job Description")

job_text = st.text_area(
    "Enter the job posting text below:",
    height=200,
    placeholder="Paste the full job description here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Job Authenticity"):
    if job_text.strip() == "":
        st.warning("⚠️ Please enter a job description.")
    else:
        # Predict probabilities
        proba = model.predict_proba([job_text])[0]

        real_prob = proba[0]
        fake_prob = proba[1]

        st.subheader("🧪 Prediction Result")

        # Decision logic (safer threshold)
        if fake_prob >= 0.6:
            st.error("🚨 This job posting is likely **FAKE**")
        else:
            st.success("✅ This job posting appears **REAL**")

        # Show confidence
        st.write("### 🔎 Confidence Scores")
        st.progress(fake_prob)
        st.write(f"**Fake:** {round(fake_prob * 100, 2)}%")
        st.write(f"**Real:** {round(real_prob * 100, 2)}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Always verify jobs manually.")
