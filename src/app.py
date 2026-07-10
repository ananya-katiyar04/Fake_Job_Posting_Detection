import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "models", "fake_job_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl"))
st.title("🛡️ Fake Job Posting Detection")
st.caption("Detect whether a job posting is Genuine or Fake using Machine Learning & NLP")
st.write(
    "This application detects whether a job posting is Genuine or Fake using Machine Learning and NLP."
)
st.markdown("---")

company = st.text_input("🏢 Company Name")

location = st.text_input("📍 Job Location")

salary = st.text_input("💰 Salary")

employment = st.selectbox(
    "💼 Employment Type",
    ["Full-Time", "Part-Time", "Internship", "Contract"]
)
job_description = st.text_area(
    "📝 Enter Job Description",
    height=200,
    placeholder="Paste the complete job description here..."
)

if st.button("🔍 Predict Job"):
    text = vectorizer.transform([job_description])

prediction = model.predict(text)
probability = model.predict_proba(text)

confidence = max(probability[0]) * 100

if prediction[0] == 1:
    st.error("❌ Fake Job Posting")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.success("✅ Genuine Job Posting")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.markdown("---")

st.subheader("🛡️ How to Identify Fake Job Postings")

st.info("""
✅ Never pay money for a job.

✅ Verify the company website.

✅ Check the recruiter's email address.

✅ Avoid offers with unrealistic salaries.

✅ Research the company before applying.
""")
st.markdown("---")
# st.caption("👨‍💻 Developed by Ananya Katiyar | Machine Learning Project")