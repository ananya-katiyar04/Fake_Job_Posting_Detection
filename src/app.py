import streamlit as st
import joblib

model = joblib.load("../models/fake_job_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

st.title("Fake Job Posting Detection")
st.write(
    "This application detects whether a job posting is Genuine or Fake using Machine Learning and NLP."
)
st.markdown("---")

job_description = st.text_area("Enter Job Description")

if st.button("Predict"):
    text = vectorizer.transform([job_description])
    prediction = model.predict(text)

    if prediction[0] == 1:
        st.error("⚠ Fake Job Posting")
    else:
        st.success("✅ Genuine Job Posting")