
import streamlit as st
import joblib
import re
import os

# Find the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load saved models and vectorizer
logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
)

naive_bayes_model = joblib.load(
    os.path.join(MODEL_DIR, "naive_bayes_model.joblib")
)

tfidf = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
)

# Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Application heading
st.set_page_config(
    page_title="Smart Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Smart Fake News Detection System")

st.write(
    "Enter a news headline and article to see how "
    "a machine learning model classifies the text."
)

st.info(
    "This application uses machine learning and text patterns. "
    "It is not a definitive fact-checking service."
)

# User inputs
title = st.text_input("Enter News Headline")

article = st.text_area(
    "Enter News Article",
    height=250
)

# Model selection
model_choice = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Multinomial Naive Bayes"
    ]
)

# Prediction button
if st.button("Detect News Category"):

    if not title.strip() and not article.strip():
        st.warning("Please enter a headline or news article.")

    else:
        cleaned_title = clean_text(title)
        cleaned_article = clean_text(article)

        content = cleaned_title + " " + cleaned_article

        transformed_text = tfidf.transform([content])

        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(transformed_text)[0]

        else:
            prediction = naive_bayes_model.predict(transformed_text)[0]

        st.subheader("Prediction Result")

        if prediction == "Fake":
            st.error("The model classified this news as: FAKE")

        else:
            st.success("The model classified this news as: REAL")

        st.caption(
            "This is a machine-learning prediction, not confirmation "
            "of whether the news is factually true or false."
        )

st.divider()

st.caption("Smart Fake News Detection System | Python, NLP and Machine Learning")
