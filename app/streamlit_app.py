
import streamlit as st
import joblib
import re
import os

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Page configuration
st.set_page_config(
    page_title="Smart Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load trained models and TF-IDF vectorizer
@st.cache_resource
def load_models():
    logistic_model = joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
    )

    naive_bayes_model = joblib.load(
        os.path.join(MODEL_DIR, "naive_bayes_model.joblib")
    )

    tfidf = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    )

    return logistic_model, naive_bayes_model, tfidf


try:
    logistic_model, naive_bayes_model, tfidf = load_models()
except Exception as e:
    st.error("Unable to load the trained models.")
    st.exception(e)
    st.stop()


# Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.write("Logistic Regression Classes:", logistic_model.classes_)
st.write("Naive Bayes Classes:", naive_bayes_model.classes_)
st.write("TF-IDF Features:", len(tfidf.get_feature_names_out()))
# Application heading
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

        # Select model
        if model_choice == "Logistic Regression":
            selected_model = logistic_model
        else:
            selected_model = naive_bayes_model

        # Generate prediction
        prediction = selected_model.predict(transformed_text)[0]

        st.subheader("Prediction Result")

        if str(prediction).strip().lower() == "fake":
            st.error("The model classified this news as: FAKE")
        elif str(prediction).strip().lower() == "real":
            st.success("The model classified this news as: REAL")
        else:
            st.write(f"Predicted class: {prediction}")

        # Display model probability estimates
        if hasattr(selected_model, "predict_proba"):

            probabilities = selected_model.predict_proba(
                transformed_text
            )[0]

            classes = selected_model.classes_

            st.subheader("Model Probability Estimates")

            for class_name, probability in zip(classes, probabilities):

                percentage = float(probability) * 100

                st.write(
                    f"**{class_name}: {percentage:.2f}%**"
                )

                st.progress(float(probability))

            st.caption(
                "These percentages are model probability estimates, "
                "not verified measures of factual accuracy."
            )

        else:
            st.warning(
                "Probability estimates are not available for this model."
            )

        st.caption(
            "This is a machine-learning prediction, not confirmation "
            "of whether the news is factually true or false."
        )

st.divider()

st.caption(
    "Smart Fake News Detection System | Python, NLP and Machine Learning"
)
