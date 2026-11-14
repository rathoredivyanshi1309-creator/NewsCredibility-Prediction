# Smart Fake News Detection System

## Project Overview

The Smart Fake News Detection System is a machine learning project
that uses Natural Language Processing (NLP) and Python to classify
news articles as Fake or Real based on their textual content.

The project uses TF-IDF vectorization and compares two machine
learning algorithms:
- Logistic Regression
- Multinomial Naive Bayes

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF
- Streamlit
- Joblib

## Dataset

The project uses the Fake and Real News Dataset by Clément Bisaillon.

Kaggle dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Download the dataset and place Fake.csv and True.csv inside the data folder.

The original CSV files are not included in this GitHub repository because
of their size.

## Project Structure

```text
Smart_Fake_News_Detection/
|
|-- notebooks/\n|   |-- fake_news_analysis.ipynb\n|\n|-- data/
|   |-- Fake.csv
|   |-- True.csv
|
|-- models/
|   |-- logistic_regression_model.joblib
|   |-- naive_bayes_model.joblib
|   |-- tfidf_vectorizer.joblib
|
|-- app/
|   |-- streamlit_app.py
|
|-- outputs/
|   |-- visualizations/
|   |-- evaluation_results/
|
|-- requirements.txt
|-- README.md
|-- .gitignore
```
