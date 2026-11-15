Smart Fake News Detection System

A Machine Learning and Natural Language Processing Approach

1. Project Overview

The Smart Fake News Detection System is a machine learning-based project developed using Python and Natural Language Processing (NLP). It aims to classify news articles as Fake or Real based on their textual content.

The system uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and compares two machine learning algorithms: Logistic Regression and Multinomial Naive Bayes.

A Streamlit-based web application allows users to enter a news headline and article and obtain a prediction with probability estimates.

2. Project Objectives

- To understand and preprocess textual news data.
- To perform Exploratory Data Analysis (EDA).
- To convert text into numerical features using TF-IDF.
- To train and evaluate machine learning classification models.
- To compare the performance of two algorithms.
- To develop an interactive web application for news classification.

3. Technologies and Libraries Used

- Programming Language: Python
- Data Processing: Pandas, NumPy
- Data Visualisation: Matplotlib
- Machine Learning: Scikit-learn
- Natural Language Processing: Text preprocessing and TF-IDF
- Model Saving: Joblib
- Web Application: Streamlit
- Development Environment: Google Colab

4. Dataset Information

The project uses the Fake and Real News Dataset created by Clément Bisaillon.

Kaggle Dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset consists of two CSV files:

- "Fake.csv" – Contains articles labelled as fake.
- "True.csv" – Contains articles labelled as real.

The original dataset files are not included in this GitHub repository because of their size. Users can download the dataset from the Kaggle source and place both CSV files inside the "data" directory.

5. Methodology

The project follows these major steps:

1. Data Collection: Loading the Fake and Real News datasets.
2. Data Preprocessing: Cleaning textual data by removing unnecessary characters, URLs, HTML elements and extra spaces.
3. Exploratory Data Analysis: Examining the dataset and generating visualisations.
4. Feature Extraction: Applying TF-IDF vectorisation with a maximum of 5,000 features and unigram-bigram combinations.
5. Data Splitting: Dividing the dataset into training and testing sets using an 80:20 ratio.
6. Model Training: Training Logistic Regression and Multinomial Naive Bayes classifiers.
7. Model Evaluation: Comparing accuracy, precision, recall, F1-score and confusion matrices.
8. Model Saving: Saving trained models and the TF-IDF vectorizer using Joblib.
9. Application Development: Integrating the trained models into a Streamlit application.

6. Machine Learning Models

Two classification algorithms were implemented:

Logistic Regression

A supervised machine learning classification algorithm used to predict whether a news article belongs to the Fake or Real category.

Multinomial Naive Bayes

A probabilistic classification algorithm commonly used for text classification tasks. It estimates class probabilities based on the extracted textual features.

7. Model Performance

The models were evaluated on a held-out test dataset.

Evaluation Metric| Logistic Regression| Multinomial Naive Bayes
Accuracy| 99.02%| 95.14%
Precision (Weighted)| 99.02%| 95.14%
Recall (Weighted)| 99.02%| 95.14%
F1-Score (Weighted)| 99.02%| 95.14%

The results indicate that Logistic Regression achieved higher performance than Multinomial Naive Bayes on this test dataset.

8. Streamlit Web Application

The project includes an interactive web application developed using Streamlit.

Main Features:

- Input fields for news headlines and article content.
- Selection between Logistic Regression and Multinomial Naive Bayes.
- Fake or Real prediction output.
- Probability estimates for both classes.
- Input validation and user-friendly interface.

The application uses the previously trained and saved machine learning models and TF-IDF vectorizer.

9. Project Structure

NewsCredibility-Prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── (Fake.csv and True.csv - downloaded separately)
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── naive_bayes_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/
│   └── fake_news_analysis.ipynb
│
├── outputs/
│   ├── visualizations/
│   │   ├── label_distribution.png
│   │   ├── article_length_histogram.png
│   │   ├── article_length_comparison.png
│   │   └── subject_distribution.png
│   │
│   └── evaluation_results/
│       ├── model_comparison.csv
│       ├── logistic_regression_confusion_matrix.png
│       └── naive_bayes_confusion_matrix.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── Smart_Fake_News_Detection_Final_Report.pdf

10. Installation and Execution

Follow these steps to run the project locally.

Step 1: Clone the repository

git clone https://github.com/rathoredivyanshi1309-creator/NewsCredibility-Prediction.git

Step 2: Open the project directory

cd NewsCredibility-Prediction

Step 3: Install the required dependencies

pip install -r requirements.txt

Step 4: Run the Streamlit application

streamlit run app/streamlit_app.py

The application will open in a browser using the local URL displayed by Streamlit.

Note: The saved models and vectorizer must be present in the "models" directory. The original CSV files are needed when reproducing the training notebook.

11. Limitations

- The system relies on textual content and does not independently verify news against trusted sources.
- Its predictions depend on the patterns learned from the training dataset.
- Unseen topics, changing language patterns and misleading but realistic content may affect predictions.
- Probability estimates are model outputs and should not be interpreted as guaranteed factual certainty.
- The dataset may not represent all forms of contemporary misinformation.

12. Conclusion

The Smart Fake News Detection System demonstrates the application of Natural Language Processing and machine learning techniques to a text classification problem.

The project successfully implements data preprocessing, TF-IDF feature extraction, model training, performance evaluation and an interactive Streamlit application.

The comparison shows that Logistic Regression performed better than Multinomial Naive Bayes on the selected test dataset. The project provides practical experience in developing an end-to-end machine learning application.

13. References

1. Clément Bisaillon, Fake and Real News Dataset, Kaggle.
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. Scikit-learn Documentation.
   https://scikit-learn.org/

3. Streamlit Documentation.
   https://docs.streamlit.io/

4. Pandas Documentation.
   https://pandas.pydata.org/docs/

5. Python Documentation.
   https://docs.python.org/3/

---

Project: Smart Fake News Detection System
Academic Session: 2026–2027
GitHub Repository: https://github.com/rathoredivyanshi1309-creator/NewsCredibility-Prediction
