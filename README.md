# IMDB Sentiment Analysis and Classification Pipeline

This project is an end-to-end sentiment analysis pipeline that processes the IMDB dataset for sentiment classification. It includes data exploration, preprocessing, visualization, and the implementation of a Naive Bayes classification model. The trained pipeline is saved for future use.

## Features

- **Data Exploration**: Load and inspect the IMDB dataset for sentiment distribution and missing values.
- **Data Visualization**: Generate sentiment distribution plots and word clouds for positive and negative reviews.
- **Text Preprocessing**: Clean and preprocess text data using steps like:
  - Case normalization
  - Removal of digits, punctuation, and HTML tags
  - Stopword removal
  - Lemmatization
  - Expansion of contractions
- **Model Building**: Use a Naive Bayes classifier within a Scikit-learn pipeline for sentiment classification.
- **Model Evaluation**: Evaluate the model with a confusion matrix and classification report.
- **Model Persistence**: Save the trained model for later use using `joblib`.

## Dataset

The dataset used is the **IMDB Dataset.csv**, which contains two columns:
- `review`: The text of the review.
- `sentiment`: The label indicating the sentiment (either "positive" or "negative").

## Prerequisites

Ensure you have the following libraries installed:
- pandas
- matplotlib
- seaborn
- wordcloud
- nltk
- sklearn
- clean-text
- joblib
- contractions

Install required libraries using:
```bash
pip install pandas matplotlib seaborn wordcloud nltk scikit-learn clean-text contractions joblib
