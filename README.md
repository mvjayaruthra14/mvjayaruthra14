Hi,I am Jayaruthra Manikandan Vani,
I am done my sentiment_analysis_chatbot using machine learning

Program explained:
1.Imports and Data Loading
   a)Imports: The necessary libraries are imported, including NLTK for natural language processing (NLP) tasks, scikit-learn for machine learning, Pandas for data manipulation, NumPy for numerical operations, and random for random number generation.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import random
   b) Data Loading: The dataset sentiment_analysis_dataset.csv is loaded into a Pandas DataFrame (df), assuming it contains columns named 'text' (containing text data) and 'sentiment' (containing sentiment labels).
df = pd.read_csv('sentiment_analysis_dataset.csv')
2.NLTK Setup
   a)NLTK Downloads: Downloads the necessary NLTK resources (vader_lexicon for sentiment analysis and punkt for tokenization).
nltk.download('vader_lexicon')
nltk.download('punkt')
3.Text Preprocessing Function
   a)Preprocessing Function: Defines a function preprocess_text that tokenizes each text, filters out non-alphabetic tokens, and joins them back into a processed string.
4.Training a Naive Bayes Classifier
   a)Model Training: Initializes a Multinomial Naive Bayes classifier (clf) and fits it to the TF-IDF transformed training data (X_train_tfidf, y_train).
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)
5.ChatBot Class for Sentiment Analysis
   a)ChatBot Class: Defines a ChatBot class capable of analyzing sentiment using NLTK's SentimentIntensityAnalyzer and responding accordingly based on sentiment polarity.
   b)Chat Interaction: Initiates a loop where the ChatBot interacts with the user by analyzing the sentiment of their input (user_input) and providing an appropriate response based on the detected sentiment.


Execution Flow:
1)Setup: Imports necessary libraries, downloads NLTK resources, and loads the dataset.
2)Preprocessing: Cleans and preprocesses the text data.
3)Model Training: Splits data, converts text to TF-IDF features, and trains a Naive Bayes classifier.
4)Evaluation: Tests the classifier's accuracy and performance metrics.
5)ChatBot Interaction: Initializes a ChatBot instance, continuously interacts with the user, analyzes sentiment, and provides appropriate responses based on the sentiment detected.


