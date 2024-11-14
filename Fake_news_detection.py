#pip install pandas numpy scikit-learn nltk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(max_iter=1000)
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub('[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        # Combine title and text for better feature extraction
        df['content'] = df['title'] + ' ' + df['text']
        
        # Preprocess the content
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        return df
    
    def train(self, X_train, y_train):
        # Transform text data into TF-IDF features
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train the classifier
        self.classifier.fit(X_train_vectorized, y_train)
    
    def predict(self, X_test):
        # Transform test data
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Make predictions
        return self.classifier.predict(X_test_vectorized)
    
    def evaluate(self, X_test, y_test):
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Generate detailed classification report
        report = classification_report(y_test, predictions)
        
        return accuracy, report

def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Load your dataset (you'll need to provide the path to your data)
    # Example dataset structure:
    # - title: news article title
    # - text: news article content
    # - label: 0 for real, 1 for fake
    df = pd.read_csv('path_to_your_dataset.csv')
    
    # Initialize the detector
    detector = FakeNewsDetector()
    
    # Prepare the data
    df = detector.prepare_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'],
        df['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Train the model
    detector.train(X_train, y_train)
    
    # Evaluate the model
    accuracy, report = detector.evaluate(X_test, y_test)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    return detector

if __name__ == "__main__":
    main()
# Example usage:
# detector = FakeNewsDetector()
# detector.train(X_train, y_train)
# predictions = detector.predict(X_test)