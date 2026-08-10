import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Download NLTK data
nltk.download('stopwords', quiet=True)

# NLP Preprocessing
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def download_datasets():
    train_url = "https://raw.githubusercontent.com/01-edu/public/master/subjects/ai/nlp-scraper/bbc_news_train.csv"
    test_url = "https://raw.githubusercontent.com/01-edu/public/master/subjects/ai/nlp-scraper/bbc_news_tests.csv"
    
    print("Downloading BBC news datasets...")
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, 'bbc_news_train.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'bbc_news_tests.csv'), index=False)
    return train_df, test_df

def plot_learning_curves(estimator, X, y, filepath):
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curves (Topic Classifier)")
    plt.xlabel("Training Examples")
    plt.ylabel("Score (Accuracy)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    plt.savefig(filepath)
    print(f"Learning curves saved to {filepath}")

if __name__ == "__main__":
    train_df, test_df = download_datasets()
    
    text_col_train = [c for c in train_df.columns if 'text' in c.lower()][0]
    label_col_train = [c for c in train_df.columns if 'category' in c.lower() or 'topic' in c.lower()][0]
    text_col_test = [c for c in test_df.columns if 'text' in c.lower()][0]
    label_col_test = [c for c in test_df.columns if 'category' in c.lower() or 'topic' in c.lower()][0]
    
    print("Preprocessing text...")
    X_train = train_df[text_col_train].apply(preprocess_text)
    y_train = train_df[label_col_train]
    X_test = test_df[text_col_test].apply(preprocess_text)
    y_test = test_df[label_col_test]
    
    print("Training model...")
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_df=0.95, min_df=2)),
        ('classifier', MultinomialNB(alpha=0.1)) 
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, 'topic_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    lc_path = os.path.join(RESULTS_DIR, 'learning_curves.png')
    plot_learning_curves(pipeline, X_train, y_train, lc_path)