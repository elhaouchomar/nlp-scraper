import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import spacy
import pickle
import string
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    sys.exit("Error: Run 'python -m spacy download en_core_web_sm' first.")

try:
    model_path = os.path.join(RESULTS_DIR, 'topic_classifier.pkl')
    with open(model_path, 'rb') as f:
        topic_classifier = pickle.load(f)
except FileNotFoundError:
    sys.exit("Error: topic_classifier.pkl not found. Run results/training_model.py first.")

sia = SentimentIntensityAnalyzer()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
scandal_keywords = [
    "environmental disaster", "massive pollution", "illegal deforestation", "toxic waste leak"
]
scandal_embeddings = embedder.encode(scandal_keywords, convert_to_tensor=True)

def process_news():
    db_file = os.path.join(DATA_DIR, 'scraped_news.db')
    
    if os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        df = pd.read_sql('SELECT * FROM scraped_news', conn)
        conn.close()
    else:
        sys.exit("Error: scraped_news data not found. Run scraper_news.py first.")
        
    enriched_data = []
    
    for index, row in df.iterrows():
        url = row['URL']
        title = str(row['Headline'])
        body = str(row['Body'])
        full_text = title + ". " + body
        
        print(f"\nEnriching {url}:\n")
        print("Cleaning document ... (optional)\n")
        
        print("---------- Detect entities ----------\n")
        doc = nlp(full_text)
        orgs = list(set([ent.text for ent in doc.ents if ent.label_ == 'ORG']))
        if not orgs:
            orgs_str = "None"
        elif len(orgs) == 1:
            orgs_str = orgs[0]
        else:
            orgs_str = f"{', '.join(orgs[:-1])} and {orgs[-1]}"
        print(f"Detected {len(orgs)} companies which are {orgs_str}")
        
        print("\n---------- Topic detection ----------\n")
        print("Text preprocessing ...\n")
        clean_text = preprocess_text(full_text)
        topic = topic_classifier.predict([clean_text])[0]
        print(f"The topic of the article is: {topic}")
        
        print("\n---------- Sentiment analysis ----------\n")
        print("Text preprocessing ... (optional)")
        sentiment_scores = sia.polarity_scores(full_text)
        compound = sentiment_scores['compound']
        sentiment = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        print(f"The article {title} has a {sentiment} sentiment")
        
        print("\n---------- Scandal detection ----------\n")
        print("Computing embeddings and distance ...\n")
        max_scandal_distance = 0.0
        
        if orgs:
            sentences = [sent.text for sent in doc.sents]
            org_sentences = [s for s in sentences if any(org in s for org in orgs)]
            
            if org_sentences:
                sent_embeddings = embedder.encode(org_sentences, convert_to_tensor=True)
                cosine_scores = util.cos_sim(scandal_embeddings, sent_embeddings)
                max_scandal_distance = cosine_scores.max().item()
                
                if max_scandal_distance > 0.35: 
                    print(f"Environmental scandal detected for {', '.join(orgs)}")
            
        enriched_data.append({
            'Unique ID': row['Unique ID'], 'URL': url, 'Date scraped': row['Date scraped'],
            'Headline': title, 'Body': body, 'Org': orgs, 'Topics': [topic],
            'Sentiment': compound, 'Scandal_distance': max_scandal_distance, 'Top_10': False
        })
        
    enriched_df = pd.DataFrame(enriched_data)
    
    # Flag Top 10
    top_10_indices = enriched_df.nlargest(10, 'Scandal_distance').index
    enriched_df.loc[top_10_indices, 'Top_10'] = True
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, 'enhanced_news.csv')
    enriched_df.to_csv(output_file, index=False)
    print(f"\nFinished processing. Results saved to {output_file}")

if __name__ == "__main__":
    process_news()