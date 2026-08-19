# NLP-Enriched News Intelligence

## Overview
The goal of this project is to build an NLP-enriched News Intelligence platform. News analysis is a trending and important topic. The analysts get their information from the news and the amount of available information is limitless. Having a platform that helps to detect the relevant information is definitely valuable.

The platform connects to a news data source, detects the entities, detects the topic of the article, analyses the sentiment and performs a scandal detection analysis.

## Role Play
You are a Natural Language Processing (NLP) specialist at a media-intelligence startup. Analysts are drowning in the daily volume of news, and your task is to build a platform that reads scraped news articles, identifies the companies mentioned, classifies each article's topic, gauges its sentiment, and flags articles that may describe an environmental scandal involving those companies.

## Learning Objectives
By the end of this project, you will be able to:
- Scrape news articles from a website and store them in a structured form.
- Detect organizations in text using named entity recognition (SpaCy).
- Train a topic classifier on a labelled news dataset and check it for overfitting with learning curves.
- Apply a pre-trained sentiment model (NLTK) to score article sentiment.
- Use word embeddings and a distance metric to flag articles likely to describe an environmental scandal.
- Assemble these steps into a single NLP enrichment pipeline.

## Instructions

### Scraper
News data source:
Find a news website that is easy to scrape. I could have chosen the website, but news websites change their scraping policy frequently.
Store the following information either in one file per day or in a SQL database: unique ID, URL, date, headline, body of the article.
Use data from the last week otherwise the volume may be too high.
There should be at least 300 articles stored in your file system or SQL database.

### NLP engine
In production architectures, the NLP engine delivers a live output based on the news that are delivered in a live stream data by the scraper. However, it required advanced Python skills which is not a prerequisite for the AI branch. To simplify this step the scraper and the NLP engine are used independently in the project. The scraper fetches the news and store them in the data structure (either the file system or the SQL database) and then, the NLP engine runs on the stored data.

1. **Entities detection**: The goal is to detect all the entities in the document (headline and body). The type of entity we focus on is ORG. This corresponds to companies and organizations. This information should be stored. Detect all companies using SpaCy NER on the body of the text.
2. **Topic detection**: The goal is to detect what topic the article covers: Tech, Sport, Business, Entertainment or Politics. To do so, a labelled dataset is provided. From this dataset, build a classifier that learns to detect the right topic in the article. Save the training process to a Python file. Save the model as `topic_classifier.pkl`. Save the plot of learning curves (`learning_curves.png`) in results.
3. **Sentiment analysis**: The goal is to detect the sentiment (positive, negative or neutral) of the news articles. To do so, use a pre-trained sentiment model (NLTK).
4. **Scandal detection**: The goal is to detect environmental disaster for the detected companies. Define keywords that correspond to environmental disaster that may be caused by companies. Compute the embeddings of the keywords. Compute the distance between the embeddings of the keywords and all sentences that contain an entity. Save a metric to unify all the distances calculated per article. Flag the top 10 articles.
5. **Source analysis (optional)**: The goal is to show insights about the news' source you scraped.

## Submission Structure
The expected structure of the project is:
```text
project
├── data
│   └── ...
├── nlp_enriched_news.py
├── requirements.txt
├── README.md
├── results
│   ├── training_model.py
│   ├── topic_classifier.pkl
│   ├── enhanced_news.csv
│   └── learning_curves.png
└── scraper_news.py
```

## Deliverable
Run the scraper until it fetches at least 300 articles:
```bash
python scraper_news.py
```
Run on these 300 articles the NLP engine. The script `nlp_enriched_news.py` should save a DataFrame with the specified structure and store the result in a csv file, `enhanced_news.csv`.
```bash
python nlp_enriched_news.py
```

---

## Results & Methodology

### Scandal Detection Methodology
**Embeddings Chosen:** 
We use the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. This is a highly efficient pre-trained model for mapping sentences to a 384-dimensional dense vector space. It allows us to capture the *semantic meaning* of an environmental disaster rather than relying strictly on exact keyword matches, which significantly reduces false negatives.

**Distance/Similarity Chosen:** 
We use **Cosine Similarity** to compute the distance between the disaster keyword embeddings and the article sentence embeddings. Cosine similarity measures the angle between vectors rather than their magnitude. This is ideal for text analysis because it accurately determines if a sentence is semantically pointing in the same direction as a "scandal," regardless of how long or short the sentence is.