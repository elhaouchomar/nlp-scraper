import os
import uuid
import time
from datetime import datetime, timedelta
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_news(target_count=1000):
    feeds = [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://feeds.bbci.co.uk/news/health/rss.xml",
            "https://feeds.bbci.co.uk/news/education/rss.xml",
            "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "https://feeds.bbci.co.uk/news/sport/rss.xml",
            "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "https://feeds.bbci.co.uk/news/business/economy/rss.xml",
            "https://feeds.bbci.co.uk/news/uk/rss.xml",
            "https://feeds.bbci.co.uk/news/england/rss.xml",
            "https://feeds.bbci.co.uk/news/northern_ireland/rss.xml",
            "https://feeds.bbci.co.uk/news/scotland/rss.xml",
            "https://feeds.bbci.co.uk/news/wales/rss.xml",
            "https://feeds.bbci.co.uk/news/business/companies/rss.xml",
            "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
            "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml",
            "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"
            ]
    
    articles = []
    seen_urls = set()
    
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    for feed_url in feeds:
        if len(articles) >= target_count:
            break
        
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            if len(articles) >= target_count:
                break
            
            url = entry.get('link', '')
            if not url or url in seen_urls:
                continue
                
            try:
                published_parsed = entry.get('published_parsed')
                if published_parsed:
                    dt = datetime.fromtimestamp(time.mktime(published_parsed))
                    if dt < seven_days_ago:
                        continue
                
                seen_urls.add(url)
                headline = entry.get('title', '')
                date = entry.get('published', '')
                
                raw_summary = entry.get('summary', '')
                body = BeautifulSoup(raw_summary, "html.parser").get_text(separator=' ')
                if not body:
                    body = headline
                    
                print(f"\n{len(articles)+1}. scraping {url}")
                print("        requesting ...")
                print("        parsing ...")
                
                save_path = os.path.join('data', 'scraped_news.db')
                print(f"        saved in {save_path}")
                
                articles.append({
                    'Unique ID': str(uuid.uuid4()),
                    'URL': url,
                    'Date scraped': date,
                    'Headline': headline,
                    'Body': body
                })

            except Exception as e:
                continue
                
    df = pd.DataFrame(articles)
    
    db_file = os.path.join(DATA_DIR, 'scraped_news.db')
    conn = sqlite3.connect(db_file)
    df.to_sql('scraped_news', conn, if_exists='replace', index=False)
    conn.close()
    
if __name__ == "__main__":
    scrape_news(1000)