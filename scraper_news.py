import os
import uuid
import feedparser
import pandas as pd
from bs4 import BeautifulSoup

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_news(target_count=300):
    # Using multiple RSS feeds to quickly gather 300+ articles
    feeds = [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
    ]
    
    articles = []
    
    for feed_url in feeds:
        if len(articles) >= target_count:
            break
        
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            if len(articles) >= target_count:
                break
            
            try:
                url = entry.get('link', '')
                headline = entry.get('title', '')
                date = entry.get('published', '')
                
                # Clean HTML tags from summary to get the body
                raw_summary = entry.get('summary', '')
                body = BeautifulSoup(raw_summary, "html.parser").get_text(separator=' ')
                if not body:
                    body = headline
                    
                print(f"\n{len(articles)+1}. scraping {url}")
                print("        requesting ...")
                print("        parsing ...")
                
                save_path = os.path.join('data', 'scraped_news.csv')
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
    output_file = os.path.join(DATA_DIR, 'scraped_news.csv')
    df.to_csv(output_file, index=False)
    print(f"\nScraping finished. {len(df)} articles saved to {output_file}")

if __name__ == "__main__":
    scrape_news(300)