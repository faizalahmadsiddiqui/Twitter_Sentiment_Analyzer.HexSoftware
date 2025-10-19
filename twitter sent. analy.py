import pandas as pd
from textblob import TextBlob
import re

FILE_PATH = 'test.csv'
TEXT_COLUMN = 'tweet'
DATA_ENCODING = 'latin1'

def clean_text(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('RT', '').strip()
    return text

def get_sentiment_category(polarity):
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    return 'Neutral'

def analyze_sentiment_from_csv(file_path, text_column, encoding):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding=encoding, usecols=['id', text_column])
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Analyzing {len(df)} tweets in the '{text_column}' column...")

    df['clean_tweet'] = df[text_column].apply(clean_text)
    df['polarity_score'] = df['clean_tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = df['polarity_score'].apply(get_sentiment_category)

    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    total_count = len(df)
    avg_polarity = df['polarity_score'].mean()

    print("\n" + "="*45)
    print("FINAL SENTIMENT ANALYSIS SUMMARY")
    print("="*45)
    print(f"Total Tweets Analyzed: {total_count:,}")
    print("-" * 25)
    
    for category in ['Positive', 'Negative', 'Neutral']:
        percentage = sentiment_counts.get(category, 0)
        print(f"{category:<10}: {int(percentage):>3}% ({df['sentiment'].value_counts().get(category, 0):,})")

    print("-" * 25)
    print(f"Overall Polarity Score: {avg_polarity:.4f}")

    if avg_polarity > 0.05:
        print("Conclusion: The overall sentiment of this dataset is POSITIVE. 👍")
    elif avg_polarity < -0.05:
        print("Conclusion: The overall sentiment of this dataset is NEGATIVE. 👎")
    else:
        print("Conclusion: The overall sentiment is NEUTRAL/MIXED. ⚖️")
    print("="*45)

analyze_sentiment_from_csv(FILE_PATH, TEXT_COLUMN, DATA_ENCODING)
