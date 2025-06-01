import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


def load_news_data(file_path):
    # Load dataset.
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded news data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading news data: {e}")
        return None


def calculate_sentiment(df):

    # Calculate sentiment scores for headlines using TextBlob.
    if df is None or 'headline' not in df:
        print("No valid data or headline column")
        return None


    # Compute polarity (-1 to 1, negative to positive)
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    print("Calculated sentiment scores")
    return df



def aggregate_daily_sentiment(df):
    
    # Aggregate sentiment scores by date and stock.
    if df is None or 'sentiment' not in df or 'date' not in df or 'stock' not in df:
        print("No valid data for aggregation")
        return None


    # Convert date to datetime and extract date only
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    # Group by date and stock, compute mean sentiment
    daily_sentiment = df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()
    print("Aggregated daily sentiment scores")
    return daily_sentiment