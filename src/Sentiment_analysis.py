import pandas as pd
from textblob import TextBlob

    
def load_stock_data(ticker, data_dir='data'):
    import yfinance as yf
    import os
    
    file_path = os.path.join(data_dir, f'{ticker}.csv')

    try:
        data = yf.download(ticker, auto_adjust=True)
        data.to_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading {ticker} data: {e}")
        return None

def calculate_sentiment(df):
    if df is None or 'headline' not in df:
        print("No valid data or headline column")
        return None
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    print("Calculated sentiment scores")
    return df

def aggregate_daily_sentiment(df):
    if df is None or 'sentiment' not in df or 'date' not in df or 'stock' not in df:
        print("No valid data for aggregation")
        return None
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    daily_sentiment = df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()
    print("Aggregated daily sentiment scores")
    return daily_sentiment