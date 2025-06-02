import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def merge_sentiment_and_returns(sentiment_df, returns_df):
    if sentiment_df is None or returns_df is None:
        print("No valid data for merging")
        return None
    
    # Rename columns for consistency
    sentiment_df = sentiment_df.rename(columns={'date': 'Date', 'stock': 'Stock'})
    returns_df = returns_df.rename(columns={'Date': 'Date'})

    # Merge on Date and Stock
    merged_df = pd.merge(
        sentiment_df, returns_df, how='inner', on='Date')
    print("Merged sentiment and returns")
    return merged_df


def calculate_correlation(merged_df, save_path='reports/correlation_plot.png'):
    if merged_df is None or 'sentiment' not in merged_df or 'Daily_Return' not in merged_df:
        print("No valid data for correlation")
        return None
    
    # Calculate Pearson correlation
    correlation = merged_df['sentiment'].corr(merged_df['Daily_Return'])
    print(f"Pearson correlation: {correlation:.4f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sentiment', y='Daily_Return', data=merged_df)
    plt.title('Sentiment vs. Daily Stock Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved correlation plot to {save_path}")
    return correlation