import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_stock_data(ticker, start_date='2024-01-01', end_date='2025-05-30'):
    
    # Load stock price data using yfinance.
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        data.to_csv(f'data/{ticker}_stock.csv')
        print(f"Loaded {ticker} data and saved to data/{ticker}_stock.csv")
        return data
    except Exception as e:
        print(f"Error loading {ticker} data: {e}")
        return None


def calculate_indicators(data, ma_window=20, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):

    #Calculate MA, RSI, and MACD indicators.

    if data is None or data.empty:
        print("No data to calculate indicators")
        return None

    # Moving Average (MA)
    data['MA'] = data['Close'].rolling(window=ma_window).mean()

    # RSI (using Pandas approximation if TA-Lib unavailable)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD (using Pandas)
    ema_fast = data['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=macd_slow, adjust=False).mean()
    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    print("Calculated indicators: MA, RSI, MACD")
    return data


def plot_price_and_ma(data, ticker='Stock', save_path='reports/price_ma_plot.png'):
   
    #Plot stock price with moving average.
   
    if data is None or 'MA' not in data:
        print("No data or MA to plot")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['MA'], label='20-Day MA', color='orange')
    plt.title(f'{ticker} Price with 20-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved price and MA plot to {save_path}")


def plot_rsi(data, ticker='Stock', save_path='reports/rsi_plot.png'):

    # Plot RSI with overbought/oversold lines.

    if data is None or 'RSI' not in data:
        print("No data or RSI to plot")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    plt.title(f'{ticker} RSI (14-Day)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved RSI plot to {save_path}")


def plot_macd(data, ticker='Stock', save_path='reports/macd_plot.png'):
   
    # Plot MACD with signal line and histogram.
   
    if data is None or 'MACD' not in data:
        print("No data or MACD to plot")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['MACD_Signal'], label='Signal Line', color='orange')
    plt.bar(data.index, data['MACD_Hist'], label='Histogram', color='gray', alpha=0.3)
    plt.title(f'{ticker} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved MACD plot to {save_path}")


def calculate_daily_returns(data):
       
    # Calculate daily stock returns.
       
    if data is None or 'Close' not in data:
        print("No valid data for returns")
        return None

        
    # Daily returns = (Close[t] - Close[t-1]) / Close[t-1]
    data['Daily_Return'] = data['Close'].pct_change()
        
    # Reset index to make date a column        
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    print("Calculated daily returns")
    return data[['Date', 'Daily_Return']]