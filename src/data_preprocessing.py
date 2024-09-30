import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    prices = stock_data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    return scaled_prices, scaler

def prepare_data(prices, window_size):
    X, y = [], []
    for i in range(window_size, len(prices)):
        X.append(prices[i - window_size:i, 0])
        y.append(prices[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    return np.reshape(X, (X.shape[0], X.shape[1], 1)), y
