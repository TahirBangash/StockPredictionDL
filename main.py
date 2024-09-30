from src.data_preprocessing import fetch_stock_data, preprocess_data, prepare_data
from src.model import create_lstm_model, train_model, save_model, load_model
from src.prediction import predict_and_visualize, identify_buy_sell_points, plot_buy_sell_points

def main():
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    scaled_prices, scaler = preprocess_data(stock_data)
    
    window_size = 60
    X, y = prepare_data(scaled_prices, window_size)
    
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    model = create_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train)
    
    model_path = 'models/lstm_model.h5'
    save_model(model, model_path)
    
    model = load_model(model_path)
    
    actual_prices = stock_data['Close'].values[split_index:]
    predicted_prices = predict_and_visualize(model, X_test, scaler, actual_prices)
    
    buying_points, selling_points = identify_buy_sell_points(predicted_prices)
    plot_buy_sell_points(predicted_prices, buying_points, selling_points)

if __name__ == '__main__':
    main()
