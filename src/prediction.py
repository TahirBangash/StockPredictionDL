import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def predict_and_visualize(model, X_test, scaler, actual_prices):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    predicted_prices = predicted_prices.flatten()

    plt.plot(actual_prices, color='blue', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('figures/stock_prediction.png')
    
    return predicted_prices


def identify_buy_sell_points(predicted_prices):
    buying_points = find_peaks(-predicted_prices)[0]
    selling_points = find_peaks(predicted_prices)[0]
    
    return buying_points, selling_points

def plot_buy_sell_points(predicted_prices, buying_points, selling_points):
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.scatter(buying_points, predicted_prices[buying_points], marker='^', color='green', label='Buy', s=100)
    plt.scatter(selling_points, predicted_prices[selling_points], marker='v', color='red', label='Sell', s=100)
    plt.legend()
    plt.savefig('figures/buyorsell.png')
