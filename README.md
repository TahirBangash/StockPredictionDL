# StockPredictionDL

## Overview

**StockPredictionDL** is a deep learning tool designed for predicting stock prices using historical stock data. The project employs Long Short-Term Memory (LSTM) networks, capable of learning from sequences such as time series data, which is ideal for predicting stock market trends. The data for this project is sourced from Yahoo Finance using the `yfinance` library.

## Features

- **LSTM Neural Networks**: Utilizes LSTM for accurate time series forecasting.
- **Automated Data Fetching**: Leverages `yfinance` to automatically download historical stock data.
- **Visual Analysis**: Generates plots comparing predicted prices with actual prices and highlights potential buy/sell points.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/TahirBangash/StockPredictionDL.git
   cd StockPredictionDL
   ```

2. **Set up a Python virtual environment**

   To isolate and manage the project's dependencies, set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
   ```

3. **Install the requirements**

   Install all necessary dependencies within the virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Project

To run the project and generate stock price predictions, execute the main script from the command line:

```bash
python main.py
```

This script will automatically download the stock data, process it, train the LSTM model, and display the prediction results as well as potential buy/sell points.

### Customizing Stock Predictions

To customize the predictions for different stocks or time periods, modify the ticker and date range parameters in the `main.py` file:

```python
# Example configuration for Apple Inc.
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'
```

Change the `ticker` to any valid stock symbol available on Yahoo Finance, and adjust the `start_date` and `end_date` to the desired time range for which you want to analyze the stock data.

### Visualizing Predictions

The predictions along with the actual stock prices are plotted using Matplotlib. The graph will be displayed automatically after running the script. The graph includes markers for recommended buy and sell points based on the model's predictions.

### Notes

- Ensure that your internet connection is active as the script needs to fetch data online.
- Adjustments in the LSTM model parameters might be necessary depending on the stock volatility and the amount of historical data available.
