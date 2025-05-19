import streamlit as st
from PIL import Image
import pytesseract
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # <- This was missing

def extract_ticker(image):
    """Extracts the stock ticker from the uploaded image using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip().split()[0].upper()
    except Exception as e:
        st.error(f"Error extracting ticker: {e}")
        return None

def fetch_data(ticker):
    """Fetches historical stock data for the given ticker."""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="6mo")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def analyze(hist):
    """Analyzes the historical stock data and predicts future prices."""
    if hist is None or hist.empty:
        st.error("No historical data available for analysis.")
        return

    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist['Close'] = hist['Close'].astype(float)

    # Prepare data
    X = np.array(range(len(hist))).reshape(-1, 1)
    y = hist['Close'].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 30 days
    future_days = np.array(range(len(hist), len(hist) + 30)).reshape(-1, 1)
    future_prices = model.predict(future_days)
    future_dates = pd.date_range(start=hist['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(hist['Date'], hist['Close'], label='Historical Prices')
    plt.plot(future_dates, future_prices, label='Predicted Prices', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app layout
st.title("📈 Stock Price Predictor from Image")

uploaded_file = st.file_uploader("Upload an image of a stock ticker", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    ticker = extract_ticker(image)
    if ticker:
        st.write(f"**Extracted Ticker:** `{ticker}`")  # <-- Small but helpful
        hist_data = fetch_data(ticker)
        analyze(hist_data)
    else:
        st.error("No valid ticker extracted. Please upload a clearer image.")


