import streamlit as st
from PIL import Image
import pytesseract
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def extract_ticker(image):
    text = pytesseract.image_to_string(image)
    return text.strip().split()[0].upper()

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="6mo")

def analyze(hist):
    hist['Timestamp'] = hist.index.astype(np.int64) // 10**9
    X = hist[['Timestamp']]
    y = hist['Close']
    model = LinearRegression().fit(X, y)
    pred_price = model.predict([[X.iloc[-1, 0] + 7 * 86400]])[0]
    st.write(f"### Predicted price in 7 days: **${pred_price:.2f}**")
    hist['Close'].plot(title="Stock Price History", figsize=(10, 4))
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    st.pyplot()

st.title("📷 Stock Analyzer from Image")

uploaded = st.file_uploader("Upload a screenshot or image with a stock ticker", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    ticker = extract_ticker(image)
