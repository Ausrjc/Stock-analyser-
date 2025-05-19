import streamlit as st
from PIL import Image
import pytesseract
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        st.error("No historical data


