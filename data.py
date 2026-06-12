"""Download and prepare stock market data."""

import re

import pandas as pd
import streamlit as st
import yfinance as yf


def clean_ticker(ticker: str) -> str:
    """Make a ticker safe to send to Yahoo Finance."""
    cleaned = ticker.strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z0-9.^=-]+", cleaned):
        raise ValueError("Enter a valid ticker, such as AAPL, TSLA, NVDA, or SPY.")
    return cleaned


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download historical daily prices and return a clean DataFrame."""
    symbol = clean_ticker(ticker)

    try:
        data = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    except Exception as error:
        raise ValueError(f"Could not download data for {symbol}. Please try again.") from error

    if data.empty:
        raise ValueError(f"No price data was found for {symbol}. Check the ticker symbol.")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("The downloaded data is missing required price information.")

    data = data[required_columns].copy()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data.dropna(subset=["Close"])
