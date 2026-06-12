"""Calculate technical indicators and create an educational stock signal."""

from typing import Any

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI, and MACD columns to stock price data."""
    result = data.copy()
    close = result["Close"]

    result["SMA_50"] = close.rolling(window=50).mean()
    result["SMA_200"] = close.rolling(window=200).mean()
    result["RSI"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    result["MACD"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()

    return result


def score_row(row: pd.Series) -> tuple[int, list[str]]:
    """Score one row of indicator data and explain each rule."""
    score = 0
    reasons = []

    if row["Close"] > row["SMA_50"]:
        score += 25
        reasons.append("The price is above its 50-day average, showing recent strength.")
    else:
        reasons.append("The price is at or below its 50-day average, showing weaker momentum.")

    if row["SMA_50"] > row["SMA_200"]:
        score += 25
        reasons.append("The 50-day average is above the 200-day average, which is a positive trend.")
    else:
        reasons.append("The 50-day average is at or below the 200-day average, which is not a positive long-term trend.")

    if 40 <= row["RSI"] <= 65:
        score += 25
        reasons.append("RSI is between 40 and 65, a balanced range in this scoring system.")
    elif row["RSI"] > 65:
        reasons.append("RSI is above 65, so the stock may have risen too quickly.")
    else:
        reasons.append("RSI is below 40, which points to weak recent momentum.")

    if row["MACD"] > row["MACD_Signal"]:
        score += 25
        reasons.append("MACD is above its signal line, a sign of positive momentum.")
    else:
        reasons.append("MACD is at or below its signal line, a sign of weaker momentum.")

    return score, reasons


def signal_from_score(score: int) -> str:
    """Turn a score from 0 to 100 into BUY, HOLD, or SELL."""
    if score >= 75:
        return "BUY"
    if score <= 25:
        return "SELL"
    return "HOLD"


def get_latest_signal(data: pd.DataFrame) -> dict[str, Any]:
    """Return the latest score, signal, indicator values, and reasons."""
    needed = ["Close", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal"]
    complete_rows = data.dropna(subset=needed)

    if complete_rows.empty:
        raise ValueError("At least 200 trading days of data are needed to calculate this signal.")

    latest = complete_rows.iloc[-1]
    score, reasons = score_row(latest)

    return {
        "signal": signal_from_score(score),
        "score": score,
        "reasons": reasons,
        "date": complete_rows.index[-1],
        "price": float(latest["Close"]),
        "sma_50": float(latest["SMA_50"]),
        "sma_200": float(latest["SMA_200"]),
        "rsi": float(latest["RSI"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_Signal"]),
    }
