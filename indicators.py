"""Technical indicator calculations used throughout the application."""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD
from ta.volume import AccDistIndexIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Adj Close"].fillna(result["Close"])
    result["Price"] = close
    result["Daily_Change_Pct"] = close.pct_change() * 100
    for days in (5, 10, 20, 50):
        result[f"Return_{days}D_Pct"] = close.pct_change(days) * 100
    for days in (20, 50, 100, 200):
        result[f"SMA_{days}"] = close.rolling(days).mean()
    result["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    result["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    result["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    result["RSI"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    result["MACD"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()
    result["MACD_Hist"] = macd.macd_diff()
    result["MACD_Hist_Improving"] = result["MACD_Hist"] > result["MACD_Hist"].shift(1)
    stochastic = StochasticOscillator(result["High"], result["Low"], close)
    result["Stochastic"] = stochastic.stoch()
    result["ROC"] = ROCIndicator(close, window=12).roc()

    bands = BollingerBands(close, window=20, window_dev=2)
    result["BB_High"] = bands.bollinger_hband()
    result["BB_Mid"] = bands.bollinger_mavg()
    result["BB_Low"] = bands.bollinger_lband()
    result["ATR"] = AverageTrueRange(result["High"], result["Low"], close, window=14).average_true_range()
    result["ATR_Pct"] = result["ATR"] / close.replace(0, np.nan) * 100
    result["Volume_Avg_20"] = result["Volume"].rolling(20).mean()
    result["Volume_Avg_50"] = result["Volume"].rolling(50).mean()
    result["Relative_Volume"] = result["Volume"] / result["Volume_Avg_20"].replace(0, np.nan)
    result["Volume_Trend"] = result["Volume_Avg_20"] / result["Volume_Avg_50"] - 1
    result["Accumulation_Distribution"] = AccDistIndexIndicator(
        result["High"], result["Low"], close, result["Volume"]
    ).acc_dist_index()
    result["High_52W"] = close.rolling(252, min_periods=120).max()
    result["Low_52W"] = close.rolling(252, min_periods=120).min()
    result["Distance_From_52W_High_Pct"] = (close / result["High_52W"] - 1) * 100
    result["Distance_From_52W_Low_Pct"] = (close / result["Low_52W"] - 1) * 100
    result["Volatility_20D_Pct"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    result["Drawdown"] = close / close.cummax() - 1
    result["Max_Drawdown_Pct"] = result["Drawdown"].expanding().min() * 100
    result["Gap_Risk_Pct"] = (
        (result["Open"] / result["Close"].shift(1) - 1).abs().rolling(20).max() * 100
    )
    result["Trend_Strength"] = ADXIndicator(
        result["High"], result["Low"], close, window=14
    ).adx()
    result["Support"] = result["Low"].rolling(20).min()
    result["Resistance"] = result["High"].rolling(20).max()
    result["Breakout_Level"] = result["High"].shift(1).rolling(20).max()
    result["Breakdown_Level"] = result["Low"].shift(1).rolling(20).min()
    result["Volume_Confirms"] = (
        (result["Daily_Change_Pct"] > 0) & (result["Relative_Volume"] > 1)
    ) | ((result["Daily_Change_Pct"] < 0) & (result["Relative_Volume"] > 1))
    result["Trend"] = np.select(
        [
            (close > result["SMA_50"]) & (result["SMA_50"] > result["SMA_200"]),
            (close < result["SMA_50"]) & (result["SMA_50"] < result["SMA_200"]),
        ],
        ["Bullish", "Bearish"],
        default="Mixed",
    )
    return result


def latest_snapshot(data: pd.DataFrame) -> pd.Series:
    required = ["Price", "SMA_50", "RSI", "MACD", "MACD_Signal", "ATR", "Support", "Resistance"]
    complete = data.dropna(subset=required)
    if complete.empty:
        raise ValueError("Not enough history to calculate a complete analysis.")
    return complete.iloc[-1]
