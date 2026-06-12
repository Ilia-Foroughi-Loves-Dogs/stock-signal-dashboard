"""Technical indicators, scoring, and plain-English signal explanations."""

from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

MIN_HISTORY_ROWS = 200


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators used by the scanner and backtest."""
    result = data.copy()
    close = result["Close"]
    result["Daily_Change_Pct"] = close.pct_change() * 100
    result["SMA_20"] = close.rolling(20).mean()
    result["SMA_50"] = close.rolling(50).mean()
    result["SMA_200"] = close.rolling(200).mean()
    result["RSI"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    result["MACD"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()
    result["Volume_Avg_20"] = result["Volume"].rolling(20).mean()
    result["Relative_Volume"] = result["Volume"] / result["Volume_Avg_20"].replace(0, np.nan)
    result["High_52W"] = close.rolling(252, min_periods=200).max()
    result["Low_52W"] = close.rolling(252, min_periods=200).min()
    result["Distance_From_52W_High_Pct"] = (close / result["High_52W"] - 1) * 100
    result["Distance_From_52W_Low_Pct"] = (close / result["Low_52W"] - 1) * 100
    result["Momentum_20D_Pct"] = close.pct_change(20) * 100
    result["ATR"] = AverageTrueRange(
        high=result["High"], low=result["Low"], close=close, window=14
    ).average_true_range()
    result["ATR_Pct"] = result["ATR"] / close * 100
    result["Trend_Status"] = result.apply(trend_status, axis=1)
    return result


def trend_status(row: pd.Series) -> str:
    """Classify moving-average alignment."""
    values = [row.get("Close"), row.get("SMA_50"), row.get("SMA_200")]
    if any(pd.isna(value) for value in values):
        return "Insufficient history"
    if row["Close"] > row["SMA_50"] > row["SMA_200"]:
        return "Bullish"
    if row["Close"] < row["SMA_50"] < row["SMA_200"]:
        return "Bearish"
    return "Mixed"


def score_row(row: pd.Series, history_rows: int = MIN_HISTORY_ROWS) -> tuple[int, dict[str, int]]:
    """Score one trading day from 0 to 100 by category."""
    close = float(row["Close"])
    volume = float(row.get("Volume", 0) or 0)
    scores = {"Trend": 0, "Momentum": 0, "Volume": 0, "Risk/Reward": 0, "Quality": 0}

    scores["Trend"] += 10 if close > row.get("SMA_50", np.nan) else 0
    scores["Trend"] += 10 if row.get("SMA_50", np.nan) > row.get("SMA_200", np.nan) else 0
    scores["Trend"] += 10 if close > row.get("SMA_200", np.nan) else 0
    scores["Momentum"] += 10 if row.get("MACD", np.nan) > row.get("MACD_Signal", np.nan) else 0
    scores["Momentum"] += 8 if 45 <= row.get("RSI", np.nan) <= 65 else 0
    scores["Momentum"] += 7 if row.get("Momentum_20D_Pct", np.nan) > 0 else 0
    scores["Volume"] += 8 if row.get("Relative_Volume", 0) > 1 else 0
    scores["Volume"] += 7 if volume > row.get("Volume_Avg_20", np.inf) else 0

    distance_high = row.get("Distance_From_52W_High_Pct", np.nan)
    scores["Risk/Reward"] += 7 if pd.notna(distance_high) and -30 <= distance_high <= -3 else 0
    scores["Risk/Reward"] += 4 if close >= row.get("SMA_50", np.inf) else 0
    scores["Risk/Reward"] += 4 if close >= row.get("SMA_200", np.inf) else 0
    scores["Risk/Reward"] += 5 if 0 < row.get("ATR_Pct", np.inf) <= 5 else 0
    scores["Quality"] += 5 if volume >= 500_000 else 0
    scores["Quality"] += 5 if history_rows >= MIN_HISTORY_ROWS else 0
    return int(sum(scores.values())), scores


def signal_from_score(score: int) -> str:
    """Map the score to a watch-oriented label."""
    if score >= 80:
        return "Strong Buy Watch"
    if score >= 65:
        return "Buy Watch"
    if score >= 45:
        return "Hold / Neutral"
    if score >= 25:
        return "Weak / Avoid"
    return "Sell / Avoid"


def _trade_levels(row: pd.Series) -> tuple[float, float, float]:
    """Create educational stop and target ideas from current volatility."""
    price = float(row["Close"])
    atr = float(row["ATR"])
    stop_candidates = [price - 2 * atr]
    sma_50 = row.get("SMA_50")
    if pd.notna(sma_50) and sma_50 < price:
        stop_candidates.append(float(sma_50) * 0.98)
    eligible_stops = [value for value in stop_candidates if value < price]
    stop = max(0.01, max(eligible_stops))
    risk = max(price - stop, atr)
    return stop, price + 2 * risk, price + 3 * risk


def explain_signal(row: pd.Series, score: int) -> dict[str, Any]:
    """Build concise strengths, risks, invalidation, stop, and target text."""
    strengths: list[str] = []
    risks: list[str] = []
    if row["Close"] > row["SMA_50"] > row["SMA_200"]:
        strengths.append("Price and moving averages are aligned in a bullish trend.")
    elif row["Close"] > row["SMA_200"]:
        strengths.append("Price remains above its long-term 200-day average.")
    else:
        risks.append("Price is below its 200-day average, weakening the long-term setup.")
    if row["MACD"] > row["MACD_Signal"]:
        strengths.append("MACD is above its signal line, showing positive momentum.")
    else:
        risks.append("MACD momentum is below its signal line.")
    if 45 <= row["RSI"] <= 65:
        strengths.append("RSI is in the preferred balanced momentum range.")
    elif row["RSI"] > 70:
        risks.append("RSI is above 70, so the move may be extended.")
    else:
        risks.append("RSI is outside the preferred 45 to 65 range.")
    if row["Relative_Volume"] > 1:
        strengths.append("Trading volume is above its 20-day average.")
    else:
        risks.append("Relative volume is below average, so participation is limited.")
    if row["ATR_Pct"] > 5:
        risks.append("ATR is above 5% of price, indicating elevated volatility.")
    if row["Distance_From_52W_High_Pct"] > -3:
        risks.append("Price is within 3% of its 52-week high, which can limit near-term reward.")

    stop, target_low, target_high = _trade_levels(row)
    return {
        "summary": f"{signal_from_score(score)} with a {score}/100 technical score.",
        "strengths": strengths or ["No major scoring strengths are currently confirmed."],
        "risks": risks or ["No exceptional technical risk flag is present; market risk still applies."],
        "invalidation": (
            f"A close below ${stop:,.2f}, especially with weakness below the 50-day SMA, "
            "would invalidate this watch setup."
        ),
        "stop_loss": stop,
        "take_profit_low": target_low,
        "take_profit_high": target_high,
    }


def get_latest_signal(data: pd.DataFrame) -> dict[str, Any]:
    """Return latest indicators, score, label, and explanation."""
    needed = [
        "Close", "SMA_20", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal",
        "ATR", "High_52W", "Low_52W",
    ]
    complete = data.dropna(subset=needed)
    if complete.empty:
        raise ValueError("At least 200 trading days are needed to calculate this signal.")
    latest = complete.iloc[-1]
    score, categories = score_row(latest, len(data))
    explanation = explain_signal(latest, score)
    return {
        "date": complete.index[-1],
        "price": float(latest["Close"]),
        "daily_change_pct": float(latest["Daily_Change_Pct"]),
        "score": score,
        "signal": signal_from_score(score),
        "category_scores": categories,
        "trend_status": latest["Trend_Status"],
        "sma_20": float(latest["SMA_20"]),
        "sma_50": float(latest["SMA_50"]),
        "sma_200": float(latest["SMA_200"]),
        "rsi": float(latest["RSI"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_Signal"]),
        "volume": int(latest["Volume"]),
        "volume_avg_20": float(latest["Volume_Avg_20"]),
        "relative_volume": float(latest["Relative_Volume"]),
        "high_52w": float(latest["High_52W"]),
        "low_52w": float(latest["Low_52W"]),
        "distance_high_pct": float(latest["Distance_From_52W_High_Pct"]),
        "distance_low_pct": float(latest["Distance_From_52W_Low_Pct"]),
        "atr": float(latest["ATR"]),
        "atr_pct": float(latest["ATR_Pct"]),
        **explanation,
    }
