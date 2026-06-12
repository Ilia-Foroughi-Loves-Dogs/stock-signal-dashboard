"""Transparent quant scoring, labels, exit alerts, and combined decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import SCORE_MAXIMUMS
from indicators import add_indicators, latest_snapshot
from risk import calculate_trade_levels


def signal_from_score(score: float) -> str:
    if score >= 90:
        return "Elite Buy Watch"
    if score >= 80:
        return "Strong Buy Watch"
    if score >= 65:
        return "Buy Watch"
    if score >= 45:
        return "Neutral / Wait"
    if score >= 30:
        return "Weak / Avoid"
    return "Sell / Exit Watch"


def _known(value: Any) -> bool:
    return value is not None and pd.notna(value)


def fundamental_score(fundamentals: dict[str, Any] | None) -> int:
    info = fundamentals or {}
    score = 0
    if _known(info.get("revenue_growth")) and info["revenue_growth"] > 0:
        score += 3
    if _known(info.get("profit_margins")) and info["profit_margins"] > 0:
        score += 3
    pe = info.get("forward_pe") or info.get("pe_ratio")
    if _known(pe) and 0 < pe <= 40:
        score += 2
    average_volume = info.get("average_volume")
    if not _known(average_volume) or average_volume >= 500_000:
        score += 2
    return score


def score_row(
    row: pd.Series, fundamentals: dict[str, Any] | None = None
) -> tuple[int, dict[str, int]]:
    price = float(row["Price"])
    levels = calculate_trade_levels(row)
    scores = {name: 0 for name in SCORE_MAXIMUMS}

    scores["Trend"] += 6 if price > row.get("SMA_20", np.inf) else 0
    scores["Trend"] += 6 if price > row.get("SMA_50", np.inf) else 0
    scores["Trend"] += 6 if price > row.get("SMA_200", np.inf) else 0
    scores["Trend"] += 6 if row.get("SMA_50", 0) > row.get("SMA_200", np.inf) else 0
    scores["Trend"] += 4 if row.get("EMA_9", 0) > row.get("EMA_21", np.inf) else 0
    scores["Trend"] += 2 if row.get("Trend_Strength", 0) >= 20 else 0

    scores["Momentum"] += 4 if 45 <= row.get("RSI", np.nan) <= 65 else 0
    scores["Momentum"] += 4 if row.get("MACD", 0) > row.get("MACD_Signal", np.inf) else 0
    scores["Momentum"] += 4 if bool(row.get("MACD_Hist_Improving", False)) else 0
    scores["Momentum"] += 4 if row.get("Return_20D_Pct", -np.inf) > 0 else 0
    scores["Momentum"] += 4 if row.get("Return_50D_Pct", -np.inf) > 0 else 0

    scores["Volume"] += 4 if row.get("Relative_Volume", 0) > 1 else 0
    scores["Volume"] += 3 if bool(row.get("Volume_Confirms", False)) else 0
    scores["Volume"] += 3 if row.get("Volume_Avg_20", 0) >= 500_000 else 0

    atr_pct = row.get("ATR_Pct", np.inf)
    distance_high = row.get("Distance_From_52W_High_Pct", np.nan)
    scores["Risk/Reward"] += 5 if 0 < atr_pct <= 5 else 0
    scores["Risk/Reward"] += 5 if pd.notna(distance_high) and -25 <= distance_high <= -2 else 0
    scores["Risk/Reward"] += 5 if levels["stop_loss"] < price and levels["risk_per_share"] / price <= 0.1 else 0
    scores["Risk/Reward"] += 5 if levels["risk_reward_ratio"] >= 2 else 0
    scores["Fundamental/Quality"] = fundamental_score(fundamentals)
    return int(sum(scores.values())), scores


def exit_alerts(data: pd.DataFrame, row: pd.Series, score: float) -> list[str]:
    alerts: list[str] = []
    previous = data.iloc[-2] if len(data) > 1 else row
    levels = calculate_trade_levels(row)
    if row["Price"] < row.get("SMA_50", -np.inf):
        alerts.append("Price closed below the 50-day SMA.")
    if row["Price"] < row.get("SMA_200", -np.inf):
        alerts.append("Price closed below the 200-day SMA.")
    if previous.get("MACD", 0) >= previous.get("MACD_Signal", 0) and row.get("MACD", 0) < row.get("MACD_Signal", 0):
        alerts.append("MACD crossed below its signal line.")
    if previous.get("RSI", 0) >= 40 and row.get("RSI", 100) < 40:
        alerts.append("RSI fell below 40 after previously holding above it.")
    if row["Price"] < row.get("Support", -np.inf):
        alerts.append("Price broke the estimated 20-day support.")
    if row["Price"] <= levels["stop_loss"]:
        alerts.append("The volatility-based stop-loss zone was reached.")
    if score < 45:
        alerts.append("Quant score dropped below 45.")
    if levels["risk_reward_ratio"] < 1:
        alerts.append("Estimated reward/risk deteriorated below 1:1.")
    if row.get("RSI", 0) > 70 and not bool(row.get("MACD_Hist_Improving", True)):
        alerts.append("Price is overextended while momentum is weakening.")
    return alerts


def get_latest_signal(
    data: pd.DataFrame, fundamentals: dict[str, Any] | None = None
) -> dict[str, Any]:
    analyzed = data if "SMA_200" in data else add_indicators(data)
    row = latest_snapshot(analyzed)
    score, categories = score_row(row, fundamentals)
    levels = calculate_trade_levels(row)
    alerts = exit_alerts(analyzed.loc[: row.name], row, score)
    strengths = []
    if row["Price"] > row.get("SMA_50", np.inf):
        strengths.append("Price is above its 50-day trend average.")
    if row.get("EMA_9", 0) > row.get("EMA_21", np.inf):
        strengths.append("Short-term EMA alignment is positive.")
    if row.get("MACD", 0) > row.get("MACD_Signal", np.inf):
        strengths.append("MACD momentum is positive.")
    if row.get("Relative_Volume", 0) > 1:
        strengths.append("Volume is above its 20-day average.")
    risks = alerts.copy()
    if row.get("ATR_Pct", 0) > 5:
        risks.append("ATR indicates unusually high volatility.")
    if row.get("Distance_From_52W_High_Pct", -100) > -2:
        risks.append("Price is very close to its 52-week high and may be extended.")
    return {
        "date": row.name,
        "price": float(row["Price"]),
        "daily_change_pct": float(row["Daily_Change_Pct"]),
        "return_5d": float(row["Return_5D_Pct"]),
        "return_10d": float(row["Return_10D_Pct"]),
        "return_20d": float(row["Return_20D_Pct"]),
        "return_50d": float(row["Return_50D_Pct"]),
        "quant_score": score,
        "score": score,
        "signal": signal_from_score(score),
        "category_scores": categories,
        "trend": str(row["Trend"]),
        "sma_20": float(row["SMA_20"]),
        "sma_50": float(row["SMA_50"]),
        "sma_100": float(row["SMA_100"]),
        "sma_200": float(row["SMA_200"]),
        "ema_9": float(row["EMA_9"]),
        "ema_21": float(row["EMA_21"]),
        "ema_50": float(row["EMA_50"]),
        "rsi": float(row["RSI"]),
        "macd": float(row["MACD"]),
        "macd_signal": float(row["MACD_Signal"]),
        "macd_hist": float(row["MACD_Hist"]),
        "stochastic": float(row["Stochastic"]),
        "roc": float(row["ROC"]),
        "relative_volume": float(row["Relative_Volume"]),
        "volume": int(row["Volume"]),
        "volume_avg_20": float(row["Volume_Avg_20"]),
        "volume_avg_50": float(row["Volume_Avg_50"]),
        "volume_trend": float(row["Volume_Trend"]),
        "atr": float(row["ATR"]),
        "atr_pct": float(row["ATR_Pct"]),
        "volatility": float(row["Volatility_20D_Pct"]),
        "max_drawdown": float(row["Max_Drawdown_Pct"]),
        "gap_risk": float(row["Gap_Risk_Pct"]),
        "trend_strength": float(row["Trend_Strength"]),
        "high_52w": float(row["High_52W"]),
        "low_52w": float(row["Low_52W"]),
        "distance_high_pct": float(row["Distance_From_52W_High_Pct"]),
        "distance_low_pct": float(row["Distance_From_52W_Low_Pct"]),
        "support": float(row["Support"]),
        "resistance": float(row["Resistance"]),
        "breakout_level": float(row["Breakout_Level"]),
        "breakdown_level": float(row["Breakdown_Level"]),
        "strengths": strengths or ["No major bullish factor is confirmed."],
        "risks": risks or ["No exceptional technical warning is active; normal market risk remains."],
        "exit_alerts": alerts,
        **levels,
    }


def combined_decision(
    signal: dict[str, Any],
    probability_up: float | None,
    fundamentals: dict[str, Any] | None,
    score_adjustment: float = 0.0,
    adjustment_reasons: list[str] | None = None,
) -> dict[str, Any]:
    # Deterministic sections total 90 points; ML contributes the final 10.
    # Missing ML remains neutral rather than penalizing symbols with weak data.
    ml_points = 5.0 if probability_up is None else probability_up * 10
    base_final = signal["quant_score"] + ml_points
    final = base_final + score_adjustment
    final = round(float(np.clip(final, 0, 100)), 1)
    confidence = round(min(95.0, 45 + abs(final - 50) * 0.8), 1)
    return {
        **signal,
        "base_final_score": round(float(np.clip(base_final, 0, 100)), 1),
        "final_score": final,
        "final_signal": signal_from_score(final),
        "confidence": confidence,
        "probability_up_10d": None if probability_up is None else probability_up,
        "probability_down_10d": None if probability_up is None else 1 - probability_up,
        "ml_score": round(ml_points, 1),
        "fundamental_score": fundamental_score(fundamentals),
        "market_score_adjustment": round(float(score_adjustment), 1),
        "market_adjustment_reasons": adjustment_reasons or [],
    }
