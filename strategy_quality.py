"""Historical signal validation and market-context measurements."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from signals import score_row

BUY_SCORE = 65
MIN_SIGNAL_SPACING = 10
MIN_STRONG_QUALITY_SIGNALS = 10


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _quality_rating(
    count: int,
    win_rate: float | None,
    average_10d: float | None,
    excess_10d: float | None,
    max_drawdown: float | None,
) -> str:
    if count < MIN_STRONG_QUALITY_SIGNALS:
        return "Not enough data"
    if (
        count >= 20
        and win_rate is not None
        and win_rate >= 60
        and average_10d is not None
        and average_10d >= 2
        and excess_10d is not None
        and excess_10d > 0
        and max_drawdown is not None
        and max_drawdown >= -15
    ):
        return "Excellent"
    if (
        win_rate is not None
        and win_rate >= 52
        and average_10d is not None
        and average_10d > 0
        and max_drawdown is not None
        and max_drawdown >= -25
    ):
        return "Good"
    return "Weak"


def validate_historical_signals(data: pd.DataFrame) -> dict[str, Any]:
    """Measure forward returns after prior occurrences of the current score rules."""
    required = [
        "Price", "SMA_200", "RSI", "MACD", "MACD_Signal", "ATR",
        "Support", "Resistance",
    ]
    eligible = data.dropna(subset=required).copy()
    empty = {
        "signal_quality": "Not enough data",
        "historical_win_rate_pct": None,
        "average_return_5d_pct": None,
        "average_return_10d_pct": None,
        "average_return_20d_pct": None,
        "max_drawdown_after_signal_pct": None,
        "historical_signal_count": 0,
        "buy_and_hold_return_pct": None,
        "buy_and_hold_average_10d_pct": None,
        "excess_return_vs_buy_hold_10d_pct": None,
    }
    if len(eligible) <= 20:
        return empty

    scores = pd.Series(
        [score_row(row)[0] for _, row in eligible.iterrows()],
        index=eligible.index,
        dtype=float,
    )
    signal_positions: list[int] = []
    last_position = -MIN_SIGNAL_SPACING
    for position, score in enumerate(scores.iloc[:-20]):
        if score >= BUY_SCORE and position - last_position >= MIN_SIGNAL_SPACING:
            signal_positions.append(position)
            last_position = position

    prices = eligible["Price"].astype(float)
    outcomes: list[dict[str, float]] = []
    for position in signal_positions:
        entry = float(prices.iloc[position])
        path = prices.iloc[position + 1:position + 21] / entry - 1
        if len(path) < 20 or entry <= 0:
            continue
        outcomes.append({
            "return_5d": float(path.iloc[4] * 100),
            "return_10d": float(path.iloc[9] * 100),
            "return_20d": float(path.iloc[19] * 100),
            "drawdown": float(min(0.0, path.min()) * 100),
        })

    if not outcomes:
        return empty

    outcome_frame = pd.DataFrame(outcomes)
    buy_hold_10d = prices.pct_change(10).dropna() * 100
    buy_hold_average_10d = _finite(buy_hold_10d.mean())
    average_10d = _finite(outcome_frame["return_10d"].mean())
    excess_10d = (
        None
        if average_10d is None or buy_hold_average_10d is None
        else average_10d - buy_hold_average_10d
    )
    count = len(outcome_frame)
    win_rate = _finite((outcome_frame["return_10d"] > 0).mean() * 100)
    max_drawdown = _finite(outcome_frame["drawdown"].min())
    return {
        "signal_quality": _quality_rating(
            count, win_rate, average_10d, excess_10d, max_drawdown
        ),
        "historical_win_rate_pct": win_rate,
        "average_return_5d_pct": _finite(outcome_frame["return_5d"].mean()),
        "average_return_10d_pct": average_10d,
        "average_return_20d_pct": _finite(outcome_frame["return_20d"].mean()),
        "max_drawdown_after_signal_pct": max_drawdown,
        "historical_signal_count": count,
        "buy_and_hold_return_pct": _finite(
            (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        ),
        "buy_and_hold_average_10d_pct": buy_hold_average_10d,
        "excess_return_vs_buy_hold_10d_pct": _finite(excess_10d),
    }


def market_regime(spy_data: pd.DataFrame | None) -> str:
    if spy_data is None or spy_data.empty:
        return "N/A"
    complete = spy_data.dropna(subset=["Price", "SMA_200"])
    if complete.empty:
        return "N/A"
    latest = complete.iloc[-1]
    return "Bullish" if latest["Price"] >= latest["SMA_200"] else "Bearish"


def relative_strength_20d(
    stock_data: pd.DataFrame,
    benchmark_data: pd.DataFrame | None,
) -> float | None:
    if benchmark_data is None or benchmark_data.empty:
        return None
    stock = stock_data["Price"].dropna()
    benchmark = benchmark_data["Price"].dropna()
    common = stock.index.intersection(benchmark.index)
    if len(common) < 21:
        return None
    stock_return = stock.loc[common].iloc[-1] / stock.loc[common].iloc[-21] - 1
    benchmark_return = (
        benchmark.loc[common].iloc[-1] / benchmark.loc[common].iloc[-21] - 1
    )
    return _finite((stock_return - benchmark_return) * 100)


def is_growth_sector(sector: str | None) -> bool:
    normalized = (sector or "").strip().lower()
    return normalized in {
        "technology",
        "information technology",
        "communication services",
        "consumer cyclical",
    }


def market_score_adjustment(
    regime: str,
    relative_strength_spy: float | None,
    relative_strength_qqq: float | None,
) -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons: list[str] = []
    if regime == "Bearish":
        adjustment -= 8
        reasons.append("Bearish SPY regime: -8")
    if relative_strength_spy is not None:
        if relative_strength_spy >= 3:
            adjustment += 3
            reasons.append("Strong 20D performance vs SPY: +3")
        elif relative_strength_spy > 0:
            adjustment += 1
            reasons.append("Positive 20D performance vs SPY: +1")
        elif relative_strength_spy <= -5:
            adjustment -= 2
            reasons.append("Weak 20D performance vs SPY: -2")
    if relative_strength_qqq is not None:
        if relative_strength_qqq >= 3:
            adjustment += 2
            reasons.append("Strong 20D performance vs QQQ: +2")
        elif relative_strength_qqq <= -5:
            adjustment -= 1
            reasons.append("Weak 20D performance vs QQQ: -1")
    return adjustment, reasons


def days_until_earnings(value: Any, now: datetime | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return None
    current = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    return int((timestamp.normalize() - current.normalize()).days)


def do_not_buy_warnings(
    decision: dict[str, Any],
    quality: dict[str, Any],
    earnings_date: Any = None,
) -> list[str]:
    warnings: list[str] = []
    earnings_days = days_until_earnings(earnings_date)
    if earnings_days is not None and 0 <= earnings_days <= 14:
        warnings.append(f"DO NOT BUY: earnings in {earnings_days} day(s).")
    average_volume = _finite(decision.get("volume_avg_20"))
    if average_volume is not None and average_volume < 500_000:
        warnings.append("DO NOT BUY: low average volume.")
    atr_pct = _finite(decision.get("atr_pct"))
    volatility = _finite(decision.get("volatility"))
    if (atr_pct is not None and atr_pct > 8) or (
        volatility is not None and volatility > 80
    ):
        warnings.append("DO NOT BUY: extreme volatility.")
    distance_high = _finite(decision.get("distance_high_pct"))
    price = _finite(decision.get("price"))
    sma_20 = _finite(decision.get("sma_20"))
    if (
        distance_high is not None
        and distance_high > -2
    ) or (
        price is not None and sma_20 is not None and price > sma_20 * 1.10
    ):
        warnings.append("DO NOT BUY: price is too extended.")
    if quality.get("signal_quality") == "Weak":
        warnings.append("DO NOT BUY: weak historical signal quality.")
    return warnings
