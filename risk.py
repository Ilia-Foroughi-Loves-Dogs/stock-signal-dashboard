"""Educational position sizing and volatility-aware trade levels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_trade_levels(row: pd.Series) -> dict[str, Any]:
    price = float(row["Price"])
    atr = max(float(row.get("ATR", 0) or 0), price * 0.005)
    support = float(row.get("Support", np.nan))
    sma_50 = float(row.get("SMA_50", np.nan))
    atr_stop = price - 2 * atr
    support_stop = support * 0.995 if pd.notna(support) and support < price else np.nan
    sma_stop = sma_50 * 0.98 if pd.notna(sma_50) and sma_50 < price else np.nan
    stop_candidates = [atr_stop]
    if pd.notna(support) and support < price:
        stop_candidates.append(support_stop)
    if pd.notna(sma_50) and sma_50 < price:
        stop_candidates.append(sma_stop)
    stop = max(0.01, max(value for value in stop_candidates if value < price))
    risk = max(price - stop, price * 0.005)
    resistance = float(row.get("Resistance", np.nan))
    target_2r = price + 2 * risk
    target_3r = price + 3 * risk
    sell_zone = max(target_2r, resistance) if pd.notna(resistance) else target_2r
    buy_low = max(stop + 0.5 * risk, price - 0.5 * atr)
    return {
        "price": price,
        "buy_zone_low": buy_low,
        "buy_zone_high": price + 0.25 * atr,
        "stop_loss": stop,
        "atr_stop": atr_stop,
        "support_stop": support_stop,
        "sma_50_stop": sma_stop,
        "risk_per_share": risk,
        "take_profit_2r": target_2r,
        "take_profit_3r": target_3r,
        "sell_zone": sell_zone,
        "risk_reward_ratio": (target_2r - price) / risk,
        "atr_pct": float(row.get("ATR_Pct", np.nan)),
    }


def position_size(
    portfolio_value: float,
    price: float,
    stop_loss: float,
    risk_percent: float = 1.0,
    max_allocation_percent: float = 15.0,
) -> dict[str, float]:
    if portfolio_value <= 0 or price <= 0 or not 0 < risk_percent <= 5:
        raise ValueError("Portfolio, price, and risk percent must be valid positive values.")
    per_share_risk = max(price - stop_loss, price * 0.005)
    risk_budget = portfolio_value * risk_percent / 100
    shares_by_risk = risk_budget / per_share_risk
    shares_by_allocation = portfolio_value * max_allocation_percent / 100 / price
    shares = max(0.0, min(shares_by_risk, shares_by_allocation))
    return {
        "risk_budget": risk_budget,
        "risk_per_share": per_share_risk,
        "max_shares": shares,
        "position_value": shares * price,
        "allocation_pct": shares * price / portfolio_value * 100,
    }


def risk_warnings(
    score: float, atr_pct: float, allocation_pct: float = 0, exposure_pct: float = 0,
    average_volume: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    if atr_pct > 5:
        warnings.append("High volatility: daily ATR exceeds 5% of price.")
    if allocation_pct > 20:
        warnings.append("Position concentration exceeds 20% of the paper portfolio.")
    if exposure_pct > 80:
        warnings.append("Portfolio exposure exceeds 80%; little fake cash remains.")
    if score < 65:
        warnings.append("The setup score is below the Buy Watch threshold.")
    if average_volume is not None and average_volume < 500_000:
        warnings.append("Low liquidity: spreads and simulated fills may be unreliable.")
    if atr_pct > 8:
        warnings.append("Gap/spread risk may be large relative to the planned stop.")
    return warnings
