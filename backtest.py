"""Educational risk-sized backtest for the deterministic quant score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import INITIAL_BACKTEST_CASH
from risk import calculate_trade_levels, position_size
from signals import score_row, signal_from_score


def _max_drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1).min() * 100)


def run_backtest(
    data: pd.DataFrame,
    initial_cash: float = INITIAL_BACKTEST_CASH,
    buy_score: int = 80,
    sell_score: int = 45,
    risk_percent: float = 1.0,
    fee_per_trade: float = 0.0,
    slippage_pct: float = 0.0,
) -> dict:
    if initial_cash <= 0:
        raise ValueError("Initial cash must be greater than zero.")
    eligible = data.dropna(
        subset=["Price", "SMA_200", "RSI", "MACD", "ATR", "Support", "Resistance"]
    ).copy()
    if eligible.empty:
        raise ValueError("At least 200 trading days are needed for this backtest.")
    eligible["Score"] = [score_row(row)[0] for _, row in eligible.iterrows()]
    eligible["Signal"] = eligible["Score"].map(signal_from_score)

    cash = float(initial_cash)
    shares = 0.0
    entry_price = stop = target = 0.0
    entry_date = None
    trades: list[dict] = []
    equity_values: list[float] = []
    for date, row in eligible.iterrows():
        price = float(row["Price"])
        if shares == 0 and row["Score"] >= buy_score:
            levels = calculate_trade_levels(row)
            sizing = position_size(cash, price, levels["stop_loss"], risk_percent, 100)
            fill_price = price * (1 + slippage_pct / 100)
            shares = min(sizing["max_shares"], max(0, cash - fee_per_trade) / fill_price)
            if shares > 0:
                entry_price, stop, target, entry_date = (
                    fill_price, levels["stop_loss"], levels["take_profit_2r"], date
                )
                cash -= shares * fill_price + fee_per_trade
        elif shares > 0 and (row["Score"] < sell_score or row["Low"] <= stop or row["High"] >= target):
            if row["Low"] <= stop:
                exit_price, reason = stop * (1 - slippage_pct / 100), "Stop-loss hit"
            elif row["High"] >= target:
                exit_price, reason = target * (1 - slippage_pct / 100), "Take-profit hit"
            else:
                exit_price, reason = price * (1 - slippage_pct / 100), "Score below 45"
            proceeds = shares * exit_price - fee_per_trade
            pnl = proceeds - shares * entry_price - fee_per_trade
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": date,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Shares": shares,
                "P/L": pnl,
                "Return %": (exit_price / entry_price - 1) * 100,
                "Exit Reason": reason,
            })
            cash += proceeds
            shares = 0.0
        equity_values.append(cash + shares * price)

    history = eligible[["Price", "Score", "Signal"]].copy()
    history["Strategy Portfolio"] = equity_values
    history["Buy and Hold Portfolio"] = initial_cash / history["Price"].iloc[0] * history["Price"]
    trade_frame = pd.DataFrame(trades)
    wins = trade_frame[trade_frame["P/L"] > 0]["P/L"] if not trade_frame.empty else pd.Series(dtype=float)
    losses = trade_frame[trade_frame["P/L"] < 0]["P/L"] if not trade_frame.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "history": history,
        "trades": trade_frame,
        "final_value": float(history["Strategy Portfolio"].iloc[-1]),
        "total_return_pct": float((history["Strategy Portfolio"].iloc[-1] / initial_cash - 1) * 100),
        "buy_and_hold_return_pct": float((history["Buy and Hold Portfolio"].iloc[-1] / initial_cash - 1) * 100),
        "max_drawdown_pct": _max_drawdown(history["Strategy Portfolio"]),
        "number_of_trades": len(trade_frame),
        "win_rate_pct": None if trade_frame.empty else float((trade_frame["P/L"] > 0).mean() * 100),
        "average_win": None if wins.empty else float(wins.mean()),
        "average_loss": None if losses.empty else float(losses.mean()),
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "open_position": shares > 0,
    }


run_portfolio_backtest = run_backtest
