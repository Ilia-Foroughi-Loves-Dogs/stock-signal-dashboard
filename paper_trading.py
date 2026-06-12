"""Local CSV-backed fake-money portfolio. No broker connection is used."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PAPER_PORTFOLIO_FILE, PAPER_TRADES_FILE
from data import clean_ticker

TRADE_COLUMNS = ["timestamp", "ticker", "side", "shares", "price", "amount"]


def load_trades(path: str | Path = PAPER_TRADES_FILE) -> pd.DataFrame:
    ledger = Path(path)
    if not ledger.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    frame = pd.read_csv(ledger)
    missing = [column for column in TRADE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Paper ledger is missing: {', '.join(missing)}")
    return frame[TRADE_COLUMNS]


def _replay(trades: pd.DataFrame, starting_cash: float) -> tuple[float, dict, float]:
    cash = float(starting_cash)
    positions: dict[str, dict[str, float]] = {}
    realized = 0.0
    for trade in trades.itertuples(index=False):
        position = positions.setdefault(str(trade.ticker), {"shares": 0.0, "cost": 0.0})
        shares, amount = float(trade.shares), float(trade.amount)
        if trade.side == "BUY":
            cash -= amount
            position["shares"] += shares
            position["cost"] += amount
        elif trade.side == "SELL" and position["shares"] > 0:
            sold = min(shares, position["shares"])
            average_cost = position["cost"] / position["shares"]
            removed_cost = sold * average_cost
            proceeds = sold * float(trade.price)
            cash += proceeds
            realized += proceeds - removed_cost
            position["shares"] -= sold
            position["cost"] -= removed_cost
    return cash, positions, realized


def portfolio_summary(
    starting_cash: float,
    current_prices: dict[str, float] | None = None,
    path: str | Path = PAPER_TRADES_FILE,
    portfolio_path: str | Path = PAPER_PORTFOLIO_FILE,
) -> dict:
    trades = load_trades(path)
    cash, positions, realized = _replay(trades, starting_cash)
    current_prices = current_prices or {}
    rows, market_value, unrealized = [], 0.0, 0.0
    for ticker, position in positions.items():
        if position["shares"] <= 1e-10:
            continue
        average_cost = position["cost"] / position["shares"]
        current = float(current_prices.get(ticker, average_cost))
        value = current * position["shares"]
        pnl = value - position["cost"]
        market_value += value
        unrealized += pnl
        rows.append({
            "Ticker": ticker,
            "Shares": position["shares"],
            "Average Cost": average_cost,
            "Current Price": current,
            "Market Value": value,
            "Unrealized P/L": pnl,
            "Unrealized P/L %": pnl / position["cost"] * 100,
        })
    frame = pd.DataFrame(rows)
    Path(portfolio_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(portfolio_path, index=False)
    return {
        "cash": cash,
        "positions": frame,
        "market_value": market_value,
        "equity": cash + market_value,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "exposure_pct": 0 if cash + market_value == 0 else market_value / (cash + market_value) * 100,
        "trades": trades,
    }


def execute_paper_order(
    ticker: str,
    side: str,
    dollar_amount: float,
    price: float,
    starting_cash: float,
    *,
    confirmed: bool,
    path: str | Path = PAPER_TRADES_FILE,
) -> dict:
    if not confirmed:
        raise PermissionError("Explicit confirmation is required for every paper order.")
    symbol, action = clean_ticker(ticker), side.strip().upper()
    if action not in {"BUY", "SELL"} or dollar_amount <= 0 or price <= 0:
        raise ValueError("Use a valid BUY/SELL side, amount, and simulated price.")
    trades = load_trades(path)
    cash, positions, _ = _replay(trades, starting_cash)
    shares = dollar_amount / price
    if action == "BUY" and dollar_amount > cash + 1e-8:
        raise ValueError(f"Not enough fake cash. Available: ${cash:,.2f}.")
    held = positions.get(symbol, {}).get("shares", 0.0)
    if action == "SELL" and shares > held + 1e-8:
        raise ValueError(f"Not enough paper shares. Current position: {held:,.4f}.")
    trade = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": symbol,
        "side": action,
        "shares": shares,
        "price": price,
        "amount": dollar_amount,
    }
    updated = pd.concat([trades, pd.DataFrame([trade])], ignore_index=True)
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger.with_suffix(".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(ledger)
    return trade
