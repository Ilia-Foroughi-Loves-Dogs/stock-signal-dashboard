"""CSV-backed paper trading ledger. No brokerage connection is used."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PAPER_TRADES_FILE
from data import clean_ticker

TRADE_COLUMNS = ["timestamp", "ticker", "side", "shares", "price", "amount"]


def load_trades(path: str | Path = PAPER_TRADES_FILE) -> pd.DataFrame:
    """Load the paper ledger or return an empty ledger."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    try:
        trades = pd.read_csv(ledger_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read the paper trade ledger: {error}") from error
    missing = [column for column in TRADE_COLUMNS if column not in trades.columns]
    if missing:
        raise ValueError(f"Paper ledger is missing columns: {', '.join(missing)}")
    return trades[TRADE_COLUMNS]


def _replay(trades: pd.DataFrame, starting_cash: float) -> tuple[float, dict, float]:
    cash = float(starting_cash)
    positions: dict[str, dict[str, float]] = {}
    realized_pnl = 0.0
    for trade in trades.itertuples(index=False):
        ticker = str(trade.ticker)
        shares = float(trade.shares)
        amount = float(trade.amount)
        position = positions.setdefault(ticker, {"shares": 0.0, "cost": 0.0})
        if trade.side == "BUY":
            cash -= amount
            position["shares"] += shares
            position["cost"] += amount
        elif trade.side == "SELL" and position["shares"] > 0:
            sold_shares = min(shares, position["shares"])
            average_cost = position["cost"] / position["shares"]
            cost_removed = average_cost * sold_shares
            proceeds = float(trade.price) * sold_shares
            cash += proceeds
            realized_pnl += proceeds - cost_removed
            position["shares"] -= sold_shares
            position["cost"] -= cost_removed
    return cash, positions, realized_pnl


def portfolio_summary(
    starting_cash: float,
    current_prices: dict[str, float] | None = None,
    path: str | Path = PAPER_TRADES_FILE,
) -> dict:
    """Calculate fake cash, positions, and paper profit/loss."""
    if starting_cash <= 0:
        raise ValueError("Starting paper cash must be greater than zero.")
    trades = load_trades(path)
    cash, positions, realized_pnl = _replay(trades, starting_cash)
    rows = []
    market_value = 0.0
    unrealized_pnl = 0.0
    current_prices = current_prices or {}
    for ticker, position in positions.items():
        shares = position["shares"]
        if shares <= 1e-10:
            continue
        average_cost = position["cost"] / shares
        current_price = float(current_prices.get(ticker, average_cost))
        value = shares * current_price
        pnl = value - position["cost"]
        market_value += value
        unrealized_pnl += pnl
        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Average Cost": average_cost,
                "Current Price": current_price,
                "Market Value": value,
                "Unrealized P/L": pnl,
                "Unrealized P/L %": pnl / position["cost"] * 100,
            }
        )
    return {
        "cash": cash,
        "positions": pd.DataFrame(rows),
        "market_value": market_value,
        "equity": cash + market_value,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
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
    """Append a validated fake buy or sell to the local CSV ledger."""
    if not confirmed:
        raise PermissionError("Explicit confirmation is required for every paper order.")
    symbol = clean_ticker(ticker)
    action = side.strip().upper()
    if action not in {"BUY", "SELL"}:
        raise ValueError("Paper order side must be BUY or SELL.")
    if dollar_amount <= 0 or price <= 0:
        raise ValueError("Paper order amount and price must be greater than zero.")

    ledger_path = Path(path)
    trades = load_trades(ledger_path)
    cash, positions, _ = _replay(trades, starting_cash)
    shares = dollar_amount / price
    if action == "BUY" and dollar_amount > cash + 1e-8:
        raise ValueError(f"Not enough paper cash. Available: ${cash:,.2f}.")
    held_shares = positions.get(symbol, {}).get("shares", 0.0)
    if action == "SELL" and shares > held_shares + 1e-8:
        raise ValueError(
            f"Not enough paper shares. {symbol} position: {held_shares:,.4f} shares."
        )

    trade = pd.DataFrame(
        [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": symbol,
            "side": action,
            "shares": shares,
            "price": price,
            "amount": dollar_amount,
        }]
    )
    updated = pd.concat([trades, trade], ignore_index=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ledger_path.with_suffix(".tmp")
    updated.to_csv(temporary_path, index=False)
    temporary_path.replace(ledger_path)
    return trade.iloc[0].to_dict()
