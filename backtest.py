"""Educational backtests for the dashboard scoring system."""

import pandas as pd

from config import INITIAL_BACKTEST_CASH
from signals import score_row, signal_from_score


def _max_drawdown(values: pd.Series) -> float:
    running_high = values.cummax()
    drawdown = values / running_high - 1
    return float(drawdown.min() * 100)


def run_backtest(
    data: pd.DataFrame,
    initial_cash: float = INITIAL_BACKTEST_CASH,
    buy_score: int = 80,
    sell_score: int = 45,
) -> dict:
    """Buy at a score of 80+ and sell below 45 using fake fractional shares."""
    if initial_cash <= 0:
        raise ValueError("Initial cash must be greater than zero.")

    needed = [
        "Close", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal",
        "Momentum_20D_Pct", "Relative_Volume", "ATR_Pct", "High_52W",
    ]
    eligible = data.dropna(subset=needed).copy()
    if eligible.empty:
        raise ValueError("At least 200 trading days are needed to run the backtest.")

    eligible["Score"] = [
        score_row(row, history_rows=data.index.get_loc(index) + 1)[0]
        for index, row in eligible.iterrows()
    ]
    eligible["Signal"] = eligible["Score"].map(signal_from_score)

    cash = float(initial_cash)
    shares = 0.0
    entry_value = 0.0
    orders = 0
    closed_returns: list[float] = []
    strategy_values: list[float] = []

    for row in eligible.itertuples():
        price = float(row.Close)
        if row.Score >= buy_score and shares == 0:
            entry_value = cash
            shares = cash / price
            cash = 0.0
            orders += 1
        elif row.Score < sell_score and shares > 0:
            proceeds = shares * price
            closed_returns.append((proceeds / entry_value - 1) * 100)
            cash = proceeds
            shares = 0.0
            entry_value = 0.0
            orders += 1
        strategy_values.append(cash + shares * price)

    first_price = float(eligible["Close"].iloc[0])
    history = eligible[["Close", "Score", "Signal"]].copy()
    history["Strategy Portfolio"] = strategy_values
    history["Buy and Hold Portfolio"] = (
        initial_cash / first_price
    ) * history["Close"]
    final_value = float(history["Strategy Portfolio"].iloc[-1])
    buy_hold_final = float(history["Buy and Hold Portfolio"].iloc[-1])
    wins = sum(result > 0 for result in closed_returns)

    return {
        "history": history,
        "final_value": final_value,
        "total_return_pct": (final_value / initial_cash - 1) * 100,
        "max_drawdown_pct": _max_drawdown(history["Strategy Portfolio"]),
        "number_of_trades": orders,
        "closed_positions": len(closed_returns),
        "win_rate_pct": (wins / len(closed_returns) * 100) if closed_returns else None,
        "buy_and_hold_final_value": buy_hold_final,
        "buy_and_hold_return_pct": (buy_hold_final / initial_cash - 1) * 100,
    }


def build_backtest(data: pd.DataFrame, days_forward: int = 20) -> pd.DataFrame:
    """Score eligible dates and calculate their later percentage return."""
    result = data.copy()
    result["Score"] = [
        score_row(row, history_rows=position + 1)[0]
        for position, (_, row) in enumerate(result.iterrows())
    ]
    result["Signal"] = result["Score"].map(signal_from_score)
    result[f"{days_forward}-Day Return"] = (
        result["Close"].shift(-days_forward) / result["Close"] - 1
    ) * 100
    columns = ["Close", "Score", "Signal", f"{days_forward}-Day Return"]
    return result.dropna(
        subset=["SMA_200", f"{days_forward}-Day Return"]
    )[columns]


def summarize_backtest(results: pd.DataFrame, days_forward: int = 20) -> pd.DataFrame:
    """Summarize forward returns by watch label."""
    return_column = f"{days_forward}-Day Return"
    summary = results.groupby("Signal")[return_column].agg(["mean", "count"])
    summary = summary.rename(
        columns={"mean": "Average Return (%)", "count": "Number of Signals"}
    )
    order = [
        "Strong Buy Watch", "Buy Watch", "Hold / Neutral",
        "Weak / Avoid", "Sell / Avoid",
    ]
    return summary.reindex(order).dropna(how="all")


# Backward-compatible name used by the original prototype.
run_portfolio_backtest = run_backtest
