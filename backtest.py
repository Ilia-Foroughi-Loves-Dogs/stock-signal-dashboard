"""Simple educational backtests for the dashboard's stock signals."""

import pandas as pd

from signals import score_row, signal_from_score


INITIAL_CASH = 10_000.0


def run_portfolio_backtest(
    data: pd.DataFrame, initial_cash: float = INITIAL_CASH
) -> dict[str, float | int | pd.DataFrame]:
    """Trade the signal with all available cash and compare it with buy-and-hold.

    Fractional shares are allowed to keep the example simple. A trade is counted
    each time the simulation buys or sells.
    """
    if initial_cash <= 0:
        raise ValueError("Initial cash must be greater than zero.")

    needed = ["Close", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal"]
    eligible = data.dropna(subset=needed).copy()
    if eligible.empty:
        raise ValueError("At least 200 trading days of data are needed to run the backtest.")

    scores_and_reasons = eligible.apply(score_row, axis=1)
    eligible["Score"] = scores_and_reasons.map(lambda result: result[0])
    eligible["Signal"] = eligible["Score"].map(signal_from_score)

    cash = float(initial_cash)
    shares = 0.0
    number_of_trades = 0
    strategy_values = []

    for row in eligible.itertuples():
        price = float(row.Close)

        if row.Signal == "BUY" and shares == 0:
            shares = cash / price
            cash = 0.0
            number_of_trades += 1
        elif row.Signal == "SELL" and shares > 0:
            cash = shares * price
            shares = 0.0
            number_of_trades += 1

        strategy_values.append(cash + shares * price)

    first_price = float(eligible["Close"].iloc[0])
    buy_and_hold_shares = initial_cash / first_price

    history = eligible[["Close", "Score", "Signal"]].copy()
    history["Strategy Portfolio"] = strategy_values
    history["Buy and Hold Portfolio"] = buy_and_hold_shares * history["Close"]

    final_value = float(history["Strategy Portfolio"].iloc[-1])
    buy_and_hold_final_value = float(history["Buy and Hold Portfolio"].iloc[-1])

    return {
        "history": history,
        "final_value": final_value,
        "total_return_pct": (final_value / initial_cash - 1) * 100,
        "number_of_trades": number_of_trades,
        "buy_and_hold_return_pct": (buy_and_hold_final_value / initial_cash - 1) * 100,
    }


def build_backtest(data: pd.DataFrame, days_forward: int = 20) -> pd.DataFrame:
    """Score every eligible day and calculate the later percentage return."""
    result = data.copy()

    result["Score"] = (
        (result["Close"] > result["SMA_50"]).astype(int) * 25
        + (result["SMA_50"] > result["SMA_200"]).astype(int) * 25
        + result["RSI"].between(40, 65).astype(int) * 25
        + (result["MACD"] > result["MACD_Signal"]).astype(int) * 25
    )
    result["Signal"] = result["Score"].apply(signal_from_score)
    result[f"{days_forward}-Day Return"] = (
        result["Close"].shift(-days_forward) / result["Close"] - 1
    ) * 100

    columns = ["Close", "Score", "Signal", f"{days_forward}-Day Return"]
    return result.dropna(subset=["SMA_200", f"{days_forward}-Day Return"])[columns]


def summarize_backtest(results: pd.DataFrame, days_forward: int = 20) -> pd.DataFrame:
    """Show average later returns and sample counts for each signal."""
    return_column = f"{days_forward}-Day Return"
    summary = (
        results.groupby("Signal")[return_column]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Average Return (%)", "count": "Number of Signals"})
    )

    signal_order = ["BUY", "HOLD", "SELL"]
    return summary.reindex(signal_order).dropna(how="all")
