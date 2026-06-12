"""Scan and rank a universe of stocks."""

from collections.abc import Callable, Iterable

import pandas as pd

from data import load_stock_data
from signals import add_indicators, get_latest_signal


def scan_tickers(
    tickers: Iterable[str],
    period: str = "2y",
    loader: Callable[[str, str], pd.DataFrame] = load_stock_data,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Analyze tickers independently so one provider error cannot stop a scan."""
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for ticker in tickers:
        try:
            signal = get_latest_signal(add_indicators(loader(ticker, period)))
            rows.append(
                {
                    "Ticker": ticker,
                    "Latest Price": signal["price"],
                    "Daily Change %": signal["daily_change_pct"],
                    "Signal": signal["signal"],
                    "Score": signal["score"],
                    "RSI": signal["rsi"],
                    "Relative Volume": signal["relative_volume"],
                    "Trend Status": signal["trend_status"],
                    "Stop-Loss Idea": signal["stop_loss"],
                    "Take-Profit Idea": (
                        f"${signal['take_profit_low']:,.2f} - "
                        f"${signal['take_profit_high']:,.2f}"
                    ),
                }
            )
        except Exception as error:
            errors[ticker] = str(error)
    if not rows:
        return pd.DataFrame(), errors
    result = pd.DataFrame(rows).sort_values(
        ["Score", "Relative Volume"], ascending=[False, False]
    )
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result.reset_index(drop=True), errors
