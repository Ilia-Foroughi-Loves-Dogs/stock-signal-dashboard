"""Multi-ticker analysis and ranking with isolated provider failures."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import pandas as pd

from data import load_company_info, load_stock_data
from indicators import add_indicators
from ml_model import train_model
from signals import combined_decision, get_latest_signal


def analyze_ticker(
    ticker: str,
    period: str = "2y",
    price_loader: Callable[[str, str], pd.DataFrame] = load_stock_data,
    info_loader: Callable[[str], dict] = load_company_info,
    run_ml: bool = True,
) -> dict:
    prices = add_indicators(price_loader(ticker, period))
    fundamentals = info_loader(ticker)
    signal = get_latest_signal(prices, fundamentals)
    ml = train_model(prices, ticker) if run_ml else {
        "accuracy": None, "probability_up_10d": None, "probability_down_10d": None
    }
    decision = combined_decision(signal, ml.get("probability_up_10d"), fundamentals)
    return {
        "ticker": ticker,
        "data": prices,
        "fundamentals": fundamentals,
        "signal": signal,
        "ml": ml,
        "decision": decision,
    }


def _result_row(ticker: str, analysis: dict) -> dict:
    decision = analysis["decision"]
    info = analysis["fundamentals"]
    probability = decision["probability_up_10d"]
    return {
        "Ticker": ticker,
        "Company": info.get("company_name", ticker),
        "Sector": info.get("sector", "Unknown"),
        "Latest Price": decision["price"],
        "Daily %": decision["daily_change_pct"],
        "Signal": decision["final_signal"],
        "Final Score": decision["final_score"],
        "Quant Score": decision["quant_score"],
        "ML Probability Up": None if probability is None else probability * 100,
        "RSI": decision["rsi"],
        "Relative Volume": decision["relative_volume"],
        "Average Volume": decision["volume_avg_20"],
        "Trend": decision["trend"],
        "Stop-Loss Idea": decision["stop_loss"],
        "Take-Profit Idea": decision["take_profit_2r"],
        "Risk/Reward": decision["risk_reward_ratio"],
    }


def _lightweight_analysis(analysis: dict) -> dict:
    """Keep scanner state small enough for large ticker universes."""
    ml = {
        key: value
        for key, value in analysis["ml"].items()
        if key not in {"model", "features"}
    }
    return {
        "ticker": analysis["ticker"],
        "fundamentals": analysis["fundamentals"],
        "signal": analysis["signal"],
        "ml": ml,
        "decision": analysis["decision"],
    }


def scan_tickers(
    tickers: Iterable[str],
    period: str = "2y",
    price_loader: Callable[[str, str], pd.DataFrame] = load_stock_data,
    info_loader: Callable[[str], dict] = load_company_info,
    run_ml: bool = True,
    progress_callback: Callable[[str, int], None] | None = None,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame(), {}, {}

    rows: list[dict] = []
    errors: dict[str, str] = {}
    analyses: dict[str, dict] = {}
    worker_count = max(1, min(int(max_workers), len(symbols)))
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="ticker-scan"
    ) as executor:
        futures: dict[Future, str] = {
            executor.submit(
                analyze_ticker,
                ticker,
                period,
                price_loader,
                info_loader,
                run_ml,
            ): ticker
            for ticker in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                analysis = future.result()
                rows.append(_result_row(ticker, analysis))
                analyses[ticker] = _lightweight_analysis(analysis)
            except Exception as error:
                message = str(error).strip() or type(error).__name__
                errors[ticker] = message
            if progress_callback:
                progress_callback(ticker, completed)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["Final Score", "Quant Score", "Relative Volume"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        result.insert(0, "Rank", range(1, len(result) + 1))
    return result, errors, analyses
