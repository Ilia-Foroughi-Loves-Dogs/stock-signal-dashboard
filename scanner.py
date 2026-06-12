"""Multi-ticker analysis and ranking with isolated provider failures."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import perf_counter

import pandas as pd

from data import load_company_info, load_stock_data
from indicators import add_indicators
from ml_model import train_model
from signals import combined_decision, get_latest_signal
from config import FAILED_TICKERS_FILE, SCANNER_RESULTS_FILE
from risk import risk_warnings


def analyze_ticker(
    ticker: str,
    period: str = "2y",
    price_loader: Callable[[str, str], pd.DataFrame] = load_stock_data,
    info_loader: Callable[[str], dict] = load_company_info,
    run_ml: bool = True,
    metadata: dict | None = None,
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
        "metadata": metadata or {},
    }


def _result_row(ticker: str, analysis: dict) -> dict:
    decision = analysis["decision"]
    info = analysis["fundamentals"]
    probability = decision["probability_up_10d"]
    metadata = analysis.get("metadata", {})
    warnings = risk_warnings(
        decision["final_score"], decision.get("atr_pct", 0),
        average_volume=decision["volume_avg_20"],
    )
    return {
        "Ticker": ticker,
        "Company": info.get("company_name", ticker),
        "Asset Type": metadata.get("asset_type", "Unknown"),
        "Exchange": metadata.get("exchange", "Unknown"),
        "Sector": info.get("sector", "Unknown"),
        "Latest Price": decision["price"],
        "Daily %": decision["daily_change_pct"],
        "Signal": decision["final_signal"],
        "Final Score": decision["final_score"],
        "Quant Score": decision["quant_score"],
        "ML Probability Up 10D": None if probability is None else probability * 100,
        "RSI": decision["rsi"],
        "Relative Volume": decision["relative_volume"],
        "Average Volume": decision["volume_avg_20"],
        "Trend": decision["trend"],
        "Stop-Loss Idea": decision["stop_loss"],
        "Take-Profit Idea": decision["take_profit_2r"],
        "Risk/Reward": decision["risk_reward_ratio"],
        "ATR %": decision.get("atr_pct"),
        "Warning": "; ".join(warnings),
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
        "metadata": analysis.get("metadata", {}),
    }


def scan_tickers(
    tickers: Iterable[str],
    period: str = "2y",
    price_loader: Callable[[str, str], pd.DataFrame] = load_stock_data,
    info_loader: Callable[[str], dict] = load_company_info,
    run_ml: bool = True,
    progress_callback: Callable[[str, int], None] | None = None,
    max_workers: int = 8,
    metadata: dict[str, dict] | None = None,
    save_results: bool = True,
    resume_results: pd.DataFrame | None = None,
    return_summary: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, str], dict[str, dict]]
    | tuple[pd.DataFrame, dict[str, str], dict[str, dict], dict[str, float | int]]
):
    started = perf_counter()
    symbols = list(dict.fromkeys(tickers))
    requested = len(symbols)
    resumed = pd.DataFrame()
    completed_symbols: set[str] = set()
    if resume_results is not None and not resume_results.empty and "Ticker" in resume_results:
        completed_symbols = set(resume_results["Ticker"].dropna().astype(str)) & set(symbols)
        resumed = resume_results[
            resume_results["Ticker"].astype(str).isin(completed_symbols)
        ].copy()
    pending_symbols = [ticker for ticker in symbols if ticker not in completed_symbols]

    if not symbols:
        empty = (pd.DataFrame(), {}, {})
        summary = {
            "total_tickers_requested": 0,
            "successful_tickers": 0,
            "failed_tickers": 0,
            "skipped_tickers": 0,
            "scan_time_seconds": 0.0,
        }
        return (*empty, summary) if return_summary else empty

    rows: list[dict] = []
    errors: dict[str, str] = {}
    analyses: dict[str, dict] = {}
    if progress_callback:
        for completed, ticker in enumerate(sorted(completed_symbols), start=1):
            progress_callback(ticker, completed)

    if pending_symbols:
        worker_count = max(1, min(int(max_workers), len(pending_symbols)))
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
                    (metadata or {}).get(ticker),
                ): ticker
                for ticker in pending_symbols
            }
            for completed, future in enumerate(
                as_completed(futures), start=len(completed_symbols) + 1
            ):
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

    new_results = pd.DataFrame(rows)
    result = pd.concat([resumed, new_results], ignore_index=True)
    if not result.empty:
        result = result.drop_duplicates("Ticker", keep="last")
        result = result.drop(columns=["Rank"], errors="ignore")
        result = result.sort_values(
            ["Final Score", "Quant Score", "Relative Volume"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        result.insert(0, "Rank", range(1, len(result) + 1))
        if save_results:
            result.to_csv(SCANNER_RESULTS_FILE, index=False)
    if save_results:
        failed = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Error": message,
                    "Timestamp UTC": datetime.now(timezone.utc).isoformat(),
                }
                for ticker, message in errors.items()
            ],
            columns=["Ticker", "Error", "Timestamp UTC"],
        )
        failed.to_csv(FAILED_TICKERS_FILE, index=False)

    summary = {
        "total_tickers_requested": requested,
        "successful_tickers": len(result),
        "failed_tickers": len(errors),
        "skipped_tickers": len(completed_symbols),
        "scan_time_seconds": round(perf_counter() - started, 2),
    }
    response = (result, errors, analyses)
    return (*response, summary) if return_summary else response
