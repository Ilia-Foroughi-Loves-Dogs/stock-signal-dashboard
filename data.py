"""Resilient Yahoo Finance price and company-data access."""

from __future__ import annotations

import re
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import pandas as pd
import yfinance as yf
from config import CACHE_DIR

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FUNDAMENTAL_FIELDS = {
    "longName": "company_name",
    "shortName": "short_name",
    "sector": "sector",
    "industry": "industry",
    "marketCap": "market_cap",
    "trailingPE": "pe_ratio",
    "forwardPE": "forward_pe",
    "priceToSalesTrailing12Months": "price_to_sales",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "profitMargins": "profit_margins",
    "grossMargins": "gross_margins",
    "debtToEquity": "debt_to_equity",
    "beta": "beta",
    "averageVolume": "average_volume",
    "freeCashflow": "free_cash_flow",
    "recommendationKey": "analyst_recommendation",
    "quoteType": "quote_type",
}


def clean_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z0-9.^=-]+", cleaned):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return cleaned


def parse_tickers(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else value
    tickers: list[str] = []
    for item in raw:
        if not str(item).strip():
            continue
        ticker = clean_ticker(str(item))
        if ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        raise ValueError("Enter at least one valid ticker.")
    return tickers


def _normalize_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)
    missing = [name for name in ["Open", "High", "Low", "Close", "Volume"] if name not in frame]
    if missing:
        raise ValueError(f"{ticker}: missing downloaded columns: {', '.join(missing)}")
    result = frame.copy()
    if "Adj Close" not in result:
        result["Adj Close"] = result["Close"]
    result = result[REQUIRED_COLUMNS]
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.apply(pd.to_numeric, errors="coerce")
    return result.dropna(subset=["Close"]).sort_index()


def _cache_path(symbol: str, period: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('^', '_')}_{period}.pkl"


def load_stock_data(
    ticker: str, period: str = "1y", retries: int = 2, use_disk_cache: bool = True
) -> pd.DataFrame:
    """Download unadjusted OHLCV and adjusted close with bounded retries."""
    symbol = clean_ticker(ticker)
    yahoo_symbol = symbol.replace(".", "-")
    cache_file = _cache_path(yahoo_symbol, period)
    if use_disk_cache and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 6 * 3600:
            try:
                return pd.read_pickle(cache_file)
            except Exception:
                pass
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = yf.download(
                yahoo_symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=15,
            )
            if frame.empty:
                raise ValueError("no price data returned")
            result = _normalize_download(frame, symbol)
            if len(result) < 30:
                raise ValueError(f"only {len(result)} usable trading days returned")
            if use_disk_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                result.to_pickle(cache_file)
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise ValueError(f"Could not download {symbol}: {last_error}") from last_error


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def load_company_info(ticker: str, retries: int = 1) -> dict[str, Any]:
    """Return best-effort fundamentals; missing fields never fail a scan."""
    symbol = clean_ticker(ticker)
    result: dict[str, Any] = {"ticker": symbol}
    last_error = ""
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(symbol).get_info() or {}
            for source, target in FUNDAMENTAL_FIELDS.items():
                value = info.get(source)
                result[target] = value if target in {
                    "company_name", "short_name", "sector", "industry",
                    "analyst_recommendation", "quote_type",
                } else _number(value)
            result["company_name"] = result.get("company_name") or result.get("short_name") or symbol
            result["sector"] = result.get("sector") or "Unknown"
            result["industry"] = result.get("industry") or "Unknown"
            return result
        except Exception as error:
            last_error = str(error)
            if attempt < retries:
                time.sleep(0.5)
    return {
        "ticker": symbol,
        "company_name": symbol,
        "sector": "Unknown",
        "industry": "Unknown",
        "info_error": last_error,
    }
