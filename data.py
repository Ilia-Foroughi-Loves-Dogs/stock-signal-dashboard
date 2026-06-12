"""Resilient Yahoo Finance price and company-data access."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import pandas as pd
import yfinance as yf
from config import CACHE_DIR, PRICE_CACHE_TTL_HOURS, PRICE_DOWNLOAD_BATCH_SIZE

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
EARNINGS_FIELDS = (
    "earningsTimestamp",
    "earningsTimestampStart",
    "earningsTimestampEnd",
)


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


def _read_cached_prices(
    symbol: str,
    period: str,
    cache_ttl_hours: float,
) -> pd.DataFrame | None:
    cache_file = _cache_path(symbol, period)
    if not cache_file.exists():
        return None
    age_seconds = time.time() - cache_file.stat().st_mtime
    if age_seconds >= max(0, cache_ttl_hours) * 3600:
        return None
    try:
        cached = pd.read_pickle(cache_file)
        return _normalize_download(cached, symbol)
    except Exception:
        return None


def _write_cached_prices(symbol: str, period: str, frame: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(symbol, period)
    temporary = cache_file.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        frame.to_pickle(temporary)
        temporary.replace(cache_file)
    finally:
        temporary.unlink(missing_ok=True)


def _retry_delay(attempt: int, backoff_seconds: float) -> float:
    return max(0, backoff_seconds) * (2 ** attempt)


def load_stock_data(
    ticker: str,
    period: str = "1y",
    retries: int = 2,
    use_disk_cache: bool = True,
    cache_ttl_hours: float = PRICE_CACHE_TTL_HOURS,
    backoff_seconds: float = 0.5,
) -> pd.DataFrame:
    """Download unadjusted OHLCV and adjusted close with bounded retries."""
    symbol = clean_ticker(ticker)
    yahoo_symbol = symbol.replace(".", "-")
    if use_disk_cache:
        cached = _read_cached_prices(yahoo_symbol, period, cache_ttl_hours)
        if cached is not None:
            return cached
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
                _write_cached_prices(yahoo_symbol, period, result)
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(_retry_delay(attempt, backoff_seconds))
    raise ValueError(f"Could not download {symbol}: {last_error}") from last_error


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _batch_ticker_frame(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if yahoo_symbol in values:
            return frame.xs(yahoo_symbol, axis=1, level=level, drop_level=True)
    raise ValueError(f"{yahoo_symbol}: ticker missing from batch response")


def load_stock_data_batch(
    tickers: Iterable[str],
    period: str = "1y",
    retries: int = 2,
    use_disk_cache: bool = True,
    cache_ttl_hours: float = PRICE_CACHE_TTL_HOURS,
    batch_size: int = PRICE_DOWNLOAD_BATCH_SIZE,
    backoff_seconds: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Load fresh cache entries and download remaining symbols in bounded batches."""
    symbols = list(dict.fromkeys(clean_ticker(ticker) for ticker in tickers))
    results: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        yahoo_symbol = symbol.replace(".", "-")
        cached = (
            _read_cached_prices(yahoo_symbol, period, cache_ttl_hours)
            if use_disk_cache else None
        )
        if cached is None:
            missing.append(symbol)
        else:
            results[symbol] = cached

    for batch in _chunks(missing, max(1, int(batch_size))):
        yahoo_symbols = [symbol.replace(".", "-") for symbol in batch]
        downloaded: pd.DataFrame | None = None
        for attempt in range(retries + 1):
            try:
                downloaded = yf.download(
                    yahoo_symbols,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=True,
                    group_by="column",
                    timeout=30,
                )
                if downloaded.empty:
                    raise ValueError("no batch price data returned")
                break
            except Exception:
                downloaded = None
                if attempt < retries:
                    time.sleep(_retry_delay(attempt, backoff_seconds))

        if downloaded is None:
            continue
        for symbol, yahoo_symbol in zip(batch, yahoo_symbols):
            try:
                ticker_frame = _batch_ticker_frame(downloaded, yahoo_symbol)
                normalized = _normalize_download(ticker_frame, symbol)
                if len(normalized) < 30:
                    continue
                results[symbol] = normalized
                if use_disk_cache:
                    _write_cached_prices(yahoo_symbol, period, normalized)
            except Exception:
                # Missing symbols fall back to the existing per-ticker retry path.
                continue
    return results


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
            earnings_timestamps = [
                _number(info.get(field)) for field in EARNINGS_FIELDS
            ]
            future_earnings = [
                value for value in earnings_timestamps
                if value is not None and value >= time.time()
            ]
            result["earnings_date"] = (
                pd.to_datetime(min(future_earnings), unit="s", utc=True).isoformat()
                if future_earnings else None
            )
            result["company_name"] = result.get("company_name") or result.get("short_name") or symbol
            result["sector"] = result.get("sector") or "Unknown"
            result["industry"] = result.get("industry") or "Unknown"
            return result
        except Exception as error:
            last_error = str(error)
            if attempt < retries:
                time.sleep(_retry_delay(attempt, 0.5))
    return {
        "ticker": symbol,
        "company_name": symbol,
        "sector": "Unknown",
        "industry": "Unknown",
        "info_error": last_error,
    }
