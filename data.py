"""Download and validate daily market data."""

import re
import time
from collections.abc import Iterable

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_ticker(ticker: str) -> str:
    """Normalize a ticker before sending it to a data provider."""
    cleaned = ticker.strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z0-9.^=-]+", cleaned):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return cleaned


def parse_tickers(value: str | Iterable[str]) -> list[str]:
    """Return unique, valid tickers while preserving input order."""
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


def _normalize_download(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output into one standard OHLCV frame."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{ticker}: downloaded data is missing {', '.join(missing)}.")

    result = data[REQUIRED_COLUMNS].copy()
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.apply(pd.to_numeric, errors="coerce")
    return result.dropna(subset=["Close"]).sort_index()


def load_stock_data(ticker: str, period: str = "2y", retries: int = 2) -> pd.DataFrame:
    """Download adjusted daily data with retries and useful errors."""
    symbol = clean_ticker(ticker)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            data = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )
            if data.empty:
                raise ValueError(f"{symbol}: no price data was returned.")
            result = _normalize_download(data, symbol)
            if result.empty:
                raise ValueError(f"{symbol}: no usable price data was returned.")
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    raise ValueError(f"Could not download {symbol}. {last_error}") from last_error
