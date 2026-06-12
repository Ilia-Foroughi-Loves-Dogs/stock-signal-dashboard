"""Build a best-effort universe of listed US stocks and ETFs."""

from __future__ import annotations

import io
import os
import re
from typing import Any

import pandas as pd
import requests

from config import DEFAULT_TICKERS

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "educational-quant-scanner contact@example.com")}


def _read_pipe(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=20, headers=HEADERS)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), sep="|")
    return frame[~frame.iloc[:, 0].astype(str).str.startswith("File Creation Time")]


def _asset_type(name: str, etf: Any) -> str:
    if str(etf).strip().upper() == "Y":
        return "ETF"
    return "ETF" if re.search(r"\b(ETF|FUND|TRUST|ISHARES|SPDR)\b", name.upper()) else "Stock"


def _clean_directory(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    symbol_col = "Symbol" if "Symbol" in frame else "ACT Symbol"
    name_col = "Security Name"
    exchange_col = "Market Category" if "Market Category" in frame else "Exchange"
    output = pd.DataFrame({
        "ticker": frame[symbol_col].astype(str).str.strip(),
        "name": frame[name_col].fillna("").astype(str).str.strip(),
        "exchange": frame.get(exchange_col, "Unknown"),
        "etf_flag": frame.get("ETF", "N"),
        "test_issue": frame.get("Test Issue", "N"),
        "source": source,
    })
    output["asset_type"] = [
        _asset_type(name, etf) for name, etf in zip(output["name"], output["etf_flag"])
    ]
    bad_name = output["name"].str.contains(
        r"\b(WARRANT|RIGHT|UNIT|PREFERRED|PFD|DEPOSITARY SHARE)\b", case=False, regex=True
    )
    good_symbol = output["ticker"].str.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}")
    output = output[(output["test_issue"] != "Y") & good_symbol & ~bad_name]
    return output[["ticker", "name", "asset_type", "exchange", "source"]]


def fallback_universe() -> pd.DataFrame:
    etfs = {"SPY", "QQQ", "DIA", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY"}
    return pd.DataFrame([{
        "ticker": ticker, "name": ticker,
        "asset_type": "ETF" if ticker in etfs else "Stock",
        "exchange": "Unknown", "source": "Fallback",
    } for ticker in DEFAULT_TICKERS])


def load_nasdaq_universe() -> pd.DataFrame:
    frames = [
        _clean_directory(_read_pipe(NASDAQ_LISTED), "Nasdaq Trader"),
        _clean_directory(_read_pipe(OTHER_LISTED), "Nasdaq Trader"),
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticker")


def load_sec_tickers() -> pd.DataFrame:
    response = requests.get(SEC_TICKERS, timeout=20, headers=HEADERS)
    response.raise_for_status()
    payload = response.json()
    fields = payload["fields"]
    frame = pd.DataFrame(payload["data"], columns=fields)
    return frame.rename(columns={"ticker": "ticker", "name": "sec_name", "exchange": "sec_exchange"})


def load_alpaca_assets() -> pd.DataFrame:
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return pd.DataFrame()
    response = requests.get(
        "https://paper-api.alpaca.markets/v2/assets",
        params={"status": "active", "asset_class": "us_equity"},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=20,
    )
    response.raise_for_status()
    assets = pd.DataFrame(response.json())
    assets = assets[assets["tradable"].fillna(False)]
    return pd.DataFrame({
        "ticker": assets["symbol"], "name": assets["name"],
        "asset_type": assets["class"].map(lambda _: "Stock"),
        "exchange": assets["exchange"], "source": "Alpaca",
    })


def build_universe(
    source: str = "Nasdaq Trader",
    asset_type: str = "Stocks + ETFs",
    include_otc: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        universe = load_nasdaq_universe()
        if source == "Alpaca (optional)":
            alpaca = load_alpaca_assets()
            if not alpaca.empty:
                universe = alpaca
            else:
                warnings.append("Alpaca keys/assets unavailable; using Nasdaq Trader.")
        try:
            sec = load_sec_tickers()
            universe = universe.merge(sec, on="ticker", how="left")
            universe["name"] = universe["name"].replace("", pd.NA).fillna(universe["sec_name"])
        except Exception as error:
            warnings.append(f"SEC enrichment skipped: {error}")
    except Exception as error:
        universe = fallback_universe()
        warnings.append(f"Online universe unavailable; using fallback list: {error}")
    if asset_type == "Stocks only":
        universe = universe[universe["asset_type"] == "Stock"]
    elif asset_type == "ETFs only":
        universe = universe[universe["asset_type"] == "ETF"]
    if not include_otc and "exchange" in universe:
        universe = universe[~universe["exchange"].astype(str).str.contains("OTC", case=False)]
    return universe.drop_duplicates("ticker").reset_index(drop=True), warnings
