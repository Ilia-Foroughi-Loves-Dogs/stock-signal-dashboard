"""Feature-engine facade shared by scanner, ML, and backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import add_indicators, latest_snapshot


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(data)


def feature_snapshot(data: pd.DataFrame, fundamentals: dict[str, Any] | None = None) -> dict:
    row = latest_snapshot(data if "SMA_200" in data else build_features(data))
    result = row.to_dict()
    result.update(fundamentals or {})
    return result


def news_sentiment(_ticker: str) -> dict:
    return {
        "available": False,
        "score": None,
        "items": [],
        "warning": "No news API configured; sentiment was skipped safely.",
    }
