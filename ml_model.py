"""Advisory 10-day direction model with time-ordered validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

FEATURE_COLUMNS = [
    "Return_5D_Pct", "Return_20D_Pct", "Return_50D_Pct", "RSI", "MACD_Hist",
    "Relative_Volume", "ATR_Pct", "Volatility_20D_Pct", "Trend_Strength",
    "Distance_From_52W_High_Pct", "Distance_From_52W_Low_Pct",
]


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["Price_vs_SMA20"] = frame["Price"] / frame["SMA_20"] - 1
    frame["Price_vs_SMA50"] = frame["Price"] / frame["SMA_50"] - 1
    frame["Price_vs_SMA200"] = frame["Price"] / frame["SMA_200"] - 1
    frame["EMA_Spread"] = frame["EMA_9"] / frame["EMA_21"] - 1
    return frame


def _model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        # Tickers are scanned concurrently, so each model must avoid spawning
        # another full CPU pool and oversubscribing the machine.
        n_jobs=1,
    )


def train_model(data: pd.DataFrame, ticker: str = "") -> dict[str, Any]:
    """Train with expanding time splits; never shuffle future rows into training."""
    frame = build_feature_frame(data)
    features = FEATURE_COLUMNS + ["Price_vs_SMA20", "Price_vs_SMA50", "Price_vs_SMA200", "EMA_Spread"]
    frame["Target_Up_10D"] = (frame["Price"].shift(-10) > frame["Price"]).astype(float)
    frame.loc[frame["Price"].shift(-10).isna(), "Target_Up_10D"] = np.nan
    train = frame.dropna(subset=features + ["Target_Up_10D"])
    if len(train) < 140 or train["Target_Up_10D"].nunique() < 2:
        return {
            "ticker": ticker,
            "model": None,
            "accuracy": None,
            "probability_up_10d": None,
            "probability_down_10d": None,
            "samples": len(train),
            "message": "Not enough balanced history for a reliable advisory model.",
        }

    splits = min(5, max(2, len(train) // 80))
    validation = TimeSeriesSplit(n_splits=splits)
    predictions: list[int] = []
    actual: list[int] = []
    x = train[features]
    y = train["Target_Up_10D"].astype(int)
    for train_index, test_index in validation.split(x):
        fold_model = _model()
        fold_model.fit(x.iloc[train_index], y.iloc[train_index])
        predictions.extend(fold_model.predict(x.iloc[test_index]).tolist())
        actual.extend(y.iloc[test_index].tolist())

    model = _model()
    model.fit(x, y)
    latest = frame.dropna(subset=features).iloc[[-1]][features]
    classes = list(model.classes_)
    probability = model.predict_proba(latest)[0]
    probability_up = float(probability[classes.index(1)]) if 1 in classes else 0.0
    return {
        "ticker": ticker,
        "model": model,
        "features": features,
        "accuracy": float(accuracy_score(actual, predictions)),
        "probability_up_10d": probability_up,
        "probability_down_10d": 1 - probability_up,
        "samples": len(train),
        "message": "Accuracy is historical classification accuracy, not expected profit.",
    }


def train_across_tickers(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    combined = []
    for ticker, data in frames.items():
        frame = data.copy()
        frame["Ticker"] = ticker
        combined.append(frame)
    if not combined:
        return {"model": None, "accuracy": None, "message": "No training data available."}
    return train_model(pd.concat(combined).sort_index(), ticker="Combined universe")
