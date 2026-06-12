"""Honest multi-horizon direction models with time-ordered validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Return_5D_Pct", "Return_10D_Pct", "Return_20D_Pct", "Return_50D_Pct",
    "RSI", "MACD_Hist", "Stochastic", "ROC", "Relative_Volume", "Volume_Trend",
    "ATR_Pct", "Volatility_20D_Pct", "Trend_Strength",
    "Distance_From_52W_High_Pct", "Distance_From_52W_Low_Pct",
]


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["Price_vs_SMA20"] = frame["Price"] / frame["SMA_20"] - 1
    frame["Price_vs_SMA50"] = frame["Price"] / frame["SMA_50"] - 1
    frame["Price_vs_SMA200"] = frame["Price"] / frame["SMA_200"] - 1
    frame["EMA_Spread"] = frame["EMA_9"] / frame["EMA_21"] - 1
    return frame.replace([np.inf, -np.inf], np.nan)


def _models() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=160, max_depth=6, min_samples_leaf=8,
            class_weight="balanced", random_state=42, n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Logistic Baseline": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
    }


def _fit_horizon(frame: pd.DataFrame, features: list[str], horizon: int) -> dict[str, Any]:
    target = f"Target_Up_{horizon}D"
    future = frame["Price"].shift(-horizon)
    frame[target] = (future > frame["Price"]).astype(float)
    frame.loc[future.isna(), target] = np.nan
    train = frame.dropna(subset=features + [target])
    if len(train) < 140 or train[target].nunique() < 2:
        return {"probability": None, "samples": len(train), "warning": "Insufficient balanced history."}
    x, y = train[features], train[target].astype(int)
    baseline = max(float(y.mean()), 1 - float(y.mean()))
    splitter = TimeSeriesSplit(n_splits=min(5, max(2, len(train) // 80)))
    candidates: list[tuple[float, str, Any, list[int], list[int]]] = []
    for name, model in _models().items():
        predictions: list[int] = []
        actual: list[int] = []
        for train_idx, test_idx in splitter.split(x):
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            predictions.extend(model.predict(x.iloc[test_idx]).tolist())
            actual.extend(y.iloc[test_idx].tolist())
        candidates.append((accuracy_score(actual, predictions), name, model, predictions, actual))
    accuracy, name, model, predictions, actual = max(candidates, key=lambda item: item[0])
    model.fit(x, y)
    latest = frame.dropna(subset=features).iloc[[-1]][features]
    classes = list(model.classes_)
    probabilities = model.predict_proba(latest)[0]
    probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    return {
        "probability": probability, "accuracy": float(accuracy),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "baseline_accuracy": baseline, "samples": len(train), "model_name": name,
        "model": model, "warning": "Historical accuracy does not guarantee profit.",
    }


def train_model(data: pd.DataFrame, ticker: str = "") -> dict[str, Any]:
    frame = build_feature_frame(data)
    features = FEATURE_COLUMNS + [
        "Price_vs_SMA20", "Price_vs_SMA50", "Price_vs_SMA200", "EMA_Spread"
    ]
    horizons = {days: _fit_horizon(frame.copy(), features, days) for days in (5, 10, 20)}
    valid = [value for value in horizons.values() if value.get("accuracy") is not None]
    accuracy = horizons[10].get("accuracy")
    confidence = None if not valid else max(
        0.0, min(1.0, sum(max(0, item["accuracy"] - item["baseline_accuracy"]) for item in valid) / len(valid) * 4)
    )
    return {
        "ticker": ticker,
        "probability_up_5d": horizons[5].get("probability"),
        "probability_up_10d": horizons[10].get("probability"),
        "probability_up_20d": horizons[20].get("probability"),
        "probability_down_10d": None if horizons[10].get("probability") is None else 1 - horizons[10]["probability"],
        "model_confidence": confidence,
        "accuracy": accuracy,
        "precision": horizons[10].get("precision"),
        "recall": horizons[10].get("recall"),
        "baseline_accuracy": horizons[10].get("baseline_accuracy"),
        "samples": horizons[10].get("samples", 0),
        "model_name": horizons[10].get("model_name"),
        "model_warning": horizons[10].get("warning"),
        "models": {days: result.get("model") for days, result in horizons.items()},
        "features": features,
    }


def train_across_tickers(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        return {"accuracy": None, "model_warning": "No training data available."}
    return train_model(pd.concat(frames.values()).sort_index(), "Combined universe")
