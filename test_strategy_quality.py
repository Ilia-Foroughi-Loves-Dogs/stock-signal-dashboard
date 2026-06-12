"""Tests for historical validation and market-aware score controls."""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from strategy_quality import (
    days_until_earnings,
    do_not_buy_warnings,
    market_regime,
    market_score_adjustment,
    relative_strength_20d,
    validate_historical_signals,
)


def _quality_data(days: int = 300, daily_gain: float = 0.5) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=days, freq="B")
    price = 100 + np.arange(days) * daily_gain
    return pd.DataFrame(
        {
            "Price": price,
            "Open": price,
            "High": price + 1,
            "Low": price - 1,
            "Volume": 1_000_000,
            "SMA_20": price - 2,
            "SMA_50": price - 4,
            "SMA_200": price - 8,
            "EMA_9": price - 1,
            "EMA_21": price - 2,
            "RSI": 55,
            "MACD": 2,
            "MACD_Signal": 1,
            "MACD_Hist_Improving": True,
            "Return_20D_Pct": 5,
            "Return_50D_Pct": 10,
            "Relative_Volume": 1.2,
            "Volume_Confirms": True,
            "Volume_Avg_20": 1_000_000,
            "ATR": 2,
            "ATR_Pct": 2,
            "Distance_From_52W_High_Pct": -10,
            "Trend_Strength": 25,
            "Support": price - 5,
            "Resistance": price + 10,
        },
        index=index,
    )


class StrategyQualityTests(unittest.TestCase):
    def test_small_sample_cannot_receive_strong_quality_rating(self) -> None:
        quality = validate_historical_signals(_quality_data(days=45))

        self.assertLess(quality["historical_signal_count"], 10)
        self.assertEqual(quality["signal_quality"], "Not enough data")

    def test_historical_validation_reports_forward_results(self) -> None:
        quality = validate_historical_signals(_quality_data())

        self.assertGreaterEqual(quality["historical_signal_count"], 20)
        self.assertEqual(quality["historical_win_rate_pct"], 100)
        self.assertGreater(quality["average_return_5d_pct"], 0)
        self.assertGreater(quality["average_return_10d_pct"], 0)
        self.assertGreater(quality["average_return_20d_pct"], 0)
        self.assertEqual(quality["max_drawdown_after_signal_pct"], 0)
        self.assertIsNotNone(quality["buy_and_hold_return_pct"])
        self.assertIn(quality["signal_quality"], {"Good", "Excellent"})

    def test_market_regime_and_relative_strength_adjust_score(self) -> None:
        stock = _quality_data(days=30, daily_gain=1.0)
        spy = _quality_data(days=30, daily_gain=0.1)
        spy.loc[spy.index[-1], "Price"] = spy["SMA_200"].iloc[-1] - 1

        regime = market_regime(spy)
        relative_strength = relative_strength_20d(stock, spy)
        adjustment, reasons = market_score_adjustment(
            regime, relative_strength, None
        )

        self.assertEqual(regime, "Bearish")
        self.assertGreater(relative_strength, 3)
        self.assertEqual(adjustment, -5)
        self.assertEqual(len(reasons), 2)

    def test_earnings_and_risk_conditions_create_do_not_buy_warnings(self) -> None:
        now = datetime(2026, 6, 11, tzinfo=timezone.utc)
        self.assertEqual(days_until_earnings("2026-06-16", now), 5)
        earnings_date = datetime.now(timezone.utc) + timedelta(days=5)
        warnings = do_not_buy_warnings(
            {
                "volume_avg_20": 100_000,
                "atr_pct": 9,
                "volatility": 90,
                "distance_high_pct": -1,
                "price": 120,
                "sma_20": 100,
            },
            {"signal_quality": "Weak"},
            earnings_date,
        )

        self.assertTrue(any("earnings" in warning for warning in warnings))
        self.assertTrue(any("low average volume" in warning for warning in warnings))
        self.assertTrue(any("extreme volatility" in warning for warning in warnings))
        self.assertTrue(any("too extended" in warning for warning in warnings))
        self.assertTrue(any("historical signal quality" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
