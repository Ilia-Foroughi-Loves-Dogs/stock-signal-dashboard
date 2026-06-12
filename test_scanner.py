"""Concurrency and failure-isolation tests for the ticker scanner."""

import threading
import time
import unittest
from unittest.mock import patch

from scanner import scan_tickers


def _analysis(ticker: str, score: float) -> dict:
    return {
        "ticker": ticker,
        "data": object(),
        "fundamentals": {"company_name": ticker, "sector": "Test"},
        "signal": {"ticker": ticker},
        "ml": {
            "model": object(),
            "features": ["feature"],
            "accuracy": 0.5,
            "probability_up_10d": 0.6,
        },
        "decision": {
            "price": 100.0,
            "daily_change_pct": 1.0,
            "final_signal": "Buy Watch",
            "final_score": score,
            "quant_score": score,
            "probability_up_10d": 0.6,
            "rsi": 50.0,
            "relative_volume": 1.2,
            "volume_avg_20": 1_000_000,
            "trend": "Up",
            "stop_loss": 95.0,
            "take_profit_2r": 110.0,
            "risk_reward_ratio": 2.0,
        },
    }


class ScannerTests(unittest.TestCase):
    def test_scan_runs_concurrently_and_reports_every_ticker(self) -> None:
        active = 0
        peak_active = 0
        lock = threading.Lock()
        progress: list[tuple[str, int]] = []

        def fake_analyze(ticker, *_args, **_kwargs):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            if ticker == "FAIL":
                raise ValueError("provider unavailable")
            return _analysis(ticker, {"AAA": 70.0, "BBB": 80.0}[ticker])

        with patch("scanner.analyze_ticker", side_effect=fake_analyze):
            results, errors, analyses = scan_tickers(
                ["AAA", "FAIL", "BBB"],
                progress_callback=lambda ticker, count: progress.append((ticker, count)),
                max_workers=3,
            )

        self.assertGreater(peak_active, 1)
        self.assertEqual(results["Ticker"].tolist(), ["BBB", "AAA"])
        self.assertEqual(errors, {"FAIL": "provider unavailable"})
        self.assertEqual(set(analyses), {"AAA", "BBB"})
        self.assertNotIn("data", analyses["AAA"])
        self.assertNotIn("model", analyses["AAA"]["ml"])
        self.assertEqual(sorted(count for _, count in progress), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
