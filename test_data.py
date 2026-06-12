"""Disk-cache and batch-download tests for market price data."""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from data import load_stock_data, load_stock_data_batch


def _prices(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=40, freq="B")
    close = np.arange(40, dtype=float) + 100 + offset
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(40, 1_000_000),
        },
        index=index,
    )


class PriceDataTests(unittest.TestCase):
    def test_cache_expiration_refreshes_stale_data(self) -> None:
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            cache_file = cache_dir / "AAA_1y.pkl"
            _prices().to_pickle(cache_file)

            with (
                patch("data.CACHE_DIR", cache_dir),
                patch("data.yf.download", return_value=_prices(10)) as download,
            ):
                fresh = load_stock_data("AAA", cache_ttl_hours=24)
                download.assert_not_called()
                old = time.time() - (25 * 3600)
                os.utime(cache_file, (old, old))
                refreshed = load_stock_data("AAA", cache_ttl_hours=24)

            download.assert_called_once()
            self.assertEqual(float(fresh["Close"].iloc[0]), 100.0)
            self.assertEqual(float(refreshed["Close"].iloc[0]), 110.0)

    def test_batch_download_fetches_multiple_tickers_once(self) -> None:
        frames = {"AAA": _prices(), "BBB": _prices(20)}
        batch = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)

        with TemporaryDirectory() as directory:
            with (
                patch("data.CACHE_DIR", Path(directory)),
                patch("data.yf.download", return_value=batch) as download,
            ):
                results = load_stock_data_batch(["AAA", "BBB"], batch_size=50)

        download.assert_called_once()
        self.assertEqual(set(results), {"AAA", "BBB"})
        self.assertEqual(float(results["BBB"]["Close"].iloc[0]), 120.0)


if __name__ == "__main__":
    unittest.main()
