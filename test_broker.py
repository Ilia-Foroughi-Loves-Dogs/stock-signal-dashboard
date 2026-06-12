"""Safety tests for the disabled broker and local paper ledger."""

import tempfile
import unittest
from pathlib import Path

from broker import connect_broker, place_order, preview_order
from paper_trading import execute_paper_order, portfolio_summary


class BrokerSafetyTests(unittest.TestCase):
    def test_broker_is_disabled(self) -> None:
        self.assertEqual(connect_broker()["status"], "disabled")
        self.assertEqual(preview_order()["status"], "disabled")

    def test_place_order_always_raises_required_message(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Real trading is disabled. This app only supports paper trading right now.",
        ):
            place_order("AAPL", "BUY", 100)


class LocalPaperTests(unittest.TestCase):
    def test_confirmation_is_required(self) -> None:
        with self.assertRaises(PermissionError):
            execute_paper_order(
                "AAPL", "BUY", 100, 10, 1_000, confirmed=False
            )

    def test_fake_buy_updates_local_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "trades.csv"
            portfolio_file = Path(directory) / "portfolio.csv"
            execute_paper_order(
                "AAPL", "BUY", 100, 10, 1_000, confirmed=True, path=ledger
            )
            summary = portfolio_summary(
                1_000, {"AAPL": 12}, ledger, portfolio_file
            )
            self.assertAlmostEqual(summary["cash"], 900)
            self.assertAlmostEqual(summary["unrealized_pnl"], 20)
            self.assertTrue(portfolio_file.exists())


if __name__ == "__main__":
    unittest.main()
