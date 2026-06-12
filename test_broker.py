"""Safety tests for the paper-only Alpaca broker boundary."""

import unittest
from unittest.mock import Mock, patch

from broker import (
    ALPACA_PAPER_BASE_URL,
    AlpacaPaperClient,
    place_order,
    validate_paper_base_url,
)
from paper_trading import execute_paper_order


class PaperUrlValidationTests(unittest.TestCase):
    def test_accepts_only_exact_paper_host(self) -> None:
        self.assertEqual(
            validate_paper_base_url(f"{ALPACA_PAPER_BASE_URL}/"),
            ALPACA_PAPER_BASE_URL,
        )

    def test_rejects_live_and_lookalike_urls(self) -> None:
        rejected_urls = [
            "https://api.alpaca.markets",
            "http://paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets.example.com",
            "https://paper-api.alpaca.markets/v2",
            "https://paper-api.alpaca.markets:443",
        ]
        for url in rejected_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_paper_base_url(url)


class AlpacaPaperOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = AlpacaPaperClient(
            "paper-key",
            "paper-secret",
            ALPACA_PAPER_BASE_URL,
        )

    def test_requires_confirmation_before_submission(self) -> None:
        with self.assertRaises(PermissionError):
            self.client.submit_market_order(
                "AAPL",
                "BUY",
                100.0,
                confirmed=False,
                client_order_id="test-order",
            )

    @patch("broker.requests.request")
    def test_submits_notional_market_order_to_paper_endpoint(
        self, request: Mock
    ) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"id": "paper-order", "status": "accepted"}
        request.return_value = response

        order = self.client.submit_market_order(
            "aapl",
            "buy",
            100.0,
            confirmed=True,
            client_order_id="test-order",
        )

        self.assertEqual(order["id"], "paper-order")
        request.assert_called_once_with(
            "POST",
            f"{ALPACA_PAPER_BASE_URL}/v2/orders",
            headers=self.client.headers,
            json={
                "symbol": "AAPL",
                "notional": "100.00",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": "test-order",
            },
            timeout=15.0,
            allow_redirects=False,
        )

    def test_generic_order_function_never_trades(self) -> None:
        with self.assertRaises(RuntimeError):
            place_order()


class LocalPaperOrderTests(unittest.TestCase):
    def test_local_order_requires_confirmation(self) -> None:
        with self.assertRaises(PermissionError):
            execute_paper_order(
                "AAPL",
                "BUY",
                100.0,
                10.0,
                1_000.0,
                confirmed=False,
            )


if __name__ == "__main__":
    unittest.main()
