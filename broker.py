"""Paper-only Alpaca integration. Live trading endpoints are always rejected."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import requests

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALLOWED_ORDER_SIDES = {"BUY", "SELL"}


def validate_paper_base_url(base_url: str) -> str:
    """Return the normalized Alpaca paper URL or reject the configuration."""
    candidate = (base_url or "").strip()
    parsed = urlparse(candidate)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != "paper-api.alpaca.markets"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Alpaca URL rejected. ALPACA_PAPER_BASE_URL must be exactly "
            f"{ALPACA_PAPER_BASE_URL}."
        )
    return ALPACA_PAPER_BASE_URL


class AlpacaPaperClient:
    """Minimal client restricted to Alpaca's paper trading API."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str,
        timeout: float = 15.0,
    ) -> None:
        if not api_key.strip() or not secret_key.strip():
            raise ValueError("Alpaca paper API key and secret key are required.")
        self.base_url = validate_paper_base_url(base_url)
        self.timeout = timeout
        self.headers = {
            "APCA-API-KEY-ID": api_key.strip(),
            "APCA-API-SECRET-KEY": secret_key.strip(),
            "Accept": "application/json",
        }

    @classmethod
    def from_env(cls) -> "AlpacaPaperClient":
        """Create a paper client from the required environment variables."""
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=os.getenv("ALPACA_PAPER_BASE_URL", ""),
        )

    def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> Any:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                headers=self.headers,
                json=json,
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach Alpaca paper trading: {error}") from error

        if not 200 <= response.status_code < 300:
            try:
                detail = response.json().get("message", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            detail = str(detail).strip() or "Unknown Alpaca API error"
            raise RuntimeError(
                f"Alpaca paper API returned {response.status_code}: {detail}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise RuntimeError("Alpaca paper API returned invalid JSON.") from error

    def get_account(self) -> dict[str, Any]:
        """Return the Alpaca paper account."""
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all open Alpaca paper positions."""
        return self._request("GET", "/v2/positions")

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        dollar_amount: float,
        *,
        confirmed: bool,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Submit a confirmed, day-only market order to the paper endpoint."""
        if not confirmed:
            raise PermissionError("Explicit confirmation is required for every paper order.")
        symbol = ticker.strip().upper()
        action = side.strip().upper()
        if not symbol or not symbol.replace(".", "").isalnum():
            raise ValueError("A valid ticker is required.")
        if action not in ALLOWED_ORDER_SIDES:
            raise ValueError("Alpaca paper order side must be BUY or SELL.")
        if dollar_amount <= 0:
            raise ValueError("Alpaca paper order amount must be greater than zero.")
        if not client_order_id.strip():
            raise ValueError("A client order ID is required.")

        return self._request(
            "POST",
            "/v2/orders",
            json={
                "symbol": symbol,
                "notional": f"{dollar_amount:.2f}",
                "side": action.lower(),
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )


def connect_broker() -> dict[str, str]:
    """Report whether a valid paper-only Alpaca configuration is present."""
    try:
        client = AlpacaPaperClient.from_env()
    except ValueError as error:
        return {"status": "not configured", "message": str(error)}
    return {
        "status": "paper configured",
        "message": f"Paper-only endpoint accepted: {client.base_url}",
    }


def place_order(*args: Any, **kwargs: Any) -> None:
    """Block generic or live order submission permanently."""
    raise RuntimeError(
        "Live trading is disabled. Use AlpacaPaperClient.submit_market_order()."
    )
