"""Disabled broker boundary. Real and external broker orders are not supported."""

from typing import Any


def connect_broker() -> dict[str, str]:
    return {
        "status": "disabled",
        "message": "Broker connectivity is disabled. Use the local paper trading ledger.",
    }


def preview_order(*args: Any, **kwargs: Any) -> dict[str, str]:
    return {
        "status": "disabled",
        "message": "External broker previews are disabled; use local paper trading.",
    }


def place_order(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "Real trading is disabled. This app only supports paper trading right now."
    )
