"""Shared configuration for the Stock Signal Dashboard."""

from pathlib import Path

APP_NAME = "Stock Signal Dashboard"
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "GOOGL", "AMZN", "NFLX",
    "AVGO", "JPM", "BAC", "V", "MA", "COST", "WMT", "DIS", "PLTR", "SOFI",
    "SPY", "QQQ",
]
SCAN_PERIODS = ["1y", "2y", "5y"]
DEFAULT_SCAN_PERIOD = "2y"
INITIAL_BACKTEST_CASH = 10_000.0
DEFAULT_PAPER_CASH = 100_000.0
DEFAULT_RISK_PER_TRADE = 1.0
PAPER_TRADES_FILE = Path(__file__).resolve().parent / "paper_trades.csv"

SIGNAL_COLORS = {
    "Strong Buy Watch": "#15803d",
    "Buy Watch": "#65a30d",
    "Hold / Neutral": "#ca8a04",
    "Weak / Avoid": "#ea580c",
    "Sell / Avoid": "#dc2626",
}
