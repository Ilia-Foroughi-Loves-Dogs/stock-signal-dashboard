"""Shared configuration for the educational AI Quant Scanner."""

from pathlib import Path

APP_NAME = "AI Quant Scanner & Paper Trading Assistant"
APP_DIR = Path(__file__).resolve().parent

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSLA", "META", "GOOGL", "AMZN",
    "NFLX", "PLTR", "SOFI", "COIN", "HOOD", "JPM", "BAC", "GS", "V", "MA",
    "COST", "WMT", "TGT", "DIS", "NKE", "SBUX", "MCD", "CMG", "SPY", "QQQ",
    "DIA", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY",
]
SCAN_PERIODS = ["1y", "2y", "5y", "max"]
DEFAULT_SCAN_PERIOD = "2y"
DEFAULT_MAX_WORKERS = 8
MAX_SCAN_WORKERS = 32
INITIAL_BACKTEST_CASH = 10_000.0
DEFAULT_PAPER_CASH = 100_000.0
DEFAULT_RISK_PER_TRADE = 1.0
DEFAULT_MAX_ALLOCATION = 15.0
PAPER_TRADES_FILE = APP_DIR / "paper_trades.csv"
PAPER_PORTFOLIO_FILE = APP_DIR / "paper_portfolio.csv"
MODEL_DIR = APP_DIR / "models"

SCORE_MAXIMUMS = {
    "Trend": 30,
    "Momentum": 25,
    "Volume": 15,
    "Risk/Reward": 20,
    "Fundamental/Quality": 10,
}

SIGNAL_COLORS = {
    "Elite Buy Watch": "#00c853",
    "Strong Buy Watch": "#22c55e",
    "Buy Watch": "#84cc16",
    "Neutral / Wait": "#eab308",
    "Weak / Avoid": "#f97316",
    "Sell / Exit Watch": "#ef4444",
}

DISCLAIMER = (
    "Educational software only. Not financial advice. Signals can be wrong. "
    "This app supports simulated paper trading only and cannot place real trades."
)
