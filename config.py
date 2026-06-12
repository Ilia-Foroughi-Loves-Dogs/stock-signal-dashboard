"""Shared configuration for the educational AI Quant Scanner."""

from pathlib import Path

APP_NAME = "Full-Market AI Quant Scanner"
APP_DIR = Path(__file__).resolve().parent

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSLA", "META", "GOOGL", "AMZN",
    "NFLX", "PLTR", "SOFI", "COIN", "HOOD", "JPM", "BAC", "GS", "V", "MA",
    "COST", "WMT", "TGT", "DIS", "NKE", "SBUX", "MCD", "CMG", "SPY", "QQQ",
    "DIA", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY",
]
SCAN_PERIODS = ["6mo", "1y", "2y", "5y"]
DEFAULT_SCAN_PERIOD = "1y"
DEFAULT_MAX_WORKERS = 8
MAX_SCAN_WORKERS = 32
INITIAL_BACKTEST_CASH = 10_000.0
DEFAULT_PAPER_CASH = 100_000.0
DEFAULT_RISK_PER_TRADE = 1.0
DEFAULT_MAX_ALLOCATION = 10.0
PAPER_TRADES_FILE = APP_DIR / "paper_trades.csv"
PAPER_PORTFOLIO_FILE = APP_DIR / "paper_portfolio.csv"
MODEL_DIR = APP_DIR / "models"
CACHE_DIR = APP_DIR / "cache"
SCANNER_RESULTS_FILE = APP_DIR / "scanner_results.csv"

SCORE_MAXIMUMS = {
    "Trend": 30,
    "Momentum": 20,
    "Volume": 10,
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
    "Educational software only. This is not financial advice. Scores, probabilities, "
    "AI explanations, and backtests do not guarantee profits. Real trading is disabled."
)
