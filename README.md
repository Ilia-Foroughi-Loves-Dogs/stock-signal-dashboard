# Full-Market AI Quant Scanner

A Python and Streamlit research dashboard that builds a broad US stock/ETF
universe, ranks educational watch setups, explains the ranking, backtests the
rules, and records fake-money paper trades.

> Educational software only. This is not financial advice. No score, model,
> backtest, or AI explanation guarantees profit. The project cannot place real
> trades.

## What It Does

- Builds a universe from Nasdaq Trader, optionally enriched by SEC tickers or
  active Alpaca assets when paper API credentials exist
- Falls back to a built-in liquid stock and ETF list when online sources fail
- Downloads daily OHLCV and best-effort fundamentals through `yfinance`
- Caches data in Streamlit and in the local `cache/` directory
- Calculates trend, momentum, volume, volatility, drawdown, support/resistance,
  breakout, and fundamental features
- Produces a transparent 0-100 quant score and watch-oriented labels
- Trains Random Forest, Gradient Boosting, and Logistic Regression candidates
  with time-ordered validation for 5, 10, and 20-session direction
- Shows model accuracy, precision, recall, and majority-class baseline
- Generates structured local explanations, with optional OpenAI explanations
  limited to top 20, worst 20, or one selected ticker
- Backtests score entries, score/stop/target exits, fake risk sizing, fees, and
  slippage against buy-and-hold
- Maintains a local CSV paper portfolio with confirmation required per trade
- Hard-disables all real broker orders

## Install

Python 3.10 or newer is recommended.

```bash
cd /Users/iliaforoughi/stock-signal-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional `.env` settings:

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
SEC_USER_AGENT=Your Name your-email@example.com
```

Alpaca is used only to discover active tradable US assets. It is never used to
submit orders.

## Run

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Start with **Full Market Scanner**, choose a scan limit of **50**, and run the
scan. This validates Yahoo/Nasdaq access and gives a realistic speed estimate.

For a larger scan, choose 500 or 1,000. To request every discovered symbol,
choose **All** and enable **Full scan mode**. Large scans can take a long time,
consume substantial CPU/network capacity, and trigger provider rate limits.
Failed symbols are isolated so the remaining scan continues.

## Workflow

1. **Full Market Scanner** builds, filters, analyzes, ranks, and exports
   `scanner_results.csv`.
2. **Best Chances** groups strong watches, risk/reward candidates, momentum
   breakouts, quality trends, and sell/exit watches.
3. **Single Stock Deep Dive** shows charts, indicators, score sections, ML
   probability, planning zones, and exit conditions.
4. **AI Analyst** explains only selected top/worst candidates.
5. **Backtest** starts with $10,000 fake cash and compares the strategy with
   buy-and-hold.
6. **Paper Trading** records confirmed fake buys/sells in `paper_trades.csv`.
7. **Risk Manager** limits risk per trade and allocation per position.

## Scoring

| Component | Points |
| --- | ---: |
| Trend | 30 |
| Momentum | 20 |
| Volume | 10 |
| Risk/reward | 20 |
| Fundamental quality | 10 |
| ML contribution | 10 |

The deterministic sections total 90 points. The 10-session model probability
contributes the final 0-10 points. Missing ML data receives a neutral 5 points;
missing fundamentals simply do not earn unavailable quality points.

Labels are: Elite Buy Watch, Strong Buy Watch, Buy Watch, Neutral / Wait,
Weak / Avoid, and Sell / Exit Watch. They are watchlist labels, not orders.

## Data And Model Limitations

- Nasdaq, SEC, Yahoo Finance, Alpaca, and OpenAI can be unavailable or
  rate-limited. Yahoo data may be delayed, incomplete, adjusted, or revised.
- Security-type classification based on free symbol directories is imperfect.
- Fundamentals are often missing for ETFs, new listings, and smaller issuers.
- Full-universe historical downloads are expensive and free providers are not
  designed for institutional-scale scanning.
- Per-ticker ML models have limited samples, can overfit, and can fail when
  classes are imbalanced. Accuracy above baseline does not imply profitability.
- The universe and data introduce survivorship, selection, and availability
  bias. Backtests use daily bars and simplified stop/target fill assumptions.
- Backtests omit taxes, dividends, borrow costs, partial fills, realistic
  spreads, market impact, and intraday path ordering when both stop and target
  are touched.
- Support, resistance, stops, targets, position sizes, and AI text are
  educational heuristics.

## Safety Boundary

`broker.place_order()` always raises:

```text
Real trading is disabled. This app only supports paper trading.
```

Future work can add better bulk market-data providers, persistent databases,
news APIs, portfolio-level walk-forward testing, calibrated probabilities,
sector-neutral ranking, and corporate-action-aware backtesting. Real-money
execution is intentionally out of scope.
