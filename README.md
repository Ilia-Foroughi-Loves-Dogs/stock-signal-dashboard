# AI Quant Scanner & Paper Trading Assistant

An educational Streamlit dashboard that scans stocks, calculates transparent
technical and quality scores, trains an advisory machine-learning model,
explains setups in plain English, backtests the rules, and tracks fake-money
paper trades.

> **Educational software only. This is not financial advice. No result,
> probability, score, backtest, or AI explanation guarantees a profit. Real
> broker connectivity and real-money trading are disabled.**

## Features

- Large built-in US stock and ETF universe plus custom comma-separated tickers
- Daily OHLC, adjusted close, volume, company profile, valuation, growth,
  margins, leverage, beta, sector, and industry when Yahoo Finance provides them
- SMA 20/50/100/200, EMA 9/21, RSI, MACD, Bollinger Bands, ATR, relative volume,
  52-week range, volatility, ADX trend strength, support, and resistance
- Transparent 0-100 quant score with trend, momentum, volume, risk/reward, and
  fundamental/quality sections
- Watch-oriented labels: Elite Buy Watch, Strong Buy Watch, Buy Watch,
  Neutral / Wait, Weak / Avoid, and Sell / Exit Watch
- Explicit exit alerts for trend breaks, MACD crosses, weak RSI, support breaks,
  stop zones, low scores, and weakening extended moves
- Random Forest probability for whether price is higher ten sessions later,
  evaluated with time-ordered walk-forward splits
- Combined decision: 50% quant, 25% ML, 15% risk/reward, 10% fundamentals
- Optional OpenAI structured explanation with a deterministic local fallback
- Risk-sized backtest against buy-and-hold with an equity curve and trade log
- Local fake-money portfolio, P/L, and trade history in CSV files
- Disabled broker module that always rejects order placement

## Install

Python 3.10 through 3.13 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional OpenAI explanations use a local `.env` file:

```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
```

The API key is optional. Without it, the app uses the local rule-based engine.

## Run

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Then open the local address printed by Streamlit, normally
`http://localhost:8501`.

## Tabs

1. **AI Market Scanner** ranks the universe and provides score, liquidity,
   sector, trend, ML probability, stop, target, and risk/reward filters.
2. **Single Stock Deep Analysis** charts price, averages, support/resistance,
   RSI, MACD, volume, score sections, fundamentals, and AI explanations.
3. **AI Buy/Sell Brain** presents top setups, weak/exit setups, attractive
   risk/reward candidates, and overextended warnings.
4. **Backtest** applies score-based entries and risk-sized exits to fake cash.
5. **Paper Trading** records confirmed simulated buys and sells locally.
6. **Risk Manager** calculates risk budget, maximum shares, allocation, ATR/SMA
   stop ideas, and 2:1 or 3:1 targets.
7. **Settings** reports provider, optional AI, storage, and broker status.

## Scoring

| Section | Points | Examples |
| --- | ---: | --- |
| Trend | 30 | Price above key SMAs, golden-cross alignment, EMA 9 above EMA 21 |
| Momentum | 25 | Balanced RSI, positive MACD, improving histogram, positive returns |
| Volume | 15 | Relative volume and participation confirm movement |
| Risk/reward | 20 | Moderate ATR, not extended, clear stop, at least 2:1 target |
| Fundamental/quality | 10 | Growth, margins, valuation, and liquidity when available |

The final score combines the quant score (50%), ML probability (25%),
risk/reward (15%), and fundamental score (10%). Missing ML data is treated as
neutral. The ML model never decides a paper trade by itself.

## Machine Learning

`ml_model.py` creates indicator features and labels each historical row based on
whether adjusted price is higher ten trading days later. A
`RandomForestClassifier` is evaluated with expanding, time-ordered splits and
then trained on all eligible historical rows for the current probability.
Displayed accuracy is historical classification accuracy, not profitability.

## Paper Trading

Paper orders use the latest downloaded daily close and require an explicit
confirmation checkbox. Trades are stored in `paper_trades.csv`; the reconstructed
portfolio is stored in `paper_portfolio.csv`. Both are local runtime files and
are ignored by Git. There is no broker API call.

`broker.py` intentionally contains only disabled placeholders. `place_order()`
always raises:

```text
Real trading is disabled. This app only supports paper trading right now.
```

A future integration may support Alpaca **paper trading only**, but it must
remain opt-in, visibly simulated, and isolated from all live endpoints.

## Limitations

- Yahoo Finance data can be delayed, missing, rate-limited, or revised.
- Fundamentals are incomplete for many ETFs and some stocks.
- Scanning the full universe with ML enabled can take several minutes.
- Models can overfit and market regimes change.
- Backtests omit slippage, commissions, taxes, dividends, liquidity limits,
  survivorship bias, and realistic intraday stop execution.
- Daily-close paper fills are simplified and are not representative of live
  execution.
- Support, resistance, stops, targets, and confidence are heuristics.

## Disclaimer

Use this project for learning and experimentation only. It does not consider
your objectives, finances, tax situation, time horizon, or risk tolerance.
Verify market data independently and consult a qualified professional before
making financial decisions.
