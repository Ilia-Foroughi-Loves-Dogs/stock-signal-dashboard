# Stock Signal Dashboard

Stock Signal Dashboard is an educational Streamlit stock scanner and paper
trading assistant. It downloads daily market data, calculates common technical
indicators, ranks a stock universe with a transparent 0-100 score, backtests the
same score, records simulated trades locally, and can optionally submit orders
to an Alpaca paper account.

> **Warning:** This project is educational software, not financial advice. It
> does not predict returns and cannot place real trades. Market data may be
> delayed, incomplete, or unavailable.

## Features

- Scan the built-in 21-stock/ETF universe or comma-separated custom tickers
- Handle bad tickers, missing history, and provider errors without stopping a scan
- Calculate price change, SMA 20/50/200, RSI, MACD, volume, relative volume,
  52-week range, ATR, momentum, and trend alignment
- Rank setups using a category-based score from 0 to 100
- Explain strengths, risks, invalidation, ATR-based stop ideas, and target zones
- Chart price, moving averages, RSI, MACD, and recent indicators
- Backtest score 80+ entries and exits below 45 against buy-and-hold
- Track fake cash, positions, cost basis, and paper profit/loss
- Store simulated orders in a local `paper_trades.csv` ledger
- Optionally show Alpaca paper balances and open positions
- Submit confirmed buy/sell market orders to Alpaca paper trading
- Reject live Alpaca URLs in the broker client
- Keep every real brokerage action hard disabled

## Install

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

## Scoring

The score has five categories:

| Category | Maximum | Main checks |
| --- | ---: | --- |
| Trend | 30 | Price above SMA 50, SMA 50 above SMA 200, price above SMA 200 |
| Momentum | 25 | MACD above signal, RSI 45-65, positive 20-day momentum |
| Volume | 15 | Relative volume above 1.0 and volume above its 20-day average |
| Risk/reward | 20 | Room below the 52-week high, major averages hold, moderate ATR |
| Quality/liquidity | 10 | At least 500,000 shares of volume and enough history |

Labels intentionally use watch-list language:

- 80-100: **Strong Buy Watch**
- 65-79: **Buy Watch**
- 45-64: **Hold / Neutral**
- 25-44: **Weak / Avoid**
- 0-24: **Sell / Avoid**

A high score is not a guarantee or an instruction to buy.

## Backtest

The backtest starts with $10,000 in fake cash. It buys with the available cash
when the daily score is at least 80 and sells the position when the score falls
below 45. Fractional shares are allowed.

Results include final value, return, maximum drawdown, order count, closed-trade
win rate when available, and a buy-and-hold comparison. The model ignores fees,
slippage, taxes, dividends, intraday execution, and liquidity constraints.
Those omissions can materially overstate real-world results.

## Paper Trading

The Paper Trading tab offers two modes:

- **Local CSV:** records simulated buys and sells at the latest downloaded daily
  close. It validates available fake cash and shares before adding an order to
  `paper_trades.csv`.
- **Alpaca Paper:** displays the paper account's cash, buying power, equity, and
  open positions. It can submit buy or sell market orders using a dollar amount.

Both modes show a separate confirmation panel before every order. Reviewing an
order does not submit it. The confirmation checkbox and **Confirm Paper Order**
button are required before the CSV ledger is updated or Alpaca is called.

The local starting cash setting is the CSV ledger's assumed opening balance, so
changing it changes the displayed local account calculation. The CSV is runtime
data and is ignored by Git. Delete it only when you intentionally want to reset
the local paper account.

## Alpaca Paper Setup

1. Create or select an Alpaca paper account and generate its paper API keys.
2. Copy `.env.example` to `.env`.
3. Set these values with paper credentials:

```bash
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
```

4. Restart Streamlit, open **Paper Trading**, and select **Alpaca Paper**.

Only `https://paper-api.alpaca.markets` is accepted. The live URL
`https://api.alpaca.markets`, alternate hosts, HTTP URLs, ports, paths, query
strings, and embedded credentials are rejected before any request is made.
Generic `place_order()` calls remain hard disabled.

Alpaca paper trading is still a simulation and can differ from live execution.
See Alpaca's [paper trading documentation](https://docs.alpaca.markets/docs/paper-trading).
Never commit `.env`, API keys, or brokerage secrets.

## Project Structure

- `app.py`: Streamlit UI and charts
- `config.py`: shared defaults and paths
- `data.py`: ticker validation and yfinance downloads
- `signals.py`: indicators, scoring, labels, and explanations
- `scanner.py`: multi-ticker ranking with per-symbol error isolation
- `backtest.py`: score strategy and buy-and-hold comparison
- `paper_trading.py`: local CSV paper ledger and portfolio calculations
- `broker.py`: strict paper-only Alpaca client and live-URL safeguards
- `test_broker.py`: paper endpoint and confirmation safety tests

## Screenshots

Add scanner, single-stock, backtest, and paper-portfolio screenshots here.

## Disclaimer

This software is provided for learning and experimentation. Technical indicators
use historical market data and can produce false or late signals. Nothing in the
application considers your objectives, finances, taxes, time horizon, or risk
tolerance. Verify all data independently and consult a qualified professional
before making financial decisions.
