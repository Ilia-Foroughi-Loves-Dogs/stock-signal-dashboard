# Stock Signal Dashboard

A beginner-friendly Streamlit app that downloads stock prices and creates an
educational **BUY**, **HOLD**, or **SELL** signal.

> Educational tool only. This is not financial advice.

## What the app shows

- Historical close price
- 50-day and 200-day simple moving averages (SMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- A signal score from 0% to 100%
- Plain-English reasons for the signal
- A $10,000 simulated signal-strategy backtest
- Final value, total return, trade count, and buy-and-hold comparison
- A chart comparing strategy and buy-and-hold portfolio values
- A simple historical check of past signals

## How the score works

The app adds 25 points for each rule that is true:

1. The current price is above the 50-day SMA.
2. The 50-day SMA is above the 200-day SMA.
3. RSI is between 40 and 65.
4. MACD is above the MACD signal line.

The final result is:

- **BUY:** score of 75 or 100
- **HOLD:** score of 50
- **SELL:** score of 0 or 25

Because every rule is worth 25 points, those are the only possible scores.

## Simple backtest

The backtest starts with $10,000 in fake cash and applies the same signal logic
used by the dashboard:

- On a **BUY** signal, it invests all available cash.
- On a **SELL** signal, it sells the full position.
- On a **HOLD** signal, it makes no change.

For simplicity, the simulation allows fractional shares and ignores transaction
fees, taxes, dividends, slippage, and order-execution delays. It compares the
signal strategy with investing the same $10,000 in the stock on the first
eligible backtest date and holding it through the end. One buy or one sell
counts as one trade.

This is an educational illustration, not evidence that the strategy will work
in the future.

## Install and run

Open a terminal in this project folder. Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use this instead:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the app:

```bash
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`. Open that
address in a web browser, enter a ticker such as `AAPL`, `TSLA`, `NVDA`, or
`SPY`. The dashboard updates automatically.

## Project files

- `app.py` builds the Streamlit page.
- `data.py` downloads and cleans stock prices.
- `signals.py` calculates indicators and the signal score.
- `backtest.py` runs the portfolio simulation and historical signal check.
- `requirements.txt` lists the Python packages.

## Important limitations

Technical indicators use past price data and cannot predict the future. This
example ignores company news, financial statements, transaction costs, taxes,
and personal risk tolerance. Do not use it as the only reason to make an
investment decision.
