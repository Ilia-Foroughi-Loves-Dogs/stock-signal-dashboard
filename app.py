"""Stock Signal Dashboard - a beginner-friendly educational Streamlit app."""

import plotly.graph_objects as go
import streamlit as st

from backtest import build_backtest, run_portfolio_backtest, summarize_backtest
from data import clean_ticker, load_stock_data
from signals import add_indicators, get_latest_signal


st.set_page_config(page_title="Stock Signal Dashboard", layout="wide")

st.title("Stock Signal Dashboard")
st.warning("Educational tool only. This is not financial advice.")
st.write(
    "Enter a stock ticker to see a simple technical-analysis score based on "
    "moving averages, RSI, and MACD."
)

with st.sidebar:
    st.header("Choose a stock")
    ticker_input = st.text_input("Ticker symbol", value="AAPL", help="Examples: AAPL, TSLA, NVDA, SPY")
    period = st.selectbox(
        "Price history",
        options=["1y", "2y", "5y", "10y"],
        index=1,
        help="The app needs at least 200 trading days to calculate the long moving average.",
    )
    st.caption("The dashboard updates automatically when you change these choices.")


def make_price_chart(data, ticker):
    """Create the close-price and moving-average chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=data.index, y=data["Close"], name="Close Price", line=dict(color="#2563EB"))
    )
    figure.add_trace(
        go.Scatter(x=data.index, y=data["SMA_50"], name="50-Day SMA", line=dict(color="#F59E0B"))
    )
    figure.add_trace(
        go.Scatter(x=data.index, y=data["SMA_200"], name="200-Day SMA", line=dict(color="#DC2626"))
    )
    figure.update_layout(
        title=f"{ticker} Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend_title="Chart lines",
        template="plotly_white",
        height=520,
    )
    return figure


def make_indicator_chart(data):
    """Create a compact MACD chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#2563EB"))
    )
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_Signal"],
            name="MACD Signal Line",
            line=dict(color="#F59E0B"),
        )
    )
    figure.update_layout(
        title="MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )
    return figure


def make_backtest_chart(history, ticker):
    """Compare the signal strategy with buying and holding the stock."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Strategy Portfolio"],
            name="Signal Strategy",
            line=dict(color="#2563EB"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Buy and Hold Portfolio"],
            name="Buy and Hold",
            line=dict(color="#6B7280"),
        )
    )
    figure.update_layout(
        title=f"{ticker} Backtest: Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    figure.update_yaxes(tickprefix="$", tickformat=",.0f")
    return figure


if ticker_input:
    try:
        ticker = clean_ticker(ticker_input)
        with st.spinner(f"Downloading {ticker} price data..."):
            prices = load_stock_data(ticker, period)
            data = add_indicators(prices)
            signal = get_latest_signal(data)

        st.caption(f"Latest market data used: {signal['date']:%B %d, %Y}")

        signal_color = {"BUY": "green", "HOLD": "orange", "SELL": "red"}[signal["signal"]]
        st.markdown(f"## Current signal: :{signal_color}[{signal['signal']}]")

        metric_columns = st.columns(4)
        metric_columns[0].metric("Signal Strength", f"{signal['score']}%")
        metric_columns[1].metric("Close Price", f"${signal['price']:,.2f}")
        metric_columns[2].metric("RSI (14 days)", f"{signal['rsi']:.1f}")
        metric_columns[3].metric("MACD", f"{signal['macd']:.2f}")

        st.progress(signal["score"] / 100, text=f"Signal score: {signal['score']} out of 100")

        st.subheader("Why did the app choose this signal?")
        for reason in signal["reasons"]:
            st.write(f"- {reason}")

        st.info(
            "The score adds 25 points for each rule that passes. "
            "75-100 is BUY, 50 is HOLD, and 0-25 is SELL."
        )

        st.plotly_chart(make_price_chart(data, ticker), use_container_width=True)

        indicator_columns = st.columns([1, 2])
        with indicator_columns[0]:
            st.subheader("Latest indicator values")
            st.metric("50-Day SMA", f"${signal['sma_50']:,.2f}")
            st.metric("200-Day SMA", f"${signal['sma_200']:,.2f}")
            st.metric("MACD Signal Line", f"{signal['macd_signal']:.2f}")
        with indicator_columns[1]:
            st.plotly_chart(make_indicator_chart(data), use_container_width=True)

        st.subheader("Simple signal backtest")
        st.write(
            "Starting with $10,000, this simulation buys with all available cash on a "
            "BUY signal and sells the full position on a SELL signal. It allows "
            "fractional shares and ignores fees, taxes, dividends, and execution delays."
        )
        portfolio_backtest = run_portfolio_backtest(data)
        backtest_metrics = st.columns(4)
        backtest_metrics[0].metric(
            "Final Portfolio Value", f"${portfolio_backtest['final_value']:,.2f}"
        )
        backtest_metrics[1].metric(
            "Total Return", f"{portfolio_backtest['total_return_pct']:.2f}%"
        )
        backtest_metrics[2].metric(
            "Number of Trades", f"{portfolio_backtest['number_of_trades']}"
        )
        backtest_metrics[3].metric(
            "Buy-and-Hold Return",
            f"{portfolio_backtest['buy_and_hold_return_pct']:.2f}%",
        )
        st.plotly_chart(
            make_backtest_chart(portfolio_backtest["history"], ticker),
            use_container_width=True,
        )
        st.caption(
            "A trade means one buy or one sell. Any open position is valued at the "
            "latest closing price. Past performance does not predict future results."
        )

        with st.expander("Educational historical check"):
            st.write(
                "This table groups past signals by the average price change over the "
                "next 20 trading days. It does not include fees, taxes, or trading rules, "
                "and past results do not predict future results."
            )
            backtest_results = build_backtest(data, days_forward=20)
            backtest_summary = summarize_backtest(backtest_results, days_forward=20)
            st.dataframe(
                backtest_summary.style.format({"Average Return (%)": "{:.2f}%"}),
                use_container_width=True,
            )

        with st.expander("How the score works"):
            st.write("The dashboard adds 25 points for each true statement:")
            st.write("- Current price is above the 50-day SMA.")
            st.write("- The 50-day SMA is above the 200-day SMA.")
            st.write("- RSI is between 40 and 65.")
            st.write("- MACD is above the MACD signal line.")

    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error("Something went wrong while analyzing this ticker. Please try again.")
        st.exception(error)

st.divider()
st.caption(
    "This dashboard is for learning about technical indicators. "
    "It does not consider company fundamentals, risk, or your personal financial situation."
)
