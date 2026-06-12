"""Streamlit interface for the educational Stock Signal Dashboard."""

import os
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from backtest import run_backtest
from broker import ALPACA_PAPER_BASE_URL, AlpacaPaperClient, connect_broker
from config import (
    APP_NAME,
    DEFAULT_PAPER_CASH,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_SCAN_PERIOD,
    DEFAULT_TICKERS,
    SCAN_PERIODS,
    SIGNAL_COLORS,
)
from data import load_stock_data, parse_tickers
from paper_trading import execute_paper_order, portfolio_summary
from scanner import scan_tickers
from signals import add_indicators, get_latest_signal

load_dotenv()
st.set_page_config(page_title=APP_NAME, layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Cache provider calls while the user explores the dashboard."""
    return load_stock_data(ticker, period)


def analyzed_data(ticker: str, period: str) -> tuple[pd.DataFrame, dict]:
    data = add_indicators(cached_stock_data(ticker, period))
    return data, get_latest_signal(data)


def price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    for column, color in [
        ("SMA_20", "#16a34a"), ("SMA_50", "#f59e0b"), ("SMA_200", "#dc2626")
    ]:
        figure.add_trace(
            go.Scatter(x=data.index, y=data[column], name=column.replace("_", " "), line_color=color)
        )
    figure.update_layout(
        title=f"{ticker} Price and Moving Averages",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        yaxis_title="Adjusted price (USD)",
    )
    return figure


def momentum_chart(data: pd.DataFrame) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    figure.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], name="RSI", line_color="#7c3aed"),
        row=1, col=1,
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=1, col=1)
    figure.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=1, col=1)
    histogram = data["MACD"] - data["MACD_Signal"]
    figure.add_trace(
        go.Bar(x=data.index, y=histogram, name="MACD Histogram", marker_color="#94a3b8"),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD", line_color="#2563eb"),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data.index, y=data["MACD_Signal"], name="Signal", line_color="#f59e0b"
        ),
        row=2, col=1,
    )
    figure.update_yaxes(title_text="RSI", row=1, col=1)
    figure.update_yaxes(title_text="MACD", row=2, col=1)
    figure.update_layout(template="plotly_white", height=560, hovermode="x unified")
    return figure


def backtest_chart(history: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Strategy Portfolio"],
            name="Score Strategy",
            line_color="#2563eb",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Buy and Hold Portfolio"],
            name="Buy and Hold",
            line_color="#64748b",
        )
    )
    figure.update_layout(
        title=f"{ticker}: Strategy vs Buy and Hold",
        template="plotly_white",
        hovermode="x unified",
        height=470,
        yaxis_title="Portfolio value (USD)",
    )
    return figure


def display_signal(signal: dict) -> None:
    color = SIGNAL_COLORS[signal["signal"]]
    st.markdown(
        f"### Signal: <span style='color:{color}'>{signal['signal']}</span>",
        unsafe_allow_html=True,
    )
    columns = st.columns(5)
    columns[0].metric("Score", f"{signal['score']}/100")
    columns[1].metric(
        "Latest Price", f"${signal['price']:,.2f}", f"{signal['daily_change_pct']:.2f}%"
    )
    columns[2].metric("RSI", f"{signal['rsi']:.1f}")
    columns[3].metric("Relative Volume", f"{signal['relative_volume']:.2f}x")
    columns[4].metric("ATR Risk", f"{signal['atr_pct']:.2f}%")
    st.progress(signal["score"] / 100)


st.title(APP_NAME)
st.warning(
    "Educational tool only. Not financial advice. This app uses delayed/public market "
    "data, simulates trades only, and cannot place real orders."
)

with st.sidebar:
    st.header("Scanner Controls")
    universe_choice = st.radio("Stock universe", ["Default universe", "Custom tickers"])
    custom_value = st.text_area(
        "Comma-separated tickers",
        value="AAPL, MSFT, NVDA",
        disabled=universe_choice == "Default universe",
    )
    period = st.selectbox(
        "Scan period",
        SCAN_PERIODS,
        index=SCAN_PERIODS.index(DEFAULT_SCAN_PERIOD),
        help="One year is the minimum; two years gives indicators more history.",
    )
    starting_cash = st.number_input(
        "Starting paper trading cash",
        min_value=1_000.0,
        value=DEFAULT_PAPER_CASH,
        step=1_000.0,
        format="%.2f",
    )
    risk_per_trade = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=10.0,
        value=DEFAULT_RISK_PER_TRADE,
        step=0.1,
    )

try:
    selected_tickers = (
        DEFAULT_TICKERS.copy()
        if universe_choice == "Default universe"
        else parse_tickers(custom_value)
    )
except ValueError as error:
    st.sidebar.error(str(error))
    selected_tickers = DEFAULT_TICKERS.copy()

scanner_tab, analysis_tab, backtest_tab, paper_tab, settings_tab = st.tabs(
    [
        "Market Scanner",
        "Single Stock Analysis",
        "Backtest",
        "Paper Trading",
        "Settings / API Keys",
    ]
)

with scanner_tab:
    st.subheader("Ranked Market Scanner")
    st.caption("Scores are technical watch-list filters, not buy recommendations.")
    if st.button("Run market scan", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting scan...")

        def scan_loader(ticker: str, scan_period: str) -> pd.DataFrame:
            completed = scan_loader.completed + 1
            scan_loader.completed = completed
            progress.progress(
                min(completed / len(selected_tickers), 1.0),
                text=f"Downloading {ticker} ({completed}/{len(selected_tickers)})",
            )
            return cached_stock_data(ticker, scan_period)

        scan_loader.completed = 0
        results, scan_errors = scan_tickers(selected_tickers, period, loader=scan_loader)
        st.session_state["scan_results"] = results
        st.session_state["scan_errors"] = scan_errors
        progress.empty()

    results = st.session_state.get("scan_results", pd.DataFrame())
    if not results.empty:
        display_results = results.copy()
        display_results["Latest Price"] = display_results["Latest Price"].map(
            lambda value: f"${value:,.2f}"
        )
        display_results["Daily Change %"] = display_results["Daily Change %"].map(
            lambda value: f"{value:.2f}%"
        )
        display_results["RSI"] = display_results["RSI"].map(lambda value: f"{value:.1f}")
        display_results["Relative Volume"] = display_results["Relative Volume"].map(
            lambda value: f"{value:.2f}x"
        )
        display_results["Stop-Loss Idea"] = display_results["Stop-Loss Idea"].map(
            lambda value: f"${value:,.2f}"
        )

        def color_signal(value: str) -> str:
            color = SIGNAL_COLORS.get(value)
            return f"color: {color}; font-weight: 600" if color else ""

        styled = display_results.style.map(color_signal, subset=["Signal"])
        st.dataframe(styled, hide_index=True, use_container_width=True, height=620)
    else:
        st.info("Run the scanner to rank the selected universe.")

    scan_errors = st.session_state.get("scan_errors", {})
    if scan_errors:
        with st.expander(f"{len(scan_errors)} ticker(s) could not be scanned"):
            for ticker, message in scan_errors.items():
                st.write(f"**{ticker}:** {message}")

with analysis_tab:
    st.subheader("Single Stock Analysis")
    analysis_ticker = st.selectbox("Ticker", selected_tickers, key="analysis_ticker")
    try:
        with st.spinner(f"Analyzing {analysis_ticker}..."):
            analysis_data, signal = analyzed_data(analysis_ticker, period)
        st.caption(f"Latest data: {signal['date']:%B %d, %Y}")
        display_signal(signal)
        st.plotly_chart(price_chart(analysis_data, analysis_ticker), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Why it scored this way")
            st.write(signal["summary"])
            for reason in signal["strengths"]:
                st.success(reason)
            st.markdown("#### Risks and invalidation")
            for risk in signal["risks"]:
                st.warning(risk)
            st.error(signal["invalidation"])
        with right:
            st.markdown("#### Score breakdown")
            categories = pd.DataFrame(
                {
                    "Category": signal["category_scores"].keys(),
                    "Points": signal["category_scores"].values(),
                    "Maximum": [30, 25, 15, 20, 10],
                }
            )
            st.dataframe(categories, hide_index=True, use_container_width=True)
            st.metric("Stop-loss idea", f"${signal['stop_loss']:,.2f}")
            st.metric(
                "Take-profit idea",
                f"${signal['take_profit_low']:,.2f} to ${signal['take_profit_high']:,.2f}",
            )
            per_share_risk = max(signal["price"] - signal["stop_loss"], 0.01)
            risk_budget = starting_cash * risk_per_trade / 100
            st.caption(
                f"At {risk_per_trade:.1f}% risk on ${starting_cash:,.0f}, an educational "
                f"position-size ceiling is about {risk_budget / per_share_risk:,.2f} shares."
            )

        st.plotly_chart(momentum_chart(analysis_data), use_container_width=True)
        recent_columns = [
            "Close", "SMA_20", "SMA_50", "SMA_200", "RSI", "MACD",
            "MACD_Signal", "Relative_Volume", "ATR",
        ]
        st.markdown("#### Recent indicators")
        st.dataframe(
            analysis_data[recent_columns].tail(10).sort_index(ascending=False).style.format(
                "{:,.2f}", na_rep="-"
            ),
            use_container_width=True,
        )
    except Exception as error:
        st.error(f"Could not analyze {analysis_ticker}: {error}")

with backtest_tab:
    st.subheader("Scoring-System Backtest")
    st.caption(
        "Starts with $10,000 fake cash, buys at score 80+, and sells below 45. "
        "Fractional shares are allowed; fees, slippage, taxes, and dividends are ignored."
    )
    backtest_ticker = st.selectbox("Backtest ticker", selected_tickers, key="backtest_ticker")
    try:
        backtest_data = add_indicators(cached_stock_data(backtest_ticker, period))
        result = run_backtest(backtest_data)
        metrics = st.columns(5)
        metrics[0].metric("Final Value", f"${result['final_value']:,.2f}")
        metrics[1].metric("Total Return", f"{result['total_return_pct']:.2f}%")
        metrics[2].metric("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
        metrics[3].metric("Orders", result["number_of_trades"])
        win_rate = result["win_rate_pct"]
        metrics[4].metric("Win Rate", "N/A" if win_rate is None else f"{win_rate:.1f}%")
        st.plotly_chart(
            backtest_chart(result["history"], backtest_ticker), use_container_width=True
        )
        st.caption(
            f"Buy-and-hold final value: ${result['buy_and_hold_final_value']:,.2f} "
            f"({result['buy_and_hold_return_pct']:.2f}%). Win rate includes closed positions only."
        )
    except Exception as error:
        st.error(f"Could not backtest {backtest_ticker}: {error}")

with paper_tab:
    st.subheader("Paper Trading")
    paper_mode = st.radio(
        "Paper trading account",
        ["Local CSV", "Alpaca Paper"],
        horizontal=True,
        help="Alpaca mode is restricted to Alpaca's paper-api host.",
    )
    if paper_mode == "Local CSV":
        st.info("Local orders are simulated and stored in paper_trades.csv.")
    else:
        st.info(
            "Alpaca orders are submitted only to https://paper-api.alpaca.markets. "
            "Live Alpaca URLs are rejected."
        )

    order_message = st.session_state.pop("paper_order_message", None)
    if order_message:
        st.success(order_message)

    paper_ticker = st.selectbox("Paper trade ticker", selected_tickers, key="paper_ticker")
    paper_amount = st.number_input(
        "Paper order dollar amount", min_value=1.0, value=1_000.0, step=100.0
    )
    paper_price = None
    if paper_mode == "Local CSV":
        try:
            quote_data = cached_stock_data(paper_ticker, "1y")
            paper_price = float(quote_data["Close"].iloc[-1])
            st.caption(f"Simulation price: ${paper_price:,.2f} from the latest daily close.")
        except Exception as error:
            st.error(f"Local paper quote unavailable: {error}")

    buy_column, sell_column = st.columns(2)
    review_disabled = paper_mode == "Local CSV" and paper_price is None
    if buy_column.button(
        "Review Paper Buy",
        type="primary",
        use_container_width=True,
        disabled=review_disabled,
    ):
        st.session_state["pending_paper_order"] = {
            "backend": paper_mode,
            "ticker": paper_ticker,
            "side": "BUY",
            "amount": float(paper_amount),
            "price": paper_price,
            "client_order_id": f"stock-signal-{uuid4().hex}",
        }
    if sell_column.button(
        "Review Paper Sell",
        use_container_width=True,
        disabled=review_disabled,
    ):
        st.session_state["pending_paper_order"] = {
            "backend": paper_mode,
            "ticker": paper_ticker,
            "side": "SELL",
            "amount": float(paper_amount),
            "price": paper_price,
            "client_order_id": f"stock-signal-{uuid4().hex}",
        }

    pending_order = st.session_state.get("pending_paper_order")
    if pending_order and pending_order["backend"] == paper_mode:
        st.markdown("#### Confirm paper order")
        st.warning(
            "Review this paper order carefully. Nothing is submitted until you "
            "check the confirmation box and click Confirm."
        )
        confirmation_details = {
            "Account": pending_order["backend"],
            "Side": pending_order["side"],
            "Ticker": pending_order["ticker"],
            "Dollar amount": f"${pending_order['amount']:,.2f}",
            "Order type": (
                "Local fill at latest daily close"
                if paper_mode == "Local CSV"
                else "Alpaca paper market order, day"
            ),
        }
        if pending_order["price"] is not None:
            confirmation_details["Simulation price"] = (
                f"${pending_order['price']:,.2f}"
            )
        st.table(pd.DataFrame([confirmation_details]))
        confirmation_key = f"confirm_{pending_order['client_order_id']}"
        confirmed = st.checkbox(
            "I confirm this is a paper order and want to submit it.",
            key=confirmation_key,
        )
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Confirm Paper Order",
            type="primary",
            use_container_width=True,
            disabled=not confirmed,
        ):
            try:
                if pending_order["backend"] == "Local CSV":
                    execute_paper_order(
                        pending_order["ticker"],
                        pending_order["side"],
                        pending_order["amount"],
                        pending_order["price"],
                        starting_cash,
                        confirmed=True,
                    )
                    message = (
                        f"Recorded local paper {pending_order['side'].lower()} "
                        f"for ${pending_order['amount']:,.2f} of "
                        f"{pending_order['ticker']}."
                    )
                else:
                    order = AlpacaPaperClient.from_env().submit_market_order(
                        pending_order["ticker"],
                        pending_order["side"],
                        pending_order["amount"],
                        confirmed=True,
                        client_order_id=pending_order["client_order_id"],
                    )
                    message = (
                        f"Submitted Alpaca paper order {order.get('id', '')} "
                        f"({order.get('status', 'accepted')})."
                    )
                st.session_state.pop("pending_paper_order", None)
                st.session_state["paper_order_message"] = message
                st.rerun()
            except Exception as error:
                st.error(f"Paper order was not submitted: {error}")
        if cancel_column.button("Cancel Pending Order", use_container_width=True):
            st.session_state.pop("pending_paper_order", None)
            st.rerun()

    if paper_mode == "Local CSV":
        try:
            initial_summary = portfolio_summary(starting_cash)
            live_prices: dict[str, float] = {}
            if not initial_summary["positions"].empty:
                for ticker in initial_summary["positions"]["Ticker"]:
                    try:
                        live_prices[ticker] = float(
                            cached_stock_data(ticker, "1y")["Close"].iloc[-1]
                        )
                    except Exception:
                        pass
            portfolio = portfolio_summary(starting_cash, live_prices)
            portfolio_metrics = st.columns(4)
            portfolio_metrics[0].metric("Fake Cash", f"${portfolio['cash']:,.2f}")
            portfolio_metrics[1].metric("Paper Equity", f"${portfolio['equity']:,.2f}")
            portfolio_metrics[2].metric(
                "Unrealized P/L", f"${portfolio['unrealized_pnl']:,.2f}"
            )
            portfolio_metrics[3].metric(
                "Realized P/L", f"${portfolio['realized_pnl']:,.2f}"
            )
            st.markdown("#### Local paper positions")
            if portfolio["positions"].empty:
                st.caption("No open paper positions.")
            else:
                st.dataframe(
                    portfolio["positions"].style.format(
                        {
                            "Shares": "{:,.4f}",
                            "Average Cost": "${:,.2f}",
                            "Current Price": "${:,.2f}",
                            "Market Value": "${:,.2f}",
                            "Unrealized P/L": "${:,.2f}",
                            "Unrealized P/L %": "{:,.2f}%",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            with st.expander("Local paper trade history"):
                st.dataframe(
                    portfolio["trades"], hide_index=True, use_container_width=True
                )
        except Exception as error:
            st.error(f"Could not load the local paper portfolio: {error}")
    else:
        try:
            alpaca_client = AlpacaPaperClient.from_env()
            account = alpaca_client.get_account()
            positions = alpaca_client.get_positions()
            account_metrics = st.columns(3)
            account_metrics[0].metric(
                "Paper Cash", f"${float(account['cash']):,.2f}"
            )
            account_metrics[1].metric(
                "Paper Buying Power", f"${float(account['buying_power']):,.2f}"
            )
            account_metrics[2].metric(
                "Paper Equity", f"${float(account['equity']):,.2f}"
            )
            st.markdown("#### Alpaca paper positions")
            if not positions:
                st.caption("No open Alpaca paper positions.")
            else:
                position_rows = [
                    {
                        "Ticker": position["symbol"],
                        "Side": position["side"],
                        "Shares": float(position["qty"]),
                        "Average Entry": float(position["avg_entry_price"]),
                        "Current Price": float(position["current_price"]),
                        "Market Value": float(position["market_value"]),
                        "Unrealized P/L": float(position["unrealized_pl"]),
                        "Unrealized P/L %": float(position["unrealized_plpc"]) * 100,
                    }
                    for position in positions
                ]
                st.dataframe(
                    pd.DataFrame(position_rows).style.format(
                        {
                            "Shares": "{:,.4f}",
                            "Average Entry": "${:,.2f}",
                            "Current Price": "${:,.2f}",
                            "Market Value": "${:,.2f}",
                            "Unrealized P/L": "${:,.2f}",
                            "Unrealized P/L %": "{:,.2f}%",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as error:
            st.error(f"Alpaca paper account unavailable: {error}")

with settings_tab:
    st.subheader("Settings / Alpaca Paper API Keys")
    broker_status = connect_broker()
    if broker_status["status"] == "paper configured":
        st.success(broker_status["message"])
    else:
        st.warning(broker_status["message"])
    st.write(
        "Alpaca is optional. The local CSV account works without API keys. Put paper "
        "credentials in a local `.env` file and never commit it."
    )
    st.code(
        "ALPACA_API_KEY=your_paper_key\n"
        "ALPACA_SECRET_KEY=your_paper_secret\n"
        f"ALPACA_PAPER_BASE_URL={ALPACA_PAPER_BASE_URL}",
        language="bash",
    )
    st.write(
        {
            "ALPACA_API_KEY configured": bool(os.getenv("ALPACA_API_KEY")),
            "ALPACA_SECRET_KEY configured": bool(os.getenv("ALPACA_SECRET_KEY")),
            "ALPACA_PAPER_BASE_URL configured": bool(
                os.getenv("ALPACA_PAPER_BASE_URL")
            ),
            "Broker status": broker_status["status"],
            "Live trading": "Hard disabled; non-paper URLs are rejected",
            "Order safeguard": "Explicit confirmation required for every order",
        }
    )

st.divider()
st.caption(
    "Technical analysis is based on historical data and can fail. Paper results do not "
    "represent live execution or future performance."
)
