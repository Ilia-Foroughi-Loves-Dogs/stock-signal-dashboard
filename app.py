"""Streamlit UI for the educational AI Quant Scanner."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from ai_engine import analyze_stock
from backtest import run_backtest
from broker import connect_broker
from config import (
    APP_NAME,
    DEFAULT_MAX_ALLOCATION,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PAPER_CASH,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_SCAN_PERIOD,
    DEFAULT_TICKERS,
    DISCLAIMER,
    MAX_SCAN_WORKERS,
    SCAN_PERIODS,
    SCORE_MAXIMUMS,
)
from data import load_company_info, load_stock_data, parse_tickers
from indicators import add_indicators
from paper_trading import execute_paper_order, portfolio_summary
from risk import position_size, risk_warnings
from scanner import analyze_ticker, scan_tickers

load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; max-width: 1500px}
    [data-testid="stMetric"] {background:#11182710; border:1px solid #64748b35;
      border-radius:12px; padding:12px}
    .quant-card {border:1px solid #64748b45; border-radius:14px; padding:14px;
      margin:8px 0; background:linear-gradient(135deg,#0f172a08,#2563eb08)}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str, period: str) -> pd.DataFrame:
    return load_stock_data(ticker, period)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_info(ticker: str) -> dict:
    return load_company_info(ticker)


def get_analysis(ticker: str, period: str, run_ml: bool = True) -> dict:
    return analyze_ticker(ticker, period, cached_prices, cached_info, run_ml)


def ai_payload(analysis: dict, backtest: dict | None = None) -> dict:
    ml = analysis["ml"]
    return {
        "ticker": analysis["ticker"],
        "decision": analysis["decision"],
        "fundamentals": analysis["fundamentals"],
        "ml": {
            "accuracy": ml.get("accuracy"),
            "probability_up_10d": ml.get("probability_up_10d"),
            "samples": ml.get("samples"),
        },
        "backtest": None if backtest is None else {
            key: value for key, value in backtest.items()
            if key not in {"history", "trades"}
        },
    }


def price_figure(data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"],
        close=data["Close"], name=ticker,
    ))
    for column, color in [
        ("SMA_20", "#22c55e"), ("SMA_50", "#f59e0b"), ("SMA_200", "#ef4444"),
        ("EMA_9", "#06b6d4"), ("EMA_21", "#8b5cf6"),
    ]:
        figure.add_trace(go.Scatter(
            x=data.index, y=data[column], name=column.replace("_", " "),
            line={"color": color, "width": 1.3},
        ))
    latest = data.dropna(subset=["Support", "Resistance"]).iloc[-1]
    figure.add_hline(y=latest["Support"], line_dash="dot", line_color="#22c55e",
                     annotation_text="Support")
    figure.add_hline(y=latest["Resistance"], line_dash="dot", line_color="#ef4444",
                     annotation_text="Resistance")
    figure.update_layout(
        title=f"{ticker} Price Structure", template="plotly_white", height=600,
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    return figure


def technical_figure(data: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.28, 0.38, 0.34],
    )
    figure.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"), row=1, col=1)
    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=1, col=1)
    figure.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=1, col=1)
    colors = ["#22c55e" if value >= 0 else "#ef4444" for value in data["MACD_Hist"].fillna(0)]
    figure.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="MACD Hist",
                            marker_color=colors), row=2, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"), row=2, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal"), row=2, col=1)
    figure.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=3, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["Volume_Avg_20"],
                                name="20D Avg Volume"), row=3, col=1)
    figure.update_layout(template="plotly_white", height=700, hovermode="x unified")
    return figure


def equity_figure(result: dict, ticker: str) -> go.Figure:
    history = result["history"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history.index, y=history["Strategy Portfolio"],
                                name="Quant Strategy"))
    figure.add_trace(go.Scatter(x=history.index, y=history["Buy and Hold Portfolio"],
                                name="Buy and Hold"))
    figure.update_layout(title=f"{ticker} Equity Curve", template="plotly_white",
                         height=480, yaxis_title="Fake portfolio value")
    return figure


def show_decision(decision: dict) -> None:
    columns = st.columns(6)
    columns[0].metric("Signal", decision["final_signal"])
    columns[1].metric("Final Score", f"{decision['final_score']:.1f}/100")
    columns[2].metric("Quant Score", f"{decision['quant_score']}/100")
    probability = decision["probability_up_10d"]
    columns[3].metric("ML Up 10D", "N/A" if probability is None else f"{probability:.1%}")
    columns[4].metric("RSI", f"{decision['rsi']:.1f}")
    columns[5].metric("Relative Volume", f"{decision['relative_volume']:.2f}x")
    st.progress(decision["final_score"] / 100)


def show_ai(analysis: dict, use_openai: bool) -> None:
    ai = analyze_stock(ai_payload(analysis), use_openai=use_openai)
    st.markdown(f"### {ai['rating']} · {ai['confidence']}% confidence")
    st.caption(ai["source"])
    st.write(ai["plain_english_summary"])
    levels = st.columns(4)
    levels[0].metric("Buy Zone", ai["buy_zone"])
    levels[1].metric("Stop-Loss", ai["stop_loss"])
    levels[2].metric("Take-Profit", ai["take_profit"])
    levels[3].metric("Sell Zone", ai["sell_zone"])
    reasons, risks, invalidation = st.columns(3)
    with reasons:
        st.markdown("**Main reasons**")
        for item in ai["main_reasons"]:
            st.success(item)
    with risks:
        st.markdown("**Main risks**")
        for item in ai["main_risks"]:
            st.warning(item)
    with invalidation:
        st.markdown("**What changes the view**")
        for item in ai["what_would_change_my_mind"]:
            st.error(item)


st.title(APP_NAME)
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Quant Controls")
    universe_mode = st.radio("Universe", ["Default universe", "Custom tickers"])
    custom_tickers = st.text_area(
        "Comma-separated tickers", "AAPL, MSFT, NVDA",
        disabled=universe_mode == "Default universe",
    )
    period = st.selectbox("History", SCAN_PERIODS, index=SCAN_PERIODS.index(DEFAULT_SCAN_PERIOD))
    enable_ml = st.toggle("Train advisory ML models", value=True)
    max_workers = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=MAX_SCAN_WORKERS,
        value=DEFAULT_MAX_WORKERS,
        help="Higher values scan faster but use more CPU and network connections.",
    )
    use_openai = st.toggle(
        "Use OpenAI explanations",
        value=False,
        disabled=not bool(os.getenv("OPENAI_API_KEY")),
        help="Falls back to deterministic local explanations when disabled or unavailable.",
    )
    starting_cash = st.number_input("Starting fake cash", min_value=1_000.0,
                                     value=DEFAULT_PAPER_CASH, step=1_000.0)
    risk_percent = st.slider("Risk per paper trade (%)", 0.25, 2.0,
                             DEFAULT_RISK_PER_TRADE, 0.25)
    max_allocation = st.slider("Max allocation per stock (%)", 5.0, 30.0,
                               DEFAULT_MAX_ALLOCATION, 1.0)

try:
    selected_tickers = (
        DEFAULT_TICKERS.copy()
        if universe_mode == "Default universe"
        else parse_tickers(custom_tickers)
    )
except ValueError as error:
    st.sidebar.error(str(error))
    selected_tickers = DEFAULT_TICKERS.copy()

tabs = st.tabs([
    "AI Market Scanner", "Single Stock Deep Analysis", "AI Buy/Sell Brain",
    "Backtest", "Paper Trading", "Risk Manager", "Settings",
])

with tabs[0]:
    st.subheader("AI Market Scanner")
    st.caption("Ranks educational watch setups. A high score is not a prediction or instruction.")
    if st.button("Run AI Quant Scan", type="primary", use_container_width=True):
        bar = st.progress(0, "Starting scan...")

        def progress(ticker: str, count: int) -> None:
            bar.progress(count / len(selected_tickers),
                         f"Analyzing {ticker} ({count}/{len(selected_tickers)})")

        try:
            results, errors, analyses = scan_tickers(
                selected_tickers,
                period,
                cached_prices,
                cached_info,
                enable_ml,
                progress,
                max_workers,
            )
        except Exception as error:
            bar.empty()
            st.error(f"Scan could not be completed: {error}")
        else:
            st.session_state["scan_results"] = results
            st.session_state["scan_errors"] = errors
            st.session_state["scan_analyses"] = analyses
            bar.empty()

    results = st.session_state.get("scan_results", pd.DataFrame())
    if results.empty:
        st.info("Run the scanner to build the ranked watch list.")
    else:
        filters = st.columns(5)
        minimum_score = filters[0].number_input("Minimum score", 0, 100, 0)
        minimum_volume = filters[1].number_input("Minimum avg volume", 0, 100_000_000, 0,
                                                  step=100_000)
        sectors = ["All"] + sorted(results["Sector"].fillna("Unknown").unique().tolist())
        sector = filters[2].selectbox("Sector", sectors)
        signals = ["All"] + sorted(results["Signal"].unique().tolist())
        signal_filter = filters[3].selectbox("Signal type", signals)
        hide_weak = filters[4].checkbox("Hide weak stocks")
        strong_only = st.checkbox("Show only Strong Buy Watch and above")
        shown = results[
            (results["Final Score"] >= minimum_score)
            & (results["Average Volume"] >= minimum_volume)
        ].copy()
        if sector != "All":
            shown = shown[shown["Sector"] == sector]
        if signal_filter != "All":
            shown = shown[shown["Signal"] == signal_filter]
        if hide_weak:
            shown = shown[shown["Final Score"] >= 45]
        if strong_only:
            shown = shown[shown["Final Score"] >= 80]
        st.dataframe(
            shown.style.format({
                "Latest Price": "${:,.2f}", "Daily %": "{:+.2f}%",
                "Final Score": "{:.1f}", "ML Probability Up": "{:.1f}%",
                "RSI": "{:.1f}", "Relative Volume": "{:.2f}x",
                "Average Volume": "{:,.0f}", "Stop-Loss Idea": "${:,.2f}",
                "Take-Profit Idea": "${:,.2f}", "Risk/Reward": "{:.1f}:1",
            }),
            hide_index=True, use_container_width=True, height=650,
        )
        st.download_button(
            "Download shown results as CSV",
            data=shown.to_csv(index=False).encode("utf-8"),
            file_name="scanner_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    errors = st.session_state.get("scan_errors", {})
    if errors:
        with st.expander(f"{len(errors)} ticker(s) could not be scanned"):
            st.json(errors)

with tabs[1]:
    st.subheader("Single Stock Deep Analysis")
    ticker = st.selectbox("Ticker", selected_tickers, key="deep_ticker")
    try:
        with st.spinner(f"Analyzing {ticker}..."):
            analysis = get_analysis(ticker, period, enable_ml)
        decision, data, info = analysis["decision"], analysis["data"], analysis["fundamentals"]
        st.caption(f"{info.get('company_name', ticker)} · {info.get('sector', 'Unknown')} · "
                   f"latest session {decision['date']:%B %d, %Y}")
        show_decision(decision)
        st.plotly_chart(price_figure(data, ticker), use_container_width=True)
        score_column, fundamental_column = st.columns(2)
        with score_column:
            st.markdown("#### Score Breakdown")
            score_frame = pd.DataFrame({
                "Section": decision["category_scores"].keys(),
                "Points": decision["category_scores"].values(),
                "Maximum": SCORE_MAXIMUMS.values(),
            })
            st.dataframe(score_frame, hide_index=True, use_container_width=True)
        with fundamental_column:
            st.markdown("#### Company Data")
            st.json({
                "Market cap": info.get("market_cap"),
                "Industry": info.get("industry"),
                "PE ratio": info.get("pe_ratio"),
                "Forward PE": info.get("forward_pe"),
                "Revenue growth": info.get("revenue_growth"),
                "Earnings growth": info.get("earnings_growth"),
                "Profit margins": info.get("profit_margins"),
                "Debt/equity": info.get("debt_to_equity"),
                "Beta": info.get("beta"),
            })
        show_ai(analysis, use_openai)
        st.plotly_chart(technical_figure(data), use_container_width=True)
    except Exception as error:
        st.error(f"Could not analyze {ticker}: {error}")

with tabs[2]:
    st.subheader("AI Buy/Sell Brain")
    results = st.session_state.get("scan_results", pd.DataFrame())
    analyses = st.session_state.get("scan_analyses", {})
    if results.empty:
        st.info("Run the AI Market Scanner first.")
    else:
        best, worst = st.columns(2)
        with best:
            st.markdown("### Top 10 Best Setups")
            for _, row in results.head(10).iterrows():
                with st.expander(
                    f"#{row['Rank']} {row['Ticker']} · {row['Signal']} · "
                    f"{row['Final Score']:.1f}"
                ):
                    decision = analyses[row["Ticker"]]["decision"]
                    st.write(" ".join(decision["strengths"]))
                    st.caption(f"Risk/reward {decision['risk_reward_ratio']:.1f}:1 · "
                               f"stop ${decision['stop_loss']:,.2f}")
        with worst:
            st.markdown("### Top 10 Worst / Exit Setups")
            for _, row in results.tail(10).sort_values("Final Score").iterrows():
                with st.expander(f"{row['Ticker']} · {row['Signal']} · {row['Final Score']:.1f}"):
                    decision = analyses[row["Ticker"]]["decision"]
                    for risk in decision["risks"]:
                        st.warning(risk)
        rr = results[results["Final Score"] >= 65].sort_values("Risk/Reward", ascending=False).head(10)
        st.markdown("### Best Risk/Reward Opportunities")
        st.dataframe(rr[["Ticker", "Signal", "Final Score", "Risk/Reward",
                         "Stop-Loss Idea", "Take-Profit Idea"]].style.format({
                             "Final Score": "{:.1f}", "Risk/Reward": "{:.1f}:1",
                             "Stop-Loss Idea": "${:,.2f}", "Take-Profit Idea": "${:,.2f}",
                         }), hide_index=True, use_container_width=True)
        risky = results[
            (results["RSI"] > 70) | (results["Risk/Reward"] < 2) | (results["Final Score"] < 45)
        ]
        st.markdown("### Too Risky / Overextended Warnings")
        st.dataframe(risky[["Ticker", "Signal", "Final Score", "RSI", "Risk/Reward"]],
                     hide_index=True, use_container_width=True)
        brain_ticker = st.selectbox("Deep-analyze a ranked ticker", results["Ticker"],
                                    key="brain_ticker")
        show_ai(analyses[brain_ticker], use_openai)

with tabs[3]:
    st.subheader("Backtest")
    st.caption("Fake $10,000 account; entries at score 80+, exits below 45 or at the stop.")
    ticker = st.selectbox("Backtest ticker", selected_tickers, key="backtest_ticker")
    try:
        data = add_indicators(cached_prices(ticker, period))
        result = run_backtest(data, risk_percent=risk_percent)
        metrics = st.columns(6)
        metrics[0].metric("Final Value", f"${result['final_value']:,.2f}")
        metrics[1].metric("Total Return", f"{result['total_return_pct']:.2f}%")
        metrics[2].metric("Buy & Hold", f"{result['buy_and_hold_return_pct']:.2f}%")
        metrics[3].metric("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
        metrics[4].metric("Trades", result["number_of_trades"])
        metrics[5].metric("Win Rate", "N/A" if result["win_rate_pct"] is None
                          else f"{result['win_rate_pct']:.1f}%")
        extra = st.columns(3)
        extra[0].metric("Average Win", "N/A" if result["average_win"] is None
                        else f"${result['average_win']:,.2f}")
        extra[1].metric("Average Loss", "N/A" if result["average_loss"] is None
                        else f"${result['average_loss']:,.2f}")
        extra[2].metric("Profit Factor", "N/A" if result["profit_factor"] is None
                        else f"{result['profit_factor']:.2f}")
        st.plotly_chart(equity_figure(result, ticker), use_container_width=True)
        st.dataframe(result["trades"], hide_index=True, use_container_width=True)
        st.warning("Historical results omit slippage, fees, taxes, and intraday gaps. "
                   "Backtest performance does not guarantee future results.")
    except Exception as error:
        st.error(f"Could not backtest {ticker}: {error}")

with tabs[4]:
    st.subheader("Paper Trading")
    st.info("Fake money only. Orders are local CSV simulations and never reach a broker.")
    ticker = st.selectbox("Paper ticker", selected_tickers, key="paper_ticker")
    side = st.radio("Action", ["BUY", "SELL"], horizontal=True)
    amount = st.number_input("Fake dollar amount", min_value=1.0, value=1_000.0, step=100.0)
    try:
        analysis = get_analysis(ticker, "1y", False)
        decision = analysis["decision"]
        price = decision["price"]
        shares = amount / price
        sizing = position_size(starting_cash, price, decision["stop_loss"],
                               risk_percent, max_allocation)
        st.table(pd.DataFrame([{
            "Ticker": ticker, "Action": f"Paper {side}", "Fake amount": f"${amount:,.2f}",
            "Current daily close": f"${price:,.2f}", "Estimated shares": f"{shares:,.4f}",
            "Stop-loss idea": f"${decision['stop_loss']:,.2f}",
            "Risk amount": f"${shares * (price - decision['stop_loss']):,.2f}",
        }]))
        if side == "BUY" and shares > sizing["max_shares"]:
            st.warning(f"This exceeds the risk manager ceiling of {sizing['max_shares']:,.4f} shares.")
        confirmed = st.checkbox("I understand this is a fake paper order.", key="paper_confirm")
        if st.button(f"Record Paper {side}", type="primary", disabled=not confirmed):
            execute_paper_order(ticker, side, amount, price, starting_cash, confirmed=True)
            st.success(f"Recorded fake {side.lower()} for {ticker}.")
            st.rerun()
    except Exception as error:
        st.error(f"Paper order preview unavailable: {error}")

    try:
        first = portfolio_summary(starting_cash)
        prices = {}
        if not first["positions"].empty:
            for symbol in first["positions"]["Ticker"]:
                try:
                    prices[symbol] = float(cached_prices(symbol, "1y")["Adj Close"].iloc[-1])
                except Exception:
                    pass
        portfolio = portfolio_summary(starting_cash, prices)
        metrics = st.columns(5)
        metrics[0].metric("Fake Cash", f"${portfolio['cash']:,.2f}")
        metrics[1].metric("Position Value", f"${portfolio['market_value']:,.2f}")
        metrics[2].metric("Paper Equity", f"${portfolio['equity']:,.2f}")
        metrics[3].metric("Unrealized P/L", f"${portfolio['unrealized_pnl']:,.2f}")
        metrics[4].metric("Realized P/L", f"${portfolio['realized_pnl']:,.2f}")
        st.markdown("#### Paper Portfolio")
        st.dataframe(portfolio["positions"], hide_index=True, use_container_width=True)
        with st.expander("Paper Trade History"):
            st.dataframe(portfolio["trades"], hide_index=True, use_container_width=True)
    except Exception as error:
        st.error(f"Could not load paper portfolio: {error}")

with tabs[5]:
    st.subheader("Risk Manager")
    ticker = st.selectbox("Risk analysis ticker", selected_tickers, key="risk_ticker")
    portfolio_value = st.number_input("Paper portfolio value", min_value=1_000.0,
                                      value=starting_cash, step=1_000.0)
    try:
        decision = get_analysis(ticker, period, False)["decision"]
        sizing = position_size(portfolio_value, decision["price"], decision["stop_loss"],
                               risk_percent, max_allocation)
        metrics = st.columns(6)
        metrics[0].metric("Max Shares", f"{sizing['max_shares']:,.4f}")
        metrics[1].metric("Position Value", f"${sizing['position_value']:,.2f}")
        metrics[2].metric("Risk Budget", f"${sizing['risk_budget']:,.2f}")
        metrics[3].metric("ATR Stop", f"${decision['stop_loss']:,.2f}")
        metrics[4].metric("2:1 Target", f"${decision['take_profit_2r']:,.2f}")
        metrics[5].metric("3:1 Target", f"${decision['take_profit_3r']:,.2f}")
        st.write(f"Maximum allocation: **{sizing['allocation_pct']:.1f}%** · "
                 f"risk per share: **${sizing['risk_per_share']:.2f}** · "
                 f"50-day SMA: **${decision['sma_50']:,.2f}**")
        for warning in risk_warnings(
            decision["final_score"], decision["atr_pct"], sizing["allocation_pct"]
        ):
            st.warning(warning)
    except Exception as error:
        st.error(f"Could not calculate risk: {error}")

with tabs[6]:
    st.subheader("Settings")
    broker = connect_broker()
    st.error(broker["message"])
    st.write({
        "OpenAI API key configured": bool(os.getenv("OPENAI_API_KEY")),
        "OpenAI model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "Broker status": broker["status"],
        "Real trading": "Hard disabled",
        "Paper storage": "paper_trades.csv and paper_portfolio.csv",
        "Price/fundamental provider": "Yahoo Finance via yfinance",
        "Default history": DEFAULT_SCAN_PERIOD,
    })
    st.markdown("#### Optional local `.env`")
    st.code("OPENAI_API_KEY=your_key\nOPENAI_MODEL=gpt-4.1-mini", language="bash")
    st.warning("Never commit API keys. OpenAI is optional; all core analysis works locally.")

st.divider()
st.caption(DISCLAIMER + " Market data can be delayed, incomplete, or inaccurate.")
