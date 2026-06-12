"""Streamlit UI for the educational full-market AI quant scanner."""

from __future__ import annotations

import os
from time import perf_counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from ai_engine import analyze_stock
from backtest import run_backtest
from broker import connect_broker
from config import (
    APP_NAME, DEFAULT_MAX_ALLOCATION, DEFAULT_MAX_WORKERS, DEFAULT_PAPER_CASH,
    DEFAULT_RISK_PER_TRADE, DISCLAIMER, FAILED_TICKERS_FILE, MAX_SCAN_WORKERS,
    PRICE_CACHE_TTL_HOURS, SCANNER_RESULTS_FILE, SCAN_PERIODS, SCORE_MAXIMUMS,
)
from data import load_company_info, load_stock_data, load_stock_data_batch
from indicators import add_indicators
from paper_trading import execute_paper_order, portfolio_summary
from risk import position_size, risk_warnings
from scanner import analyze_ticker, scan_tickers
from universe import build_universe

load_dotenv()
st.set_page_config(page_title=APP_NAME, layout="wide")


@st.cache_data(ttl=PRICE_CACHE_TTL_HOURS * 3600, show_spinner=False)
def cached_prices(ticker: str, period: str) -> pd.DataFrame:
    return load_stock_data(ticker, period)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_info(ticker: str) -> dict:
    return load_company_info(ticker)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_universe(source: str, asset_type: str, include_otc: bool):
    return build_universe(source, asset_type, include_otc)


def analysis_for(ticker: str, period: str, ml: bool = True) -> dict:
    return analyze_ticker(ticker, period, cached_prices, cached_info, ml)


def ai_payload(analysis: dict) -> dict:
    return {
        "ticker": analysis["ticker"],
        "decision": analysis["decision"],
        "fundamentals": analysis["fundamentals"],
        "ml": {key: value for key, value in analysis["ml"].items()
               if key not in {"models", "features"}},
    }


def show_ai(analysis: dict, use_openai: bool) -> None:
    result = analyze_stock(ai_payload(analysis), use_openai)
    st.subheader(f"{result['ticker']}: {result['rating']}")
    st.caption(f"{result['source']} | confidence {result['confidence']}%")
    st.write(result["plain_english_summary"])
    columns = st.columns(5)
    columns[0].metric("Best buy zone", result["best_buy_zone"])
    columns[1].metric("Stop", result["stop_loss"])
    columns[2].metric("Target 1", result["take_profit_1"])
    columns[3].metric("Target 2", result["take_profit_2"])
    columns[4].metric("Exit zone", result["sell_or_exit_zone"])
    st.warning(f"Avoid zone: {result['avoid_zone']}")
    reasons, risks, changes = st.columns(3)
    reasons.write({"Main reasons": result["main_reasons"]})
    risks.write({"Main risks": result["main_risks"]})
    changes.write({"What changes the view": result["what_would_change_my_mind"]})
    st.info(result["beginner_explanation"])


def price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"],
        close=data["Close"], name=ticker,
    ))
    for column, color in [
        ("SMA_20", "#16a34a"), ("SMA_50", "#f59e0b"), ("SMA_200", "#dc2626"),
        ("EMA_9", "#0891b2"), ("EMA_21", "#7c3aed"),
    ]:
        figure.add_scatter(x=data.index, y=data[column], name=column, line={"color": color})
    row = data.dropna(subset=["Support", "Resistance"]).iloc[-1]
    figure.add_hline(y=row["Support"], line_dash="dot", line_color="green")
    figure.add_hline(y=row["Resistance"], line_dash="dot", line_color="red")
    figure.update_layout(height=580, xaxis_rangeslider_visible=False, hovermode="x unified")
    return figure


def technical_chart(data: pd.DataFrame) -> go.Figure:
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True)
    figure.add_scatter(x=data.index, y=data["RSI"], name="RSI", row=1, col=1)
    figure.add_scatter(x=data.index, y=data["MACD"], name="MACD", row=2, col=1)
    figure.add_scatter(x=data.index, y=data["MACD_Signal"], name="Signal", row=2, col=1)
    figure.add_bar(x=data.index, y=data["Volume"], name="Volume", row=3, col=1)
    figure.update_layout(height=650, hovermode="x unified")
    return figure


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_money(value: object) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"${number:.2f}"


def safe_percent(value: object) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:+.2f}%"


def safe_number(value: object) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.1f}"


def safe_ratio(value: object) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.2f}"


def result_table(frame: pd.DataFrame) -> None:
    formatters = {
        "Latest Price": safe_money,
        "Daily %": safe_percent,
        "Final Score": safe_number,
        "ML Probability Up 10D": safe_percent,
        "RSI": safe_number,
        "Relative Volume": safe_ratio,
        "ATR %": safe_percent,
        "Risk/Reward": safe_ratio,
        "Stop-Loss Idea": safe_money,
        "Take-Profit Idea": safe_money,
    }
    available_formatters = {
        column: formatter
        for column, formatter in formatters.items()
        if column in frame.columns
    }
    styled_frame = frame.style.format(available_formatters, na_rep="N/A")
    st.dataframe(
        styled_frame,
        hide_index=True,
        use_container_width=True,
        height=620,
    )


st.title(APP_NAME)
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Global Settings")
    period = st.selectbox("History period", SCAN_PERIODS, index=1)
    max_workers = st.slider("Max workers", 1, MAX_SCAN_WORKERS, DEFAULT_MAX_WORKERS)
    enable_ml = st.toggle("Train ML models", value=True)
    use_openai = st.toggle(
        "Use optional OpenAI analyst", value=False,
        disabled=not bool(os.getenv("OPENAI_API_KEY")),
    )
    risk_percent = st.slider("Risk per fake trade (%)", 0.25, 2.0,
                             DEFAULT_RISK_PER_TRADE, 0.25)
    max_allocation = st.slider("Max allocation (%)", 1.0, 20.0,
                               DEFAULT_MAX_ALLOCATION, 1.0)
    starting_cash = st.number_input("Starting fake cash", 1_000.0, value=DEFAULT_PAPER_CASH)

tabs = st.tabs([
    "Full Market Scanner", "Best Chances", "Single Stock Deep Dive", "AI Analyst",
    "Backtest", "Paper Trading", "Risk Manager", "Settings",
])

with tabs[0]:
    st.subheader("Full Market Scanner")
    controls = st.columns(4)
    source = controls[0].selectbox("Universe source", ["Nasdaq Trader", "Alpaca (optional)"])
    asset_type = controls[1].selectbox("Asset type", ["Stocks + ETFs", "Stocks only", "ETFs only"])
    scan_choice = controls[2].selectbox("Scan limit", ["50", "100", "500", "1000", "All"])
    include_otc = controls[3].checkbox("Include OTC", value=False)
    filters = st.columns(4)
    min_price = filters[0].number_input("Minimum price", 0.0, value=1.0)
    max_price = filters[1].number_input("Maximum price", 0.0, value=10_000.0)
    min_volume = filters[2].number_input("Minimum average volume", 0, value=500_000, step=100_000)
    min_market_cap = filters[3].number_input("Minimum market cap", 0, value=0, step=100_000_000)
    minimum_score = st.slider("Minimum score", 0, 100, 0)
    options = st.columns(2)
    full_scan = options[0].checkbox(
        "Full scan mode",
        value=False,
        help="Large scans can take a long time and providers may rate-limit requests.",
    )
    resume_scan = options[1].checkbox(
        "Resume previous scan results",
        value=False,
        help="Skip selected tickers already present in scanner_results.csv.",
    )
    if st.button("Run market scan", type="primary", use_container_width=True):
        scan_started = perf_counter()
        universe, universe_warnings = cached_universe(source, asset_type, include_otc)
        limit = len(universe) if scan_choice == "All" else int(scan_choice)
        if scan_choice == "All" and not full_scan:
            limit = min(1000, len(universe))
            universe_warnings.append("Enable full scan mode to scan more than 1,000 symbols.")
        selected = universe.head(limit)
        metadata = selected.set_index("ticker")[["asset_type", "exchange"]].to_dict("index")
        bar = st.progress(0, "Preparing scan")
        resume_results = None
        if resume_scan and SCANNER_RESULTS_FILE.exists():
            try:
                resume_results = pd.read_csv(SCANNER_RESULTS_FILE)
            except Exception as error:
                universe_warnings.append(f"Previous scan results could not be loaded: {error}")

        def progress(ticker: str, count: int) -> None:
            bar.progress(count / len(selected), f"{ticker}: {count}/{len(selected)}")

        completed = set()
        if resume_results is not None and "Ticker" in resume_results:
            completed = set(resume_results["Ticker"].dropna().astype(str))
        pending = [
            ticker for ticker in selected["ticker"].tolist()
            if ticker not in completed
        ]
        bar.progress(0, f"Loading price data for {len(pending)} tickers")
        prefetched_prices = load_stock_data_batch(pending, period)

        def scan_price_loader(ticker: str, scan_period: str) -> pd.DataFrame:
            if ticker in prefetched_prices:
                return prefetched_prices[ticker].copy()
            return cached_prices(ticker, scan_period)

        results, errors, analyses, summary = scan_tickers(
            selected["ticker"], period, scan_price_loader, cached_info, enable_ml,
            progress, max_workers, metadata, resume_results=resume_results,
            return_summary=True,
        )
        summary["scan_time_seconds"] = round(perf_counter() - scan_started, 2)
        if not results.empty:
            market_caps = {ticker: item["fundamentals"].get("market_cap")
                           for ticker, item in analyses.items()}
            results["Market Cap"] = results["Ticker"].map(market_caps)
            results = results[
                (results["Latest Price"] >= min_price)
                & (results["Latest Price"] <= max_price)
                & (results["Average Volume"] >= min_volume)
                & (results["Final Score"] >= minimum_score)
                & ((results["Market Cap"].fillna(0) >= min_market_cap) | (min_market_cap == 0))
            ].reset_index(drop=True)
            results["Rank"] = range(1, len(results) + 1)
        st.session_state.update(
            scan_results=results, scan_errors=errors, scan_analyses=analyses,
            scan_summary=summary, universe_warnings=universe_warnings,
        )
        bar.empty()
    summary = st.session_state.get("scan_summary")
    if summary:
        summary_columns = st.columns(5)
        summary_columns[0].metric("Requested", summary["total_tickers_requested"])
        summary_columns[1].metric("Successful", summary["successful_tickers"])
        summary_columns[2].metric("Failed", summary["failed_tickers"])
        summary_columns[3].metric("Skipped", summary["skipped_tickers"])
        summary_columns[4].metric("Scan time", f"{summary['scan_time_seconds']:.1f}s")
    results = st.session_state.get("scan_results", pd.DataFrame())
    if results.empty:
        st.info("Run a 50-symbol scan first to validate provider access and performance.")
    else:
        result_table(results)
        st.download_button("Download scanner results", results.to_csv(index=False),
                           "scanner_results.csv", "text/csv")
    for warning in st.session_state.get("universe_warnings", []):
        st.warning(warning)
    errors = st.session_state.get("scan_errors", {})
    if errors:
        failed_frame = pd.DataFrame(
            [{"Ticker": ticker, "Error": error} for ticker, error in errors.items()]
        )
        st.download_button(
            "Download failed tickers",
            failed_frame.to_csv(index=False),
            FAILED_TICKERS_FILE.name,
            "text/csv",
        )
        with st.expander(f"{len(errors)} symbols failed without stopping the scan"):
            st.dataframe(failed_frame, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Best Chances")
    results = st.session_state.get("scan_results", pd.DataFrame())
    analyses = st.session_state.get("scan_analyses", {})
    if results.empty:
        st.info("Run the Full Market Scanner first.")
    else:
        groups = {
            "Elite / Strong Buy Watch": results[results["Final Score"] >= 80].head(10),
            "Best risk/reward": results.sort_values("Risk/Reward", ascending=False).head(10),
            "Momentum breakouts": results.sort_values(
                ["Relative Volume", "Daily %"], ascending=False
            ).head(10),
            "High-quality trends": results.sort_values(
                ["Quant Score", "Final Score"], ascending=False
            ).head(10),
            "Sell / Exit Watch": results.sort_values("Final Score").head(10),
        }
        for title, frame in groups.items():
            st.markdown(f"### {title}")
            for row in frame.itertuples(index=False):
                ticker = getattr(row, "Ticker")
                if ticker not in analyses:
                    continue
                decision = analyses[ticker]["decision"]
                with st.expander(f"{ticker} | {decision['final_signal']} | {decision['final_score']:.1f}"):
                    st.write("Why ranked:", " ".join(decision["strengths"]))
                    st.write("Risk:", " ".join(decision["risks"]))
                    st.write(f"Invalidation: close below ${decision['stop_loss']:.2f} "
                             f"or score below 45.")

with tabs[2]:
    st.subheader("Single Stock Deep Dive")
    deep_ticker = st.text_input("Ticker", "AAPL").strip().upper()
    if st.button("Analyze ticker", key="deep_button"):
        try:
            st.session_state["deep_analysis"] = analysis_for(deep_ticker, period, enable_ml)
        except Exception as error:
            st.error(str(error))
    analysis = st.session_state.get("deep_analysis")
    if analysis:
        decision, data = analysis["decision"], analysis["data"]
        metrics = st.columns(5)
        metrics[0].metric("Signal", decision["final_signal"])
        metrics[1].metric("Final score", f"{decision['final_score']:.1f}")
        metrics[2].metric("Price", f"${decision['price']:.2f}")
        probability = decision["probability_up_10d"]
        metrics[3].metric("ML up 10D", "N/A" if probability is None else f"{probability:.1%}")
        metrics[4].metric("Risk/reward", f"{decision['risk_reward_ratio']:.1f}:1")
        st.plotly_chart(price_chart(data, analysis["ticker"]), use_container_width=True)
        score = pd.DataFrame({
            "Section": list(decision["category_scores"]),
            "Points": list(decision["category_scores"].values()),
            "Maximum": list(SCORE_MAXIMUMS.values()),
        })
        st.dataframe(score, hide_index=True)
        show_ai(analysis, use_openai)
        st.plotly_chart(technical_chart(data), use_container_width=True)

with tabs[3]:
    st.subheader("AI Analyst")
    results = st.session_state.get("scan_results", pd.DataFrame())
    analyses = st.session_state.get("scan_analyses", {})
    if results.empty:
        st.info("Run a scan first. AI is intentionally limited to top 20, worst 20, or one selected stock.")
    else:
        allowed = list(dict.fromkeys(
            results.head(20)["Ticker"].tolist() + results.tail(20)["Ticker"].tolist()
        ))
        ticker = st.selectbox("Top/worst ranked ticker", allowed)
        if ticker in analyses:
            show_ai(analyses[ticker], use_openai)

with tabs[4]:
    st.subheader("Backtest")
    ticker = st.text_input("Backtest ticker", "SPY")
    fees = st.number_input("Fee per fake trade", 0.0, value=0.0)
    slippage = st.number_input("Slippage (%)", 0.0, value=0.05, step=0.01)
    if st.button("Run backtest"):
        try:
            data = add_indicators(cached_prices(ticker.upper(), period))
            result = run_backtest(data, risk_percent=risk_percent,
                                  fee_per_trade=fees, slippage_pct=slippage)
            metrics = st.columns(6)
            metrics[0].metric("Total return", f"{result['total_return_pct']:.2f}%")
            metrics[1].metric("Buy/hold", f"{result['buy_and_hold_return_pct']:.2f}%")
            metrics[2].metric("Max drawdown", f"{result['max_drawdown_pct']:.2f}%")
            metrics[3].metric("Win rate", "N/A" if result["win_rate_pct"] is None else f"{result['win_rate_pct']:.1f}%")
            metrics[4].metric("Profit factor", "N/A" if result["profit_factor"] is None else f"{result['profit_factor']:.2f}")
            metrics[5].metric("Trades", result["number_of_trades"])
            chart = go.Figure()
            chart.add_scatter(x=result["history"].index,
                              y=result["history"]["Strategy Portfolio"], name="Strategy")
            chart.add_scatter(x=result["history"].index,
                              y=result["history"]["Buy and Hold Portfolio"], name="Buy and hold")
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(result["trades"], hide_index=True, use_container_width=True)
            st.warning("Backtests are simplified and do not predict future performance.")
        except Exception as error:
            st.error(str(error))

with tabs[5]:
    st.subheader("Paper Trading")
    st.info("Fake money only. Every simulated order requires confirmation.")
    ticker = st.text_input("Paper ticker", "AAPL", key="paper_ticker").upper()
    side = st.radio("Paper action", ["BUY", "SELL"], horizontal=True)
    amount = st.number_input("Fake dollar amount", 1.0, value=1_000.0)
    confirmed = st.checkbox("Confirm this is a fake paper order")
    try:
        price = float(cached_prices(ticker, "1y")["Adj Close"].iloc[-1])
        st.write(f"Latest daily close: ${price:.2f}")
        if st.button("Record paper trade", disabled=not confirmed):
            execute_paper_order(ticker, side, amount, price, starting_cash, confirmed=True)
            st.success("Fake trade recorded locally.")
        portfolio = portfolio_summary(starting_cash, {ticker: price})
        columns = st.columns(4)
        columns[0].metric("Fake cash", f"${portfolio['cash']:,.2f}")
        columns[1].metric("Equity", f"${portfolio['equity']:,.2f}")
        columns[2].metric("Realized P/L", f"${portfolio['realized_pnl']:,.2f}")
        columns[3].metric("Unrealized P/L", f"${portfolio['unrealized_pnl']:,.2f}")
        st.dataframe(portfolio["positions"], hide_index=True, use_container_width=True)
        st.dataframe(portfolio["trades"], hide_index=True, use_container_width=True)
    except Exception as error:
        st.error(str(error))

with tabs[6]:
    st.subheader("Risk Manager")
    ticker = st.text_input("Risk ticker", "AAPL", key="risk_ticker").upper()
    portfolio_value = st.number_input("Fake portfolio value", 1_000.0, value=100_000.0)
    if st.button("Calculate risk plan"):
        try:
            decision = analysis_for(ticker, period, False)["decision"]
            sizing = position_size(portfolio_value, decision["price"], decision["stop_loss"],
                                   risk_percent, max_allocation)
            st.write({
                "suggested_stop_loss": decision["stop_loss"],
                "atr_stop": decision["atr_stop"],
                "50_sma_stop": decision["sma_50_stop"],
                "support_stop": decision["support_stop"],
                "take_profit_2_to_1": decision["take_profit_2r"],
                "take_profit_3_to_1": decision["take_profit_3r"],
                "max_position_size_shares": sizing["max_shares"],
                "suggested_fake_dollar_amount": sizing["position_value"],
                "risk_per_share": sizing["risk_per_share"],
            })
            for warning in risk_warnings(
                decision["final_score"], decision["atr_pct"], sizing["allocation_pct"],
                average_volume=decision["volume_avg_20"],
            ):
                st.warning(warning)
        except Exception as error:
            st.error(str(error))

with tabs[7]:
    st.subheader("Settings")
    broker = connect_broker()
    st.error(broker["message"])
    st.write({
        "OpenAI configured": bool(os.getenv("OPENAI_API_KEY")),
        "OpenAI model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "Real trading": "Hard disabled",
        "Universe": "Nasdaq Trader with optional SEC/Alpaca enrichment",
        "Price data": "Yahoo Finance via yfinance",
        "Local cache": "cache/",
        "Scanner export": "scanner_results.csv",
    })

st.divider()
st.caption(DISCLAIMER + " Market data may be delayed, incomplete, revised, or unavailable.")
