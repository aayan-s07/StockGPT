import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from ddgs import DDGS

warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Stock Intelligence", page_icon="📈", layout="wide")

st.title("📈 India Stock Intelligence")
st.caption("Research assistant for NSE/BSE-listed shares • educational research, not a guaranteed prediction or personalised financial advice")

st.warning(
    "Markets are uncertain. This app cannot guarantee profits or detect every price-moving event. "
    "Data may be delayed, incomplete, adjusted, or unavailable. Verify important facts against NSE/BSE filings "
    "and company reports before making any decision."
)

PERIODS = {
    "1 day": 1,
    "3 days": 3,
    "7 days": 7,
    "1 month (~21 sessions)": 21,
    "3 months (~63 sessions)": 63,
    "6 months (~126 sessions)": 126,
    "1 year (~252 sessions)": 252,
}

def safe_num(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None

def fmt_num(value, digits=2):
    value = safe_num(value)
    return "Not available" if value is None else f"{value:,.{digits}f}"

def fmt_pct(value, digits=2):
    value = safe_num(value)
    return "Not available" if value is None else f"{value * 100:.{digits}f}%"

def search_tickers(query):
    try:
        results = yf.Search(query, max_results=12, news_count=0).quotes
        candidates = []
        for item in results or []:
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or symbol
            exchange = item.get("exchange") or item.get("exchDisp") or ""
            if symbol and (symbol.endswith(".NS") or symbol.endswith(".BO") or exchange in ("NSI", "BSE", "NSE")):
                candidates.append({"label": f"{name} — {symbol} ({exchange})", "symbol": symbol, "name": name})
        # Ensure direct ticker input is also possible.
        q = query.strip().upper().replace(" ", "")
        if q.endswith(".NS") or q.endswith(".BO"):
            candidates.insert(0, {"label": f"Use entered ticker — {q}", "symbol": q, "name": q})
        seen, unique = set(), []
        for c in candidates:
            if c["symbol"] not in seen:
                unique.append(c); seen.add(c["symbol"])
        return unique
    except Exception:
        q = query.strip().upper().replace(" ", "")
        if q.endswith(".NS") or q.endswith(".BO"):
            return [{"label": q, "symbol": q, "name": q}]
        return []

def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high, low = df["High"], df["Low"]
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    df["BB_upper"] = mid + 2 * sd
    df["BB_lower"] = mid - 2 * sd
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["DailyReturn"] = close.pct_change()
    return df

def get_fundamentals(ticker):
    t = yf.Ticker(ticker)
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}
    try:
        financials = t.financials
    except Exception:
        financials = pd.DataFrame()
    try:
        balance = t.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cashflow = t.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    def find_row(frame, names):
        if frame is None or frame.empty:
            return None
        for name in names:
            if name in frame.index:
                row = frame.loc[name].dropna()
                if len(row):
                    return row
        return None

    revenue = find_row(financials, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income = find_row(financials, ["Net Income", "Net Income Common Stockholders"])
    operating_cash = find_row(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = find_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    debt = None
    equity = None
    if not balance.empty:
        debt_row = find_row(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        equity_row = find_row(balance, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        if debt_row is not None: debt = safe_num(debt_row.iloc[0])
        if equity_row is not None: equity = safe_num(equity_row.iloc[0])
    fcf = None
    if operating_cash is not None and capex is not None:
        ocf_value = safe_num(operating_cash.iloc[0])
        capex_value = safe_num(capex.iloc[0])
        if ocf_value is not None and capex_value is not None:
            fcf = ocf_value + capex_value  # Yahoo commonly records capex as a negative value.

    rows = [
        ("Current/last price", info.get("currentPrice") or info.get("regularMarketPrice"), "₹", "Latest provider quote; may be delayed."),
        ("Market capitalisation", info.get("marketCap"), "₹", "Market value of listed equity; provider currency/coverage may vary."),
        ("P/E (trailing)", info.get("trailingPE"), "x", "Price divided by trailing 12-month earnings per share. Compare within the same industry."),
        ("P/E (forward)", info.get("forwardPE"), "x", "Price divided by estimated future EPS; estimates can be wrong."),
        ("Price/book", info.get("priceToBook"), "x", "Price relative to accounting book value; especially useful for financial firms."),
        ("Price/sales", info.get("priceToSalesTrailing12Months"), "x", "Market value relative to revenue; ignores profitability."),
        ("EV/EBITDA", info.get("enterpriseToEbitda"), "x", "Enterprise value relative to operating earnings before selected costs."),
        ("EPS (trailing)", info.get("trailingEps"), "₹", "Trailing earnings per share."),
        ("EPS (forward)", info.get("forwardEps"), "₹", "Forecast earnings per share, if available."),
        ("Dividend yield", info.get("dividendYield"), "%", "Cash dividend yield; not guaranteed and price changes affect it."),
        ("ROE", info.get("returnOnEquity"), "%", "Profit relative to equity; high leverage can inflate it."),
        ("Profit margin", info.get("profitMargins"), "%", "Net income divided by revenue."),
        ("Operating margin", info.get("operatingMargins"), "%", "Operating profit relative to revenue."),
        ("Debt/equity (provider)", info.get("debtToEquity"), "provider ratio", "Definitions and units can differ; verify against statements."),
        ("Beta", info.get("beta"), "x", "Historical sensitivity to a benchmark, not a forecast."),
        ("Analyst target (provider)", info.get("targetMeanPrice"), "₹", "If supplied; not a promise or independently verified fair value."),
        ("Revenue (latest annual, provider statements)", safe_num(revenue.iloc[0]) if revenue is not None else None, "₹", "Latest available annual statement value."),
        ("Net income (latest annual, provider statements)", safe_num(net_income.iloc[0]) if net_income is not None else None, "₹", "Latest available annual statement value."),
        ("Free cash flow (rough, latest annual)", fcf, "₹", "Operating cash flow plus capex when capex is recorded as a negative number."),
    ]
    return t, info, pd.DataFrame(rows, columns=["Metric", "Value", "Unit", "Meaning"])

def news_and_web(ticker, company_name):
    articles = []
    try:
        tnews = yf.Ticker(ticker).get_news(count=12)
        for item in tnews or []:
            content = item.get("content", item)
            title = content.get("title") or item.get("title") or "Untitled"
            url = content.get("canonicalUrl") or content.get("clickThroughUrl") or item.get("link") or {}
            if isinstance(url, dict): url = url.get("url", "")
            provider = content.get("provider") or {}
            publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
            articles.append({"Source": publisher or "Yahoo Finance", "Headline": title, "URL": url})
    except Exception:
        pass
    web_results = []
    try:
        query = f'"{company_name}" stock India NSE BSE results announcement'
        for r in DDGS().text(query, max_results=10):
            web_results.append({"Title": r.get("title", ""), "URL": r.get("href", ""), "Snippet": r.get("body", "")})
    except Exception as e:
        web_results.append({"Title": "Web search unavailable", "URL": "", "Snippet": str(e)})
    return articles, web_results

def build_forecast(df):
    # Transparent scenario ranges from historical volatility, not a trained predictive model.
    close = df["Close"].dropna()
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 60:
        return pd.DataFrame(), "Not enough history for a useful volatility estimate (need roughly 60 trading sessions)."
    price = float(close.iloc[-1])
    daily_vol = float(rets.tail(252).std())
    # Conservative drift estimate: blend recent 20-session momentum with a long-run daily drift estimate,
    # then cap it so a tiny sample cannot create extreme forecasts.
    recent = float(rets.tail(20).mean()) if len(rets) >= 20 else 0.0
    longrun = float(rets.tail(252).mean())
    drift = float(np.clip(0.35 * recent + 0.65 * longrun, -0.002, 0.002))
    rows = []
    for label, days in PERIODS.items():
        mu = drift * days
        sigma = daily_vol * math.sqrt(days)
        expected = price * math.exp(mu)
        low80 = price * math.exp(mu - 1.282 * sigma)
        high80 = price * math.exp(mu + 1.282 * sigma)
        low95 = price * math.exp(mu - 1.96 * sigma)
        high95 = price * math.exp(mu + 1.96 * sigma)
        rows.append({
            "Horizon": label, "Sessions": days, "Model midpoint (₹)": expected,
            "Illustrative 80% range low (₹)": low80, "Illustrative 80% range high (₹)": high80,
            "Illustrative 95% range low (₹)": low95, "Illustrative 95% range high (₹)": high95,
            "Midpoint change": expected / price - 1, "Historical daily volatility": daily_vol,
        })
    return pd.DataFrame(rows), None

def simple_technical_read(df):
    last = df.iloc[-1]
    price = safe_num(last.get("Close"))
    rsi = safe_num(last.get("RSI14"))
    sma20, sma50, sma200 = [safe_num(last.get(k)) for k in ("SMA20", "SMA50", "SMA200")]
    macd, sig = safe_num(last.get("MACD")), safe_num(last.get("MACD_signal"))
    vol, vavg = safe_num(last.get("Volume")), safe_num(last.get("VolAvg20"))
    notes = []
    if rsi is not None:
        notes.append(f"RSI(14) is {rsi:.1f}: " + ("high momentum/possibly overbought; not automatically a sell" if rsi >= 70 else "weak momentum/possibly oversold; not automatically a buy" if rsi <= 30 else "between the common 30/70 reference levels"))
    if price and sma20:
        notes.append("Price is " + ("above" if price > sma20 else "below") + " its 20-session SMA.")
    if price and sma50:
        notes.append("Price is " + ("above" if price > sma50 else "below") + " its 50-session SMA.")
    if sma50 and sma200:
        notes.append("50-session SMA is " + ("above" if sma50 > sma200 else "below") + " the 200-session SMA.")
    if macd is not None and sig is not None:
        notes.append("MACD is " + ("above" if macd > sig else "below") + " its signal line.")
    if vol is not None and vavg:
        notes.append(f"Latest volume is {vol:,.0f}; 20-session average is {vavg:,.0f}.")
    return notes

query = st.text_input("Enter a company name or Yahoo Finance ticker", value="Sagility")
if st.button("Research stock", type="primary"):
    with st.spinner("Finding ticker candidates…"):
        candidates = search_tickers(query)
    if not candidates:
        st.error("No NSE/BSE ticker candidates found. Try a more specific company name or enter a ticker such as TCS.NS or 500570.BO.")
        st.stop()
    st.session_state["candidates"] = candidates
    st.session_state["query"] = query

if "candidates" in st.session_state:
    candidates = st.session_state["candidates"]
    labels = [c["label"] for c in candidates]
    chosen_label = st.selectbox("Confirm the exact listing", labels)
    chosen = candidates[labels.index(chosen_label)]
    ticker = chosen["symbol"]
    st.caption(f"Selected ticker: {ticker}. Confirm this is the intended company before interpreting the results.")
    if st.button("Run full analysis"):
        with st.spinner("Downloading market data and calculating indicators…"):
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                st.error("Price history unavailable for this ticker. Check the symbol or try later.")
                st.stop()
            hist = hist.dropna(subset=["Close"])
            df = add_indicators(hist)
            t, info, fundamentals = get_fundamentals(ticker)
            news, web_results = news_and_web(ticker, chosen.get("name", query))
            forecast, forecast_error = build_forecast(df)

        latest = df.iloc[-1]
        price = float(latest["Close"])
        prev = df["Close"].iloc[-2] if len(df) > 1 else price
        change = price / prev - 1 if prev else 0
        st.subheader(f"{chosen.get('name', query)} ({ticker})")
        a, b, c, d = st.columns(4)
        a.metric("Latest available close", f"₹{price:,.2f}", f"{change*100:+.2f}% vs previous close")
        b.metric("RSI (14)", fmt_num(latest.get("RSI14"), 1))
        b.caption("Momentum indicator, not a standalone buy/sell signal.")
        c.metric("20-session volatility", fmt_pct(df["DailyReturn"].tail(20).std() * np.sqrt(252)))
        d.metric("Data through", str(df.index[-1].date()))

        st.subheader("Price chart and moving averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
        for col in ["SMA20", "SMA50", "SMA200"]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        fig.update_layout(height=480, xaxis_title="Date", yaxis_title="₹", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Technical analysis", "Fundamentals", "Forecast scenarios", "News & web research", "Data & limitations"])
        with tab1:
            st.markdown("### Indicator interpretation")
            for note in simple_technical_read(df):
                st.write("• " + note)
            tech_cols = ["Close", "SMA20", "SMA50", "SMA200", "RSI14", "MACD", "MACD_signal", "BB_upper", "BB_lower", "ATR14", "Volume", "VolAvg20"]
            st.dataframe(df[tech_cols].tail(30).sort_index(ascending=False), use_container_width=True)
            st.caption("Indicators use daily candles. Intraday prices, splits, low liquidity and data errors can change the picture.")
        with tab2:
            st.markdown("### Provider-reported fundamentals")
            st.dataframe(fundamentals, use_container_width=True, hide_index=True)
            if not t.financials.empty:
                st.markdown("### Annual income statement (provider data)")
                st.dataframe(t.financials, use_container_width=True)
            if not t.balance_sheet.empty:
                st.markdown("### Balance sheet")
                st.dataframe(t.balance_sheet, use_container_width=True)
            if not t.cashflow.empty:
                st.markdown("### Cash flow statement")
                st.dataframe(t.cashflow, use_container_width=True)
            st.info("A low P/E does not automatically mean cheap; it can reflect declining earnings or risk. Compare peers, accounting quality, growth, capital intensity and the latest official filings.")
        with tab3:
            st.markdown("### Statistical scenario ranges — not price targets")
            if forecast_error:
                st.warning(forecast_error)
            else:
                st.dataframe(forecast, use_container_width=True, hide_index=True)
                st.caption(
                    "These ranges are generated from historical daily-return drift and volatility using a simple log-normal scenario model. "
                    "They are NOT AI predictions, guarantees, calibrated probabilities, or buy/sell recommendations. Real markets have jumps, news shocks, regime changes and fat tails."
                )
                selected_horizon = st.selectbox("Inspect horizon", list(PERIODS.keys()), key="horizon")
                row = forecast[forecast["Horizon"] == selected_horizon].iloc[0]
                st.write(f"Model midpoint: ₹{row['Model midpoint (₹)']:,.2f} ({row['Midpoint change']*100:+.2f}% vs last close)")
                st.write(f"Illustrative 80% range: ₹{row['Illustrative 80% range low (₹)']:,.2f} – ₹{row['Illustrative 80% range high (₹)']:,.2f}")
                st.write(f"Illustrative 95% range: ₹{row['Illustrative 95% range low (₹)']:,.2f} – ₹{row['Illustrative 95% range high (₹)']:,.2f}")
        with tab4:
            st.markdown("### Latest provider news")
            if news:
                for item in news:
                    st.markdown(f"**{item['Headline']}**")
                    st.caption(item["Source"])
                    if item["URL"]:
                        st.markdown(f"[Open article]({item['URL']})")
                    st.divider()
            else:
                st.info("No provider news returned. This does not mean there is no news.")
            st.markdown("### Public web search results")
            for item in web_results:
                st.markdown(f"**[{item['Title'] or 'Untitled'}]({item['URL']})**")
                st.write(item["Snippet"])
                st.divider()
            st.caption("Search results are not exhaustive and can include stale, duplicate, promotional or inaccurate material. Open and verify the original source.")
        with tab5:
            st.markdown("### Important limitations")
            st.write("- This prototype does not inspect every website or every possible price-moving variable.")
            st.write("- It does not have guaranteed real-time NSE/BSE data; the free sources can be delayed, rate-limited or unavailable.")
            st.write("- It does not reliably parse every NSE/BSE corporate filing, concall transcript, legal proceeding, order book, FII/DII flow or global macroeconomic series.")
            st.write("- Yahoo Finance/yfinance is an unofficial community tool for personal research; respect its terms and data-provider conditions.")
            st.write("- The news search is keyword-based, not a verified sentiment engine. Headlines need human verification.")
            st.write("- Missing fundamentals are shown as unavailable, not zero.")
            st.write("- Do not use the output alone to trade. Test any strategy on historical data with transaction costs, slippage and out-of-sample validation.")
            st.markdown("### Official sources to verify")
            st.markdown("- [NSE India](https://www.nseindia.com/)")
            st.markdown("- [BSE India](https://www.bseindia.com/)")
            st.markdown("- [SEBI Investor](https://investor.sebi.gov.in/)")
        st.success("Analysis completed. Treat this as a research dashboard, not a guaranteed prediction.")
        st.download_button("Download price history CSV", df.to_csv().encode("utf-8"), file_name=f"{ticker.replace('.', '_')}_history.csv", mime="text/csv")
        st.download_button("Download fundamentals CSV", fundamentals.to_csv(index=False).encode("utf-8"), file_name=f"{ticker.replace('.', '_')}_fundamentals.csv", mime="text/csv")

st.markdown("---")
st.caption("Built for learning and research. No broker connection, no order placement, and no automatic trading.")
