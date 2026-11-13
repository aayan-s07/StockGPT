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

st.set_page_config(page_title="StockGPT • Market Playground", page_icon="🌍", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f7f8ff 0%,#ffffff 42%);}
.block-container {padding-top:1.5rem; max-width:1450px;}
.hero {padding:1.4rem 1.6rem;border-radius:22px;background:linear-gradient(120deg,#3425a8,#5b4be7 55%,#00a6a6);color:white;margin-bottom:1rem;}
.hero h1 {color:white;margin:0;font-size:2.3rem;} .hero p {color:#e9e7ff;margin:.35rem 0 0 0;}
div[data-testid="stMetric"] {background:white;border:1px solid #e7e8f3;padding:14px;border-radius:16px;box-shadow:0 4px 14px #2020500a;}
div[data-testid="stTabs"] button {border-radius:12px 12px 0 0;}
</style>
<div class="hero"><h1>🌍 StockGPT</h1><p>Your global-market research playground — built for Indian investors 🇮🇳</p></div>
""", unsafe_allow_html=True)
st.caption("Global ticker lookup • INR/USD conversions • transparent research signals • intraday scenario estimates")


st.warning(
    "Markets are uncertain. This app cannot guarantee profits or detect every price-moving event. "
    "Free-provider quotes may be delayed, rate-limited or incomplete; intraday estimates are scenarios, not reliable promises. "
    "Verify important facts against official exchange filings before making decisions."
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
            if symbol:
                currency = item.get("currency") or ""
                candidates.append({"label": f"{name} — {symbol} ({exchange or 'listing'}; {currency or 'currency unknown'})", "symbol": symbol, "name": name})
        # Ensure direct ticker input is also possible.
        raw = query.strip()
        q = raw.upper().replace(" ", "")
        if raw and " " not in raw and ("." in raw or raw.isupper()):
            candidates.insert(0, {"label": f"Try entered ticker — {q}", "symbol": q, "name": q})
        seen, unique = set(), []
        for c in candidates:
            if c["symbol"] not in seen:
                unique.append(c); seen.add(c["symbol"])
        return unique
    except Exception:
        raw = query.strip()
        q = raw.upper().replace(" ", "")
        if q and " " not in raw:
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

    quote_currency = str(info.get("currency") or ("INR" if ticker.endswith((".NS", ".BO")) else "provider currency"))
    rows = [
        ("Current/last price", info.get("currentPrice") or info.get("regularMarketPrice"), quote_currency, "Latest provider quote; may be delayed."),
        ("Market capitalisation", info.get("marketCap"), quote_currency, "Market value of listed equity; provider currency/coverage may vary."),
        ("P/E (trailing)", info.get("trailingPE"), "x", "Price divided by trailing 12-month earnings per share. Compare within the same industry."),
        ("P/E (forward)", info.get("forwardPE"), "x", "Price divided by estimated future EPS; estimates can be wrong."),
        ("Price/book", info.get("priceToBook"), "x", "Price relative to accounting book value; especially useful for financial firms."),
        ("Price/sales", info.get("priceToSalesTrailing12Months"), "x", "Market value relative to revenue; ignores profitability."),
        ("EV/EBITDA", info.get("enterpriseToEbitda"), "x", "Enterprise value relative to operating earnings before selected costs."),
        ("EPS (trailing)", info.get("trailingEps"), quote_currency, "Trailing earnings per share."),
        ("EPS (forward)", info.get("forwardEps"), quote_currency, "Forecast earnings per share, if available."),
        ("Dividend yield", info.get("dividendYield"), "%", "Cash dividend yield; not guaranteed and price changes affect it."),
        ("ROE", info.get("returnOnEquity"), "%", "Profit relative to equity; high leverage can inflate it."),
        ("Profit margin", info.get("profitMargins"), "%", "Net income divided by revenue."),
        ("Operating margin", info.get("operatingMargins"), "%", "Operating profit relative to revenue."),
        ("Debt/equity (provider)", info.get("debtToEquity"), "provider ratio", "Definitions and units can differ; verify against statements."),
        ("Beta", info.get("beta"), "x", "Historical sensitivity to a benchmark, not a forecast."),
        ("Analyst target (provider)", info.get("targetMeanPrice"), quote_currency, "If supplied; not a promise or independently verified fair value."),
        ("Revenue (latest annual, provider statements)", safe_num(revenue.iloc[0]) if revenue is not None else None, quote_currency, "Latest available annual statement value."),
        ("Net income (latest annual, provider statements)", safe_num(net_income.iloc[0]) if net_income is not None else None, quote_currency, "Latest available annual statement value."),
        ("Free cash flow (rough, latest annual)", fcf, quote_currency, "Operating cash flow plus capex when capex is recorded as a negative number."),
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


def _signal_row(metric, value, signal, reason, score):
    return {"Metric": metric, "Latest value": value, "Signal": signal, "Signal basis": reason, "Score": score}


def build_technical_signals(df):
    """Transparent, rule-based readings of common daily technical indicators."""
    last = df.iloc[-1]
    price = safe_num(last.get("Close"))
    rows = []

    for label, key in [("Price vs SMA20", "SMA20"), ("Price vs SMA50", "SMA50"), ("Price vs SMA200", "SMA200")]:
        average = safe_num(last.get(key))
        if price is None or average is None:
            rows.append(_signal_row(label, "N/A", "N/A", "Insufficient history to calculate this comparison.", None))
        elif price > average:
            rows.append(_signal_row(label, f"₹{price:,.2f} vs ₹{average:,.2f}", "BUY", f"Price is above {key}, a positive trend condition in this simple rule set.", 1))
        elif price < average:
            rows.append(_signal_row(label, f"₹{price:,.2f} vs ₹{average:,.2f}", "SELL", f"Price is below {key}, a negative trend condition in this simple rule set.", -1))
        else:
            rows.append(_signal_row(label, f"₹{price:,.2f} vs ₹{average:,.2f}", "HOLD", "Price and moving average are equal; no direction from this comparison.", 0))

    sma50, sma200 = safe_num(last.get("SMA50")), safe_num(last.get("SMA200"))
    if sma50 is None or sma200 is None:
        rows.append(_signal_row("SMA50 vs SMA200", "N/A", "N/A", "Insufficient history for the long-term moving-average comparison.", None))
    elif sma50 > sma200:
        rows.append(_signal_row("SMA50 vs SMA200", f"₹{sma50:,.2f} vs ₹{sma200:,.2f}", "BUY", "SMA50 is above SMA200, a commonly watched positive trend condition.", 1))
    elif sma50 < sma200:
        rows.append(_signal_row("SMA50 vs SMA200", f"₹{sma50:,.2f} vs ₹{sma200:,.2f}", "SELL", "SMA50 is below SMA200, a commonly watched negative trend condition.", -1))
    else:
        rows.append(_signal_row("SMA50 vs SMA200", f"₹{sma50:,.2f} vs ₹{sma200:,.2f}", "HOLD", "The moving averages are equal.", 0))

    rsi = safe_num(last.get("RSI14"))
    if rsi is None:
        rows.append(_signal_row("RSI (14)", "N/A", "N/A", "RSI is not available yet.", None))
    elif rsi >= 70:
        rows.append(_signal_row("RSI (14)", f"{rsi:.1f}", "SELL", "High RSI can mean momentum is stretched; strong trends can remain overbought.", -1))
    elif rsi >= 55:
        rows.append(_signal_row("RSI (14)", f"{rsi:.1f}", "BUY", "RSI is in a positive-momentum zone under this heuristic.", 1))
    elif rsi <= 30:
        rows.append(_signal_row("RSI (14)", f"{rsi:.1f}", "BUY", "Potentially oversold; this can precede a bounce or further declines.", 1))
    elif rsi < 40:
        rows.append(_signal_row("RSI (14)", f"{rsi:.1f}", "SELL", "RSI indicates relatively weak momentum under this heuristic.", -1))
    else:
        rows.append(_signal_row("RSI (14)", f"{rsi:.1f}", "HOLD", "RSI is in a middle zone without a strong directional reading.", 0))

    macd, macd_sig = safe_num(last.get("MACD")), safe_num(last.get("MACD_signal"))
    if macd is None or macd_sig is None:
        rows.append(_signal_row("MACD vs signal", "N/A", "N/A", "MACD comparison is not available.", None))
    elif macd > macd_sig:
        rows.append(_signal_row("MACD vs signal", f"{macd:.3f} vs {macd_sig:.3f}", "BUY", "MACD is above its signal line, a positive momentum condition.", 1))
    elif macd < macd_sig:
        rows.append(_signal_row("MACD vs signal", f"{macd:.3f} vs {macd_sig:.3f}", "SELL", "MACD is below its signal line, a negative momentum condition.", -1))
    else:
        rows.append(_signal_row("MACD vs signal", f"{macd:.3f} vs {macd_sig:.3f}", "HOLD", "MACD equals its signal line.", 0))

    upper, lower = safe_num(last.get("BB_upper")), safe_num(last.get("BB_lower"))
    if price is None or upper is None or lower is None:
        rows.append(_signal_row("Bollinger Bands", "N/A", "N/A", "Bollinger Bands are not available.", None))
    elif price < lower:
        rows.append(_signal_row("Bollinger Bands", f"Price ₹{price:,.2f}; lower ₹{lower:,.2f}", "BUY", "Price is below the lower band: possible oversold condition, but not a confirmed reversal.", 1))
    elif price > upper:
        rows.append(_signal_row("Bollinger Bands", f"Price ₹{price:,.2f}; upper ₹{upper:,.2f}", "SELL", "Price is above the upper band: potentially stretched, though strong trends can continue.", -1))
    else:
        rows.append(_signal_row("Bollinger Bands", f"Price ₹{price:,.2f}", "HOLD", "Price is inside the bands; this indicator gives no extreme reading.", 0))

    return pd.DataFrame(rows)


def build_fundamental_signals(fundamentals, info, current_price):
    """Add deliberately broad valuation/quality heuristics; these are not sector-adjusted."""
    rows = []
    for _, item in fundamentals.iterrows():
        metric = str(item.get("Metric", ""))
        value = safe_num(item.get("Value"))
        signal, reason, score = "N/A", "Value unavailable; no signal assigned.", None
        if value is not None:
            signal, reason, score = "HOLD", "This metric alone does not establish direction.", 0
            if metric in ("P/E (trailing)", "P/E (forward)"):
                if value <= 0:
                    signal, reason, score = "SELL", "Non-positive P/E can indicate losses or unusable earnings; inspect the accounts.", -1
                elif value <= 20:
                    signal, reason, score = "BUY", "P/E is in a lower broad range; compare with sector peers and growth.", 1
                elif value <= 35:
                    signal, reason, score = "HOLD", "P/E is in a middle broad range; valuation depends on growth and sector.", 0
                else:
                    signal, reason, score = "SELL", "High P/E may leave less room for disappointment; some fast-growing firms justify it.", -1
            elif metric == "Price/book":
                if value <= 0:
                    signal, reason, score = "SELL", "Non-positive price/book needs careful balance-sheet review.", -1
                elif value <= 2.5:
                    signal, reason, score = "BUY", "Price/book is in a lower broad range; asset quality still matters.", 1
                elif value <= 5:
                    signal, reason, score = "HOLD", "Middle broad range; sector and asset quality matter substantially.", 0
                else:
                    signal, reason, score = "SELL", "High price/book may indicate a demanding valuation; sector context matters.", -1
            elif metric == "Price/sales":
                if value <= 0:
                    signal, reason, score = "SELL", "Non-positive price/sales is not a conventional positive valuation reading.", -1
                elif value <= 2:
                    signal, reason, score = "BUY", "Lower price/sales range; profitability and sector context still matter.", 1
                elif value <= 5:
                    signal, reason, score = "HOLD", "Middle price/sales range; compare with similar companies.", 0
                else:
                    signal, reason, score = "SELL", "High price/sales can imply demanding expectations; growth varies by sector.", -1
            elif metric == "EV/EBITDA":
                if value <= 0:
                    signal, reason, score = "SELL", "Non-positive EV/EBITDA can reflect weak or negative EBITDA; inspect the accounts.", -1
                elif value <= 10:
                    signal, reason, score = "BUY", "Lower broad EV/EBITDA range; debt, capex and sector context matter.", 1
                elif value <= 18:
                    signal, reason, score = "HOLD", "Middle broad range; compare with industry peers.", 0
                else:
                    signal, reason, score = "SELL", "Higher broad EV/EBITDA range may imply a demanding valuation.", -1
            elif metric in ("EPS (trailing)", "EPS (forward)"):
                if value > 0:
                    signal, reason, score = "BUY", "Positive EPS under this basic profitability check; trend and quality still matter.", 1
                else:
                    signal, reason, score = "SELL", "Non-positive EPS; check whether losses are temporary or structural.", -1
            elif metric == "Dividend yield":
                if value >= 0.025:
                    signal, reason, score = "BUY", "Dividend yield is at least 2.5%; verify sustainability and payout coverage.", 1
                else:
                    signal, reason, score = "HOLD", "Low or zero yield is not inherently negative; growth firms may reinvest cash.", 0
            elif metric == "ROE":
                if value >= 0.15:
                    signal, reason, score = "BUY", "ROE is at least 15%; check leverage and one-off effects.", 1
                elif value >= 0.08:
                    signal, reason, score = "HOLD", "Moderate ROE under this broad rule; compare with sector peers.", 0
                else:
                    signal, reason, score = "SELL", "ROE is below 8% or negative; investigate profitability and equity changes.", -1
            elif metric in ("Profit margin", "Operating margin"):
                if value >= 0.15:
                    signal, reason, score = "BUY", "Margin is at least 15% under this broad rule; sector norms differ.", 1
                elif value >= 0.05:
                    signal, reason, score = "HOLD", "Margin is positive but below 15%; compare the trend and sector.", 0
                else:
                    signal, reason, score = "SELL", "Low or negative margin; review trend, industry norms and one-off costs.", -1
            elif metric == "Debt/equity (provider)":
                if value <= 100:
                    signal, reason, score = "BUY", "Provider debt/equity is at or below 100; definitions differ and financial firms need special treatment.", 1
                elif value <= 200:
                    signal, reason, score = "HOLD", "Moderate-to-high provider debt/equity; check industry norms and debt maturity.", 0
                else:
                    signal, reason, score = "SELL", "High provider debt/equity under this broad rule; verify units and business model.", -1
            elif metric == "Analyst target (provider)":
                if current_price and current_price > 0:
                    upside = value / current_price - 1
                    if upside >= 0.10:
                        signal, reason, score = "BUY", f"Provider consensus target is {upside:+.1%} above last close; targets can be wrong or stale.", 1
                    elif upside <= -0.10:
                        signal, reason, score = "SELL", f"Provider consensus target is {upside:+.1%} versus last close; targets can be wrong or stale.", -1
                    else:
                        signal, reason, score = "HOLD", f"Provider target is {upside:+.1%} versus last close, within the ±10% neutral band.", 0
                else:
                    signal, reason, score = "N/A", "Current price unavailable for comparison.", None
            elif metric == "Net income (latest annual, provider statements)":
                if value > 0:
                    signal, reason, score = "BUY", "Latest available annual net income is positive; growth and earnings quality still matter.", 1
                else:
                    signal, reason, score = "SELL", "Latest available annual net income is zero or negative; inspect the trend and one-offs.", -1
            elif metric == "Free cash flow (rough, latest annual)":
                if value > 0:
                    signal, reason, score = "BUY", "Estimated free cash flow is positive; confirm capex sign and statement completeness.", 1
                else:
                    signal, reason, score = "SELL", "Estimated free cash flow is zero or negative; check investment cycles and data quality.", -1
            elif metric == "Revenue (latest annual, provider statements)":
                signal, reason, score = "HOLD", "Revenue size alone is not directional; revenue growth and margins are needed.", 0
            elif metric in ("Current/last price", "Market capitalisation", "Beta"):
                signal, reason, score = "HOLD", "Descriptive metric only; it does not indicate buy or sell on its own.", 0

        enriched = item.to_dict()
        enriched["Signal"] = signal
        enriched["Signal basis"] = reason
        enriched["Score"] = score
        rows.append(enriched)
    return pd.DataFrame(rows)


def build_overall_signal(technical_signals, fundamental_signals):
    """Combine rule-based scores; output is a research heuristic, not a probability."""
    tech_scores = pd.to_numeric(technical_signals["Score"], errors="coerce").dropna()
    tech_score = float(tech_scores.mean()) if len(tech_scores) else 0.0
    core_metrics = {
        "P/E (trailing)", "P/E (forward)", "Price/book", "Price/sales", "EV/EBITDA",
        "ROE", "Profit margin", "Operating margin", "Debt/equity (provider)",
        "Net income (latest annual, provider statements)", "Free cash flow (rough, latest annual)",
        "Analyst target (provider)",
    }
    core = fundamental_signals[fundamental_signals["Metric"].isin(core_metrics)]
    fund_scores = pd.to_numeric(core["Score"], errors="coerce").dropna()
    fund_score = float(fund_scores.mean()) if len(fund_scores) >= 3 else None
    if fund_score is None:
        combined = tech_score
        method = "Technical indicators only; fewer than 3 usable core fundamental checks were available."
    else:
        combined = 0.60 * tech_score + 0.40 * fund_score
        method = f"60% technical score + 40% fundamental score; {len(fund_scores)} core fundamental checks were available."
    if combined >= 0.25:
        label = "BUY"
    elif combined <= -0.25:
        label = "SELL"
    else:
        label = "HOLD"
    return {
        "label": label,
        "score": float(np.clip(combined, -1, 1)),
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "technical_count": int(len(tech_scores)),
        "fundamental_count": int(len(fund_scores)),
        "method": method,
    }

def get_fx_rate(from_currency, to_currency):
    """Approximate FX conversion from Yahoo Finance daily history; provider quotes may be delayed."""
    if not from_currency or not to_currency:
        return None
    if from_currency.upper() == to_currency.upper():
        return 1.0
    pair = f"{from_currency.upper()}{to_currency.upper()}=X"
    try:
        fx = yf.Ticker(pair).history(period="5d", interval="1d")
        if not fx.empty:
            return safe_num(fx["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

def intraday_scenarios(ticker):
    """Short-horizon volatility scenarios using available 1-minute candles; not validated predictions."""
    try:
        intraday = yf.Ticker(ticker).history(period="7d", interval="1m", auto_adjust=True)
    except Exception:
        intraday = pd.DataFrame()
    if intraday.empty or "Close" not in intraday or intraday["Close"].dropna().shape[0] < 30:
        return pd.DataFrame(), "Intraday candles unavailable or insufficient. The provider may not support 1-minute data for this listing."
    close = intraday["Close"].dropna()
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 20:
        return pd.DataFrame(), "Not enough intraday observations for even a rough volatility scenario."
    # Exclude overnight gaps by using observed candle volatility. Drift is heavily shrunk toward zero.
    recent = returns.tail(120)
    drift_per_min = float(np.clip(recent.mean() * 0.15, -0.00015, 0.00015))
    vol_per_min = float(recent.std())
    price_now = float(close.iloc[-1])
    horizons = [("1 minute", 1), ("1 hour", 60), ("3 hours", 180), ("6 hours", 360)]
    rows=[]
    for label, minutes in horizons:
        mu=drift_per_min*minutes
        sigma=vol_per_min*math.sqrt(minutes)
        midpoint=price_now*math.exp(mu)
        lo=price_now*math.exp(mu-1.0*sigma)
        hi=price_now*math.exp(mu+1.0*sigma)
        rows.append({"Horizon":label,"Minutes":minutes,"Scenario midpoint":midpoint,
                     "Illustrative range low":lo,"Illustrative range high":hi,
                     "Midpoint change %":(midpoint/price_now-1)*100,
                     "Recent candle volatility %":vol_per_min*100})
    return pd.DataFrame(rows), None

NSE_UNIVERSE = [
    ("RELIANCE.NS","Reliance Industries"),("TCS.NS","Tata Consultancy Services"),
    ("HDFCBANK.NS","HDFC Bank"),("ICICIBANK.NS","ICICI Bank"),("INFY.NS","Infosys"),
    ("ITC.NS","ITC"),("BHARTIARTL.NS","Bharti Airtel"),("SBIN.NS","State Bank of India"),
    ("LT.NS","Larsen & Toubro"),("HINDUNILVR.NS","Hindustan Unilever"),
    ("TATAMOTORS.NS","Tata Motors"),("SUNPHARMA.NS","Sun Pharma"),
    ("KOTAKBANK.NS","Kotak Mahindra Bank"),("AXISBANK.NS","Axis Bank"),("MARUTI.NS","Maruti Suzuki"),
    ("TRENT.NS","Trent"),("BEL.NS","Bharat Electronics"),("ZOMATO.NS","Eternal (formerly Zomato)"),
    ("TATAPOWER.NS","Tata Power"),("ONGC.NS","ONGC")
]

def quick_nse_watchlist():
    """Scan a small liquid NSE watchlist, not the entire market; rank by simple daily technical score."""
    rows=[]
    for ticker,name in NSE_UNIVERSE:
        try:
            h=yf.Ticker(ticker).history(period="9mo", interval="1d", auto_adjust=True)
            if h.empty or len(h)<55: continue
            close=h["Close"].dropna()
            sma20=close.rolling(20).mean().iloc[-1]
            sma50=close.rolling(50).mean().iloc[-1]
            ret5=close.pct_change(5).iloc[-1]
            vol20=close.pct_change().tail(20).std()
            score=(1 if close.iloc[-1]>sma20 else -1)+(1 if close.iloc[-1]>sma50 else -1)+(1 if sma20>sma50 else -1)+(1 if ret5>0 else -1)
            signal="BUY setup" if score>=2 else "SELL / weak" if score<=-2 else "Mixed / watch"
            rows.append({"NSE stock":name,"Ticker":ticker,"Last close (₹)":float(close.iloc[-1]),
                         "5-session move %":float(ret5*100),"20-session volatility %":float(vol20*100),
                         "Rule-based setup":signal,"Trend score":score})
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Trend score","5-session move %"],ascending=False)

with st.expander("🔎 Find a stock anywhere in the world", expanded=True):
    query = st.text_input("Company name or ticker", value="Reliance Industries", placeholder="e.g. Reliance Industries, Apple, MSFT, TCS.NS, 7203.T")
    st.caption("Examples: Indian NSE ticker TCS.NS • US ticker AAPL • UK ticker SHEL.L • Japan ticker 7203.T. Search results depend on provider coverage.")

if st.button("Research stock", type="primary"):
    with st.spinner("Finding ticker candidates…"):
        candidates = search_tickers(query)
    if not candidates:
        st.error("No ticker candidates found. Try the exact company name or provider ticker, such as AAPL, MSFT, TCS.NS, or 7203.T.")
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
        currency = str(info.get("currency") or ("INR" if ticker.endswith((".NS", ".BO")) else "USD"))
        fx_inr = get_fx_rate(currency, "INR")
        fx_usd = get_fx_rate(currency, "USD")
        prev = df["Close"].iloc[-2] if len(df) > 1 else price
        change = price / prev - 1 if prev else 0
        technical_signals = build_technical_signals(df)
        fundamental_signals = build_fundamental_signals(fundamentals, info, price)
        overall = build_overall_signal(technical_signals, fundamental_signals)
        st.subheader(f"{chosen.get('name', query)} ({ticker})")
        a, b, c, d = st.columns(4)
        a.metric(f"Latest available close ({currency})", f"{currency} {price:,.2f}", f"{change*100:+.2f}% vs previous close")
        if fx_inr is not None:
            st.caption(f"Approx. converted price: ₹{price*fx_inr:,.2f} INR")
        if fx_usd is not None:
            st.caption(f"Approx. converted price: ${price*fx_usd:,.2f} USD")
        b.metric("RSI (14)", fmt_num(latest.get("RSI14"), 1))
        b.caption("Momentum indicator, not a standalone buy/sell signal.")
        c.metric("20-session volatility", fmt_pct(df["DailyReturn"].tail(20).std() * np.sqrt(252)))
        d.metric("Data through", str(df.index[-1].date()))

        st.subheader("Overall research signal")
        signal_left, signal_right = st.columns([1, 1.7])
        with signal_left:
            if overall["label"] == "BUY":
                st.success("BUY — bullish conditions under this rule set")
            elif overall["label"] == "SELL":
                st.error("SELL — bearish conditions under this rule set")
            else:
                st.info("HOLD — mixed or neutral conditions under this rule set")
            st.caption("This is a rule-based research label, not personal financial advice or a probability of profit.")
            st.write(overall["method"])
            st.caption(f"Technical checks used: {overall['technical_count']} • Core fundamental checks used: {overall['fundamental_count']}")
        with signal_right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall["score"] * 100,
                number={"suffix": "/100", "valueformat": ".0f"},
                title={"text": "Heuristic score (not probability)"},
                gauge={
                    "axis": {"range": [-100, 100], "tickvals": [-100, -50, 0, 50, 100]},
                    "bar": {"color": "#555555"},
                    "steps": [
                        {"range": [-100, -25], "color": "#f5c6cb"},
                        {"range": [-25, 25], "color": "#e9ecef"},
                        {"range": [25, 100], "color": "#c3e6cb"},
                    ],
                    "threshold": {"line": {"color": "#222222", "width": 4}, "thickness": 0.8, "value": overall["score"] * 100},
                },
            ))
            gauge.update_layout(height=250, margin={"l": 25, "r": 25, "t": 55, "b": 10})
            st.plotly_chart(gauge, use_container_width=True)
        st.warning("Signals use broad, fixed rules. Valuation thresholds are not sector-adjusted, provider data can be incomplete, and a BUY/SELL label does not predict what happens next. Review the reasons beside each metric.")
        st.subheader("🗣️ StockGPT's plain-English take")
        if overall["label"] == "BUY":
            explainer = f"The available indicators lean bullish for {chosen.get('name', ticker)}, but this is only a rule-based setup. Check the latest news, liquidity, valuation and a planned risk limit before making any decision."
        elif overall["label"] == "SELL":
            explainer = f"The available indicators lean bearish for {chosen.get('name', ticker)} under these rules. That does not prove the price will fall next; check whether the move is news-driven and compare the setup with your own plan."
        else:
            explainer = f"The evidence for {chosen.get('name', ticker)} is mixed. StockGPT does not see a strong enough combined signal under its current rules, so this is a watch-and-research situation rather than a confident directional call."
        st.info(explainer)
        st.caption("This friendly explanation is generated from the app's rules; it is not a human analyst or a live conversational AI model.")

        st.subheader("⚡ Intraday scenarios (1 min / 1h / 3h / 6h)")
        intraday_df, intraday_error = intraday_scenarios(ticker)
        st.caption(f"Intraday data fetched at {datetime.now().astimezone().strftime('%d %b %Y, %H:%M:%S %Z')}. Click Research stock / Run full analysis again to refresh; this is not a guaranteed live feed.")
        if intraday_error:
            st.info(intraday_error)
        else:
            st.dataframe(intraday_df, use_container_width=True, hide_index=True)
            st.caption("Ranges are volatility-based scenarios from recent 1-minute candles. Midpoints are not calibrated predictions; short-term noise and news can dominate.")
        st.subheader("Price chart and moving averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
        for col in ["SMA20", "SMA50", "SMA200"]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        fig.update_layout(height=480, xaxis_title="Date", yaxis_title=f"Price ({currency})", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Technical analysis", "Fundamentals", "Forecast scenarios", "NSE idea board 🇮🇳", "News & web research", "Data & limitations"])
        with tab1:
            st.markdown("### Technical indicator signals")
            st.dataframe(technical_signals.drop(columns=["Score"]), use_container_width=True, hide_index=True)
            st.caption("BUY/SELL/HOLD here are indicator readings generated by fixed rules. They are not guaranteed trade instructions.")
            st.markdown("### Indicator interpretation")
            for note in simple_technical_read(df):
                st.write("• " + note)
            tech_cols = ["Close", "SMA20", "SMA50", "SMA200", "RSI14", "MACD", "MACD_signal", "BB_upper", "BB_lower", "ATR14", "Volume", "VolAvg20"]
            st.dataframe(df[tech_cols].tail(30).sort_index(ascending=False), use_container_width=True)
            st.caption("Indicators use daily candles. Intraday prices, splits, low liquidity and data errors can change the picture.")
        with tab2:
            st.markdown("### Fundamentals with signal beside each metric")
            st.dataframe(fundamental_signals.drop(columns=["Score"]), use_container_width=True, hide_index=True)
            st.caption("Fundamental signals use broad generic thresholds, not sector-specific valuation models. N/A means the provider did not return a usable value.")
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
            st.markdown("### 🇮🇳 NSE watchlist — rule-based ideas")
            st.write("A small watchlist scan of selected NSE-listed companies. It is not the whole NSE universe and does not know your risk tolerance, entry price or trading costs.")
            if st.button("Scan selected NSE shares", type="primary"):
                with st.spinner("Checking recent daily candles for selected NSE listings…"):
                    scan = quick_nse_watchlist()
                if scan.empty:
                    st.warning("No watchlist data returned. Try again later.")
                else:
                    st.dataframe(scan, use_container_width=True, hide_index=True)
                    st.caption("These are technical setups, not personalized recommendations. Market-cap classification is not included because this quick scan does not reliably verify current market cap.")
            st.markdown("### What the ranking means")
            st.write("The scan ranks a curated list using price versus 20/50-session averages and 5-session momentum. It does not estimate a dependable time-to-profit or guarantee a winning stock.")
        with tab5:
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
        with tab6:
            st.markdown("### Important limitations")
            st.write("- Global ticker discovery depends on Yahoo Finance coverage and ticker conventions; this is not every listed share in every country.")
            st.write("- It does not have a guaranteed real-time feed. Free sources may be delayed, rate-limited or unavailable; the displayed timestamp is fetch time, not proof of exchange-live data.")
            st.write("- It does not reliably parse every NSE/BSE corporate filing, concall transcript, legal proceeding, order book, FII/DII flow or global macroeconomic series.")
            st.write("- Currency conversions use provider FX history when available and are approximate; fees, taxes, spreads and live exchange rates may differ. Yahoo Finance/yfinance is an unofficial community tool; respect provider terms.")
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
