# India Stock Intelligence — beginner setup

## What this prototype does
- Searches for likely NSE/BSE ticker candidates using Yahoo Finance search.
- Downloads available daily price history.
- Calculates RSI, SMA, EMA, MACD, Bollinger Bands, ATR, volume averages and volatility.
- Displays available provider fundamentals and financial statements.
- Retrieves available Yahoo Finance news and performs a public web search.
- Produces transparent historical-volatility scenario ranges for several horizons.
- Exports price history and fundamentals as CSV files.

## Important limitations
This is a prototype, not a professional-grade investment model. Free public data is incomplete, may be delayed, and can fail. Web search is not exhaustive. Scenario ranges are not reliable probabilities or guaranteed forecasts. Always verify company facts and announcements against official NSE/BSE filings and company investor-relations material. No automatic trading or broker connection is included.

`yfinance` is an unofficial community library that uses publicly available Yahoo Finance interfaces; it is intended for research/personal use and is not endorsed by Yahoo. Review data-provider terms before deployment or commercial use.

## Run locally
1. Install Python 3.11 or later.
2. Put `app.py` and `requirements.txt` in the same folder.
3. In a terminal in that folder, run:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Your browser opens the local app.

## Run free in a browser
Recommended: Streamlit Community Cloud.
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Choose **Create app**, select the repository and `app.py`, then deploy.
5. The app gets a public URL. Do not put private API keys or passwords in source code.

Google Colab is another free way to experiment with Python notebooks, but it is not designed to host a continuously available web app; sessions can stop.

## How to use
- Enter a company name, or a Yahoo-style ticker such as `TCS.NS` or `500570.BO`.
- Click **Research stock**.
- Confirm the exact listing from the candidates.
- Click **Run full analysis**.
- Review data coverage and the source of every important claim before relying on it.

## Suggested next upgrades
1. Add official NSE/BSE corporate announcements with careful rate-limit and terms compliance.
2. Add industry-peer comparison and sector benchmarks.
3. Add a proper news sentiment model with article deduplication and source-quality scoring.
4. Add walk-forward backtesting including brokerage, taxes, spread and slippage.
5. Validate forecast calibration on unseen periods and compare against simple baselines.
6. Store timestamps and source links for every datum; flag stale or missing values.
7. Keep a paper-trading journal before considering real-money use.
