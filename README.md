# StockGPT Global Upgrade

A Streamlit research dashboard for global stock tickers with an India-first experience.

## Features
- Search for global listings through Yahoo Finance search, including NSE symbols such as `TCS.NS`, US symbols such as `AAPL`, and other supported exchange suffixes.
- Approximate INR and USD conversion using available Yahoo Finance FX history.
- Daily technical indicators and transparent BUY/SELL/HOLD heuristic signals.
- Available fundamentals with a signal and explanation for each metric.
- Short-horizon (1-minute, 1-hour, 3-hour, 6-hour) volatility scenarios when 1-minute candle data is available.
- Longer-horizon historical-volatility scenario ranges.
- Small curated NSE watchlist scan for technical setups (not the full NSE universe).
- News/search results, charts, CSV downloads, and a friendlier plain-English summary.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to the root of your GitHub repository.
2. In Streamlit Community Cloud, select that repository and branch.
3. Set **Main file path** to `app.py`.
4. Deploy.

## Important limitations
- This is a research prototype, not a broker terminal or guaranteed live market-data service. The free data source may be delayed, rate-limited, incomplete, or unavailable. The timestamp indicates fetch time, not proof of exchange-live data.
- Global coverage depends on Yahoo Finance search/ticker coverage. It does not guarantee every listed share in every country.
- Intraday scenarios are simple volatility ranges based on recent one-minute candles, when provided. They are not trained, calibrated, or validated price predictions.
- No model can reliably promise a profitable trade or an exact time to sell. This app deliberately does not claim a high probability of profit without backtesting evidence.
- NSE idea board scans only a small curated list of liquid/known NSE tickers; it is not a comprehensive market-wide screener and does not verify cap classifications dynamically.
- Currency conversions are approximate and may differ from live FX, brokerage spreads, taxes, and fees.
- Technical/fundamental rules are generic, not sector-adjusted, and not personalized investment advice.
- Yahoo Finance/yfinance is an unofficial community tool. Respect its terms and data-provider restrictions.
