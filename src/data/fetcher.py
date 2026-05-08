import time
import logging
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_sp500_tickers(url: str) -> list:
    import io
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (compatible; stock-trading-ml/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info(f"Found {len(tickers)} S&P 500 tickers")
    return tickers


def download_price_data(
    tickers: list,
    start_date: str,
    end_date: str,
    cache_dir: str,
) -> pd.DataFrame:
    cache_path = Path(cache_dir) / f"prices_{start_date}_{end_date}.pkl"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Loading prices from cache")
        return pd.read_pickle(cache_path)

    logger.info(f"Downloading price data for {len(tickers)} tickers + SPY + ^VIX")

    all_tickers = tickers + ["SPY", "^VIX"]
    batch_size = 100
    closes = []

    for i in tqdm(range(0, len(all_tickers), batch_size), desc="Downloading batches"):
        batch = all_tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
            closes.append(close)
        except Exception as e:
            logger.warning(f"Batch {i // batch_size} failed ({e}), trying individually")
            for ticker in batch:
                try:
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False,
                    )
                    closes.append(data[["Close"]].rename(columns={"Close": ticker}))
                except Exception:
                    pass
        time.sleep(0.5)

    prices = pd.concat(closes, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Drop tickers with >30% missing data
    prices = prices.loc[:, prices.isna().mean() < 0.3]
    prices = prices.ffill()

    prices.to_pickle(cache_path)
    logger.info(f"Saved {len(prices.columns)} tickers over {len(prices)} trading days")
    return prices
