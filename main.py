import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.engine import plot_results, run_backtest
from src.data.fetcher import download_price_data, get_sp500_tickers
from src.features.engineer import compute_features
from src.models.trainer import walk_forward_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="S&P 500 ML Long-Term Trading Pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--clear-cache", action="store_true", help="Delete all cached data and restart")
    p.add_argument(
        "--step",
        choices=["data", "features", "train", "backtest", "all"],
        default="all",
        help="Run a specific pipeline step only",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if args.clear_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")

    features_cache = Path(cache_dir) / "features.pkl"
    predictions_cache = Path(cache_dir) / "predictions.pkl"

    # ── Step 1: Data ──────────────────────────────────────────────────────
    logger.info("━━ Step 1/4 — Data Ingestion ━━")
    tickers = get_sp500_tickers(cfg["data"]["sp500_url"])
    prices = download_price_data(
        tickers,
        cfg["data"]["start_date"],
        cfg["data"]["end_date"],
        cache_dir,
    )
    logger.info(f"Prices: {prices.shape[1]} tickers × {len(prices)} days")

    if args.step == "data":
        return

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    logger.info("━━ Step 2/4 — Feature Engineering ━━")
    if features_cache.exists() and not args.clear_cache:
        features_df = pd.read_pickle(features_cache)
        logger.info(f"Loaded features from cache: {features_df.shape}")
    else:
        features_df = compute_features(prices, cfg["features"])
        features_df.to_pickle(features_cache)

    if args.step == "features":
        print(features_df.describe())
        return

    # ── Step 3: Walk-Forward Training ────────────────────────────────────
    logger.info("━━ Step 3/4 — Walk-Forward Training ━━")
    if predictions_cache.exists() and not args.clear_cache:
        predictions = pd.read_pickle(predictions_cache)
        logger.info(f"Loaded predictions from cache: {predictions.shape}")
    else:
        predictions = walk_forward_train(features_df, cfg["model"])
        predictions.to_pickle(predictions_cache)

    if args.step == "train":
        print(predictions.head(10).to_string())
        return

    # ── Step 4: Backtest ─────────────────────────────────────────────────
    logger.info("━━ Step 4/4 — Backtesting ━━")
    results = run_backtest(predictions, prices, cfg["backtest"])
    plot_results(results)


if __name__ == "__main__":
    main()
