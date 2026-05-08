import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EXCLUDED = {"SPY", "^VIX"}


def compute_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    logger.info("Computing features...")

    momentum_windows = cfg.get("momentum_windows", [1, 3, 6, 12])
    vol_window = cfg.get("volatility_window", 12)

    monthly = prices.resample("ME").last()
    returns = monthly.pct_change()

    stock_cols = [c for c in returns.columns if c not in _EXCLUDED]
    stocks = returns[stock_cols]
    spy = returns["SPY"] if "SPY" in returns.columns else None
    vix = monthly["^VIX"] if "^VIX" in monthly.columns else None

    log_r = np.log1p(stocks.clip(lower=-0.99))
    feat: dict[str, pd.DataFrame] = {}

    # Compound momentum over w months (uses returns up to and including current month)
    for w in momentum_windows:
        feat[f"mom_{w}m"] = np.expm1(log_r.rolling(w).sum())

    # Annualised realised volatility
    feat["vol_12m"] = stocks.rolling(vol_window).std() * np.sqrt(12)

    # Price position relative to rolling SMA
    mp = monthly[stock_cols]
    feat["price_to_sma6"] = mp / mp.rolling(6).mean() - 1
    feat["price_to_sma12"] = mp / mp.rolling(12).mean() - 1

    # Excess momentum over SPY
    if spy is not None:
        spy_log = np.log1p(spy.clip(lower=-0.99))
        for w in [3, 12]:
            spy_mom = np.expm1(spy_log.rolling(w).sum())
            feat[f"rel_mom_{w}m"] = feat[f"mom_{w}m"].subtract(spy_mom, axis=0)

    # VIX as macro regime signal
    if vix is not None:
        feat["vix"] = pd.DataFrame(
            np.tile(vix.values.reshape(-1, 1), (1, len(stock_cols))),
            index=monthly.index,
            columns=stock_cols,
        )

    # --- Targets ---
    target_return = stocks.shift(-1)
    if spy is not None:
        target_excess = stocks.shift(-1).subtract(spy.shift(-1), axis=0)
    else:
        target_excess = target_return.copy()

    # Stack all features into long (date, ticker) format
    parts = []
    for name, frame in feat.items():
        s = frame[stock_cols].stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        s.name = name
        parts.append(s)

    features = pd.concat(parts, axis=1)

    for name, wide in [("target_return", target_return), ("target_excess", target_excess)]:
        s = wide[stock_cols].stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        s.name = name
        features = features.join(s)

    features = features.reset_index()
    features = features.dropna(subset=["target_return"])

    feat_cols = [c for c in features.columns if c not in {"date", "ticker", "target_return", "target_excess"}]
    # Keep rows that have at least half the features
    features = features.dropna(subset=feat_cols, thresh=max(1, len(feat_cols) // 2))

    logger.info(f"Features: {len(features):,} samples | {len(feat_cols)} feature columns")
    return features
