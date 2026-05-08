import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_META = {"date", "ticker", "target_return", "target_excess"}


def _feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _META]


def walk_forward_train(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Walk-forward prediction: retrain every `retrain_freq` months on all
    preceding data, then predict the upcoming month for all stocks.
    """
    target_col = cfg.get("target_col", "target_excess")
    min_train = cfg.get("min_train_months", 24)
    retrain_freq = cfg.get("retrain_freq", 3)
    lgbm_cfg = cfg.get("lgbm", {})

    lgbm_params = {
        "n_estimators": lgbm_cfg.get("n_estimators", 200),
        "learning_rate": lgbm_cfg.get("learning_rate", 0.05),
        "max_depth": lgbm_cfg.get("max_depth", 4),
        "num_leaves": lgbm_cfg.get("num_leaves", 31),
        "min_child_samples": lgbm_cfg.get("min_child_samples", 20),
        "subsample": lgbm_cfg.get("subsample", 0.8),
        "colsample_bytree": lgbm_cfg.get("colsample_bytree", 0.8),
        "reg_alpha": lgbm_cfg.get("reg_alpha", 0.1),
        "reg_lambda": lgbm_cfg.get("reg_lambda", 0.1),
        "random_state": lgbm_cfg.get("random_state", 42),
        "verbose": -1,
    }

    feat_cols = _feature_cols(df)
    dates = sorted(df["date"].unique())
    all_preds = []
    model = None
    last_train_idx = -retrain_freq  # force first training

    logger.info(f"Walk-forward: {len(dates)} months | features: {feat_cols}")

    for i, date in enumerate(dates):
        if i < min_train:
            continue

        # Retrain if due
        if (i - last_train_idx) >= retrain_freq:
            train_df = df[df["date"].isin(dates[:i])].dropna(subset=feat_cols + [target_col])
            if len(train_df) < 50:
                continue
            X_tr = train_df[feat_cols]
            y_tr = train_df[target_col].values

            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_tr, y_tr)
            last_train_idx = i

            if i % 12 == 0:
                logger.info(f"  Retrained at month {i}/{len(dates)} ({date.strftime('%Y-%m')}), "
                             f"train rows={len(train_df):,}")

        if model is None:
            continue

        test_df = df[df["date"] == date].copy()
        X_te = test_df[feat_cols]
        test_df["predicted_excess"] = model.predict(X_te)
        test_df["predicted_rank"] = test_df["predicted_excess"].rank(pct=True)

        all_preds.append(
            test_df[["date", "ticker", "predicted_excess", "predicted_rank",
                      "target_return", "target_excess"]]
        )

    predictions = pd.concat(all_preds, ignore_index=True)
    logger.info(f"Walk-forward complete: {len(predictions):,} predictions over "
                f"{predictions['date'].nunique()} months")

    # Save feature importances
    if model is not None:
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        Path("results").mkdir(exist_ok=True)
        imp.to_csv("results/feature_importance.csv")
        logger.info("Top 5 features:\n" + imp.head().to_string())

    return predictions
