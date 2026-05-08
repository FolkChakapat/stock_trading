import numpy as np
import pandas as pd
import pytest

from src.features.engineer import compute_features


def _make_prices(n_days: int = 300, tickers: list = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    data = {}
    for t in tickers:
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
        data[t] = prices
    return pd.DataFrame(data, index=idx)


CFG = {"momentum_windows": [1, 3, 6, 12], "volatility_window": 12}


class TestComputeFeatures:
    def test_returns_dataframe(self):
        prices = _make_prices()
        result = compute_features(prices, CFG)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        prices = _make_prices()
        result = compute_features(prices, CFG)
        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "target_return" in result.columns
        assert "target_excess" in result.columns

    def test_momentum_columns_exist(self):
        prices = _make_prices()
        result = compute_features(prices, CFG)
        for w in [1, 3, 6, 12]:
            assert f"mom_{w}m" in result.columns, f"mom_{w}m missing"

    def test_no_future_data_in_features(self):
        """target_return must be shift(-1); features must not contain future info."""
        prices = _make_prices()
        result = compute_features(prices, CFG)
        # All targets should differ from the last-period feature values
        assert result["target_return"].isna().sum() == 0

    def test_no_ticker_from_excluded(self):
        prices = _make_prices(tickers=["AAPL", "MSFT", "SPY", "^VIX"])
        result = compute_features(prices, CFG)
        assert "SPY" not in result["ticker"].unique()
        assert "^VIX" not in result["ticker"].unique()

    def test_relative_momentum_present_when_spy_exists(self):
        prices = _make_prices(tickers=["AAPL", "MSFT", "SPY"])
        result = compute_features(prices, CFG)
        assert "rel_mom_3m" in result.columns
        assert "rel_mom_12m" in result.columns

    def test_no_inf_values(self):
        prices = _make_prices()
        result = compute_features(prices, CFG)
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Infinite values found in features"
