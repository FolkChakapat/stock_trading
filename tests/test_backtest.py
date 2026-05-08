import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _compute_metrics, run_backtest


def _make_predictions(dates, tickers) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "predicted_excess": rng.normal(),
                "predicted_rank": rng.random(),
                "target_return": rng.normal(0.005, 0.05),
                "target_excess": rng.normal(0, 0.03),
            })
    return pd.DataFrame(rows)


def _make_prices(dates, tickers) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range(dates[0], dates[-1], freq="B")
    data = {}
    for t in tickers:
        p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
        data[t] = p
    return pd.DataFrame(data, index=idx)


DATES = pd.date_range("2020-01-31", periods=24, freq="ME")
TICKERS = [f"STOCK{i}" for i in range(50)] + ["SPY"]


class TestRunBacktest:
    def test_returns_dict_with_keys(self):
        preds = _make_predictions(DATES, TICKERS[:-1])
        prices = _make_prices(DATES, TICKERS)
        result = run_backtest(preds, prices, {"top_n": 10, "transaction_cost": 0.001, "initial_capital": 100_000})
        assert "portfolio" in result
        assert "benchmark" in result
        assert "metrics" in result

    def test_portfolio_starts_at_initial_capital(self):
        preds = _make_predictions(DATES, TICKERS[:-1])
        prices = _make_prices(DATES, TICKERS)
        result = run_backtest(preds, prices, {"top_n": 10, "transaction_cost": 0.001, "initial_capital": 50_000})
        assert result["portfolio"].iloc[0] == pytest.approx(50_000, rel=0.1)

    def test_portfolio_length_matches_prediction_dates(self):
        preds = _make_predictions(DATES, TICKERS[:-1])
        prices = _make_prices(DATES, TICKERS)
        result = run_backtest(preds, prices, {"top_n": 10, "transaction_cost": 0.001, "initial_capital": 100_000})
        assert len(result["portfolio"]) == len(DATES)

    def test_portfolio_values_are_positive(self):
        preds = _make_predictions(DATES, TICKERS[:-1])
        prices = _make_prices(DATES, TICKERS)
        result = run_backtest(preds, prices, {"top_n": 10, "transaction_cost": 0.001, "initial_capital": 100_000})
        assert (result["portfolio"] > 0).all()


class TestComputeMetrics:
    def test_metrics_keys_present(self):
        p = pd.Series([100, 105, 110, 108, 115])
        b = pd.Series([100, 103, 106, 104, 110])
        m = _compute_metrics(p, b)
        assert "Portfolio CAGR" in m
        assert "Sharpe Ratio" in m
        assert "Max Drawdown" in m

    def test_drawdown_is_non_positive(self):
        p = pd.Series([100, 105, 110, 108, 115])
        b = pd.Series([100, 103, 106, 104, 110])
        m = _compute_metrics(p, b)
        dd = float(m["Max Drawdown"].strip("%")) / 100
        assert dd <= 0
