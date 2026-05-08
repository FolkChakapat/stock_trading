import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def run_backtest(predictions: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> dict:
    top_n = cfg.get("top_n", 30)
    transaction_cost = cfg.get("transaction_cost", 0.001)
    initial_capital = cfg.get("initial_capital", 100_000)

    monthly = prices.resample("ME").last()
    returns = monthly.pct_change()
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None

    dates = sorted(predictions["date"].unique())

    port_values = []
    spy_values = []
    prev_holdings: set = set()

    port_val = initial_capital
    spy_val = initial_capital

    for date in dates:
        month_preds = predictions[predictions["date"] == date].dropna(subset=["predicted_rank"])
        top_stocks = month_preds.nlargest(top_n, "predicted_rank")["ticker"].tolist()

        # Portfolio return: equal-weight mean of top-N stocks
        valid = [s for s in top_stocks if s in returns.columns and date in returns.index]
        if valid:
            avg_return = float(returns.loc[date, valid].mean())
            if pd.isna(avg_return):
                avg_return = 0.0
        else:
            avg_return = 0.0

        # Transaction cost proportional to turnover
        turnover = len(set(top_stocks) - prev_holdings) / max(top_n, 1)
        port_val *= 1 + avg_return - turnover * transaction_cost

        # Benchmark
        if spy_returns is not None and date in spy_returns.index:
            spy_r = float(spy_returns.loc[date])
            if pd.isna(spy_r):
                spy_r = 0.0
        else:
            spy_r = 0.0
        spy_val *= 1 + spy_r

        port_values.append((date, port_val))
        spy_values.append((date, spy_val))
        prev_holdings = set(top_stocks)

    portfolio = pd.Series(dict(port_values), name="Portfolio")
    benchmark = pd.Series(dict(spy_values), name="SPY")

    metrics = _compute_metrics(portfolio, benchmark)

    return {
        "portfolio": portfolio,
        "benchmark": benchmark,
        "metrics": metrics,
        "predictions": predictions,
    }


def _compute_metrics(portfolio: pd.Series, benchmark: pd.Series) -> dict:
    port_r = portfolio.pct_change(fill_method=None).dropna()
    bench_r = benchmark.pct_change(fill_method=None).dropna()
    n_years = len(port_r) / 12

    def cagr(s):
        return (s.iloc[-1] / s.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1

    def max_dd(s):
        return ((s - s.cummax()) / s.cummax()).min()

    sharpe = (port_r.mean() / port_r.std()) * np.sqrt(12) if port_r.std() > 0 else 0.0
    hit_rate = (port_r > bench_r).mean()

    return {
        "Portfolio CAGR": f"{cagr(portfolio):.2%}",
        "Benchmark CAGR (SPY)": f"{cagr(benchmark):.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd(portfolio):.2%}",
        "Benchmark Max Drawdown": f"{max_dd(benchmark):.2%}",
        "Total Return": f"{(portfolio.iloc[-1] / portfolio.iloc[0] - 1):.2%}",
        "Benchmark Total Return": f"{(benchmark.iloc[-1] / benchmark.iloc[0] - 1):.2%}",
        "Monthly Hit Rate vs SPY": f"{hit_rate:.2%}",
    }


def plot_results(results: dict):
    portfolio = results["portfolio"]
    benchmark = results["benchmark"]
    metrics = results["metrics"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Normalise to 100
    p = portfolio / portfolio.iloc[0] * 100
    b = benchmark / benchmark.iloc[0] * 100

    axes[0].plot(p.index, p.values, label="ML Portfolio", color="#2196F3", linewidth=2)
    axes[0].plot(b.index, b.values, label="SPY Benchmark", color="#FF9800", linewidth=2)
    axes[0].set_title("Cumulative Returns — ML Portfolio vs SPY", fontsize=14)
    axes[0].set_ylabel("Value (base = 100)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    dd_port = (portfolio - portfolio.cummax()) / portfolio.cummax() * 100
    dd_bench = (benchmark - benchmark.cummax()) / benchmark.cummax() * 100

    axes[1].fill_between(dd_port.index, dd_port.values, 0, alpha=0.6, color="#2196F3", label="ML Portfolio")
    axes[1].fill_between(dd_bench.index, dd_bench.values, 0, alpha=0.4, color="#FF9800", label="SPY Benchmark")
    axes[1].set_title("Drawdown (%)", fontsize=14)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    out = "results/backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<35} {v}")
    print("=" * 55)
    print(f"  Chart → {out}")
