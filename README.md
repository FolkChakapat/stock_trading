# S&P 500 ML Trading Pipeline

A long-term machine learning trading system that predicts which S&P 500 stocks will outperform the index, rebalances monthly, and backtests against SPY.

---

## How It Works

```
Wikipedia (S&P 500 list)
        ↓
  Data Ingestion          yfinance — 10 years of daily prices (478 tickers + SPY + VIX)
        ↓
Feature Engineering       Momentum · Volatility · SMA ratios · Relative strength vs SPY · VIX
        ↓
Walk-Forward Training     LightGBM retrained every 3 months on all prior data (no lookahead)
        ↓
Backtesting               Equal-weight top-30 portfolio · Monthly rebalance · Transaction costs
        ↓
Results                   CAGR · Sharpe · Max Drawdown · Chart (PNG)
```

---

## Features

| Feature | Description |
|---|---|
| `mom_1m / 3m / 6m / 12m` | Compound return over 1–12 months |
| `vol_12m` | 12-month annualised volatility |
| `price_to_sma6 / sma12` | Price position relative to 6/12-month SMA |
| `rel_mom_3m / 12m` | Momentum relative to SPY benchmark |
| `vix` | VIX level as macro regime signal |

**Target:** next month's return minus SPY return (excess return)

---

## Project Structure

```
stock_trading/
├── main.py                    # Pipeline orchestrator
├── config.yaml                # All tuneable parameters
├── requirements.txt
├── src/
│   ├── data/
│   │   └── fetcher.py         # S&P 500 ticker list + yfinance price download (batched + cached)
│   ├── features/
│   │   └── engineer.py        # Vectorised feature engineering (no lookahead bias)
│   ├── models/
│   │   └── trainer.py         # Walk-forward LightGBM training
│   ├── backtest/
│   │   └── engine.py          # Portfolio simulation + performance metrics
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_features.py       # 7 unit tests for feature engineering
│   └── test_backtest.py       # 6 unit tests for backtest engine
└── results/                   # Output charts and feature importance (git-ignored)
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/FolkChakapat/stock_trading.git
cd stock_trading

# 2. Set up environment
python3 -m venv .env
source .env/bin/activate          # Windows: .env\Scripts\activate
pip install -r requirements.txt

# 3. Run full pipeline  (~20 min first run — downloads 10 years of data)
python main.py

# Results saved to results/backtest_results.png
```

### Run individual steps

```bash
python main.py --step data        # Download & cache prices only
python main.py --step features    # Compute features only
python main.py --step train       # Walk-forward training only
python main.py --step backtest    # Backtest only (uses cached predictions)
python main.py --clear-cache      # Force re-download everything
```

### Run tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters are in `config.yaml`:

```yaml
data:
  start_date: "2015-01-01"
  end_date: "2024-12-31"

features:
  momentum_windows: [1, 3, 6, 12]   # months
  volatility_window: 12

model:
  min_train_months: 24    # warm-up period before first prediction
  retrain_freq: 3         # retrain every N months
  lgbm:
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 4

backtest:
  top_n: 30               # number of stocks to hold
  transaction_cost: 0.001 # 0.1% per position turned over
  initial_capital: 100000
```

---

## Design Decisions

**No lookahead bias** — features at time `t` use only data up to and including end of month `t`. Target is always month `t+1`.

**Walk-forward validation** — the model never trains on future data. At each prediction date, it trains on all history up to that point.

**Survivorship bias caveat** — uses the *current* S&P 500 constituent list. A production system would need historical constituent data (e.g. from Compustat) to avoid inflating backtest returns.

**Caching** — downloaded prices, computed features, and predictions are all cached to `data/cache/` so re-runs are instant.

---

## Requirements

- Python 3.10+
- pandas >= 2.2.0
- lightgbm >= 4.3.0
- yfinance >= 0.2.40
- scikit-learn >= 1.3.0

---

## Roadmap

- [ ] Add fundamental features (P/E, ROE, EPS growth)
- [ ] Add FRED macro data (yield curve, CPI, Fed funds rate)
- [ ] Hyperparameter optimisation with Optuna
- [ ] Sector-neutral portfolio construction
- [ ] Historical S&P 500 constituent list to remove survivorship bias
