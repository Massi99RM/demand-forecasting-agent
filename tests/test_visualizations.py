"""
Tests for visualizations.py.

These tests verify that each plotting function:
1. Creates a valid PNG file at the expected path
2. Doesn't crash on valid input
3. Raises helpful errors on invalid input (wrong store/item)

We don't check what the charts LOOK like (that's a visual test you'd
do manually). We check that they're generated, saved, and non-empty.

Run:  python tests/test_visualizations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CFG
from src.visualizations import (
    plot_sales_trend,
    plot_forecast_vs_actual,
    plot_feature_importance,
    plot_demand_distribution,
    plot_weekly_pattern,
    plot_store_comparison,
    plot_volatility_ranking,
)


def make_synthetic_df() -> pd.DataFrame:
    """Create synthetic data for chart testing."""
    dates = pd.date_range("2017-01-01", "2017-12-31", freq="D")
    np.random.seed(42)
    rows = []

    for store in [1, 2, 3]:
        for item in [1, 2]:
            base = 30 + store * 10 + item * 5
            for date in dates:
                weekday_effect = 8 if date.dayofweek >= 5 else 0
                month_effect = 15 if date.month == 12 else 0
                noise = np.random.normal(0, 4)
                sales = max(0, int(base + weekday_effect + month_effect + noise))
                rows.append({"date": date, "store": store, "item": item, "sales": sales})

    return pd.DataFrame(rows)


def _check_file(path_str: str, label: str):
    """Verify a file was created and is non-empty."""
    path = Path(path_str)
    assert path.exists(), f"{label}: file not created at {path}"
    assert path.stat().st_size > 1000, f"{label}: file suspiciously small ({path.stat().st_size} bytes)"
    print(f"  ✓ {label}: {path.name} ({path.stat().st_size:,} bytes)")


def test_sales_trend(df):
    print("\n── Testing plot_sales_trend() ──")
    path = plot_sales_trend(df, store=1, item=1)
    _check_file(path, "Sales trend")


def test_forecast_vs_actual(df):
    print("\n── Testing plot_forecast_vs_actual() ──")
    subset = df[(df["store"] == 1) & (df["item"] == 1)].iloc[:60]
    y_true = subset["sales"].values
    y_pred = y_true + np.random.normal(0, 3, len(y_true))  # simulated predictions
    path = plot_forecast_vs_actual(subset["date"], y_true, y_pred, store=1, item=1)
    _check_file(path, "Forecast vs actual")


def test_feature_importance():
    print("\n── Testing plot_feature_importance() ──")
    importances = [
        ("sales_lag_7", 0.33),
        ("is_weekend", 0.11),
        ("month", 0.07),
        ("sales_rolling_mean_7", 0.06),
        ("sales_lag_1", 0.05),
        ("day_of_week", 0.05),
        ("sales_lag_28", 0.04),
        ("quarter", 0.03),
    ]
    path = plot_feature_importance(importances, top_n=8)
    _check_file(path, "Feature importance")


def test_demand_distribution(df):
    print("\n── Testing plot_demand_distribution() ──")
    path = plot_demand_distribution(df, store=1, item=1)
    _check_file(path, "Demand distribution")


def test_weekly_pattern(df):
    print("\n── Testing plot_weekly_pattern() ──")
    path = plot_weekly_pattern(df, store=1, item=1)
    _check_file(path, "Weekly pattern")


def test_store_comparison(df):
    print("\n── Testing plot_store_comparison() ──")
    path = plot_store_comparison(df, item=1)
    _check_file(path, "Store comparison")


def test_volatility_ranking(df):
    print("\n── Testing plot_volatility_ranking() ──")
    path = plot_volatility_ranking(df, top_n=5)
    _check_file(path, "Volatility ranking")


def test_invalid_inputs(df):
    print("\n── Testing error handling ──")

    try:
        plot_sales_trend(df, store=99, item=99)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Invalid store-item raises ValueError")

    try:
        plot_demand_distribution(df, store=99, item=99)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Invalid demand distribution raises ValueError")


if __name__ == "__main__":
    print("=" * 55)
    print("  TESTS: visualizations.py")
    print("=" * 55)

    df = make_synthetic_df()
    print(f"  Synthetic data: {df.shape}")

    test_sales_trend(df)
    test_forecast_vs_actual(df)
    test_feature_importance()
    test_demand_distribution(df)
    test_weekly_pattern(df)
    test_store_comparison(df)
    test_volatility_ranking(df)
    test_invalid_inputs(df)

    # Clean up generated files
    import glob
    pngs = glob.glob(str(CFG.OUTPUT_DIR / "*.png"))
    for f in pngs:
        Path(f).unlink()
    print(f"\n  Cleaned up {len(pngs)} test chart files")

    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)