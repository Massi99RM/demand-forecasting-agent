"""
Test using synthetic data that mimics the Kaggle dataset structure.
Verifies that config.py and data_loader.py work correctly if
the real dataset isn't downloaded yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CFG
from src.data_loader import load_data, get_data_summary, get_item_summary


def create_synthetic_data(path: Path):
    """
    Create a small synthetic dataset with the same structure as
    the Kaggle Store Item Demand Forecasting dataset.

    2 stores × 3 items × 365 days = 2,190 rows
    (real dataset: 10 stores × 50 items × 1,826 days = 913,000 rows)
    """
    dates = pd.date_range("2017-01-01", "2017-12-31", freq="D")
    rows = []

    np.random.seed(42)
    for store in [1, 2]:
        for item in [1, 2, 3]:
            # Base demand varies by store and item
            base = 20 + store * 5 + item * 3

            for date in dates:
                # Weekly seasonality: weekends are higher
                weekday_effect = 5 if date.dayofweek >= 5 else 0
                # Monthly seasonality: December has a holiday spike
                month_effect = 10 if date.month == 12 else 0
                # Random noise
                noise = np.random.randint(-5, 6)

                sales = max(0, base + weekday_effect + month_effect + noise)
                rows.append({
                    "date": date,
                    "store": store,
                    "item": item,
                    "sales": sales,
                })

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Created synthetic data: {len(df)} rows at {path}")
    return df


def test_config():
    """Verify Config loads and has expected fields."""
    print("\n── Testing config.py ──")

    assert CFG.FORECAST_HORIZON == 30, "Default horizon should be 30"
    assert CFG.TARGET == "sales", "Target should be 'sales'"
    assert CFG.TRAIN_END_DATE == "2017-09-30"
    assert CFG.TEST_START_DATE == "2017-10-01"
    assert len(CFG.LAG_FEATURES) == 4
    assert len(CFG.ROLLING_WINDOWS) == 3
    assert CFG.MODEL_PARAMS["n_estimators"] == 500

    # Test that frozen=True prevents mutation
    try:
        CFG.FORECAST_HORIZON = 60
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected — config is immutable

    print("  ✓ All config checks passed")


def test_load_data(data_path: Path):
    """Test data loading and validation."""
    print("\n── Testing load_data() ──")

    df = load_data(path=data_path)

    # Check shape
    assert len(df) == 2190, f"Expected 2190 rows, got {len(df)}"
    assert list(df.columns) == ["date", "store", "item", "sales"]

    # Check dtypes
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["sales"])

    # Check sorting (must be sorted by store, item, date)
    assert df.iloc[0]["store"] == 1
    assert df.iloc[0]["item"] == 1
    assert df["date"].iloc[0] <= df["date"].iloc[1]

    # Check no nulls
    assert df.isnull().sum().sum() == 0

    print(f"  ✓ Loaded {len(df)} rows, 4 columns")
    print(f"  ✓ Dtypes correct: {dict(df.dtypes)}")
    print(f"  ✓ Sorted correctly")
    print(f"  ✓ No null values")

    return df


def test_data_summary(df: pd.DataFrame):
    """Test the summary function."""
    print("\n── Testing get_data_summary() ──")

    summary = get_data_summary(df)

    assert summary["n_stores"] == 2
    assert summary["n_items"] == 3
    assert summary["n_time_series"] == 6
    assert summary["n_rows"] == 2190
    assert summary["date_frequency"] == "daily"
    assert "mean" in summary["sales_stats"]
    assert "std" in summary["sales_stats"]

    print(f"  ✓ {summary['n_stores']} stores, {summary['n_items']} items")
    print(f"  ✓ {summary['n_time_series']} time series, {summary['n_rows']} rows")
    print(f"  ✓ Frequency: {summary['date_frequency']}")
    print(f"  ✓ Sales mean: {summary['sales_stats']['mean']}")
    print(f"  ✓ Date range: {summary['date_range']}")


def test_item_summary(df: pd.DataFrame):
    """Test per-item summary."""
    print("\n── Testing get_item_summary() ──")

    info = get_item_summary(df, store=1, item=1)

    assert info["store"] == 1
    assert info["item"] == 1
    assert info["n_days"] == 365
    assert info["trend_direction"] in ("increasing", "decreasing", "stable")
    assert isinstance(info["has_weekly_pattern"], bool)
    assert len(info["weekday_avg_sales"]) == 7

    print(f"  ✓ Store 1, Item 1: {info['n_days']} days")
    print(f"  ✓ Trend: {info['trend_direction']} ({info['trend_pct_change']}%)")
    print(f"  ✓ Weekly pattern: {info['has_weekly_pattern']}")
    print(f"  ✓ Weekday averages: {info['weekday_avg_sales']}")

    # Test error handling for invalid store-item
    try:
        get_item_summary(df, store=99, item=99)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Invalid store-item correctly raises ValueError")


def test_missing_file():
    """Test that a helpful error is shown for missing data."""
    print("\n── Testing missing file error ──")

    try:
        load_data(path=Path("/nonexistent/path/train.csv"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "kaggle" in str(e).lower(), "Error should mention Kaggle"
        print(f"  ✓ FileNotFoundError with helpful message")


if __name__ == "__main__":
    print("=" * 50)
    print("  SMOKE TEST: config.py + data_loader.py")
    print("=" * 50)

    test_config()

    # Create synthetic data for testing
    test_data_path = Path("/tmp/test_train.csv")
    print("\n── Creating synthetic test data ──")
    create_synthetic_data(test_data_path)

    df = test_load_data(test_data_path)
    test_data_summary(df)
    test_item_summary(df)
    test_missing_file()

    # Clean up
    test_data_path.unlink()

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED ✓")
    print("=" * 50)