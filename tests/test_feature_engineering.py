"""
Tests for feature_engineering.py.

These tests verify:
1. Correctness — features are computed right
2. No data leakage — features only use past information
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CFG
from src.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_holiday_features,
    build_features,
    get_feature_names,
)


def make_simple_df() -> pd.DataFrame:
    """
    Creates a minimal DataFrame

    1 store, 1 item, 10 days with sales = [10, 20, 30, ..., 100].
    Simple sequential values, easier to check:
      - lag_1 of day 3 (sales=30) should be 20 (day 2's sales)
      - rolling_mean_3 shifted by 1 for day 5 should be mean(20,30,40) = 30
    """
    dates = pd.date_range("2017-01-02", periods=10, freq="D")  # Mon Jan 2
    return pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })


def make_multigroup_df() -> pd.DataFrame:
    """
    Create a DataFrame with 2 store-item groups to test that
    features are computed within groups, not across them.
    """
    dates = pd.date_range("2017-01-02", periods=5, freq="D")

    group1 = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": [100, 200, 300, 400, 500],
    })
    group2 = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 2,
        "sales": [10, 20, 30, 40, 50],
    })

    return pd.concat([group1, group2]).sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)


def test_time_features():
    """Verify time features are extracted correctly."""
    print("\n── Testing add_time_features() ──")

    df = make_simple_df()
    result = add_time_features(df)

    # Jan 2, 2017 is a Monday (dayofweek=0)
    row0 = result.iloc[0]
    assert row0["day_of_week"] == 0, f"Expected Monday=0, got {row0['day_of_week']}"
    assert row0["day_of_month"] == 2
    assert row0["month"] == 1
    assert row0["quarter"] == 1
    assert row0["year"] == 2017
    assert row0["is_weekend"] == 0  # Monday is not weekend

    # Jan 7, 2017 is a Saturday (dayofweek=5)
    row5 = result.iloc[5]  # 6th day = Jan 7
    assert row5["day_of_week"] == 5
    assert row5["is_weekend"] == 1  # Saturday is weekend

    # Original data should not be modified
    assert "day_of_week" not in df.columns, "Original df was modified!"

    expected_new_cols = {
        "day_of_week", "day_of_month", "month", "week_of_year",
        "quarter", "year", "is_weekend", "is_month_start", "is_month_end",
    }
    actual_new_cols = set(result.columns) - set(df.columns)
    assert expected_new_cols == actual_new_cols, (
        f"Missing columns: {expected_new_cols - actual_new_cols}"
    )

    print("  ✓ Day of week correct (Monday=0, Saturday=5)")
    print("  ✓ Weekend detection correct")
    print("  ✓ Month, quarter, year correct")
    print("  ✓ Original DataFrame not modified")
    print(f"  ✓ Added {len(expected_new_cols)} time features")


def test_lag_features():
    """Verify lag features and data leakage protection."""
    print("\n── Testing add_lag_features() ──")

    df = make_simple_df()
    result = add_lag_features(df, lags=(1, 3))

    # lag_1: each row should have the PREVIOUS row's sales
    # Row 0 (sales=10): lag_1 = NaN (no previous day)
    # Row 1 (sales=20): lag_1 = 10
    # Row 2 (sales=30): lag_1 = 20
    assert pd.isna(result.iloc[0]["sales_lag_1"]), "First row lag_1 should be NaN"
    assert result.iloc[1]["sales_lag_1"] == 10, "lag_1 of row 1 should be 10"
    assert result.iloc[2]["sales_lag_1"] == 20, "lag_1 of row 2 should be 20"

    # lag_3: row should have sales from 3 days ago
    # Row 3 (sales=40): lag_3 = 10 (row 0's sales)
    # Row 4 (sales=50): lag_3 = 20 (row 1's sales)
    assert pd.isna(result.iloc[0]["sales_lag_3"])
    assert pd.isna(result.iloc[1]["sales_lag_3"])
    assert pd.isna(result.iloc[2]["sales_lag_3"])
    assert result.iloc[3]["sales_lag_3"] == 10
    assert result.iloc[4]["sales_lag_3"] == 20

    print("  ✓ lag_1 values correct")
    print("  ✓ lag_3 values correct")
    print("  ✓ NaN in correct positions (no future leakage)")


def test_lag_features_multigroup():
    """
    Verify lags don't leak across store-item groups.
    """
    print("\n── Testing lag features across groups ──")

    df = make_multigroup_df()
    result = add_lag_features(df, lags=(1,))

    # Item 1 rows are indices 0-4, Item 2 rows are indices 5-9
    # Item 2, first row (index 5, sales=10): lag_1 should be NaN
    # not 500 (which is item 1's last sales value)
    item2_first = result.iloc[5]
    assert pd.isna(item2_first["sales_lag_1"]), (
        f"Item 2's first lag_1 should be NaN, got {item2_first['sales_lag_1']}. "
        f"This means lags are leaking across groups!"
    )

    # Item 2, second row (index 6, sales=20): lag_1 should be 10
    item2_second = result.iloc[6]
    assert item2_second["sales_lag_1"] == 10, (
        f"Item 2's second lag_1 should be 10, got {item2_second['sales_lag_1']}"
    )

    print("  ✓ No cross-group contamination in lag features")
    print("  ✓ Each store-item pair has independent lag computation")


def test_rolling_features():
    """Verify rolling features and the critical shift(1)."""
    print("\n── Testing add_rolling_features() ──")

    df = make_simple_df()
    result = add_rolling_features(df, windows=(3,))

    # Rolling mean with window=3, then shifted by 1:
    # Row 0 (sales=10): shifted, so NaN
    # Row 1 (sales=20): shift(1) → row 0's rolling = mean(10) = 10.0
    # Row 2 (sales=30): shift(1) → row 1's rolling = mean(10,20) = 15.0
    # Row 3 (sales=40): shift(1) → row 2's rolling = mean(10,20,30) = 20.0
    # Row 4 (sales=50): shift(1) → row 3's rolling = mean(20,30,40) = 30.0

    col = "sales_rolling_mean_3"

    assert pd.isna(result.iloc[0][col]), "Row 0 should be NaN (shifted)"
    assert result.iloc[1][col] == 10.0, f"Row 1: expected 10.0, got {result.iloc[1][col]}"
    assert result.iloc[2][col] == 15.0, f"Row 2: expected 15.0, got {result.iloc[2][col]}"
    assert result.iloc[3][col] == 20.0, f"Row 3: expected 20.0, got {result.iloc[3][col]}"
    assert result.iloc[4][col] == 30.0, f"Row 4: expected 30.0, got {result.iloc[4][col]}"

    print("  ✓ Rolling mean values correct")
    print("  ✓ shift(1) applied — no current-row leakage")


def test_rolling_no_leakage():
    """
    Verify that for any row, the rolling mean feature does not include
    that row's actual sales value. This is the data leakage test.

    For each row, the rolling_mean_N (shifted by 1) should
    be computable from only the previous rows' sales values.
    """
    print("\n── Testing rolling features: NO DATA LEAKAGE ──")

    df = make_simple_df()
    result = add_rolling_features(df, windows=(3,))

    sales = df["sales"].values
    col = "sales_rolling_mean_3"

    for i in range(len(result)):
        rolling_val = result.iloc[i][col]

        if pd.isna(rolling_val):
            # NaN is fine for the first row (no history)
            continue

        # The rolling mean at row i (after shift) should be the mean
        # of the window ending at row i-1 (not row i).
        # Window of 3 ending at i-1: rows [max(0, i-3), i-1]
        window_start = max(0, i - 3)
        window_end = i  # exclusive (up to but not including current row)
        expected = np.mean(sales[window_start:window_end])

        assert abs(rolling_val - expected) < 0.01, (
            f"Row {i}: rolling_mean_3 = {rolling_val}, but based on "
            f"past sales {sales[window_start:window_end]}, expected {expected}. "
            f"DATA LEAKAGE DETECTED!"
        )

    print("  ✓ No data leakage in rolling features (verified row by row)")


def test_rolling_multigroup():
    """Verify rolling features don't leak across groups."""
    print("\n── Testing rolling features across groups ──")

    df = make_multigroup_df()
    result = add_rolling_features(df, windows=(3,))

    col = "sales_rolling_mean_3"

    # Item 2's first row (index 5) should be NaN after shift
    assert pd.isna(result.iloc[5][col]), (
        f"Item 2's first rolling mean should be NaN, got {result.iloc[5][col]}. "
        f"Rolling features are leaking across groups!"
    )

    print("  ✓ No cross-group contamination in rolling features")


def test_holiday_features():
    """Verify holiday detection."""
    print("\n── Testing add_holiday_features() ──")

    # Create a DataFrame that spans a known USA holiday
    # For example, July 4, 2017 is Independence Day
    dates = pd.date_range("2017-07-01", "2017-07-07", freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": range(10, 80, 10),
    })

    result = add_holiday_features(df)

    # July 4 should be a holiday
    july4_row = result[result["date"] == "2017-07-04"].iloc[0]
    assert july4_row["is_holiday"] == 1, "July 4 should be flagged as holiday"

    # July 3 should not be a holiday
    july3_row = result[result["date"] == "2017-07-03"].iloc[0]
    assert july3_row["is_holiday"] == 0, "July 3 should NOT be a holiday"

    # July 3: days_to_holiday should be 1 (tomorrow is July 4)
    assert july3_row["days_to_holiday"] == 1, (
        f"July 3 should be 1 day to holiday, got {july3_row['days_to_holiday']}"
    )

    # July 4: days_to_holiday should be 0
    assert july4_row["days_to_holiday"] == 0, (
        f"July 4 should be 0 days to holiday, got {july4_row['days_to_holiday']}"
    )

    print("  ✓ Holiday detection correct (July 4 = holiday)")
    print("  ✓ days_to_holiday correct")
    print("  ✓ Non-holidays correctly identified")


def test_build_features():
    """Test the full pipeline orchestrator."""
    print("\n── Testing build_features() ──")

    # Create a larger synthetic dataset (needs enough rows for lag_28)
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    np.random.seed(42)

    df = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": np.random.randint(10, 100, size=60),
    })

    result = build_features(df)

    # Should have no NaN values
    assert result.isnull().sum().sum() == 0, "build_features output has NaN values!"

    # Should have fewer rows (NaN rows dropped)
    assert len(result) < len(df), "Should have dropped NaN rows"

    # Should have more columns
    assert len(result.columns) > len(df.columns), "Should have added features"

    # Feature names should exclude date and sales
    feature_names = get_feature_names(result)
    assert "date" not in feature_names
    assert "sales" not in feature_names
    assert "sales_lag_1" in feature_names
    assert "day_of_week" in feature_names

    print(f"  ✓ Input: {len(df)} rows, {len(df.columns)} columns")
    print(f"  ✓ Output: {len(result)} rows, {len(result.columns)} columns")
    print(f"  ✓ No NaN values in output")
    print(f"  ✓ {len(feature_names)} feature columns (excluding date & sales)")


def test_original_not_modified():
    """Verify that no function modifies the input DataFrame."""
    print("\n── Testing immutability ──")

    df = make_simple_df()
    original_cols = list(df.columns)
    original_shape = df.shape

    _ = add_time_features(df)
    assert list(df.columns) == original_cols, "add_time_features modified input"

    _ = add_lag_features(df)
    assert list(df.columns) == original_cols, "add_lag_features modified input"

    _ = add_rolling_features(df)
    assert list(df.columns) == original_cols, "add_rolling_features modified input"

    _ = add_holiday_features(df)
    assert list(df.columns) == original_cols, "add_holiday_features modified input"

    assert df.shape == original_shape, "Input DataFrame shape changed"

    print("  ✓ All functions return new DataFrames (input never modified)")


if __name__ == "__main__":
    print("=" * 55)
    print("  TESTS: feature_engineering.py")
    print("=" * 55)

    test_time_features()
    test_lag_features()
    test_lag_features_multigroup()
    test_rolling_features()
    test_rolling_no_leakage()
    test_rolling_multigroup()
    test_holiday_features()
    test_build_features()
    test_original_not_modified()

    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)