"""
Data Loader — load, validate, and summarize the raw dataset.
"""

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
from config import CFG


def load_data(path: Path = None) -> pd.DataFrame:
    """
    Load the raw CSV and perform basic validation.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file. Defaults to CFG.RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64), store (int), item (int), sales (int)
        Sorted by store, item, date.

    Raises
    ------
    FileNotFoundError
        If the CSV doesn't exist
    ValueError
        If expected columns are missing or data has unexpected nulls.
    """
    if path is None:
        path = CFG.RAW_DATA_PATH

    # ── Check file exists ────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n\n"
            f"To get the data:\n"
            f"1. Go to https://www.kaggle.com/c/demand-forecasting-kernels-only/data\n"
            f"2. Download 'train.csv'\n"
            f"3. Place it in: {CFG.DATA_DIR}/\n"
        )

    # ── Load CSV ─────────────────────────────────────────────────────
    df = pd.read_csv(path, parse_dates=["date"])

    # ── Validate expected columns ────────────────────────────────────
    expected_cols = {"date", "store", "item", "sales"}
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Found columns: {sorted(actual_cols)}\n"
            f"Make sure you're using the correct dataset "
            f"(Store Item Demand Forecasting from Kaggle)."
        )

    # ── Validate no nulls ────────────────────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Unexpected null values found: {bad_cols}\n"
            f"This dataset should have no missing values. "
            f"Re-download from Kaggle if this persists."
        )

    # ── Validate data types ──────────────────────────────────────────
    if not np.issubdtype(df["sales"].dtype, np.number):
        raise ValueError(
            f"'sales' column has dtype {df['sales'].dtype}, expected numeric. "
            f"Check the CSV for formatting issues."
        )

    # ── Sort ─────────────────────────────────────────────────────────
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or featured) DataFrame.

    Returns
    -------
    dict
        A structured summary with keys:
        - date_range: (min_date, max_date) as strings
        - n_stores: number of unique stores
        - n_items: number of unique items
        - n_rows: total row count
        - n_time_series: stores × items (number of individual series)
        - date_frequency: inferred frequency (should be "daily")
        - sales_stats: dict with mean, median, std, min, max, q25, q75
        - stores: list of store IDs
        - items: list of item IDs
    """

    sales = df["sales"]

    summary = {
        "date_range": (
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ),
        "n_stores": int(df["store"].nunique()),
        "n_items": int(df["item"].nunique()),
        "n_rows": len(df),
        "n_time_series": int(df["store"].nunique() * df["item"].nunique()),
        "date_frequency": _infer_frequency(df),
        "sales_stats": {
            "mean": round(float(sales.mean()), 2),
            "median": round(float(sales.median()), 2),
            "std": round(float(sales.std()), 2),
            "min": int(sales.min()),
            "max": int(sales.max()),
            "q25": round(float(sales.quantile(0.25)), 2),
            "q75": round(float(sales.quantile(0.75)), 2),
        },
        "stores": sorted(df["store"].unique().tolist()),
        "items": sorted(df["item"].unique().tolist()),
    }

    return summary


def get_item_summary(df: pd.DataFrame, store: int, item: int) -> dict:
    """
    Get detailed statistics for a specific store-item pair.

    Parameters
    ----------
    df : pd.DataFrame
    store : int
        Store ID (1-10 in this dataset).
    item : int
        Item ID (1-50 in this dataset).

    Returns
    -------
    dict
        Keys: store, item, n_days, date_range, sales_stats,
        trend_direction, has_weekly_pattern

    Raises
    ------
    ValueError
        If the store-item pair doesn't exist in the dataset.
    """
    mask = (df["store"] == store) & (df["item"] == item)
    subset = df.loc[mask]

    if subset.empty:
        raise ValueError(
            f"No data found for store={store}, item={item}. "
            f"Valid stores: {sorted(df['store'].unique())}. "
            f"Valid items: {sorted(df['item'].unique())}."
        )

    sales = subset["sales"]

    # ── Trend detection ──────────────────────────────────────────────
    midpoint = len(sales) // 2
    first_half_mean = sales.iloc[:midpoint].mean()
    second_half_mean = sales.iloc[midpoint:].mean()
    pct_change = (second_half_mean - first_half_mean) / first_half_mean * 100

    if pct_change > 10:
        trend = "increasing"
    elif pct_change < -10:
        trend = "decreasing"
    else:
        trend = "stable"

    # ── Weekly pattern detection ─────────────────────────────────────
    weekday_means = subset.groupby(subset["date"].dt.dayofweek)["sales"].mean()
    cv = weekday_means.std() / weekday_means.mean()
    has_weekly_pattern = bool(cv > 0.1)

    return {
        "store": store,
        "item": item,
        "n_days": len(subset),
        "date_range": (
            str(subset["date"].min().date()),
            str(subset["date"].max().date()),
        ),
        "sales_stats": {
            "mean": round(float(sales.mean()), 2),
            "median": round(float(sales.median()), 2),
            "std": round(float(sales.std()), 2),
            "min": int(sales.min()),
            "max": int(sales.max()),
            "cv": round(float(sales.std() / sales.mean()), 3),
        },
        "trend_direction": trend,
        "trend_pct_change": round(pct_change, 1),
        "has_weekly_pattern": has_weekly_pattern,
        "weekday_avg_sales": {
            day_name: round(val, 1)
            for day_name, val in zip(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                weekday_means.values,
            )
        },
    }


def _infer_frequency(df: pd.DataFrame) -> str:
    """
    Infer the time frequency of the dataset.

    Checks the most common gap between consecutive dates for the first
    store-item pair.
    """
    # Take the first store-item pair as a sample
    first_store = df["store"].iloc[0]
    first_item = df["item"].iloc[0]
    sample = df[(df["store"] == first_store) & (df["item"] == first_item)]

    if len(sample) < 2:
        return "unknown"

    # Most common gap between consecutive dates
    gaps = sample["date"].diff().dropna()
    most_common_gap = gaps.mode().iloc[0]

    if most_common_gap == pd.Timedelta(days=1):
        return "daily"
    elif most_common_gap == pd.Timedelta(weeks=1):
        return "weekly"
    elif most_common_gap >= pd.Timedelta(days=28) and most_common_gap <= pd.Timedelta(
        days=31
    ):
        return "monthly"
    else:
        return f"~{most_common_gap.days} days"


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    try:
        df = load_data()
        print(f"  Loaded {len(df):,} rows")

        print("\nDataset summary:")
        summary = get_data_summary(df)
        for key, val in summary.items():
            if key not in ("stores", "items"):  # skip long lists
                print(f"  {key}: {val}")

        print("\nSample item summary (store=1, item=1):")
        item_info = get_item_summary(df, store=1, item=1)
        for key, val in item_info.items():
            print(f"  {key}: {val}")

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        print("\n✓ All checks passed!")

    except FileNotFoundError as e:
        print(f"\n✗ {e}")

    except ValueError as e:
        print(f"\n✗ {e}")
