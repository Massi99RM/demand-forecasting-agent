"""
Feature Engineering — transform raw data into model-ready features.
"""
import sys
from pathlib import Path
 
# Ensure the project root is on Python's path.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
from config import CFG


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the date column.

    These features encode calendar patterns that drive retail demand:
    - day_of_week: 0=Monday..6=Sunday. Retail has strong weekday vs
      weekend patterns.
    - day_of_month: 1-31. Captures payday effects — many people get
      paid on the 1st or 15th, causing demand spikes.
    - month: 1-12. Captures seasonal patterns
    - week_of_year: 1-52. Finer granularity than month for seasonality.
    - quarter: 1-4. Useful for quarterly business cycles.
    - is_weekend: binary. Simplifies the weekday pattern for the model.
    - is_month_start / is_month_end: binary. Captures boundary effects
      (e.g., end-of-month inventory clearance sales).
    - year: Captures long-term trend (demand growing year over year).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'date' column of type datetime64.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new time feature columns added.
    """
    date_col = df["date"]

    df = df.copy()  # avoid modifying the caller's DataFrame

    df["day_of_week"] = date_col.dt.dayofweek        # 0-6
    df["day_of_month"] = date_col.dt.day              # 1-31
    df["month"] = date_col.dt.month                   # 1-12
    df["week_of_year"] = date_col.dt.isocalendar().week.astype(int)  # 1-52
    df["quarter"] = date_col.dt.quarter               # 1-4
    df["year"] = date_col.dt.year                     # e.g., 2017
    df["is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)  # 0 or 1
    df["is_month_start"] = date_col.dt.is_month_start.astype(int)
    df["is_month_end"] = date_col.dt.is_month_end.astype(int)

    return df


def add_lag_features(df: pd.DataFrame,lags: tuple = None,) -> pd.DataFrame:
    """
    Add lagged sales values as features.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by (store, item, date) and contain 'sales'.
    lags : tuple of int, optional
        Which lag days to create. Defaults to CFG.LAG_FEATURES.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new columns: sales_lag_1, sales_lag_7, etc.
    """
    if lags is None:
        lags = CFG.LAG_FEATURES

    df = df.copy()

    # Group by store-item so each time series gets its own lags
    grouped = df.groupby(["store", "item"])["sales"]

    for lag in lags:
        # .shift(lag) moves values down by `lag` positions.
        # Row i gets the sales value from row (i - lag).
        # For lag=1: today's row gets yesterday's sales.
        # This creates NaN for the first `lag` rows of each
        # group (there's no "7 days ago" for the first week).
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame,windows: tuple = None,) -> pd.DataFrame:
    """
    Add rolling mean and rolling std features.

    Rolling features smooth out daily noise to reveal underlying trends.
    They answer: "what's the average sales been recently?" and "how
    volatile have sales been recently?"

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by (store, item, date) and contain 'sales'.
    windows : tuple of int, optional
        Window sizes. Defaults to CFG.ROLLING_WINDOWS.

    Returns
    -------
    pd.DataFrame
        New columns: sales_rolling_mean_7, sales_rolling_std_7, etc.
    """
    if windows is None:
        windows = CFG.ROLLING_WINDOWS

    df = df.copy()

    grouped = df.groupby(["store", "item"])["sales"]

    for window in windows:
        # Compute rolling stats within each group
        # min_periods=1 means the rolling window starts computing as
        # soon as it has at least 1 data point (instead of returning
        # NaN until the full window is filled). This reduces the number
        # of NaN rows lost at the start of each series.
        df[f"sales_rolling_mean_{window}"] = grouped.transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))  

        df[f"sales_rolling_std_{window}"] = grouped.transform(lambda x: x.rolling(window=window, min_periods=1).std().shift(1))  

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday indicator features.

    Holidays cause predictable demand spikes (and sometimes dips) in
    retail.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'date' column.

    Returns
    -------
    pd.DataFrame
        New columns: is_holiday, days_to_holiday, days_from_holiday.
    """
    try:
        import holidays as holidays_lib
    except ImportError:
        # If the holidays package isn't installed, then skip it
        # The model will still work — it just won't have holiday info.
        print(
            "Warning: 'holidays' package not installed. "
            "Skipping holiday features. Install with: pip install holidays"
        )
        df = df.copy()
        df["is_holiday"] = 0
        df["days_to_holiday"] = 0
        df["days_from_holiday"] = 0
        return df

    df = df.copy()

    # Get all years in the dataset
    years = df["date"].dt.year.unique().tolist()

    # Include one extra year on each end for proximity calculations.
    # Without this, dates at the start of the earliest year would have
    # no "previous holiday" to measure distance from.
    all_years = list(range(min(years) - 1, max(years) + 2))
    us_holidays = holidays_lib.US(years=all_years)

    # Convert holiday dates to a sorted list for efficient searching
    holiday_dates = sorted(us_holidays.keys())
    holiday_dates_ts = pd.to_datetime(holiday_dates)

    # ── is_holiday ───────────────────────────────────────────────────
    # Convert dates to date objects (not datetime) for comparison with
    # the holidays library, which returns date objects.
    df["is_holiday"] = df["date"].dt.date.isin(us_holidays).astype(int)

    # ── days_to_holiday / days_from_holiday ──────────────────────────
    # For each date, find the nearest future and past holiday.

    unique_dates = df["date"].drop_duplicates().sort_values()
    date_to_next = {}
    date_to_prev = {}

    holiday_ts_array = holiday_dates_ts.values  # numpy array for searchsorted

    for date in unique_dates:
        idx = np.searchsorted(holiday_ts_array, date.to_numpy())

        # Days to next holiday
        if idx < len(holiday_ts_array):
            delta_next = (pd.Timestamp(holiday_ts_array[idx]) - date).days
            date_to_next[date] = max(0, delta_next)
        else:
            date_to_next[date] = 365  # fallback: no future holiday found

        # Days from previous holiday
        if idx > 0:
            delta_prev = (date - pd.Timestamp(holiday_ts_array[idx - 1])).days
            date_to_prev[date] = max(0, delta_prev)
        else:
            date_to_prev[date] = 365  # fallback: no past holiday found

    df["days_to_holiday"] = df["date"].map(date_to_next)
    df["days_from_holiday"] = df["date"].map(date_to_prev)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate the full feature engineering pipeline.

    Call this once on the raw DataFrame to get a model-ready dataset.
    The order matters:
    1. Time features first (no dependencies)
    2. Lag features (depend on sales column being present)
    3. Rolling features (depend on sales column being present)
    4. Holiday features (no dependencies)
    5. Drop NaN rows last (lags and rolling create NaNs at the start)

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_data().

    Returns
    -------
    pd.DataFrame
        Fully featured DataFrame, NaN rows dropped, ready for model.
    """
    print("Building features...")

    print("  Adding time features...")
    df = add_time_features(df)

    print("  Adding lag features...")
    df = add_lag_features(df)

    print("  Adding rolling features...")
    df = add_rolling_features(df)

    print("  Adding holiday features...")
    df = add_holiday_features(df)

    # ── Drop NaN rows ────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    n_dropped = n_before - n_after
    pct_dropped = n_dropped / n_before * 100
    print(f"  Dropped {n_dropped:,} NaN rows ({pct_dropped:.1f}% of data)")

    # ── Final validation ─────────────────────────────────────────────
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        raise ValueError(
            f"Still have {remaining_nulls} null values after dropna(). "
            f"This shouldn't happen — check feature engineering logic."
        )

    feature_cols = [
        c for c in df.columns if c not in CFG.EXCLUDE_COLS
    ]
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")
    print(f"  Final shape: {df.shape}")

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excluding target and date).

    Utility function used by the model and tools layers to know which
    columns to feed into XGBoost.

    Parameters
    ----------
    df : pd.DataFrame
        The featured DataFrame (after build_features).

    Returns
    -------
    list of str
        Column names to use as model input features.
    """
    return [
        col for col in df.columns
        if col not in CFG.EXCLUDE_COLS
    ]


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_data

    print("Loading data...")
    df = load_data()
    print(f"  Raw shape: {df.shape}")

    df_featured = build_features(df)
    print(f"\nFeatured shape: {df_featured.shape}")
    print(f"\nSample row (first non-NaN):")
    print(df_featured.iloc[0].to_string())
    print(f"\nFeature names:")
    print(get_feature_names(df_featured))