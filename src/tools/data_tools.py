"""
Data Tools — let the agent explore and understand the dataset.
"""

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_core.tools import tool
from src.agent import state
from src.data_loader import load_data, get_data_summary, get_item_summary
from src.feature_engineering import build_features, get_feature_names


def _ensure_data_loaded():
    """Load and feature-engineer data if not already done."""
    if not state.is_data_loaded:
        state.raw_df = load_data()
        state.featured_df = build_features(state.raw_df)
        state.feature_names = get_feature_names(state.featured_df)
        state.is_data_loaded = True


@tool
def explore_dataset() -> str:
    """Load the demand forecasting dataset and return a comprehensive summary.

    Call this FIRST when starting any analysis. It loads the data, runs
    feature engineering, and returns: date range, number of stores,
    number of items, total rows, sales statistics (mean, median, std,
    min, max), and the list of available stores and items.

    This tool takes no arguments.
    """
    _ensure_data_loaded()
    summary = get_data_summary(state.raw_df)

    result = (
        f"Dataset loaded successfully.\n\n"
        f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}\n"
        f"Stores: {summary['n_stores']} (IDs: {summary['stores']})\n"
        f"Items: {summary['n_items']} (IDs: 1 through {max(summary['items'])})\n"
        f"Total time series: {summary['n_time_series']}\n"
        f"Total rows: {summary['n_rows']:,}\n"
        f"Frequency: {summary['date_frequency']}\n\n"
        f"Sales statistics:\n"
        f"  Mean: {summary['sales_stats']['mean']}\n"
        f"  Median: {summary['sales_stats']['median']}\n"
        f"  Std: {summary['sales_stats']['std']}\n"
        f"  Min: {summary['sales_stats']['min']}, Max: {summary['sales_stats']['max']}\n"
        f"  25th percentile: {summary['sales_stats']['q25']}\n"
        f"  75th percentile: {summary['sales_stats']['q75']}\n\n"
        f"Features engineered: {len(state.feature_names)}\n"
        f"Feature list: {state.feature_names}\n\n"
        f"Data is ready. You can now get item details, train a model, "
        f"or analyze demand patterns."
    )

    return result


@tool
def get_item_details(store: int, item: int) -> str:
    """Get detailed statistics for a specific store-item combination.

    Returns: number of days, date range, sales statistics (mean, median,
    std, min, max, coefficient of variation), trend direction, whether
    a weekly pattern exists, and average sales by day of week.

    Args:
        store: Store ID (integer, typically 1-10).
        item: Item ID (integer, typically 1-50).
    """
    _ensure_data_loaded()

    try:
        info = get_item_summary(state.raw_df, store=store, item=item)
    except ValueError as e:
        return f"Error: {str(e)}"

    result = (
        f"Store {store}, Item {item}\n\n"
        f"Data: {info['n_days']} days ({info['date_range'][0]} to {info['date_range'][1]})\n\n"
        f"Sales statistics:\n"
        f"  Mean: {info['sales_stats']['mean']} units/day\n"
        f"  Median: {info['sales_stats']['median']} units/day\n"
        f"  Std: {info['sales_stats']['std']}\n"
        f"  Min: {info['sales_stats']['min']}, Max: {info['sales_stats']['max']}\n"
        f"  Coefficient of Variation: {info['sales_stats']['cv']}\n\n"
        f"Trend: {info['trend_direction']} ({info['trend_pct_change']:+.1f}% over the period)\n"
        f"Weekly pattern: {'Yes' if info['has_weekly_pattern'] else 'No'}\n\n"
        f"Average sales by day of week:\n"
    )

    for day, avg in info["weekday_avg_sales"].items():
        bar = "█" * int(avg / 2)
        result += f"  {day}: {avg:5.1f} {bar}\n"

    return result
