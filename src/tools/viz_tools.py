"""
Visualization Tools — generate charts the agent can reference.
"""

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from langchain_core.tools import tool
from src.agent import state
from src.data_loader import load_data
from src.feature_engineering import build_features, get_feature_names
from src.model import predict
from src.visualizations import (
    plot_sales_trend,
    plot_forecast_vs_actual,
    plot_weekly_pattern,
    plot_volatility_ranking,
)
from config import CFG


def _ensure_data_loaded():
    """Load and feature-engineer data if not already done."""
    if not state.is_data_loaded:
        state.raw_df = load_data()
        state.featured_df = build_features(state.raw_df)
        state.feature_names = get_feature_names(state.featured_df)
        state.is_data_loaded = True


@tool
def plot_sales_history(store: int, item: int) -> str:
    """Generate a sales trend chart for a specific store-item pair.

    Creates a line chart showing daily sales and a 30-day moving
    average over the full history. Saves the chart as a PNG file
    and returns the file path.

    Args:
        store: Store ID (integer, typically 1-10).
        item: Item ID (integer, typically 1-50).
    """
    _ensure_data_loaded()

    try:
        path = plot_sales_trend(state.raw_df, store=store, item=item)
        return (
            f"Sales trend chart generated for Store {store}, Item {item}.\n"
            f"Saved to: {path}\n\n"
            f"The chart shows daily sales (thin blue line) and a 30-day "
            f"moving average (orange line) revealing the underlying trend "
            f"and seasonal patterns."
        )
    except ValueError as e:
        return f"Error: {str(e)}"


@tool
def plot_forecast_chart(store: int, item: int) -> str:
    """Generate a forecast vs actual chart for a specific store-item pair.

    Creates a dual-panel chart: top panel shows actual vs predicted
    sales with error shading, bottom panel shows the error over time.
    Only covers the test period.

    The model must be trained first.

    Args:
        store: Store ID (integer, typically 1-10).
        item: Item ID (integer, typically 1-50).
    """
    if not state.is_model_trained:
        return (
            "Error: No trained model available. "
            "Please call train_forecast_model first."
        )

    _ensure_data_loaded()

    # Filter test data to this specific store-item pair
    df_test = state.featured_df[
        state.featured_df["date"] >= CFG.TEST_START_DATE
    ]
    mask = (df_test["store"] == store) & (df_test["item"] == item)
    df_item = df_test[mask]

    if df_item.empty:
        return f"Error: No test data for store={store}, item={item}."

    # Generate predictions for this item only
    X_item = df_item[state.feature_names]
    y_true = df_item["sales"].values
    y_pred = predict(state.model, X_item)
    dates = df_item["date"]

    try:
        path = plot_forecast_vs_actual(
            dates, y_true, y_pred, store=store, item=item
        )
        mae = np.mean(np.abs(y_true - y_pred))
        return (
            f"Forecast vs actual chart generated for Store {store}, Item {item}.\n"
            f"Saved to: {path}\n\n"
            f"Top panel: blue = actual sales, orange dashed = predictions, "
            f"red shading = forecast error.\n"
            f"Bottom panel: error bars (green = over-predicted, "
            f"red = under-predicted).\n"
            f"Average error: {mae:.2f} units/day."
        )
    except Exception as e:
        return f"Error generating chart: {str(e)}"


@tool
def plot_weekly_pattern_chart(store: int, item: int) -> str:
    """Generate a weekly demand pattern chart for a specific store-item pair.

    Creates a bar chart showing average sales by day of week, with
    weekend bars highlighted. Error bars show daily variability.

    Args:
        store: Store ID (integer, typically 1-10).
        item: Item ID (integer, typically 1-50).
    """
    _ensure_data_loaded()

    try:
        path = plot_weekly_pattern(state.raw_df, store=store, item=item)
        return (
            f"Weekly pattern chart generated for Store {store}, Item {item}.\n"
            f"Saved to: {path}\n\n"
            f"Blue bars = weekdays, orange bars = weekends. "
            f"Error bars show standard deviation (daily variability)."
        )
    except ValueError as e:
        return f"Error: {str(e)}"


@tool
def plot_volatility_chart(top_n: int = 15) -> str:
    """Generate a volatility ranking chart showing the most unpredictable products.

    Creates a horizontal bar chart ranking products by Coefficient
    of Variation (CV). Red = high volatility, amber = moderate,
    green = low.

    Args:
        top_n: Number of most-volatile products to show. Default 15.
    """
    _ensure_data_loaded()

    try:
        path = plot_volatility_ranking(state.raw_df, top_n=top_n)
        return (
            f"Volatility ranking chart generated (top {top_n} products).\n"
            f"Saved to: {path}\n\n"
            f"Products ranked by CV (std/mean). "
            f"Red = high volatility (CV > 0.3), "
            f"amber = moderate (0.2-0.3), green = stable (< 0.2)."
        )
    except Exception as e:
        return f"Error generating chart: {str(e)}"