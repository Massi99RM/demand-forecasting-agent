"""
Analysis Tools — answer higher-level business questions.
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
from src.model import predict, evaluate_model
from config import CFG


def _ensure_data_loaded():
    """Load and feature-engineer data if not already done."""
    if not state.is_data_loaded:
        state.raw_df = load_data()
        state.featured_df = build_features(state.raw_df)
        state.feature_names = get_feature_names(state.featured_df)
        state.is_data_loaded = True


@tool
def find_volatile_products(top_n: int = 10) -> str:
    """Find the products with the most unpredictable demand.

    Ranks all store-item pairs by Coefficient of Variation (CV = std / mean).
    Higher CV means more volatile demand and harder to forecast accurately.

    Use this when asked: "which products are hardest to predict?",
    "which items have unstable demand?", or "what should we focus on?"

    Args:
        top_n: Number of most-volatile products to return. Default 10.
    """
    _ensure_data_loaded()

    item_stats = (
        state.raw_df.groupby(["store", "item"])["sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    item_stats["cv"] = item_stats["std"] / item_stats["mean"]
    top_volatile = item_stats.nlargest(top_n, "cv")
    bottom_stable = item_stats.nsmallest(5, "cv")

    result = f"Top {top_n} most volatile products (by CV):\n\n"
    result += f"{'Rank':<6} {'Store':<8} {'Item':<8} {'CV':<8} {'Mean':<10} {'Std':<10}\n"
    result += "-" * 52 + "\n"

    for rank, (_, row) in enumerate(top_volatile.iterrows(), 1):
        result += (
            f"{rank:<6} {int(row['store']):<8} {int(row['item']):<8} "
            f"{row['cv']:<8.3f} {row['mean']:<10.1f} {row['std']:<10.1f}\n"
        )

    result += (
        f"\nFor reference, the 5 most STABLE products:\n"
    )
    for _, row in bottom_stable.iterrows():
        result += (
            f"  Store {int(row['store'])}, Item {int(row['item'])}: "
            f"CV={row['cv']:.3f} (mean={row['mean']:.1f})\n"
        )

    result += (
        f"\nCV interpretation:\n"
        f"  < 0.20 = Stable (easy to forecast)\n"
        f"  0.20-0.30 = Moderate volatility\n"
        f"  > 0.30 = High volatility (hard to forecast)"
    )

    return result


@tool
def simulate_demand_spike(store: int, item: int, spike_pct: float) -> str:
    """Simulate a demand spike and show the impact on forecasts.

    Takes a store-item pair and a percentage spike (e.g., 20 for a 20%
    increase), applies it to the actual test data, and shows how the
    current model's predictions would compare against the spiked demand.

    Use this for what-if scenarios: "what if demand increases by 30%?"
    or "how bad would our forecast be during a 50% surge?"

    The model must be trained first.

    Args:
        store: Store ID (integer, typically 1-10).
        item: Item ID (integer, typically 1-50).
        spike_pct: Percentage increase in demand (e.g., 20 for +20%).
    """
    if not state.is_model_trained:
        return (
            "Error: No trained model available. "
            "Please call train_forecast_model first."
        )

    _ensure_data_loaded()

    # Get test data for this item
    df_test = state.featured_df[
        state.featured_df["date"] >= CFG.TEST_START_DATE
    ]
    mask = (df_test["store"] == store) & (df_test["item"] == item)
    df_item = df_test[mask]

    if df_item.empty:
        return f"Error: No test data for store={store}, item={item}."

    X_item = df_item[state.feature_names]
    y_actual = df_item["sales"].values
    y_pred = predict(state.model, X_item)

    # Apply spike to actual demand
    spike_factor = 1 + spike_pct / 100
    y_spiked = y_actual * spike_factor

    # Compute metrics: normal vs spiked
    metrics_normal = evaluate_model(y_actual, y_pred)
    metrics_spiked = evaluate_model(y_spiked, y_pred)

    # Calculate the inventory shortfall
    # If predicted < spiked_actual, we'd run out of stock
    shortfall_units = np.maximum(y_spiked - y_pred, 0).sum()
    shortfall_days = np.sum(y_spiked > y_pred)

    mae_diff = metrics_spiked["mae"] - metrics_normal["mae"]
    mape_diff = metrics_spiked["mape"] - metrics_normal["mape"]

    result = (
        f"Demand spike simulation: Store {store}, Item {item}\n"
        f"Scenario: +{spike_pct:.0f}% demand increase\n\n"
        f"                    {'Normal':>12} {'With Spike':>12} {'Change':>10}\n"
        f"{'─'*48}\n"
        f"Avg daily demand    {y_actual.mean():>12.1f} {y_spiked.mean():>12.1f} "
        f"{'+'}{spike_pct:.0f}%\n"
        f"Model MAE           {metrics_normal['mae']:>12.2f} {metrics_spiked['mae']:>12.2f} "
        f"{'+' if mae_diff >= 0 else ''}{mae_diff:.2f}\n"
        f"Model MAPE          {metrics_normal['mape']:>11.1f}% {metrics_spiked['mape']:>11.1f}% "
        f"{'+' if mape_diff >= 0 else ''}{mape_diff:.1f}%\n\n"
        f"Inventory impact:\n"
        f"  Total shortfall: {shortfall_units:.0f} units over the test period\n"
        f"  Days understocked: {int(shortfall_days)} / {len(y_actual)} days "
        f"({shortfall_days / len(y_actual) * 100:.0f}%)\n"
        f"  Avg daily shortfall: {shortfall_units / len(y_actual):.1f} units\n\n"
        f"Recommendation: "
    )

    if spike_pct <= 10:
        result += "Minor spike. Current safety stock may absorb this."
    elif spike_pct <= 30:
        result += "Moderate spike. Consider increasing safety stock by the spike percentage."
    else:
        result += "Major spike. Requires supply chain intervention — expedited orders, alternative suppliers, or demand management."

    return result


@tool
def compare_stores(item: int) -> str:
    """Compare demand for a specific item across all stores.

    Shows average daily demand, standard deviation, and coefficient of
    variation for each store, ranked by volume. Useful for inventory
    allocation decisions.

    Use this when asked: "which store sells the most of item X?" or
    "how does demand compare across locations?"

    Args:
        item: Item ID (integer, typically 1-50).
    """
    _ensure_data_loaded()

    subset = state.raw_df[state.raw_df["item"] == item]
    if subset.empty:
        return f"Error: No data for item={item}."

    store_stats = (
        subset.groupby("store")["sales"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    store_stats["cv"] = store_stats["std"] / store_stats["mean"]
    store_stats = store_stats.sort_values("mean", ascending=False)

    result = (
        f"Store comparison for Item {item}\n\n"
        f"{'Store':<8} {'Avg/Day':<10} {'Std':<8} {'CV':<8} "
        f"{'Min':<8} {'Max':<8} {'Volume':<12}\n"
        f"{'─'*64}\n"
    )

    overall_mean = subset["sales"].mean()

    for _, row in store_stats.iterrows():
        vs_avg = ((row["mean"] - overall_mean) / overall_mean) * 100
        volume_label = "▲" if vs_avg > 10 else "▼" if vs_avg < -10 else "─"
        result += (
            f"{int(row['store']):<8} {row['mean']:<10.1f} {row['std']:<8.1f} "
            f"{row['cv']:<8.3f} {int(row['min']):<8} {int(row['max']):<8} "
            f"{volume_label} {vs_avg:+.0f}% vs avg\n"
        )

    result += (
        f"\nOverall average for Item {item}: {overall_mean:.1f} units/day\n"
        f"▲ = significantly above average, ▼ = significantly below, "
        f"─ = near average"
    )

    return result