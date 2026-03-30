"""
Visualizations — generate charts and plots for the agent.
"""
import sys
from pathlib import Path
 
# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from config import CFG


# ── Style Constants ──────────────────────────────────────────────────
# Consistent visual identity across all charts.
COLORS = {
    "primary": "#2563EB",     # blue — main data
    "secondary": "#F59E0B",   # amber — comparisons / predictions
    "accent": "#10B981",      # green — positive / good
    "danger": "#EF4444",      # red — negative / bad
    "neutral": "#6B7280",     # gray — gridlines, secondary text
    "background": "#FAFAFA",  # light gray background
}

FIGSIZE_WIDE = (12, 5)    # for time series, trend charts
FIGSIZE_SQUARE = (8, 6)   # for bar charts, distributions
DPI = 150                  # good quality without huge file sizes


def _setup_style():
    """Apply consistent style to all charts."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })


def _save_and_close(fig: plt.Figure, filename: str) -> str:
    """
    Save figure to outputs directory and close it to free memory.

    Returns the file path as a string (for the agent to reference).
    """
    output_path = CFG.OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_sales_trend(df: pd.DataFrame,store: int,item: int,) -> str:
    """
    Plot daily sales over time for a specific store-item pair.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'store', 'item', 'sales' columns.
    store : int
        Store ID.
    item : int
        Item ID.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    _setup_style()

    subset = df[(df["store"] == store) & (df["item"] == item)].copy()
    if subset.empty:
        raise ValueError(f"No data for store={store}, item={item}")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Plot raw daily sales as a thin line
    ax.plot(
        subset["date"], subset["sales"],
        color=COLORS["primary"], linewidth=0.7, alpha=0.6,
        label="Daily sales",
    )

    # Add a 30-day rolling average to show the trend clearly.
    rolling_avg = subset["sales"].rolling(window=30, min_periods=1).mean()
    ax.plot(
        subset["date"], rolling_avg,
        color=COLORS["secondary"], linewidth=2.0,
        label="30-day moving average",
    )

    ax.set_title(f"Sales trend — Store {store}, Item {item}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily sales (units)")
    ax.legend(loc="upper left", framealpha=0.9)

    # Format x-axis dates nicely
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    return _save_and_close(fig, f"sales_trend_s{store}_i{item}.png")


def plot_forecast_vs_actual(dates: pd.Series,y_true: np.ndarray,y_pred: np.ndarray,store: int = None,item: int = None,) -> str:
    """
    Overlay predicted vs actual sales.

    Parameters
    ----------
    dates : pd.Series
        Date values for the x-axis.
    y_true : array-like
        Actual sales values.
    y_pred : array-like
        Predicted sales values.
    store : int, optional
        Store ID (for the title).
    item : int, optional
        Item ID (for the title).

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    _setup_style()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)

    # ── Top panel: actual vs predicted ───────────────────────────────
    ax_main = axes[0]
    ax_main.plot(
        dates, y_true,
        color=COLORS["primary"], linewidth=1.2,
        label="Actual", alpha=0.8,
    )
    ax_main.plot(
        dates, y_pred,
        color=COLORS["secondary"], linewidth=1.2,
        label="Predicted", linestyle="--", alpha=0.8,
    )

    # Shade the error region between actual and predicted.
    # This makes forecast errors immediately visible as colored area.
    ax_main.fill_between(
        dates, y_true, y_pred,
        alpha=0.15, color=COLORS["danger"],
        label="Forecast error",
    )

    title = "Forecast vs actual"
    if store is not None and item is not None:
        title += f" — Store {store}, Item {item}"
    ax_main.set_title(title)
    ax_main.set_ylabel("Sales (units)")
    ax_main.legend(loc="upper left", framealpha=0.9)

    # ── Bottom panel: error over time ────────────────────────────────
    # Shows where the model struggles. Spikes in error reveal
    # periods the model can't handle (e.g., holiday spikes,
    # anomalous demand events).
    ax_err = axes[1]
    errors = np.asarray(y_true) - np.asarray(y_pred)
    ax_err.bar(
        dates, errors,
        color=[COLORS["accent"] if e >= 0 else COLORS["danger"] for e in errors],
        alpha=0.6, width=1.0,
    )
    ax_err.axhline(y=0, color=COLORS["neutral"], linewidth=0.8)
    ax_err.set_ylabel("Error")
    ax_err.set_xlabel("Date")

    fig.tight_layout()

    suffix = f"_s{store}_i{item}" if store and item else ""
    return _save_and_close(fig, f"forecast_vs_actual{suffix}.png")


def plot_feature_importance(importances: list[tuple[str, float]],top_n: int = 15,) -> str:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : list of (feature_name, importance_score)
        From model.get_feature_importance(). Should be sorted descending.
    top_n : int
        How many features to display.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    _setup_style()

    # Take top N and reverse for horizontal bar (top feature at top)
    data = importances[:top_n]
    names = [name for name, _ in data][::-1]
    scores = [score for _, score in data][::-1]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    bars = ax.barh(
        names, scores,
        color=COLORS["primary"], alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )

    # Add value labels on each bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}",
            va="center", fontsize=10, color=COLORS["neutral"],
        )

    ax.set_title("Feature importance (gain)")
    ax.set_xlabel("Relative importance")

    fig.tight_layout()
    return _save_and_close(fig, "feature_importance.png")


def plot_demand_distribution(df: pd.DataFrame,store: int,item: int,) -> str:
    """
    Histogram plus box plot showing demand variability for a product.

    Parameters
    ----------
    df : pd.DataFrame
    store : int
    item : int

    Returns
    -------
    str
        Path to saved PNG.
    """
    _setup_style()

    subset = df[(df["store"] == store) & (df["item"] == item)]
    if subset.empty:
        raise ValueError(f"No data for store={store}, item={item}")

    sales = subset["sales"]

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[1, 3],
        sharex=True, gridspec_kw={"hspace": 0.05},
        layout="constrained",
    )

    # Box plot (top)
    bp = ax_box.boxplot(
        sales, vert=False, widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=COLORS["primary"], alpha=0.3),
        medianprops=dict(color=COLORS["danger"], linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax_box.set_yticks([])
    ax_box.set_title(f"Demand distribution — Store {store}, Item {item}")

    # Histogram (bottom)
    ax_hist.hist(
        sales, bins=40, color=COLORS["primary"],
        alpha=0.7, edgecolor="white", linewidth=0.5,
    )

    # Add mean and median lines
    mean_val = sales.mean()
    median_val = sales.median()
    ax_hist.axvline(mean_val, color=COLORS["secondary"], linewidth=2, linestyle="--", label=f"Mean: {mean_val:.1f}")
    ax_hist.axvline(median_val, color=COLORS["danger"], linewidth=2, linestyle="-", label=f"Median: {median_val:.1f}")

    ax_hist.set_xlabel("Daily sales (units)")
    ax_hist.set_ylabel("Frequency (days)")
    ax_hist.legend(loc="upper right", framealpha=0.9)

    # Add stats annotation
    stats_text = (
        f"Std: {sales.std():.1f}\n"
        f"CV: {sales.std() / sales.mean():.2f}\n"
        f"Min: {sales.min()} / Max: {sales.max()}"
    )
    ax_hist.text(
        0.98, 0.95, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top", horizontalalignment="right",
        fontsize=10, color=COLORS["neutral"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    return _save_and_close(fig, f"demand_dist_s{store}_i{item}.png")


def plot_weekly_pattern(df: pd.DataFrame,store: int,item: int,) -> str:
    """
    Bar chart showing average sales by day of week.

    Parameters
    ----------
    df : pd.DataFrame
    store : int
    item : int

    Returns
    -------
    str
        Path to saved PNG.
    """
    _setup_style()

    subset = df[(df["store"] == store) & (df["item"] == item)].copy()
    if subset.empty:
        raise ValueError(f"No data for store={store}, item={item}")

    # Compute mean and std by day of week
    subset["dow"] = subset["date"].dt.dayofweek
    day_stats = subset.groupby("dow")["sales"].agg(["mean", "std"])

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        day_names, day_stats["mean"],
        color=[COLORS["primary"]] * 5 + [COLORS["secondary"]] * 2,
        alpha=0.8, edgecolor="white", linewidth=0.5,
    )

    # Error bars show the std — how much daily demand varies within
    # each weekday. Large error bars mean "Saturday averages 50, but
    # it could easily be 30 or 70."
    ax.errorbar(
        day_names, day_stats["mean"], yerr=day_stats["std"],
        fmt="none", color=COLORS["neutral"], capsize=4, linewidth=1.2,
    )

    # Add value labels on bars
    for bar, mean_val in zip(bars, day_stats["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{mean_val:.1f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title(f"Average sales by day of week — Store {store}, Item {item}")
    ax.set_ylabel("Average daily sales (units)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return _save_and_close(fig, f"weekly_pattern_s{store}_i{item}.png")


def plot_store_comparison(df: pd.DataFrame,item: int,) -> str:
    """
    Compare average demand for the same item across all stores.

    Parameters
    ----------
    df : pd.DataFrame
    item : int

    Returns
    -------
    str
        Path to saved PNG.
    """
    _setup_style()

    subset = df[df["item"] == item]
    if subset.empty:
        raise ValueError(f"No data for item={item}")

    store_stats = subset.groupby("store")["sales"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        store_stats["store"].astype(str),
        store_stats["mean"],
        color=COLORS["primary"], alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )

    ax.errorbar(
        store_stats["store"].astype(str),
        store_stats["mean"],
        yerr=store_stats["std"],
        fmt="none", color=COLORS["neutral"], capsize=4, linewidth=1.2,
    )

    for bar, mean_val in zip(bars, store_stats["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{mean_val:.1f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title(f"Average daily sales by store — Item {item}")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average daily sales (units)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return _save_and_close(fig, f"store_comparison_i{item}.png")


def plot_volatility_ranking(df: pd.DataFrame,top_n: int = 15,) -> str:
    """
    Rank items by demand volatility (coefficient of variation).

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        How many most-volatile items to show.

    Returns
    -------
    str
        Path to saved PNG.
    """
    _setup_style()

    # Compute CV per store-item pair
    item_stats = (
        df.groupby(["store", "item"])["sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    item_stats["cv"] = item_stats["std"] / item_stats["mean"]
    item_stats["label"] = "S" + item_stats["store"].astype(str) + "-I" + item_stats["item"].astype(str)

    # Sort by CV descending and take top N
    top_volatile = item_stats.nlargest(top_n, "cv")

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    colors = [COLORS["danger"] if cv > 0.3 else COLORS["secondary"] if cv > 0.2 else COLORS["accent"]
              for cv in top_volatile["cv"]]

    bars = ax.barh(
        top_volatile["label"][::-1],
        top_volatile["cv"][::-1],
        color=colors[::-1], alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )

    for bar, cv in zip(bars, top_volatile["cv"][::-1]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{cv:.3f}",
            va="center", fontsize=10, color=COLORS["neutral"],
        )

    ax.set_title(f"Top {top_n} most volatile products (by CV)")
    ax.set_xlabel("Coefficient of Variation (std / mean)")

    # Add a legend for the color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["danger"], alpha=0.8, label="High (CV > 0.3)"),
        Patch(facecolor=COLORS["secondary"], alpha=0.8, label="Medium (0.2-0.3)"),
        Patch(facecolor=COLORS["accent"], alpha=0.8, label="Low (CV < 0.2)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    return _save_and_close(fig, "volatility_ranking.png")


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_data
    from src.feature_engineering import build_features
    from src.model import (
        prepare_train_test, train_model, 
        predict, get_feature_importance,
    )
    from src.feature_engineering import get_feature_names

    print("Loading and preparing data...")
    df = load_data()
    df_featured = build_features(df)

    print("\nTraining model...")
    X_train, y_train, X_test, y_test = prepare_train_test(df_featured)
    model = train_model(X_train, y_train)
    preds = predict(model, X_test)

    print("\nGenerating visualizations...")

    path = plot_sales_trend(df, store=1, item=1)
    print(f"  ✓ Sales trend: {path}")

    test_dates = df_featured.loc[X_test.index, "date"]
    path = plot_forecast_vs_actual(test_dates, y_test.values, preds, store=1, item=1)
    print(f"  ✓ Forecast vs actual: {path}")

    feature_names = get_feature_names(df_featured)
    importances = get_feature_importance(model, feature_names)
    path = plot_feature_importance(importances)
    print(f"  ✓ Feature importance: {path}")

    path = plot_demand_distribution(df, store=1, item=1)
    print(f"  ✓ Demand distribution: {path}")

    path = plot_weekly_pattern(df, store=1, item=1)
    print(f"  ✓ Weekly pattern: {path}")

    path = plot_store_comparison(df, item=1)
    print(f"  ✓ Store comparison: {path}")

    path = plot_volatility_ranking(df, top_n=10)
    print(f"  ✓ Volatility ranking: {path}")

    print(f"\nAll charts saved to: {CFG.OUTPUT_DIR}")