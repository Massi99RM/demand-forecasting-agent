"""
Forecast Tools — train models, make predictions, and explain results.
"""

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_core.tools import tool
from src.agent import state
from src.data_loader import load_data
from src.feature_engineering import build_features, get_feature_names
from src.model import (
    prepare_train_test,
    train_model,
    predict,
    evaluate_model,
    get_feature_importance,
    evaluate_by_item,
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
def train_forecast_model() -> str:
    """Train an XGBoost demand forecasting model on the dataset.

    This runs the full pipeline: loads data (if needed), engineers
    features, splits into train/test by date, trains XGBoost with
    early stopping, and evaluates on the test set.

    Returns training results including: number of training/test samples,
    best iteration (early stopping), and evaluation metrics (MAE, RMSE,
    MAPE) on the test set.

    The trained model is cached — if already trained, returns the
    existing metrics without retraining. Call this before making any
    predictions.

    This tool takes no arguments.
    """
    _ensure_data_loaded()

    # Return cached results if model already trained
    if state.is_model_trained and state.model is not None:
        metrics = state.training_metrics
        return (
            f"Model is already trained (cached from earlier).\n\n"
            f"Test set performance:\n"
            f"  MAE:  {metrics['mae']:.2f} units (avg absolute error)\n"
            f"  RMSE: {metrics['rmse']:.2f} units (penalizes large errors)\n"
            f"  MAPE: {metrics['mape']:.1f}% (avg percentage error)\n\n"
            f"Ready for predictions. Use predict_demand, "
            f"simulate_demand_spike, or get_model_explanation."
        )

    # Prepare train/test split
    X_train, y_train, X_test, y_test = prepare_train_test(state.featured_df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate on test set
    preds = predict(model, X_test)
    metrics = evaluate_model(y_test, preds)

    # Per-item breakdown (top 3 best and worst)
    df_test = state.featured_df[
        state.featured_df["date"] >= CFG.TEST_START_DATE
    ]
    item_eval = evaluate_by_item(df_test, preds, top_n=3)

    # Cache everything in state
    state.model = model
    state.is_model_trained = True
    state.training_metrics = metrics

    # Format results
    result = (
        f"Model trained successfully!\n\n"
        f"Training: {len(X_train):,} samples (up to {CFG.TRAIN_END_DATE})\n"
        f"Test: {len(X_test):,} samples ({CFG.TEST_START_DATE} to end)\n"
        f"Best iteration: {model.best_iteration} / "
        f"{CFG.MODEL_PARAMS['n_estimators']}\n\n"
        f"Test set performance:\n"
        f"  MAE:  {metrics['mae']:.2f} units (avg absolute error)\n"
        f"  RMSE: {metrics['rmse']:.2f} units (penalizes large errors)\n"
        f"  MAPE: {metrics['mape']:.1f}% (avg percentage error)\n\n"
        f"Best forecasted items:\n"
    )

    for item in item_eval["best_items"]:
        result += (
            f"  Store {item['store']}, Item {item['item']}: "
            f"MAE={item['mae']:.2f}, MAPE={item['mape']:.1f}%\n"
        )

    result += f"\nWorst forecasted items:\n"
    for item in item_eval["worst_items"]:
        result += (
            f"  Store {item['store']}, Item {item['item']}: "
            f"MAE={item['mae']:.2f}, MAPE={item['mape']:.1f}%\n"
        )

    result += (
        f"\nModel is ready. You can now use predict_demand to forecast "
        f"specific store-item pairs, or get_model_explanation to see "
        f"which features drive predictions."
    )

    return result


@tool
def predict_demand(store: int, item: int) -> str:
    """Predict future demand for a specific store-item pair.

    Uses the trained XGBoost model to forecast demand for the test
    period (the last 3 months of data). Returns daily predictions
    with dates, actual vs predicted values, and error metrics for
    this specific item.

    The model must be trained first — call train_forecast_model before
    using this tool.

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

    # Filter test data for this store-item pair
    df_test = state.featured_df[
        state.featured_df["date"] >= CFG.TEST_START_DATE
    ]
    mask = (df_test["store"] == store) & (df_test["item"] == item)
    df_item = df_test[mask]

    if df_item.empty:
        return f"Error: No test data for store={store}, item={item}."

    # Predict
    X_item = df_item[state.feature_names]
    y_true = df_item["sales"].values
    y_pred = predict(state.model, X_item)
    dates = df_item["date"].values

    # Compute metrics for this specific item
    metrics = evaluate_model(y_true, y_pred)

    # Format results — show first 14 days and last 7 days of predictions
    result = (
        f"Demand forecast for Store {store}, Item {item}\n\n"
        f"Period: {str(dates[0])[:10]} to {str(dates[-1])[:10]} "
        f"({len(dates)} days)\n\n"
        f"Performance on this item:\n"
        f"  MAE:  {metrics['mae']:.2f} units\n"
        f"  RMSE: {metrics['rmse']:.2f} units\n"
        f"  MAPE: {metrics['mape']:.1f}%\n\n"
        f"Daily predictions (first 14 days):\n"
        f"{'Date':<12} {'Actual':>8} {'Predicted':>10} {'Error':>8}\n"
        f"{'-'*40}\n"
    )

    show_n = min(14, len(dates))
    for i in range(show_n):
        date_str = str(dates[i])[:10]
        error = y_true[i] - y_pred[i]
        result += (
            f"{date_str:<12} {y_true[i]:>8.0f} {y_pred[i]:>10.1f} "
            f"{error:>+8.1f}\n"
        )

    if len(dates) > 14:
        result += f"... ({len(dates) - 14} more days)\n"

    # Summary stats
    result += (
        f"\nSummary:\n"
        f"  Avg actual demand: {y_true.mean():.1f} units/day\n"
        f"  Avg predicted demand: {y_pred.mean():.1f} units/day\n"
        f"  Total actual: {y_true.sum():.0f} units\n"
        f"  Total predicted: {y_pred.sum():.0f} units\n"
    )

    return result


@tool
def get_model_explanation() -> str:
    """Explain what features drive the demand forecasting model.

    Returns the top 15 most important features ranked by gain
    (average loss reduction when the feature is used in a split).
    Higher importance means the feature has more influence on
    predictions.

    The model must be trained first — call train_forecast_model before
    using this tool.

    This tool takes no arguments.
    """
    if not state.is_model_trained:
        return (
            "Error: No trained model available. "
            "Please call train_forecast_model first."
        )

    importances = get_feature_importance(
        state.model, state.feature_names, top_n=15
    )

    result = (
        f"Feature importance (ranked by gain)\n\n"
        f"{'Rank':<6} {'Feature':<30} {'Importance':>12}\n"
        f"{'-'*50}\n"
    )

    for rank, (feat, score) in enumerate(importances, 1):
        bar = "█" * int(score * 40)
        result += f"{rank:<6} {feat:<30} {score:>10.1%}  {bar}\n"

    result += (
        f"\nInterpretation:\n"
        f"- Rolling mean features capture recent demand trends\n"
        f"- Lag features capture periodic patterns (weekly, monthly)\n"
        f"- Time features capture seasonality and calendar effects\n"
        f"- Holiday features capture special event impacts"
    )

    return result