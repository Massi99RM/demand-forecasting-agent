"""
Tests for the tools layer (Phase 2).

These tests verify that:
1. All tools are callable and return strings
2. Tools auto-load data when needed
3. Tools that require a trained model fail gracefully before training
4. Tools produce meaningful output after proper setup
5. The shared AgentState is correctly updated across tool calls
"""

import sys
import shutil

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Create synthetic data before importing tools
from config import CFG

_BACKUP_PATH = None  # module-level, set during setup


def create_synthetic_data():
    """Create synthetic dataset, backing up any existing real data first."""
    global _BACKUP_PATH

    # If a real dataset already exists, back it up before overwriting
    if CFG.RAW_DATA_PATH.exists():
        _BACKUP_PATH = CFG.RAW_DATA_PATH.with_suffix(".csv.backup")
        shutil.move(str(CFG.RAW_DATA_PATH), str(_BACKUP_PATH))
        print(f"  Backed up existing data to {_BACKUP_PATH}")

    dates = pd.date_range("2016-01-01", "2017-12-31", freq="D")
    np.random.seed(42)
    rows = []
    for store in [1, 2]:
        for item in [1, 2, 3]:
            base = 30 + store * 10 + item * 5
            for date in dates:
                weekday_effect = 8 if date.dayofweek >= 5 else 0
                month_effect = 15 if date.month == 12 else 0
                noise = np.random.normal(0, 3)
                sales = max(0, int(base + weekday_effect + month_effect + noise))
                rows.append(
                    {"date": date, "store": store, "item": item, "sales": sales}
                )
    df = pd.DataFrame(rows)
    CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFG.RAW_DATA_PATH, index=False)
    print(f"  Created synthetic data: {len(df)} rows at {CFG.RAW_DATA_PATH}")


# Set up synthetic data
print("── Setting up test data ──")
create_synthetic_data()

# Import tools
from src.agent import state
from src.tools.data_tools import explore_dataset, get_item_details
from src.tools.forecast_tools import (
    train_forecast_model,
    predict_demand,
    get_model_explanation,
)
from src.tools.analysis_tools import (
    find_volatile_products,
    simulate_demand_spike,
    compare_stores,
)
from src.tools.viz_tools import (
    plot_sales_history,
    plot_forecast_chart,
    plot_weekly_pattern_chart,
    plot_volatility_chart,
    plot_feature_importance_chart,
    plot_demand_distribution_chart,
    plot_store_comparison_chart,
)


def restore_real_data():
    """Delete synthetic data and restore the user's real data if it existed."""
    if CFG.RAW_DATA_PATH.exists():
        CFG.RAW_DATA_PATH.unlink()

    if _BACKUP_PATH is not None and _BACKUP_PATH.exists():
        shutil.move(str(_BACKUP_PATH), str(CFG.RAW_DATA_PATH))
        print(f"  Restored original data from backup")
    else:
        print(f"  Cleaned up synthetic data (no backup to restore)")


def _ensure_data_loaded_for_test():
    """Ensure data is loaded."""
    if not state.is_data_loaded:
        explore_dataset.invoke({})


def _ensure_model_trained_for_test():
    """Ensure model is trained."""
    _ensure_data_loaded_for_test()
    if not state.is_model_trained:
        train_forecast_model.invoke({})


def test_explore_dataset():
    print("\n── Testing explore_dataset ──")
    state.reset()

    result = explore_dataset.invoke({})

    assert isinstance(result, str), "Tool should return a string"
    assert "loaded successfully" in result.lower(), f"Unexpected result: {result[:100]}"
    assert state.is_data_loaded, "State should show data as loaded"
    assert state.raw_df is not None, "raw_df should be populated"
    assert state.featured_df is not None, "featured_df should be populated"
    assert len(state.feature_names) > 0, "feature_names should be populated"

    print(f"  ✓ Returns string with dataset summary")
    print(f"  ✓ State updated: is_data_loaded={state.is_data_loaded}")
    print(f"  ✓ Features: {len(state.feature_names)}")


def test_get_item_details():
    print("\n── Testing get_item_details ──")

    result = get_item_details.invoke({"store": 1, "item": 1})

    assert isinstance(result, str)
    assert "store 1" in result.lower()
    assert "trend" in result.lower()

    # Test invalid item
    result_bad = get_item_details.invoke({"store": 99, "item": 99})
    assert "error" in result_bad.lower(), "Should return error for invalid item"

    print(f"  ✓ Returns detailed item summary")
    print(f"  ✓ Invalid store-item returns error message")


def test_predict_before_training():
    print("\n── Testing predict_demand BEFORE training ──")
    state.reset()
    # Load data but don't train
    explore_dataset.invoke({})

    result = predict_demand.invoke({"store": 1, "item": 1})
    assert "error" in result.lower(), "Should fail without trained model"
    assert "train_forecast_model" in result, "Should suggest training first"

    print(f"  ✓ Correctly refuses to predict without a model")


def test_explanation_before_training():
    print("\n── Testing get_model_explanation BEFORE training ──")

    result = get_model_explanation.invoke({})
    assert "error" in result.lower()

    print(f"  ✓ Correctly refuses without a model")


def test_train_model():
    print("\n── Testing train_forecast_model ──")

    result = train_forecast_model.invoke({})

    assert isinstance(result, str)
    assert "trained successfully" in result.lower(), f"Unexpected: {result[:100]}"
    assert state.is_model_trained, "State should show model as trained"
    assert state.model is not None, "Model should be cached"
    assert len(state.training_metrics) > 0, "Metrics should be stored"
    assert "mae" in result.lower()
    assert "rmse" in result.lower()
    assert "mape" in result.lower()

    print(f"  ✓ Model trained and cached in state")
    print(f"  ✓ Metrics: {state.training_metrics}")
    print(f"  ✓ Output includes MAE, RMSE, MAPE")


def test_predict_after_training():
    print("\n── Testing predict_demand AFTER training ──")
    _ensure_model_trained_for_test()

    result = predict_demand.invoke({"store": 1, "item": 1})

    assert "forecast" in result.lower() or "demand" in result.lower()
    assert "actual" in result.lower()
    assert "predicted" in result.lower()
    # Make sure it's not an error response
    assert not result.startswith("Error"), f"Got error: {result[:200]}"

    print(f"  ✓ Predictions generated successfully")
    print(f"  ✓ Output includes actual vs predicted comparison")


def test_model_explanation():
    print("\n── Testing get_model_explanation ──")

    result = get_model_explanation.invoke({})

    assert "error" not in result.lower()
    assert "importance" in result.lower()
    assert "lag" in result.lower() or "rolling" in result.lower()

    print(f"  ✓ Feature importances returned")


def test_find_volatile():
    print("\n── Testing find_volatile_products ──")

    result = find_volatile_products.invoke({"top_n": 5})

    assert isinstance(result, str)
    assert "volatile" in result.lower()
    assert "cv" in result.lower()

    print(f"  ✓ Volatility ranking returned")


def test_simulate_spike():
    print("\n── Testing simulate_demand_spike ──")

    result = simulate_demand_spike.invoke(
        {
            "store": 1,
            "item": 1,
            "spike_pct": 20.0,
        }
    )

    assert not result.startswith("Error"), f"Got error: {result[:200]}"
    assert "spike" in result.lower() or "simulation" in result.lower()
    assert "shortfall" in result.lower()

    print(f"  ✓ Demand spike simulation completed")


def test_compare_stores():
    print("\n── Testing compare_stores ──")

    result = compare_stores.invoke({"item": 1})

    assert isinstance(result, str)
    assert "store" in result.lower()
    assert "avg" in result.lower() or "mean" in result.lower()

    print(f"  ✓ Store comparison returned")


def test_viz_sales():
    print("\n── Testing plot_sales_history ──")

    result = plot_sales_history.invoke({"store": 1, "item": 1})
    assert "saved to" in result.lower() or "generated" in result.lower()

    print(f"  ✓ Sales trend chart generated")


def test_viz_forecast():
    print("\n── Testing plot_forecast_chart ──")

    result = plot_forecast_chart.invoke({"store": 1, "item": 1})
    assert "saved to" in result.lower() or "generated" in result.lower()
    assert not result.startswith("Error"), f"Got error: {result[:200]}"

    print(f"  ✓ Forecast chart generated (single item, not spaghetti)")


def test_viz_weekly():
    print("\n── Testing plot_weekly_pattern_chart ──")

    result = plot_weekly_pattern_chart.invoke({"store": 1, "item": 1})
    assert "saved to" in result.lower() or "generated" in result.lower()

    print(f"  ✓ Weekly pattern chart generated")


def test_viz_volatility():
    print("\n── Testing plot_volatility_chart ──")

    result = plot_volatility_chart.invoke({"top_n": 5})
    assert "saved to" in result.lower() or "generated" in result.lower()

    print(f"  ✓ Volatility ranking chart generated")


def test_viz_feature_importance():
    print("\n── Testing plot_feature_importance_chart ──")
    _ensure_model_trained_for_test()  # needs a trained model

    result = plot_feature_importance_chart.invoke({"top_n": 10})
    assert "saved to" in result.lower() or "generated" in result.lower()
    assert not result.startswith("Error"), f"Got error: {result[:200]}"

    print(f"  ✓ Feature importance chart generated")


def test_viz_feature_importance_before_training():
    print("\n── Testing plot_feature_importance_chart BEFORE training ──")
    state.reset()
    explore_dataset.invoke({})  # data loaded but no model

    result = plot_feature_importance_chart.invoke({"top_n": 10})
    assert "error" in result.lower(), "Should fail without trained model"
    assert "train_forecast_model" in result

    print(f"  ✓ Correctly refuses without a trained model")


def test_viz_demand_distribution():
    print("\n── Testing plot_demand_distribution_chart ──")
    _ensure_data_loaded_for_test()

    result = plot_demand_distribution_chart.invoke({"store": 1, "item": 1})
    assert "saved to" in result.lower() or "generated" in result.lower()
    assert not result.startswith("Error"), f"Got error: {result[:200]}"

    # Test invalid input
    result_bad = plot_demand_distribution_chart.invoke({"store": 99, "item": 99})
    assert "error" in result_bad.lower()

    print(f"  ✓ Demand distribution chart generated")
    print(f"  ✓ Invalid store-item returns error")


def test_viz_store_comparison():
    print("\n── Testing plot_store_comparison_chart ──")
    _ensure_data_loaded_for_test()

    result = plot_store_comparison_chart.invoke({"item": 1})
    assert "saved to" in result.lower() or "generated" in result.lower()
    assert not result.startswith("Error"), f"Got error: {result[:200]}"

    # Test invalid input
    result_bad = plot_store_comparison_chart.invoke({"item": 999})
    assert "error" in result_bad.lower()

    print(f"  ✓ Store comparison chart generated")
    print(f"  ✓ Invalid item returns error")


if __name__ == "__main__":
    import glob

    print("=" * 55)
    print("  TESTS: Tools Layer (Phase 2)")
    print("=" * 55)

    try:
        test_explore_dataset()
        test_get_item_details()
        test_predict_before_training()
        test_explanation_before_training()
        test_train_model()
        test_predict_after_training()
        test_model_explanation()
        test_find_volatile()
        test_simulate_spike()
        test_compare_stores()
        test_viz_sales()
        test_viz_forecast()
        test_viz_weekly()
        test_viz_volatility()
        test_viz_feature_importance_before_training()
        test_viz_feature_importance()
        test_viz_demand_distribution()
        test_viz_store_comparison()

        pngs = glob.glob(str(CFG.OUTPUT_DIR / "*.png"))
        for f in pngs:
            Path(f).unlink()
        print(f"\n  Cleaned up {len(pngs)} chart files")

    finally:
        restore_real_data()

    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)
