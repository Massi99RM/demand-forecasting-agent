"""
Tests for model.py.

These tests verify:
1. Train/test split is strictly time-based (no leakage)
2. Model trains and produces reasonable predictions
3. Evaluation metrics are computed correctly
4. Feature importance extraction works
5. Predictions are non-negative (demand can't be < 0)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CFG
from src.feature_engineering import build_features, get_feature_names
from src.model import (
    prepare_train_test,
    train_model,
    predict,
    evaluate_model,
    get_feature_importance,
    evaluate_by_item,
)


def make_synthetic_featured_df() -> pd.DataFrame:
    """
    Create a synthetic dataset large enough to test the full pipeline.

    2 stores × 2 items × 730 days (2 years) = 2,920 rows.
    After feature engineering drops ~28 NaN rows per series,
    approximately ~2,808 rows — enough for train/test splitting.
    """
    dates = pd.date_range("2016-01-01", "2017-12-31", freq="D")
    np.random.seed(42)
    rows = []

    for store in [1, 2]:
        for item in [1, 2]:
            base = 30 + store * 10 + item * 5
            for date in dates:
                weekday_effect = 8 if date.dayofweek >= 5 else 0
                month_effect = 15 if date.month == 12 else 0
                trend = (date - dates[0]).days * 0.01  # slight uptrend
                noise = np.random.normal(0, 3)
                sales = max(0, int(base + weekday_effect + month_effect + trend + noise))
                rows.append({"date": date, "store": store, "item": item, "sales": sales})

    df = pd.DataFrame(rows)
    return build_features(df)


def test_train_test_split():
    """Verify the time-based split has no overlap and correct sizes."""
    print("\n── Testing prepare_train_test() ──")

    df = make_synthetic_featured_df()
    X_train, y_train, X_test, y_test = prepare_train_test(df)

    # Sizes should be non-zero
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"

    # X and y should be aligned
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # No date overlap: all train dates <= TRAIN_END_DATE
    train_dates = df.loc[X_train.index, "date"]
    test_dates = df.loc[X_test.index, "date"]
    assert train_dates.max() <= pd.Timestamp(CFG.TRAIN_END_DATE), (
        f"Train data extends past {CFG.TRAIN_END_DATE}"
    )
    assert test_dates.min() >= pd.Timestamp(CFG.TEST_START_DATE), (
        f"Test data starts before {CFG.TEST_START_DATE}"
    )

    # No overlap
    assert train_dates.max() < test_dates.min(), "Train and test dates overlap!"

    # Feature columns should NOT include date or sales
    assert "date" not in X_train.columns
    assert "sales" not in X_train.columns
    assert CFG.TARGET not in X_train.columns

    print(f"  ✓ Train: {len(X_train)} rows, ends {train_dates.max().date()}")
    print(f"  ✓ Test: {len(X_test)} rows, starts {test_dates.min().date()}")
    print(f"  ✓ No date overlap between train and test")
    print(f"  ✓ Target column excluded from features")

    return X_train, y_train, X_test, y_test, df


def test_train_model(X_train, y_train):
    """Verify model trains and uses early stopping."""
    print("\n── Testing train_model() ──")

    model = train_model(X_train, y_train)

    # Model should be fitted
    assert hasattr(model, "best_iteration"), "Model doesn't have best_iteration"
    assert model.best_iteration > 0, "Model didn't train any rounds"

    # Early stopping should have kicked in (stopped before 500 rounds)
    # This isn't guaranteed but very likely on this data
    assert model.best_iteration <= 500, "Unexpected: used all 500 rounds"

    print(f"  ✓ Model trained, stopped at iteration {model.best_iteration}")
    print(f"  ✓ Best validation score: {model.best_score:.4f}")

    return model


def test_predict(model, X_test):
    """Verify predictions are reasonable."""
    print("\n── Testing predict() ──")

    preds = predict(model, X_test)

    # Should return an array of same length as input
    assert len(preds) == len(X_test), (
        f"Prediction count {len(preds)} != test count {len(X_test)}"
    )

    # All predictions should be >= 0 (demand can't be negative)
    assert (preds >= 0).all(), "Found negative predictions!"

    # Predictions should be in a reasonable range (not all zeros, not extreme)
    assert preds.mean() > 0, "All predictions are zero"
    assert preds.max() < 10000, f"Suspiciously high prediction: {preds.max()}"

    print(f"  ✓ {len(preds)} predictions generated")
    print(f"  ✓ All predictions >= 0")
    print(f"  ✓ Prediction range: [{preds.min():.1f}, {preds.max():.1f}]")
    print(f"  ✓ Prediction mean: {preds.mean():.1f}")

    return preds


def test_evaluate_model_correctness():
    """Verify metrics are computed correctly with known values."""
    print("\n── Testing evaluate_model() correctness ──")

    # Known values where it's possible to compute metrics by hand:
    # actual:    [10, 20, 30]
    # predicted: [12, 18, 33]
    # errors:    [ 2,  2,  3]
    # MAE = (2 + 2 + 3) / 3 = 2.333...
    # MSE = (4 + 4 + 9) / 3 = 5.666...  → RMSE = 2.380...
    # APE = (2/10 + 2/20 + 3/30) = (0.2 + 0.1 + 0.1) = 0.4
    # MAPE = 0.4 / 3 × 100 = 13.33...%

    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    metrics = evaluate_model(y_true, y_pred)

    assert abs(metrics["mae"] - 2.3333) < 0.01, f"MAE wrong: {metrics['mae']}"
    assert abs(metrics["rmse"] - 2.3805) < 0.01, f"RMSE wrong: {metrics['rmse']}"
    assert abs(metrics["mape"] - 13.33) < 0.1, f"MAPE wrong: {metrics['mape']}"
    assert metrics["n_samples"] == 3

    print(f"  ✓ MAE = {metrics['mae']} (expected ~2.33)")
    print(f"  ✓ RMSE = {metrics['rmse']} (expected ~2.38)")
    print(f"  ✓ MAPE = {metrics['mape']}% (expected ~13.33%)")


def test_evaluate_model_zero_handling():
    """Verify MAPE handles zero actual values gracefully."""
    print("\n── Testing evaluate_model() with zeros ──")

    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([5.0, 12.0, 18.0])

    metrics = evaluate_model(y_true, y_pred)

    # MAPE should only use non-zero rows: |2/10| + |2/20| = 0.2 + 0.1
    # MAPE = 0.3 / 2 × 100 = 15.0%
    assert abs(metrics["mape"] - 15.0) < 0.1, f"MAPE with zeros wrong: {metrics['mape']}"
    assert "mape_note" in metrics, "Should have a note about excluded zeros"

    print(f"  ✓ MAPE = {metrics['mape']}% (zeros excluded correctly)")
    print(f"  ✓ Note: {metrics['mape_note']}")


def test_feature_importance(model, df):
    """Verify feature importance extraction."""
    print("\n── Testing get_feature_importance() ──")

    feature_names = get_feature_names(df)
    importances = get_feature_importance(model, feature_names, top_n=10)

    # Should return a list of (name, score) tuples
    assert len(importances) > 0, "No importances returned"
    assert len(importances) <= 10, "Returned more than top_n"

    # All scores should be positive and sum to <= 1.0 (top_n subset)
    for name, score in importances:
        assert isinstance(name, str)
        assert score > 0, f"Non-positive importance for {name}"

    # Top feature should have the highest score
    scores = [s for _, s in importances]
    assert scores == sorted(scores, reverse=True), "Not sorted by importance"

    # Check that known strong features are present
    feature_names_returned = [name for name, _ in importances]
    # Lag and time features should be important for this synthetic data
    has_temporal = any(
        "lag" in name or "rolling" in name or "day" in name
        for name in feature_names_returned
    )
    assert has_temporal, (
        f"Expected temporal features in top 10, got: {feature_names_returned}"
    )

    print(f"  ✓ {len(importances)} features returned (sorted by importance)")
    print(f"  ✓ Top 3: {importances[:3]}")
    print(f"  ✓ Temporal features present in top features")


def test_evaluate_by_item(df, preds):
    """Verify per-item evaluation breakdown."""
    print("\n── Testing evaluate_by_item() ──")

    df_test = df[df["date"] >= CFG.TEST_START_DATE].copy()

    # Make sure preds aligns with df_test
    assert len(preds) == len(df_test), (
        f"Predictions ({len(preds)}) != test rows ({len(df_test)})"
    )

    result = evaluate_by_item(df_test, preds, top_n=2)

    assert "best_items" in result
    assert "worst_items" in result
    assert len(result["best_items"]) <= 2
    assert len(result["worst_items"]) <= 2
    assert result["total_items_evaluated"] == 4  # 2 stores × 2 items

    # Best items should have lower MAE than worst items
    best_mae = result["best_items"][0]["mae"]
    worst_mae = result["worst_items"][0]["mae"]
    assert best_mae <= worst_mae, "Best item has higher MAE than worst!"

    print(f"  ✓ {result['total_items_evaluated']} items evaluated")
    print(f"  ✓ Best item MAE: {best_mae}")
    print(f"  ✓ Worst item MAE: {worst_mae}")


def test_predict_non_negative():
    """
    Verify predictions are clipped to >= 0 even when model would
    predict negative values.
    """
    print("\n── Testing non-negative prediction clipping ──")

    # Train on data that could cause negative predictions
    # (very low values with high variance)
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 5, n))  # low values

    model = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
    model.fit(X, y)

    # Predict on values that might push predictions negative
    X_extreme = pd.DataFrame({
        "feature1": [-10.0, -20.0, -30.0],
        "feature2": [-10.0, -20.0, -30.0],
    })
    preds = predict(model, X_extreme)

    assert (preds >= 0).all(), f"Found negative predictions: {preds}"
    print(f"  ✓ Predictions clipped to >= 0: {preds}")


if __name__ == "__main__":
    from xgboost import XGBRegressor  # import for the non-negative test

    print("=" * 55)
    print("  TESTS: model.py")
    print("=" * 55)

    # Build synthetic featured data
    print("\n── Preparing synthetic data ──")
    df = make_synthetic_featured_df()
    print(f"  Featured data: {df.shape}")

    # Run tests in pipeline order
    X_train, y_train, X_test, y_test, df = test_train_test_split()
    model = test_train_model(X_train, y_train)
    preds = test_predict(model, X_test)
    test_evaluate_model_correctness()
    test_evaluate_model_zero_handling()
    test_feature_importance(model, df)
    test_evaluate_by_item(df, preds)
    test_predict_non_negative()

    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)