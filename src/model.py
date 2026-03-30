"""
Model — train, evaluate, predict, and explain demand forecasts.
"""
import sys
from pathlib import Path
 
# Ensure the project root is on Python's path 
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from config import CFG
from src.feature_engineering import get_feature_names


def prepare_train_test(df: pd.DataFrame,train_end: str = None,test_start: str = None,) -> tuple:
    """
    Split featured data into train and test sets by date.

    Parameters
    ----------
    df : pd.DataFrame
        Featured DataFrame from build_features().
    train_end : str, optional
        Last date (inclusive) in training set. Format: "YYYY-MM-DD".
    test_start : str, optional
        First date (inclusive) in test set. Format: "YYYY-MM-DD".

    Returns
    -------
    tuple of (X_train, y_train, X_test, y_test)
        X = feature matrices (pd.DataFrame)
        y = target vectors (pd.Series)
    """
    if train_end is None:
        train_end = CFG.TRAIN_END_DATE
    if test_start is None:
        test_start = CFG.TEST_START_DATE

    feature_cols = get_feature_names(df)

    # ── Split by date ────────────────────────────────────────────────
    train_mask = df["date"] <= train_end
    test_mask = df["date"] >= test_start

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, CFG.TARGET]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, CFG.TARGET]

    # ── Sanity checks ────────────────────────────────────────────────
    if len(X_train) == 0:
        raise ValueError(
            f"Training set is empty! Check TRAIN_END_DATE ({train_end}). "
            f"Data date range: {df['date'].min()} to {df['date'].max()}"
        )
    if len(X_test) == 0:
        raise ValueError(
            f"Test set is empty! Check TEST_START_DATE ({test_start}). "
            f"Data date range: {df['date'].min()} to {df['date'].max()}"
        )

    # Verify no date overlap (the gap between train_end and test_start
    # can be zero or positive, but never negative)
    train_max = df.loc[train_mask, "date"].max()
    test_min = df.loc[test_mask, "date"].min()
    if train_max >= test_min:
        raise ValueError(
            f"Train/test overlap detected! Train ends {train_max}, "
            f"test starts {test_min}. Adjust dates in config.py."
        )

    print(f"  Train: {len(X_train):,} rows ({df.loc[train_mask, 'date'].min().date()} to {train_max.date()})")
    print(f"  Test:  {len(X_test):,} rows ({test_min.date()} to {df.loc[test_mask, 'date'].max().date()})")
    print(f"  Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test


def train_model(X_train: pd.DataFrame,y_train: pd.Series,X_val: pd.DataFrame = None,y_val: pd.Series = None,params: dict = None,) -> XGBRegressor:
    """
    Train an XGBoost regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame, optional
        Validation features for early stopping. If None, uses last 20%
        of training data (by row order, which is chronological).
    y_val : pd.Series, optional
        Validation target.
    params : dict, optional
        XGBoost hyperparameters. Defaults to CFG.MODEL_PARAMS.

    Returns
    -------
    XGBRegressor
        Trained model.
    """
    if params is None:
        params = CFG.MODEL_PARAMS.copy()

    # Extract early_stopping_rounds
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    # ── Create validation set if not provided ────────────────────────
    if X_val is None or y_val is None:
        # Take the last 20% of training data as validation.
        val_size = int(len(X_train) * 0.2)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        print(f"  Auto-split: {len(X_train_fit):,} train, {len(X_val):,} validation")
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    # ── Train ────────────────────────────────────────────────────────
    model = XGBRegressor(
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )

    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,  # suppress per-round output
    )

    # Report training outcome
    # best_iteration says when early stopping kicked in.
    # best_score is the validation metric at that iteration.
    best_iteration = model.best_iteration
    best_score = model.best_score
    n_estimators = params.get("n_estimators", 500)
    print(f"  Best iteration: {best_iteration} / {n_estimators}")
    print(f"  Best validation RMSE: {best_score:.4f}")

    return model


def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions from a trained model.

    Parameters
    ----------
    model : XGBRegressor
        Trained model from train_model().
    X : pd.DataFrame
        Feature matrix (same columns as training data).

    Returns
    -------
    np.ndarray
        Predicted sales values, clipped to >= 0.
    """
    preds = model.predict(X)

    # Demand can't be negative.
    preds = np.clip(preds, 0, None)

    return preds


def evaluate_model(y_true: np.ndarray | pd.Series,y_pred: np.ndarray,) -> dict:
    """
    Compute evaluation metrics for demand forecasting.

    Parameters
    ----------
    y_true : array-like
        Actual sales values.
    y_pred : array-like
        Predicted sales values.

    Returns
    -------
    dict
        Keys: mae, rmse, mape (as percentage), n_samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # ── MAPE: handle zero actual values ──────────────────────────────
    nonzero_mask = y_true != 0
    n_zeros = (~nonzero_mask).sum()

    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = float("inf")  # all zeros — MAPE is meaningless

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 2),
        "n_samples": len(y_true),
    }

    if n_zeros > 0:
        metrics["mape_note"] = (
            f"MAPE excludes {n_zeros} zero-demand samples "
            f"({n_zeros / len(y_true) * 100:.1f}% of test set)"
        )

    return metrics


def get_feature_importance(model: XGBRegressor,feature_names: list[str],top_n: int = 15,) -> list[tuple[str, float]]:
    """
    Extract and rank feature importances from the trained model.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    feature_names : list of str
        Feature column names (in the same order as training data).
    top_n : int
        How many top features to return.

    Returns
    -------
    list of (feature_name, importance_score)
        Sorted by importance descending. Scores are normalized to sum
        to 1.0 for easier interpretation.
    """
    # 'gain' = average loss reduction per split using this feature
    raw_importance = model.get_booster().get_score(importance_type="gain")

    # XGBoost uses internal feature names (f0, f1, ...) if the model
    # wasn't trained with feature names. Map them to the used column names.
    importance = {}
    for key, value in raw_importance.items():
        if key.startswith("f"):
            idx = int(key[1:])
            if idx < len(feature_names):
                importance[feature_names[idx]] = value
        else:
            importance[key] = value

    # Normalize so scores sum to 1.0
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    # Sort descending and take top N
    sorted_importance = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    return sorted_importance


def evaluate_by_item(df_test: pd.DataFrame,y_pred: np.ndarray,top_n: int = 5,) -> dict:
    """
    Break down model performance by individual store-item pairs.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test set with 'store', 'item', 'sales' columns.
    y_pred : np.ndarray
        Model predictions aligned with df_test rows.
    top_n : int
        How many best/worst items to return.

    Returns
    -------
    dict
        Keys: best_items, worst_items — each a list of dicts with
        store, item, mae, rmse, mape, n_samples.
    """
    df_eval = df_test[["store", "item", "sales"]].copy()
    df_eval["prediction"] = y_pred
    df_eval["abs_error"] = np.abs(df_eval["sales"] - df_eval["prediction"])

    # Compute per-item metrics
    item_metrics = []
    for (store, item), group in df_eval.groupby(["store", "item"]):
        y_t = group["sales"].values
        y_p = group["prediction"].values
        metrics = evaluate_model(y_t, y_p)
        metrics["store"] = int(store)
        metrics["item"] = int(item)
        item_metrics.append(metrics)

    # Sort by MAE
    item_metrics.sort(key=lambda x: x["mae"])

    return {
        "best_items": item_metrics[:top_n],
        "worst_items": item_metrics[-top_n:][::-1],  # worst first
        "total_items_evaluated": len(item_metrics),
    }


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_data
    from src.feature_engineering import build_features

    print("Loading and featuring data...")
    df = load_data()
    df = build_features(df)

    print("\nPreparing train/test split...")
    X_train, y_train, X_test, y_test = prepare_train_test(df)

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating on test set...")
    preds = predict(model, X_test)
    metrics = evaluate_model(y_test, preds)
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    print("\nTop 10 feature importances:")
    feature_names = get_feature_names(df)
    importances = get_feature_importance(model, feature_names, top_n=10)
    for feat, score in importances:
        bar = "█" * int(score * 50)
        print(f"  {feat:30s} {score:.4f} {bar}")

    print("\nPer-item evaluation (top 3 best, top 3 worst):")
    df_test = df[df["date"] >= CFG.TEST_START_DATE]
    item_eval = evaluate_by_item(df_test, preds, top_n=3)
    print("  Best forecasted items:")
    for item in item_eval["best_items"]:
        print(f"    Store {item['store']}, Item {item['item']}: MAE={item['mae']}, MAPE={item['mape']}%")
    print("  Worst forecasted items:")
    for item in item_eval["worst_items"]:
        print(f"    Store {item['store']}, Item {item['item']}: MAE={item['mae']}, MAPE={item['mape']}%")