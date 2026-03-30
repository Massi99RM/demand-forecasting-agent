"""
Configuration for the entire project.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """
    Immutable project configuration.
    """

    # ── Paths ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = field(default=None)       # set in __post_init__
    RAW_DATA_PATH: Path = field(default=None)  # set in __post_init__
    OUTPUT_DIR: Path = field(default=None)      # set in __post_init__

    # ── Train / Test Split ───────────────────────────────────────────────
    # The dataset spans 2013-01-01 to 2017-12-31 (5 years).
    # Last 3 months (Q4 2017) used as test data.
    TRAIN_END_DATE: str = "2017-09-30"
    TEST_START_DATE: str = "2017-10-01"

    # ── Forecast Horizon ─────────────────────────────────────────────────
    # How many days ahead the agent can predict when asked.
    # 30 days = ~1 month
    FORECAST_HORIZON: int = 30

    # ── Feature Engineering ──────────────────────────────────────────────
    # LAG_FEATURES: which past days to use as features.
    #   1  = yesterday's sales (captures momentum / short-term trend)
    #   7  = same weekday last week (captures weekly seasonality)
    #   14 = two weeks ago (smooths out one-off weekly spikes)
    #   28 = four weeks ago (captures monthly patterns)
    LAG_FEATURES: tuple = (1, 7, 14, 28)

    # ROLLING_WINDOWS: window sizes for rolling mean and std.
    #   7  = weekly average (smooths daily noise)
    #   14 = biweekly average (smooths weekly noise)
    #   30 = monthly average (captures the underlying trend)
    ROLLING_WINDOWS: tuple = (7, 14, 30)

    # ── Model Hyperparameters ───────────────────────────────────────────
    # n_estimators=500: number of boosting rounds.
    #
    # max_depth=6: how deep each tree can grow.
    #
    # learning_rate=0.05: step size for each boosting round.
    #
    # subsample=0.8: use 80% of rows per tree. Adds randomness to prevent
    #   overfitting.
    #
    # colsample_bytree=0.8: use 80% of features per tree.
    #   Forces trees to learn from different feature subsets.
    #
    # early_stopping_rounds=50: if validation loss doesn't improve for
    #   50 rounds, stop training. Prevents overfitting and saves time.
    MODEL_PARAMS: dict = field(default=None)  # set in __post_init__

    # ── Target Column ────────────────────────────────────────────────────
    TARGET: str = "sales"

    # ── Features to Exclude from Model Input ─────────────────────────────
    # These columns are in the DataFrame but should not be fed to XGBoost:
    # 'date' is used for splitting.
    # 'sales' is the target.
    EXCLUDE_COLS: tuple = ("date", "sales")

    def __post_init__(self):
        object.__setattr__(self, "DATA_DIR", self.PROJECT_ROOT / "data" / "raw")
        object.__setattr__(self, "RAW_DATA_PATH", self.DATA_DIR / "train.csv")
        object.__setattr__(self, "OUTPUT_DIR", self.PROJECT_ROOT / "outputs")
        object.__setattr__(
            self,
            "MODEL_PARAMS",
            {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 50,
                "random_state": 42,
                "n_jobs": -1,           # use all CPU cores
                "verbosity": 0,         # suppress XGBoost warnings
            },
        )

        # Create output directory if it doesn't exist
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Global instance ──────────────────────────────────────────────────────
CFG = Config()