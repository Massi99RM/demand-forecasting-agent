"""
Agent State
"""

import pandas as pd
from xgboost import XGBRegressor
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """
    Mutable state shared across all agent tools.

    Attributes
    ----------
    raw_df : pd.DataFrame or None
        The raw dataset loaded by data_tools.explore_dataset.
    featured_df : pd.DataFrame or None
        The dataset after feature engineering.
    model : XGBRegressor or None
        The trained forecasting model.
    feature_names : list[str]
        Column names used as model input features.
    is_data_loaded : bool
        Whether data has been loaded.
    is_model_trained : bool
        Whether a model has been trained.
    training_metrics : dict
        Evaluation metrics from the most recent training run.
    """

    raw_df: pd.DataFrame | None = None
    featured_df: pd.DataFrame | None = None
    model: XGBRegressor | None = None
    feature_names: list[str] = field(default_factory=list)
    is_data_loaded: bool = False
    is_model_trained: bool = False
    training_metrics: dict = field(default_factory=dict)

    def reset(self):
        """Clear all state — useful for testing or restarting."""

        default = AgentState()
        for fname in self.__dataclass_fields__:
            setattr(self, fname, getattr(default, fname))


# Global instance — all tools import and use this same object.
# This is safe for the single-user CLI. For a multi-user service,
# state would need to be per-session (e.g., keyed by conversation ID).
state = AgentState()
