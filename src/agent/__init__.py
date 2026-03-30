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
    feature_names: list = field(default_factory=list)
    is_data_loaded: bool = False
    is_model_trained: bool = False
    training_metrics: dict = field(default_factory=dict)

    def reset(self):
        """Clear all state — useful for testing or restarting."""
        self.raw_df = None
        self.featured_df = None
        self.model = None
        self.feature_names = []
        self.is_data_loaded = False
        self.is_model_trained = False
        self.training_metrics = {}


# Global instance — all tools import and use this same object.
state = AgentState()