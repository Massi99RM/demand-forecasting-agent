"""
Tools package — agent tool wrappers around the ML pipeline.
"""

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

all_tools = [
    explore_dataset,
    get_item_details,
    train_forecast_model,
    predict_demand,
    get_model_explanation,
    find_volatile_products,
    simulate_demand_spike,
    compare_stores,
    plot_sales_history,
    plot_forecast_chart,
    plot_weekly_pattern_chart,
    plot_volatility_chart,
    plot_feature_importance_chart,
    plot_demand_distribution_chart,
    plot_store_comparison_chart,
]
