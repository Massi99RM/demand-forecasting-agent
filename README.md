# Demand Forecasting Agent

An AI agent that acts as an intelligent supply chain analyst. Given a retail dataset, it autonomously explores data, engineers features, trains forecasting models, and answers natural language questions about demand — combining machine learning fundamentals with modern AI agent architecture.

## The Problem

Supply chain planning depends on accurate demand forecasts. Traditional approaches require analysts to manually load data, run models, interpret metrics, and generate reports — a time-consuming pipeline that doesn't scale. This project builds an AI agent that handles the entire workflow through natural conversation.

## The Solution

Instead of a static forecasting script, this project implements a **reasoning agent** that decides what to do based on the question asked. Ask "which products are most volatile?" and it runs a volatility analysis. Ask "what if demand spikes 30%?" and it simulates the scenario with inventory impact calculations. The agent chains multiple tools together when needed — finding the hardest product to forecast, predicting its demand, and generating a chart, all from a single request.

## Architecture

```
User (natural language question)
            │
            ▼
┌──────────────────────────────┐
│     Agent (LangGraph)        │
│  Claude LLM ←→ Agent State   │
│  Reasoning + tool selection  │
└──────────┬───────────────────┘
           │ tool calls
    ┌──────┼──────────────┐
    ▼      ▼      ▼       ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│ Data ││Fore- ││Analy-││ Viz  │
│Tools ││cast  ││sis   ││Tools │
│      ││Tools ││Tools ││      │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘
   └───────┴───────┴───────┘
           │
    ┌──────┴──────────────┐
    │   ML Pipeline        │
    │  (pure Python)       │
    │                      │
    │  data_loader         │
    │  feature_engineering │
    │  model (XGBoost)     │
    │  visualizations      │
    └──────────────────────┘
```

**Key design decision:** The ML pipeline has zero dependency on LangChain. The `model.py` and `feature_engineering.py` modules are pure Python with pandas/scikit-learn/XGBoost. The tools layer is a thin wrapper that converts these into agent-callable functions. This means the ML code can be tested independently, and if the agent framework changes, only the wrapper needs updating.

## Features

- **Data exploration** — Loads and summarizes 913K rows of retail sales data (10 stores × 50 items × 5 years)
- **Automated feature engineering** — Generates 24 features: time-based (day of week, month, quarter), lag features (1/7/14/28 day), rolling statistics (mean and std over 7/14/30 day windows), and holiday indicators
- **XGBoost forecasting** — Trains with early stopping, time-based train/test split, and per-item evaluation
- **Natural language Q&A** — Ask questions like "predict demand for item 5 in store 1" or "which products are hardest to forecast?"
- **What-if simulation** — Simulate demand spikes and see inventory shortfall impact
- **Store comparison** — Compare demand patterns across locations for inventory allocation
- **Visualization** — Generates sales trends, forecast vs actual charts, weekly patterns, and volatility rankings

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python | Industry standard for ML |
| ML Model | XGBoost | Best performance on structured tabular data with engineered features |
| Data | pandas, NumPy | Standard data manipulation |
| Evaluation | scikit-learn | MAE, RMSE, MAPE metrics |
| Agent Framework | LangGraph | Explicit state management, stable API, replaces deprecated AgentExecutor |
| LLM | Claude (Anthropic) | Strong reasoning and tool-calling capabilities |
| Visualization | matplotlib | Reliable static chart generation |
| Features | holidays | US holiday calendar for demand signals |

## Tech Decisions

**XGBoost over LSTM/Neural Networks:** Tree-based models consistently outperform deep learning on structured tabular data with hand-crafted features. XGBoost trains in seconds, provides interpretable feature importances, and doesn't require normalization or sequence windowing. LSTMs would add significant complexity for marginal benefit on this data type.

**Time-based split over random split:** The train/test split is by date (train up to Sept 2017, test Oct-Dec 2017), not random. Random splitting would leak future information into training data, predicting forward in time.

**Shift inside groupby transform:** Rolling features use `.transform(lambda x: x.rolling(...).mean().shift(1))` with the shift inside the transform. Placing shift outside would cause cross-group leakage — the first row of one product would incorrectly use another product's last rolling value.

**Model caching in agent state:** The trained XGBoost model is cached in memory after the first training call. Subsequent prediction and analysis requests reuse the cached model instantly instead of retraining.

## Dataset

[Store Item Demand Forecasting] from Kaggle.

- 10 stores × 50 items × 1,826 days = 913,000 rows
- Daily sales data from 2013-01-01 to 2017-12-31
- Clean data with zero missing values
- No built-in features (price, promotions) — all 24 features are engineered

## Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | 5.93 | Forecast is off by ~6 units on average |
| RMSE | 7.68 | Typical error with large misses penalized more |
| MAPE | 13.0% | Average percentage error (acceptable for retail) |

Top features by importance: `sales_rolling_mean_7` (34%), `sales_rolling_mean_14` (26%), `sales_lag_7` (25%) — confirming strong weekly seasonality and recent trend dependence.

## Project Structure

```
demand-forecasting-agent/
│
├── main.py                      # Entry point — run the agent
├── config.py                    # All hyperparameters and paths
├── requirements.txt             # Dependencies
├── .env.example                 # API key template
│
├── src/
│   ├── __init__.py              # Package marker
│   ├── data_loader.py           # Load and validate raw data
│   ├── feature_engineering.py   # Feature creation (lags, rolling, holidays)
│   ├── model.py                 # XGBoost training, prediction, evaluation
│   ├── visualizations.py        # Chart generation (7 plot types)
│   │
│   ├── tools/                   # LangChain tool wrappers
│   │   ├── __init__.py          # Tool registry (all_tools list)            
│   │   ├── data_tools.py        # explore_dataset, get_item_details
│   │   ├── forecast_tools.py    # train_forecast_model, predict_demand
│   │   ├── analysis_tools.py    # find_volatile, simulate_spike, compare_stores
│   │   └── viz_tools.py         # Chart generation tools
│   │
│   └── agent/
│       ├── __init__.py          # Shared AgentState (model cache, data cache)
│       ├── graph.py             # LangGraph ReAct agent definition
│       └── prompts.py           # System prompt for the LLM
│
├── tests/                       # Test suites for each module
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_visualizations.py
│   ├── test_tools.py
│   └── test_agent.py
│
├── data/raw/                    # Place train.csv here
└── outputs/                     # Agent-generated charts
```

## How to Run

### Prerequisites
- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com/))
> **Note:** The agent is configured for Claude (Anthropic). The agent architecture is LLM-agnostic — switching to other API's requires only changing the `get_llm()` function in `main.py`and the env file.

### Setup

```bash
# Clone the repository
git clone https://github.com/Massi99RM/demand-forecasting-agent.git
cd demand-forecasting-agent

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Download dataset
# Go to https://www.kaggle.com/c/demand-forecasting-kernels-only/data
# Download train.csv and place it in data/raw/
```

### Run the agent

```bash
python main.py
```

### Run tests (no API key needed)

```bash
python tests/test_data_loader.py
python tests/test_feature_engineering.py
python tests/test_model.py
python tests/test_visualizations.py
python tests/test_tools.py
python tests/test_agent.py
```

## Example Conversation

```
📊 You: What data do we have?
🤖 Agent: Here's what we're working with:
   - 5 years of daily sales data (2013-2017)
   - 10 stores × 50 items = 500 time series
   - 913,000 total records
   - Average daily sales: 52 units per item
   ...

📊 You: Train a forecasting model
🤖 Agent: Model trained successfully!
   - MAE: 5.93 units (forecast off by ~6 units on average)
   - MAPE: 13.0% (acceptable for retail forecasting)
   ...

📊 You: Which products are most volatile?
🤖 Agent: Store 6, Item 5 is the most unpredictable (CV: 0.373).
   Item 5 appears across multiple stores in the top volatile list,
   suggesting inherently unstable demand patterns...

📊 You: What if demand for item 5 spikes 50%?
🤖 Agent: The model can't handle it without intervention.
   - 89% of days would be understocked
   - ~700 unit shortfall over 3 months
   Recommendation: increase safety stock by 60-70%...
```