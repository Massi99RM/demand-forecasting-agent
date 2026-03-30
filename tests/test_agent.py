"""
Tests for the agent layer.

These tests verify:
1. The system prompt is non-empty and contains key instructions
2. All 12 tools are registered and importable
3. The agent graph builds correctly (without calling the API)
4. main.py imports cleanly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.prompts import SYSTEM_PROMPT
from src.tools import all_tools


def test_system_prompt():
    """Verify the system prompt contains critical instructions."""
    print("\n── Testing system prompt ──")

    assert len(SYSTEM_PROMPT) > 500, (
        f"System prompt too short ({len(SYSTEM_PROMPT)} chars)."
    )

    prompt_lower = SYSTEM_PROMPT.lower()

    assert "never" in prompt_lower and "invent" in prompt_lower, (
        "Prompt must tell the agent to never invent numbers"
    )
    assert "mae" in prompt_lower, "Prompt should explain MAE"
    assert "rmse" in prompt_lower, "Prompt should explain RMSE"
    assert "mape" in prompt_lower, "Prompt should explain MAPE"

    print(f"  ✓ Prompt length: {len(SYSTEM_PROMPT)} chars")
    print(f"  ✓ Contains anti-hallucination instructions")
    print(f"  ✓ Contains metric explanations")


def test_all_tools_registered():
    """Verify all 12 tools are importable and have proper metadata."""
    print("\n── Testing tool registration ──")

    assert len(all_tools) == 12, (
        f"Expected 12 tools, got {len(all_tools)}: {[t.name for t in all_tools]}"
    )

    expected_names = {
        "explore_dataset", "get_item_details",
        "train_forecast_model", "predict_demand", "get_model_explanation",
        "find_volatile_products", "simulate_demand_spike", "compare_stores",
        "plot_sales_history", "plot_forecast_chart",
        "plot_weekly_pattern_chart", "plot_volatility_chart",
    }

    actual_names = {t.name for t in all_tools}
    missing = expected_names - actual_names
    extra = actual_names - expected_names

    assert not missing, f"Missing tools: {missing}"
    assert not extra, f"Unexpected tools: {extra}"

    print(f"  ✓ All 12 tools registered:")
    for t in all_tools:
        assert t.name, f"Tool missing name"
        assert t.description and len(t.description) > 50, (
            f"Tool '{t.name}' description too short for the LLM"
        )
        print(f"    • {t.name}")


def test_agent_graph_builds():
    """
    Verify the agent graph constructs without errors.
    Uses a mock LLM to avoid needing an API key.
    """
    print("\n── Testing agent graph construction ──")

    from unittest.mock import MagicMock
    from src.agent.graph import build_agent

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    try:
        agent = build_agent(mock_llm)
        assert agent is not None, "build_agent returned None"
        print(f"  ✓ Agent graph built successfully")
        print(f"  ✓ Type: {type(agent).__name__}")
    except Exception as e:
        print(f"  ✗ Failed to build agent: {e}")
        raise


def test_main_imports():
    """Verify main.py can be imported without side effects."""
    print("\n── Testing main.py imports ──")

    import main as main_module

    assert hasattr(main_module, "main"), "main.py must have main()"
    assert hasattr(main_module, "get_llm"), "main.py must have get_llm()"
    assert hasattr(main_module, "print_banner"), "main.py must have print_banner()"

    print(f"  ✓ main.py imports cleanly")
    print(f"  ✓ main(), get_llm(), print_banner() all defined")


if __name__ == "__main__":
    print("=" * 55)
    print("  TESTS: Agent Layer (Phase 3)")
    print("=" * 55)

    test_system_prompt()
    test_all_tools_registered()
    test_agent_graph_builds()
    test_main_imports()

    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)