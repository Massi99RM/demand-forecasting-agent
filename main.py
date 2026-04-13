"""
Entry point for the Demand Forecasting Agent.

Run this to start an interactive conversation with the agent

The agent will:
  1. Accept your natural language questions
  2. Reason about which tools to use
  3. Load data, train models, and generate charts as needed
  4. Respond with insights and explanations

REQUIREMENTS:
  - Kaggle dataset at data/raw/train.csv
  - ANTHROPIC_API_KEY environment variable set, must be set in a .env file
  - Dependencies installed: pip install -r requirements.txt
  """

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from src.agent.graph import build_agent, run_agent_query

# Load .env file if it exists
load_dotenv()


def get_llm():
    """
    Initialize the Claude LLM.

    To change the model, swap this function's contents and update the import at the top of this file.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print(
            "\n✗ ANTHROPIC_API_KEY not found.\n\n"
            "Set it in one of these ways:\n"
            "  1. Environment variable: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  2. Create a .env file in the project root with:\n"
            "     ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Get your API key at: https://console.anthropic.com/\n"
        )
        sys.exit(1)

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,  # deterministic for reproducible demos
        max_tokens=4096,
    )


def print_banner():
    """Print a welcome banner."""
    print()
    print("=" * 60)
    print("  🔮 Demand Forecasting Agent")
    print("  An AI-powered supply chain analyst")
    print("=" * 60)
    print()
    print("  Ask me anything about demand forecasting!")
    print("  Examples:")
    print("    • What data do we have?")
    print("    • Train a forecasting model")
    print("    • Predict demand for item 5 in store 1")
    print("    • Which products are most volatile?")
    print("    • What if demand spikes by 30% for item 10?")
    print("    • Show me the sales trend for store 2, item 15")
    print()
    print("  Type 'quit' or 'exit' to stop.")
    print("─" * 60)


def main():
    """Run the interactive agent loop."""
    print_banner()

    # Initialize
    print("  Initializing agent...")
    llm = get_llm()
    agent = build_agent(llm)
    print("  ✓ Agent ready!\n")

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n📊 You: ").strip()

            # Check for exit
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye! 👋\n")
                break

            # Run the agent
            print("\n🤖 Agent: (thinking...)\n")
            response = run_agent_query(agent, user_input)
            print(f"🤖 Agent: {response}")

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n  ✗ Error: {e}\n")
            print("  Try a different question, or type 'quit' to exit.")


if __name__ == "__main__":
    main()
