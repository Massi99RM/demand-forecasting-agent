"""
Agent Graph — the LangGraph reasoning loop.

The loop:
  User message → agent (thinks) → calls a tool → tools (executes)
  → agent (reads result, thinks again) → calls another tool OR
  → responds to the user
"""

import sys
from pathlib import Path

# Ensure the project root is on Python's path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langgraph.prebuilt import create_react_agent
from src.tools import all_tools
from src.agent.prompts import SYSTEM_PROMPT


def build_agent(llm):
    """
    Build the LangGraph ReAct agent.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM to use for reasoning.

    Returns
    -------
    CompiledGraph
        A LangGraph agent ready to invoke with messages.
    """
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run_agent_query(agent, query: str) -> str:
    """
    Send a single query to the agent and return the final response.

    Parameters
    ----------
    agent : CompiledGraph
        The agent from build_agent().
    query : str
        The user's question in natural language.

    Returns
    -------
    str
        The agent's final text response.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    # Extract the last AI message from the conversation
    messages = result["messages"]

    # Walk backwards to find the last AI text response
    for msg in reversed(messages):
        # LangGraph messages can be dicts or message objects
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if msg.type == "ai" and isinstance(msg.content, str) and msg.content.strip():
                return msg.content
        elif isinstance(msg, dict):
            if msg.get("role") == "assistant" and msg.get("content", "").strip():
                return msg["content"]

    return "No response generated."