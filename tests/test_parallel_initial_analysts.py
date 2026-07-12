from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from fxxkstock.graph.analyst_execution import build_analyst_execution_plan
from fxxkstock.graph.parallel_analysts import (
    create_parallel_initial_analysts_node,
)


@tool
def market_lookup() -> str:
    """Return fixture market data."""
    return "market_lookup result"


@tool
def news_lookup() -> str:
    """Return fixture news data."""
    return "news_lookup result"


def _analyst_with_one_tool_round(label: str, report_key: str):
    def analyst_node(state):
        if any(isinstance(message, ToolMessage) for message in state["messages"]):
            report = f"{label} report"
            return {
                "messages": [AIMessage(content=report)],
                report_key: report,
            }

        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"call-{label}",
                            "name": f"{label}_lookup",
                            "args": {},
                        }
                    ],
                )
            ],
            report_key: "",
        }

    return analyst_node


def test_parallel_initial_analysts_runs_tool_loops_and_merges_reports():
    plan = build_analyst_execution_plan(["market", "news"])

    node = create_parallel_initial_analysts_node(
        plan,
        {
            "market": _analyst_with_one_tool_round("market", "market_report"),
            "news": _analyst_with_one_tool_round("news", "news_report"),
        },
        {
            "market": ToolNode([market_lookup]),
            "news": ToolNode([news_lookup]),
        },
        max_workers=2,
        max_tool_rounds=3,
    )

    result = node(
        {"messages": [HumanMessage(content="002364.SZ")]},
        config={"configurable": {"thread_id": "test-run"}},
        runtime=Runtime(),
    )

    assert result["sender"] == "Parallel Initial Analysts"
    assert result["market_report"] == "market report"
    assert result["news_report"] == "news report"
    assert result["parallel_initial_analysts_total_seconds"] >= 0
    timings = {
        item["key"]: item for item in result["parallel_initial_analyst_timings"]
    }
    assert timings["market"]["label"] == "Market Analyst"
    assert timings["market"]["tool_rounds"] == 1
    assert timings["market"]["duration_seconds"] >= 0
    assert timings["news"]["label"] == "News Analyst"
    assert timings["news"]["tool_rounds"] == 1
    assert timings["news"]["duration_seconds"] >= 0
    assert [message.content for message in result["messages"] if message.content] == [
        "market_lookup result",
        "market report",
        "news_lookup result",
        "news report",
    ]
