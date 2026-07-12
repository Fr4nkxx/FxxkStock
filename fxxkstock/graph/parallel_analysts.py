import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .analyst_execution import AnalystExecutionPlan, AnalystNodeSpec

AgentNode = Callable[[dict[str, Any]], dict[str, Any]]


def create_parallel_initial_analysts_node(
    plan: AnalystExecutionPlan,
    analyst_nodes: dict[str, AgentNode],
    tool_nodes: dict[str, ToolNode],
    *,
    max_workers: int = 4,
    max_tool_rounds: int = 12,
) -> AgentNode:
    """Run independent initial analysts concurrently and merge report fields.

    This is intentionally scoped to the first analyst layer. Each analyst gets
    an isolated copy of the incoming state and produces only its own report, so
    downstream evidence/debate nodes see the same report keys as the serial
    graph while avoiding concurrent writes to shared graph fields such as
    ``sender``.
    """

    worker_count = max(1, min(max_workers, len(plan.specs)))

    def parallel_initial_analysts_node(
        state: dict[str, Any],
        config: RunnableConfig | None = None,
        runtime: Runtime | None = None,
    ) -> dict[str, Any]:
        node_started_at = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_single_analyst,
                    spec,
                    analyst_nodes[spec.key],
                    tool_nodes[spec.key],
                    state,
                    config=config,
                    runtime=runtime,
                    max_tool_rounds=max_tool_rounds,
                ): spec
                for spec in plan.specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results[spec.key] = future.result()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Parallel initial analyst failed: {spec.agent_node}"
                    ) from exc

        merged: dict[str, Any] = {"sender": "Parallel Initial Analysts"}
        merged_messages: list[BaseMessage] = []
        timings: list[dict[str, Any]] = []
        for spec in plan.specs:
            result = results[spec.key]
            merged[spec.report_key] = result.get(spec.report_key, "")
            merged_messages.extend(result.get("messages", []))
            if isinstance(result.get("timing"), dict):
                timings.append(result["timing"])
        if merged_messages:
            merged["messages"] = merged_messages
        if timings:
            merged["parallel_initial_analyst_timings"] = timings
            merged["parallel_initial_analysts_total_seconds"] = round(
                time.perf_counter() - node_started_at,
                3,
            )
        return merged

    return parallel_initial_analysts_node


def _run_single_analyst(
    spec: AnalystNodeSpec,
    analyst_node: AgentNode,
    tool_node: ToolNode,
    state: dict[str, Any],
    *,
    config: RunnableConfig | None,
    runtime: Runtime | None,
    max_tool_rounds: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    branch_state = dict(state)
    branch_state["messages"] = list(state.get("messages") or [])
    emitted_messages: list[BaseMessage] = []
    report = branch_state.get(spec.report_key, "")
    tool_rounds = 0

    for _ in range(max_tool_rounds):
        analyst_update = analyst_node(branch_state)
        analyst_messages = list(analyst_update.get("messages") or [])
        if analyst_messages:
            branch_state["messages"] = branch_state["messages"] + analyst_messages
            emitted_messages.extend(analyst_messages)

        if analyst_update.get(spec.report_key):
            report = analyst_update[spec.report_key]
            break

        last_message = analyst_messages[-1] if analyst_messages else None
        if not getattr(last_message, "tool_calls", None):
            break

        tool_rounds += 1
        tool_update = tool_node.invoke(
            branch_state,
            config=config,
            runtime=runtime or Runtime(),
        )
        tool_messages = list(tool_update.get("messages") or [])
        if tool_messages:
            branch_state["messages"] = branch_state["messages"] + tool_messages
            emitted_messages.extend(tool_messages)
    else:
        raise RuntimeError(
            f"{spec.agent_node} exceeded {max_tool_rounds} tool rounds"
        )

    return {
        "messages": emitted_messages,
        spec.report_key: report,
        "timing": {
            "key": spec.key,
            "label": spec.agent_node,
            "report_key": spec.report_key,
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            "tool_rounds": tool_rounds,
            "message_count": len(emitted_messages),
        },
    }
