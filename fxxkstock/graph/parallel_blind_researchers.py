"""Concurrent execution for the prompt-isolated Blind Bull/Bear first pass."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

ResearchNode = Callable[[dict[str, Any]], dict[str, Any]]


def _branch_state(state: dict[str, Any]) -> dict[str, Any]:
    branch = dict(state)
    branch["investment_debate_state"] = dict(state.get("investment_debate_state") or {})
    return branch


def _run_researcher(
    key: str,
    label: str,
    node: ResearchNode,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    started_at = time.perf_counter()
    result = node(_branch_state(state))
    return (
        key,
        result,
        {
            "key": key,
            "label": label,
            "duration_seconds": round(time.perf_counter() - started_at, 3),
        },
    )


def create_parallel_blind_researchers_node(
    blind_bull_node: ResearchNode,
    blind_bear_node: ResearchNode,
) -> ResearchNode:
    """Run both independent blind theses and rebuild the serial final state.

    The two prompts intentionally share only pre-debate evidence. The serial
    graph uses ordering solely to append both completed arguments to the debate
    state, so this node performs that merge deterministically after both calls.
    """

    specs = (
        ("blind_bull", "Blind Bull", blind_bull_node),
        ("blind_bear", "Blind Bear", blind_bear_node),
    )

    def parallel_blind_researchers_node(state: dict[str, Any]) -> dict[str, Any]:
        node_started_at = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        timings: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="blind-research",
        ) as executor:
            futures = {
                executor.submit(_run_researcher, key, label, node, state): (key, label)
                for key, label, node in specs
            }
            for future in as_completed(futures):
                key, label = futures[future]
                try:
                    result_key, result, timing = future.result()
                except Exception as exc:  # noqa: BLE001
                    detail = " ".join(str(exc).split()) or exc.__class__.__name__
                    raise RuntimeError(
                        f"Parallel blind researcher failed: {label}: "
                        f"{exc.__class__.__name__}: {detail[:360]}"
                    ) from exc
                results[result_key] = result
                timings[result_key] = timing

        blind_bull = str(results["blind_bull"].get("blind_bull_argument") or "")
        blind_bear = str(results["blind_bear"].get("blind_bear_argument") or "")
        if not blind_bull or not blind_bear:
            missing = "Blind Bull" if not blind_bull else "Blind Bear"
            raise RuntimeError(f"Parallel blind researcher returned no argument: {missing}")

        debate = dict(state.get("investment_debate_state") or {})
        debate.update(
            {
                "bull_history": blind_bull,
                "bear_history": blind_bear,
                "history": f"{blind_bull}\n\n{blind_bear}",
                "current_response": "",
                "count": 0,
            }
        )
        return {
            "sender": "Parallel Blind Researchers",
            "blind_bull_argument": blind_bull,
            "blind_bear_argument": blind_bear,
            "investment_debate_state": debate,
            "parallel_blind_researcher_timings": [timings[key] for key, _, _ in specs],
            "parallel_blind_researchers_total_seconds": round(
                time.perf_counter() - node_started_at,
                3,
            ),
        }

    return parallel_blind_researchers_node
