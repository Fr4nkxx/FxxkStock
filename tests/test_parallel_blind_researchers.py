from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from fxxkstock.agents.researchers.blind_researchers import (
    create_blind_bear_researcher,
    create_blind_bull_researcher,
)
from fxxkstock.graph.conditional_logic import ConditionalLogic
from fxxkstock.graph.parallel_blind_researchers import (
    create_parallel_blind_researchers_node,
)
from fxxkstock.graph.setup import GraphSetup


def _state() -> dict:
    return {
        "market_report": "market facts",
        "sentiment_report": "sentiment facts",
        "news_report": "news facts",
        "fundamentals_report": "fundamental facts",
        "evidence_ledger": {"markdown": "E1"},
        "researchability_assessment": {"markdown": "grade B"},
        "instrument_context": "Ticker: 600353.SS",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }


class DeterministicBlindLLM:
    def invoke(self, prompt: str) -> AIMessage:
        if "Blind Bull Analyst" in prompt:
            return AIMessage(content="independent bull thesis")
        if "Blind Bear Analyst" in prompt:
            return AIMessage(content="independent bear thesis")
        raise AssertionError("unexpected prompt")


def test_parallel_blind_merge_matches_serial_final_state():
    llm = DeterministicBlindLLM()
    bull_node = create_blind_bull_researcher(llm)
    bear_node = create_blind_bear_researcher(llm)
    initial = _state()

    serial_after_bull = {**initial, **bull_node(initial)}
    serial_final = {**serial_after_bull, **bear_node(serial_after_bull)}
    parallel = create_parallel_blind_researchers_node(bull_node, bear_node)(initial)

    assert parallel["blind_bull_argument"] == serial_final["blind_bull_argument"]
    assert parallel["blind_bear_argument"] == serial_final["blind_bear_argument"]
    assert parallel["investment_debate_state"] == serial_final["investment_debate_state"]
    assert [item["key"] for item in parallel["parallel_blind_researcher_timings"]] == [
        "blind_bull",
        "blind_bear",
    ]


def test_parallel_blind_calls_reach_model_concurrently():
    barrier = threading.Barrier(2)

    class BarrierLLM(DeterministicBlindLLM):
        def invoke(self, prompt: str) -> AIMessage:
            barrier.wait(timeout=1)
            return super().invoke(prompt)

    llm = BarrierLLM()
    node = create_parallel_blind_researchers_node(
        create_blind_bull_researcher(llm),
        create_blind_bear_researcher(llm),
    )

    result = node(_state())

    assert result["sender"] == "Parallel Blind Researchers"
    assert result["parallel_blind_researchers_total_seconds"] >= 0


def test_parallel_blind_error_identifies_failed_branch():
    def bull_node(state):
        raise ValueError("provider concurrency rejected")

    def bear_node(state):
        return {"blind_bear_argument": "Blind Bear Analyst: thesis"}

    node = create_parallel_blind_researchers_node(bull_node, bear_node)

    with pytest.raises(
        RuntimeError,
        match="Parallel blind researcher failed: Blind Bull: ValueError: provider concurrency rejected",
    ):
        node(_state())


def test_graph_setup_keeps_parallel_and_serial_paths_exclusive():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    tool_nodes = {"market": lambda state: state}
    logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    parallel_workflow = GraphSetup(
        llm,
        llm,
        tool_nodes,
        logic,
        parallel_blind_researchers=True,
    ).setup_graph(["market"])
    serial_workflow = GraphSetup(
        llm,
        llm,
        tool_nodes,
        logic,
        parallel_blind_researchers=False,
    ).setup_graph(["market"])

    assert "Parallel Blind Researchers" in parallel_workflow.nodes
    assert "Blind Bull" not in parallel_workflow.nodes
    assert "Blind Bear" not in parallel_workflow.nodes
    assert ("Researchability Assessor", "Parallel Blind Researchers") in (parallel_workflow.edges)
    assert ("Parallel Blind Researchers", "Bull Researcher") in (parallel_workflow.edges)
    assert "Parallel Blind Researchers" not in serial_workflow.nodes
    assert "Blind Bull" in serial_workflow.nodes
    assert "Blind Bear" in serial_workflow.nodes


def test_graph_setup_passes_falsification_structured_method(monkeypatch):
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    falsification_factory = MagicMock(return_value=lambda state: state)
    monkeypatch.setattr(
        "fxxkstock.graph.setup.create_falsification_auditor",
        falsification_factory,
    )

    GraphSetup(
        llm,
        llm,
        {"market": lambda state: state},
        ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
        falsification_structured_method="json_mode",
    ).setup_graph(["market"])

    falsification_factory.assert_called_once_with(
        llm,
        structured_method="json_mode",
    )


def test_graph_setup_rejects_unknown_falsification_structured_method():
    with pytest.raises(ValueError, match="falsification_structured_method"):
        GraphSetup(
            MagicMock(),
            MagicMock(),
            {},
            ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
            falsification_structured_method="unknown",
        )
