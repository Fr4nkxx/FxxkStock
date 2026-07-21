"""Structured anti-bias graph nodes and routing."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from fxxkstock.agents.managers.anti_bias import (
    create_falsification_auditor,
    create_research_manager_revision,
    create_researchability_assessor,
)
from fxxkstock.agents.schemas import (
    FalsificationAudit,
    PortfolioRating,
    ResearchabilityAssessment,
    ResearchPlan,
    render_falsification_audit,
    render_researchability,
)
from fxxkstock.agents.utils.ticker_memory import TickerMemoryStore
from fxxkstock.graph.conditional_logic import ConditionalLogic


def _state():
    return {
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "analysis_mode": "full",
        "market_report": "Market evidence.",
        "sentiment_report": "Sentiment evidence.",
        "news_report": "News evidence.",
        "fundamentals_report": "Fundamental evidence.",
        "investment_plan": "**Recommendation**: Buy",
        "investment_debate_state": {
            "history": "Bull and bear debate.",
            "bull_history": "Bull.",
            "bear_history": "Bear.",
            "current_response": "",
            "judge_decision": "**Recommendation**: Buy",
            "count": 2,
        },
        "researchability_assessment": {
            "markdown": "**Information Grade**: B",
        },
    }


def _llm_with_result(result):
    structured = MagicMock()
    structured.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_researchability_schema_and_node_render():
    assessment = ResearchabilityAssessment(
        information_grade="b",
        source_diversity="medium",
        consensus_risk="high",
        critical_missing_data=["Customer concentration"],
        inferred_claims=["Demand persistence"],
        research_limitations=["Only one channel check"],
        recommended_posture="Use conservative sizing.",
    )
    result = create_researchability_assessor(_llm_with_result(assessment))(_state())
    payload = result["researchability_assessment"]
    assert payload["information_grade"] == "B"
    assert "**信息等级 / Information Grade**: B" in payload["markdown"]
    assert "Customer concentration" in render_researchability(assessment)


@pytest.mark.unit
def test_critical_falsification_finding_forces_single_revision_route():
    audit = FalsificationAudit(
        strongest_counter_thesis="Demand may be pulled forward.",
        critical_findings=["The decisive demand claim has no independent source."],
        requires_revision=False,
        revision_instructions=["Reduce conviction."],
    )
    result = create_falsification_auditor(_llm_with_result(audit))(_state())
    payload = result["falsification_audit"]
    assert payload["requires_revision"] is True
    assert result["initial_investment_plan"] == "**Recommendation**: Buy"
    diagnostics = result["falsification_auditor_diagnostics"]
    assert diagnostics["structured_success"] is True
    assert diagnostics["model_attempts"] == 1
    assert diagnostics["fallback_used"] is False
    assert diagnostics["input_characters"]["prompt"] > 0
    assert (
        ConditionalLogic().should_revise_research(result)
        == "Research Manager Revision"
    )
    assert "Demand may be pulled forward" in render_falsification_audit(audit)


@pytest.mark.unit
def test_unstructured_audit_is_advisory_and_does_not_auto_revise():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")
    llm.invoke.return_value = MagicMock(content="Free-text challenge.")
    result = create_falsification_auditor(llm)(_state())
    assert result["falsification_audit"]["status"] == "unavailable"
    assert result["falsification_audit"]["requires_revision"] is False
    diagnostics = result["falsification_auditor_diagnostics"]
    assert diagnostics["structured_available"] is False
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "structured_output_unavailable"
    assert ConditionalLogic().should_revise_research(result) == "Trader"


@pytest.mark.unit
def test_falsification_diagnostics_record_structured_failure_and_fallback():
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("structured output returned no result")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content="Free-text challenge.")

    result = create_falsification_auditor(llm)(_state())

    diagnostics = result["falsification_auditor_diagnostics"]
    assert diagnostics["structured_attempts"] == 1
    assert diagnostics["fallback_attempts"] == 1
    assert diagnostics["model_attempts"] == 2
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "ValueError"


@pytest.mark.unit
def test_falsification_recovers_structured_audit_from_raw_json():
    structured = MagicMock()
    structured.invoke.return_value = {
        "parsed": None,
        "raw": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FalsificationAudit",
                    "args": {
                        "strongest_counter_thesis": "Demand may reverse.",
                        "conflicting_or_ignored_evidence": [],
                        "hidden_assumptions": [],
                        "bias_flags": [],
                        "falsification_triggers": ["Revenue misses"],
                        "critical_findings": [
                            "A decisive claim lacks support."
                        ],
                        "requires_revision": False,
                        "revision_instructions": ["Reduce conviction."],
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        "parsing_error": ValueError("tool call missing"),
    }
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    result = create_falsification_auditor(llm)(_state())

    assert result["falsification_audit"]["status"] == "available"
    assert result["falsification_audit"]["requires_revision"] is True
    diagnostics = result["falsification_auditor_diagnostics"]
    assert diagnostics["structured_recovered_from_raw"] is True
    assert diagnostics["model_attempts"] == 1
    assert diagnostics["fallback_used"] is False
    llm.invoke.assert_not_called()


@pytest.mark.unit
def test_falsification_reuses_unparsed_raw_response_without_second_call():
    structured = MagicMock()
    structured.invoke.return_value = {
        "parsed": None,
        "raw": AIMessage(content="A useful free-text counter-case."),
        "parsing_error": ValueError("tool call missing; token=secret-value"),
    }
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    result = create_falsification_auditor(llm)(_state())

    assert result["falsification_audit"]["status"] == "unavailable"
    assert result["falsification_audit"]["requires_revision"] is False
    assert "A useful free-text counter-case." in result["falsification_audit"][
        "markdown"
    ]
    diagnostics = result["falsification_auditor_diagnostics"]
    assert diagnostics["model_attempts"] == 1
    assert diagnostics["fallback_attempts"] == 0
    assert diagnostics["fallback_used"] is True
    assert diagnostics["raw_output_reused"] is True
    assert diagnostics["fallback_reason"] == "structured_output_unparsed"
    assert "secret-value" not in diagnostics["fallback_error"]
    llm.invoke.assert_not_called()


@pytest.mark.unit
def test_falsification_json_mode_is_explicit_and_schema_guided():
    audit = FalsificationAudit(
        strongest_counter_thesis="Demand may reverse.",
        requires_revision=False,
    )
    structured = MagicMock()
    structured.invoke.return_value = audit
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    result = create_falsification_auditor(
        llm,
        structured_method="json_mode",
    )(_state())

    llm.with_structured_output.assert_called_once_with(
        FalsificationAudit,
        include_raw=True,
        method="json_mode",
    )
    prompt = structured.invoke.call_args.args[0]
    assert "Return exactly one JSON object" in prompt
    assert '"strongest_counter_thesis"' in prompt
    assert result["falsification_auditor_diagnostics"]["structured_method"] == (
        "json_mode"
    )


@pytest.mark.unit
def test_revision_replaces_plan_and_preserves_initial_plan():
    revised = ResearchPlan(
        recommendation=PortfolioRating.HOLD,
        rationale="Audit invalidated the decisive demand claim.",
        strategic_actions="Wait for independent confirmation.",
    )
    state = _state()
    state["initial_investment_plan"] = state["investment_plan"]
    state["falsification_audit"] = {
        "markdown": "Critical issue.",
        "requires_revision": True,
    }
    result = create_research_manager_revision(_llm_with_result(revised))(state)
    assert "**Recommendation**: Hold" in result["investment_plan"]
    assert result["investment_debate_state"]["judge_decision"] == result["investment_plan"]
    assert result["falsification_audit"]["revision_status"] == "applied"
    assert state["initial_investment_plan"] == "**Recommendation**: Buy"


@pytest.mark.unit
def test_ticker_memory_persists_audit_summary(tmp_path):
    store = TickerMemoryStore({"ticker_memory_dir": str(tmp_path)})
    state = {
        "researchability_assessment": {
            "information_grade": "B",
            "research_limitations": ["Limited channel checks"],
        },
        "falsification_audit": {
            "strongest_counter_thesis": "Demand may reverse.",
            "falsification_triggers": ["Revenue misses"],
        },
        "final_trade_decision": (
            "**Data Confidence**: High\n"
            "**Data Confidence Reason**: Fresh.\n"
            "**Thesis Confidence**: Medium\n"
            "**Thesis Confidence Reason**: Mixed.\n"
            "**Execution Confidence**: Low\n"
            "**Execution Confidence Reason**: No trigger."
        ),
    }
    snapshot = store.update_from_state("NVDA", "2026-07-01", state)
    assert snapshot["anti_bias"]["information_grade"] == "B"
    assert snapshot["anti_bias"]["falsification_triggers"] == ["Revenue misses"]
    assert snapshot["anti_bias"]["confidence"]["execution"]["level"] == "Low"
