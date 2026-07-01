"""Structured researchability and falsification nodes for the decision graph."""

from __future__ import annotations

import logging
from typing import Any

from fxxkstock.agents.schemas import (
    FalsificationAudit,
    ResearchabilityAssessment,
    ResearchPlan,
    render_falsification_audit,
    render_research_plan,
    render_researchability,
)
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


def _model_payload(model: Any, markdown: str) -> dict[str, Any]:
    return {
        "status": "available",
        "markdown": markdown,
        **model.model_dump(mode="json"),
    }


def create_researchability_assessor(llm):
    structured_llm = bind_structured(
        llm, ResearchabilityAssessment, "Researchability Assessor"
    )

    def researchability_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        prompt = f"""You are the Researchability Assessor. Before the bull/bear debate,
evaluate how safely an AI system can research this instrument from the evidence
collected in this run. Do not recommend Buy/Sell and do not confuse abundant
coverage with investment certainty.

{instrument}

Asset type: {state.get("asset_type", "stock")}
Analysis mode: {state.get("analysis_mode", "full")}

MARKET REPORT:
{state.get("market_report", "")}

SENTIMENT REPORT:
{state.get("sentiment_report", "")}

NEWS REPORT:
{state.get("news_report", "")}

FUNDAMENTALS REPORT:
{state.get("fundamentals_report", "")}

Grade A when evidence is broad, current, and independently corroborated.
Grade B when meaningful gaps or inference remain. Grade C when decisive facts
are sparse. Explicitly identify homogeneous-source consensus risk."""

        if structured_llm is not None:
            try:
                assessment = structured_llm.invoke(prompt)
                if assessment is None:
                    raise ValueError("structured output returned no result")
                return {
                    "researchability_assessment": _model_payload(
                        assessment, render_researchability(assessment)
                    )
                }
            except Exception as exc:
                logger.warning("Researchability structured output failed: %s", exc)

        response = llm.invoke(
            prompt
            + "\n\nStructured output is unavailable. Write a concise free-text "
            "research limitation assessment without inventing an A/B/C grade."
        )
        text = str(getattr(response, "content", response)).strip()
        markdown = (
            "# 可研究性评估 / Researchability Assessment\n\n"
            "**Information Grade**: Unavailable\n\n"
            "> Structured assessment unavailable; downstream agents must use "
            "the evidence conservatively.\n\n"
            f"{text}"
        )
        return {
            "researchability_assessment": {
                "status": "unavailable",
                "information_grade": None,
                "markdown": markdown,
                "research_limitations": [
                    "Structured researchability assessment was unavailable."
                ],
            }
        }

    return researchability_node


def create_falsification_auditor(llm):
    structured_llm = bind_structured(llm, FalsificationAudit, "Falsification Auditor")

    def falsification_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        researchability = (state.get("researchability_assessment") or {}).get(
            "markdown", ""
        )
        initial_plan = state.get("investment_plan", "")
        prompt = f"""You are an independent Falsification Auditor. Challenge the
Research Manager's initial plan rather than improving its rhetoric. Identify the
strongest counter-case, ignored evidence, hidden assumptions, cognitive biases,
and observable facts that would invalidate the thesis.

Set requires_revision=true only for a material data conflict, an unsupported
decisive claim, ignored strong counterevidence, or an asset/strategy mismatch.
Bias flags alone are not critical. Do not choose the final rating.

{instrument}

RESEARCHABILITY ASSESSMENT:
{researchability}

ANALYST REPORTS:
Market: {state.get("market_report", "")}
Sentiment: {state.get("sentiment_report", "")}
News: {state.get("news_report", "")}
Fundamentals: {state.get("fundamentals_report", "")}

BULL/BEAR DEBATE:
{state.get("investment_debate_state", {}).get("history", "")}

RESEARCH MANAGER INITIAL PLAN:
{initial_plan}"""

        if structured_llm is not None:
            try:
                audit = structured_llm.invoke(prompt)
                if audit is None:
                    raise ValueError("structured output returned no result")
                if audit.critical_findings:
                    audit.requires_revision = True
                return {
                    "initial_investment_plan": initial_plan,
                    "falsification_audit": _model_payload(
                        audit, render_falsification_audit(audit)
                    ),
                }
            except Exception as exc:
                logger.warning("Falsification structured output failed: %s", exc)

        response = llm.invoke(
            prompt
            + "\n\nStructured output is unavailable. Write a concise free-text "
            "audit. It will be advisory and cannot trigger automatic revision."
        )
        text = str(getattr(response, "content", response)).strip()
        return {
            "initial_investment_plan": initial_plan,
            "falsification_audit": {
                "status": "unavailable",
                "requires_revision": False,
                "falsification_triggers": [],
                "markdown": (
                    "# 证伪审计 / Falsification Audit\n\n"
                    "**Requires Revision**: No (structured routing unavailable)\n\n"
                    f"{text}"
                ),
            },
        }

    return falsification_node


def create_research_manager_revision(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager Revision")

    def revision_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        initial_plan = state.get("initial_investment_plan") or state.get(
            "investment_plan", ""
        )
        audit = (state.get("falsification_audit") or {}).get("markdown", "")
        debate = state.get("investment_debate_state", {}).get("history", "")
        prompt = f"""As the Research Manager, revise the initial investment plan
exactly once in response to the independent falsification audit. Correct every
critical issue, preserve supported conclusions, and do not mention drafts or the
revision process. Use the five-tier Buy/Overweight/Hold/Underweight/Sell scale.

{instrument}

DEBATE:
{debate}

INITIAL PLAN:
{initial_plan}

FALSIFICATION AUDIT:
{audit}""" + get_report_instructions()

        revision_status = "applied"
        try:
            if structured_llm is None:
                raise ValueError("structured output unavailable")
            plan = structured_llm.invoke(prompt)
            if plan is None:
                raise ValueError("structured output returned no result")
            revised = render_research_plan(plan)
        except Exception as exc:
            logger.warning(
                "Research Manager revision failed; retaining initial plan: %s", exc
            )
            revised = initial_plan
            revision_status = "failed"

        debate_state = dict(state.get("investment_debate_state") or {})
        debate_state["judge_decision"] = revised
        debate_state["current_response"] = revised
        audit_state = dict(state.get("falsification_audit") or {})
        audit_state["revision_status"] = revision_status
        audit_state["markdown"] = (
            audit_state.get("markdown", "")
            + f"\n\n**Revision Status**: {revision_status.capitalize()}"
        ).strip()
        return {
            "investment_plan": revised,
            "investment_debate_state": debate_state,
            "falsification_audit": audit_state,
        }

    return revision_node
