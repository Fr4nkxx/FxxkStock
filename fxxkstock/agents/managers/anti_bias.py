"""Structured researchability and falsification nodes for the decision graph."""

from __future__ import annotations

import logging
from typing import Any

from fxxkstock.agents.schemas import (
    EvidenceLedger,
    FalsificationAudit,
    ResearchabilityAssessment,
    ResearchPlan,
    render_falsification_audit,
    normalize_evidence_ledger,
    render_evidence_ledger,
    render_research_plan,
    render_researchability,
)
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
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


def create_evidence_ledger_builder(llm):
    structured_llm = bind_structured(llm, EvidenceLedger, "Evidence Ledger Builder")

    def evidence_ledger_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        prompt = f"""You are the Evidence Ledger Builder. Extract at most 20
decisive, non-duplicative claims from the four analyst reports. Preserve source
references and dates. Distinguish observed facts, calculations, inference and
opinion. A claim is supported only when it has verifiable references and at
least two genuinely independent sources; syndicated copies count as one.
Record material counter-evidence. Use provisional claim IDs; the application
will assign stable E01..E20 IDs.

{instrument}

MARKET REPORT:
{state.get("market_report", "")}

SENTIMENT REPORT:
{state.get("sentiment_report", "")}

NEWS REPORT:
{state.get("news_report", "")}

FUNDAMENTALS REPORT:
{state.get("fundamentals_report", "")}

Write all human-readable claim text, counter-evidence, and missing-evidence
descriptions in the configured report language. Keep only enum values and
claim_id in their schema-defined form.{get_language_instruction()}"""
        try:
            if structured_llm is None:
                raise ValueError("structured output unavailable")
            ledger = structured_llm.invoke(prompt)
            if ledger is None:
                raise ValueError("structured output returned no result")
            ledger = normalize_evidence_ledger(ledger)
            return {
                "evidence_ledger": _model_payload(
                    ledger, render_evidence_ledger(ledger)
                )
            }
        except Exception as exc:
            logger.warning("Evidence ledger structured output failed: %s", exc)
            return {
                "evidence_ledger": {
                    "status": "unavailable",
                    "claims": [],
                    "omitted_or_missing_evidence": [
                        "Structured evidence extraction was unavailable."
                    ],
                    "markdown": (
                        "# 证据账本 / Evidence Ledger\n\n"
                        "> 结构化证据账本不可用。下游 Agent 必须直接使用分析师"
                        "报告，且不得虚构 claim_id。"
                    ),
                }
            }

    return evidence_ledger_node


def create_researchability_assessor(llm):
    structured_llm = bind_structured(
        llm, ResearchabilityAssessment, "Researchability Assessor"
    )

    def researchability_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        evidence_ledger = (state.get("evidence_ledger") or {}).get("markdown", "")
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

EVIDENCE LEDGER:
{evidence_ledger}

Grade A when evidence is broad, current, and independently corroborated.
Grade B when meaningful gaps or inference remain. Grade C when decisive facts
are sparse. Explicitly identify homogeneous-source consensus risk.
Write every explanatory text field in the configured report language; keep
only the A/B/C and Low/Medium/High enum values unchanged.
{get_language_instruction()}"""

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
            "**信息等级**: 未评估\n\n"
            "> 结构化评估不可用；下游 Agent 必须保守使用现有证据。\n\n"
            f"{text}"
        )
        return {
            "researchability_assessment": {
                "status": "unavailable",
                "information_grade": None,
                "markdown": markdown,
                "research_limitations": [
                    "结构化可研究性评估不可用。"
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
        evidence_ledger = (state.get("evidence_ledger") or {}).get("markdown", "")
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

EVIDENCE LEDGER (cite claim_id for ledger claims; identify omissions explicitly):
{evidence_ledger}

ANALYST REPORTS:
Market: {state.get("market_report", "")}
Sentiment: {state.get("sentiment_report", "")}
News: {state.get("news_report", "")}
Fundamentals: {state.get("fundamentals_report", "")}

BULL/BEAR DEBATE:
{state.get("investment_debate_state", {}).get("history", "")}

RESEARCH MANAGER INITIAL PLAN:
{initial_plan}

Write every explanatory audit field in the configured report language. Keep
only boolean and schema enum values in their required machine-readable form.
{get_language_instruction()}"""

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
                    "**是否需要修正**: 否（结构化路由不可用）\n\n"
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
