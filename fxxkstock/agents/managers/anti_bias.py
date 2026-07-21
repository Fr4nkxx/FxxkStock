"""Structured researchability and falsification nodes for the decision graph."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser

from fxxkstock.agents.schemas import (
    EvidenceLedger,
    FalsificationAudit,
    ResearchabilityAssessment,
    ResearchPlan,
    normalize_evidence_ledger,
    render_evidence_ledger,
    render_falsification_audit,
    render_research_plan,
    render_researchability,
)
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_report_instructions,
)
from fxxkstock.agents.utils.diagnostics import append_stage_replay_context
from fxxkstock.agents.utils.structured import (
    bind_structured,
    extract_response_text,
    summarize_diagnostic_error,
)

logger = logging.getLogger(__name__)


def _model_payload(model: Any, markdown: str) -> dict[str, Any]:
    return {
        "status": "available",
        "markdown": markdown,
        **model.model_dump(mode="json"),
    }


def _tool_arguments_text(response: Any) -> str:
    """Extract model-supplied tool arguments when message content is empty."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        additional = getattr(response, "additional_kwargs", None) or {}
        tool_calls = additional.get("tool_calls") or []

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("args")
        if arguments is None:
            function = call.get("function") or {}
            if isinstance(function, dict):
                arguments = function.get("arguments")
        if isinstance(arguments, dict) and arguments:
            return json.dumps(arguments, ensure_ascii=False, indent=2)
        if isinstance(arguments, str) and arguments.strip():
            return arguments.strip()
    return ""


def _advisory_falsification_payload(text: str) -> dict[str, Any]:
    """Render a non-structured audit that cannot trigger automatic revision."""
    return {
        "status": "unavailable",
        "requires_revision": False,
        "falsification_triggers": [],
        "markdown": (
            "# 证伪审计 / Falsification Audit\n\n"
            "**是否需要修正 / Requires Revision**: 否（结构化结果不可用）\n\n"
            f"{text}"
        ),
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
        diagnostics: dict[str, Any] = {
            "agent": "Evidence Ledger Builder",
            "sequence": 1,
            "input_characters": {
                "prompt": len(prompt),
                "instrument_context": len(instrument),
                "market_report": len(state.get("market_report", "")),
                "sentiment_report": len(state.get("sentiment_report", "")),
                "news_report": len(state.get("news_report", "")),
                "fundamentals_report": len(state.get("fundamentals_report", "")),
            },
            "structured_available": structured_llm is not None,
            "structured_attempts": 0,
            "fallback_attempts": 0,
            "fallback_used": False,
        }
        started_at = time.perf_counter()
        try:
            if structured_llm is None:
                raise ValueError("structured output unavailable")
            diagnostics["structured_attempts"] = 1
            ledger = structured_llm.invoke(prompt)
            if ledger is None:
                raise ValueError("structured output returned no result")
            ledger = normalize_evidence_ledger(ledger)
            markdown = render_evidence_ledger(ledger)
            elapsed = round(time.perf_counter() - started_at, 3)
            diagnostics.update(
                {
                    "structured_success": True,
                    "structured_duration_seconds": elapsed,
                    "model_attempts": 1,
                    "output_characters": len(markdown),
                    "total_model_duration_seconds": elapsed,
                }
                )
            return {
                "evidence_ledger": _model_payload(ledger, markdown),
                "evidence_ledger_builder_diagnostics": diagnostics,
                "stage_replay_contexts": append_stage_replay_context(
                    state,
                    "evidence",
                    {"instrument_context": instrument},
                ),
            }
        except Exception as exc:
            logger.warning("Evidence ledger structured output failed: %s", exc)
            elapsed = round(time.perf_counter() - started_at, 3)
            diagnostics.update(
                {
                    "structured_success": False,
                    "structured_duration_seconds": elapsed,
                    "model_attempts": int(diagnostics["structured_attempts"]),
                    "output_characters": 0,
                    "total_model_duration_seconds": elapsed,
                    "model_error": summarize_diagnostic_error(exc),
                }
            )
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
                },
                "evidence_ledger_builder_diagnostics": diagnostics,
                "stage_replay_contexts": append_stage_replay_context(
                    state,
                    "evidence",
                    {"instrument_context": instrument},
                ),
            }

    return evidence_ledger_node


def create_researchability_assessor(llm):
    structured_llm = bind_structured(llm, ResearchabilityAssessment, "Researchability Assessor")

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
            prompt + "\n\nStructured output is unavailable. Write a concise free-text "
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
                "research_limitations": ["结构化可研究性评估不可用。"],
            }
        }

    return researchability_node


def create_falsification_auditor(
    llm,
    *,
    structured_method: str | None = None,
):
    raw_parser = PydanticOutputParser(pydantic_object=FalsificationAudit)
    structured_llm = bind_structured(
        llm,
        FalsificationAudit,
        "Falsification Auditor",
        include_raw=True,
        method=structured_method,
    )
    json_mode_instruction = ""
    if structured_method == "json_mode":
        json_mode_instruction = (
            "\n\nReturn exactly one JSON object matching this schema. Do not "
            "wrap it in a markdown code fence.\n"
            f"{raw_parser.get_format_instructions()}"
        )

    def falsification_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        researchability = (state.get("researchability_assessment") or {}).get("markdown", "")
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
{get_language_instruction()}{json_mode_instruction}"""

        diagnostics: dict[str, Any] = {
            "agent": "Falsification Auditor",
            "input_characters": {
                "prompt": len(prompt),
                "instrument_context": len(instrument),
                "researchability": len(researchability),
                "evidence_ledger": len(evidence_ledger),
                "market_report": len(state.get("market_report", "")),
                "sentiment_report": len(state.get("sentiment_report", "")),
                "news_report": len(state.get("news_report", "")),
                "fundamentals_report": len(state.get("fundamentals_report", "")),
                "debate_history": len(state.get("investment_debate_state", {}).get("history", "")),
                "initial_plan": len(initial_plan),
            },
            "structured_available": structured_llm is not None,
            "structured_method": structured_method or "provider_default",
            "structured_attempts": 0,
            "fallback_attempts": 0,
            "fallback_used": False,
        }
        model_started_at = time.perf_counter()

        if structured_llm is not None:
            structured_started_at = time.perf_counter()
            diagnostics["structured_attempts"] = 1
            try:
                result = structured_llm.invoke(prompt)
                raw_text = ""
                parsing_error: object | None = None
                if isinstance(result, dict) and {
                    "parsed",
                    "raw",
                    "parsing_error",
                }.intersection(result):
                    audit = result.get("parsed")
                    raw_message = result.get("raw")
                    raw_text = extract_response_text(raw_message) or _tool_arguments_text(
                        raw_message
                    )
                    parsing_error = result.get("parsing_error")
                    diagnostics.update(
                        {
                            "raw_response_available": raw_message is not None,
                            "raw_content_characters": len(raw_text),
                            "raw_output_reused": False,
                        }
                    )
                else:
                    audit = result

                if isinstance(audit, dict):
                    try:
                        audit = FalsificationAudit.model_validate(audit)
                    except Exception as exc:
                        parsing_error = parsing_error or exc
                        audit = None
                if audit is None and raw_text:
                    try:
                        audit = raw_parser.parse(raw_text)
                        diagnostics["structured_recovered_from_raw"] = True
                    except Exception as exc:
                        parsing_error = parsing_error or exc
                        elapsed = round(time.perf_counter() - structured_started_at, 3)
                        diagnostics.update(
                            {
                                "structured_success": False,
                                "structured_duration_seconds": elapsed,
                                "fallback_reason": "structured_output_unparsed",
                                "fallback_error": summarize_diagnostic_error(parsing_error),
                                "fallback_used": True,
                                "fallback_attempts": 0,
                                "fallback_duration_seconds": 0.0,
                                "raw_output_reused": True,
                                "model_attempts": 1,
                                "output_characters": len(raw_text),
                                "total_model_duration_seconds": round(
                                    time.perf_counter() - model_started_at,
                                    3,
                                ),
                            }
                        )
                        logger.warning(
                            "Falsification structured output was not parsed; "
                            "reusing the first response as an advisory audit: %s",
                            summarize_diagnostic_error(parsing_error),
                        )
                        return {
                            "initial_investment_plan": initial_plan,
                            "falsification_audit": _advisory_falsification_payload(raw_text),
                            "falsification_auditor_diagnostics": diagnostics,
                        }
                if audit is None:
                    raise ValueError("structured output returned no parsed or raw result")
                if not isinstance(audit, FalsificationAudit):
                    audit = FalsificationAudit.model_validate(audit)
                if audit.critical_findings:
                    audit.requires_revision = True
                markdown = render_falsification_audit(audit)
                diagnostics.update(
                    {
                        "structured_success": True,
                        "structured_duration_seconds": round(
                            time.perf_counter() - structured_started_at,
                            3,
                        ),
                        "model_attempts": 1,
                        "output_characters": len(markdown),
                        "total_model_duration_seconds": round(
                            time.perf_counter() - model_started_at,
                            3,
                        ),
                    }
                )
                return {
                    "initial_investment_plan": initial_plan,
                    "falsification_audit": _model_payload(audit, markdown),
                    "falsification_auditor_diagnostics": diagnostics,
                }
            except Exception as exc:
                diagnostics.update(
                    {
                        "structured_success": False,
                        "structured_duration_seconds": round(
                            time.perf_counter() - structured_started_at,
                            3,
                        ),
                        "fallback_reason": type(exc).__name__,
                        "fallback_error": summarize_diagnostic_error(exc),
                    }
                )
                logger.warning(
                    "Falsification structured output failed: %s",
                    summarize_diagnostic_error(exc),
                )
        else:
            diagnostics.update(
                {
                    "structured_success": False,
                    "structured_duration_seconds": 0.0,
                    "fallback_reason": "structured_output_unavailable",
                }
            )

        fallback_started_at = time.perf_counter()
        diagnostics["fallback_attempts"] = 1
        diagnostics["fallback_used"] = True
        response = llm.invoke(
            prompt + "\n\nStructured output is unavailable. Write a concise free-text "
            "audit. It will be advisory and cannot trigger automatic revision."
        )
        text = extract_response_text(response)
        diagnostics.update(
            {
                "fallback_duration_seconds": round(
                    time.perf_counter() - fallback_started_at,
                    3,
                ),
                "model_attempts": int(diagnostics["structured_attempts"]) + 1,
                "output_characters": len(text),
                "total_model_duration_seconds": round(
                    time.perf_counter() - model_started_at,
                    3,
                ),
            }
        )
        return {
            "initial_investment_plan": initial_plan,
            "falsification_audit": _advisory_falsification_payload(text),
            "falsification_auditor_diagnostics": diagnostics,
        }

    return falsification_node


def create_research_manager_revision(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager Revision")

    def revision_node(state) -> dict:
        instrument = get_instrument_context_from_state(state)
        initial_plan = state.get("initial_investment_plan") or state.get("investment_plan", "")
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
            logger.warning("Research Manager revision failed; retaining initial plan: %s", exc)
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
