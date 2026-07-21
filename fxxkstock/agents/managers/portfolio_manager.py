"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import logging
import time
from datetime import date

from langchain_core.output_parsers import PydanticOutputParser

from fxxkstock.agents.schemas import PortfolioDecision, render_pm_decision
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.diagnostics import (
    append_stage_replay_context,
    prompt_characters,
)
from fxxkstock.agents.utils.position import render_position_context
from fxxkstock.agents.utils.structured import bind_structured, summarize_diagnostic_error
from fxxkstock.dataflows.market_data_validator import (
    find_current_price_conflicts,
    render_current_market_context,
)

logger = logging.getLogger(__name__)


def create_portfolio_manager(
    llm,
    *,
    structured_method: str | None = None,
):
    structured_llm = bind_structured(
        llm,
        PortfolioDecision,
        "Portfolio Manager",
        method=structured_method,
    )
    json_mode_instruction = ""
    if structured_method == "json_mode":
        parser = PydanticOutputParser(pydantic_object=PortfolioDecision)
        json_mode_instruction = (
            "\n\nReturn exactly one JSON object matching this schema. Do not "
            "wrap it in a markdown code fence.\n"
            f"{parser.get_format_instructions()}"
        )

    def portfolio_manager_node(state) -> dict:
        diagnostics = {
            "agent": "Portfolio Manager",
            "sequence": 1,
            "input_characters": {"prompt": 0, "total_prompt": 0},
            "structured_available": structured_llm is not None,
            "structured_method": structured_method or "provider_default",
            "structured_attempts": 0,
            "fallback_attempts": 0,
            "fallback_used": False,
            "attempts": [],
        }
        model_duration = 0.0

        def invoke(prompt_value, name):
            nonlocal model_duration
            input_size = prompt_characters(prompt_value)
            if not diagnostics["input_characters"]["prompt"]:
                diagnostics["input_characters"]["prompt"] = input_size
            diagnostics["input_characters"]["total_prompt"] += input_size
            if structured_llm is not None:
                diagnostics["structured_attempts"] += 1
                started_at = time.perf_counter()
                try:
                    result = structured_llm.invoke(prompt_value)
                    if result is None:
                        raise ValueError("structured output returned no result")
                    output = render_pm_decision(result)
                    elapsed = time.perf_counter() - started_at
                    model_duration += elapsed
                    diagnostics["attempts"].append(
                        {
                            "label": name,
                            "transport": "structured",
                            "input_characters": input_size,
                            "output_characters": len(output),
                            "duration_seconds": round(elapsed, 3),
                            "success": True,
                        }
                    )
                    return output, result.model_dump(mode="json")
                except Exception as exc:
                    elapsed = time.perf_counter() - started_at
                    model_duration += elapsed
                    error = summarize_diagnostic_error(exc)
                    diagnostics.update(
                        {
                            "fallback_used": True,
                            "fallback_reason": type(exc).__name__,
                            "fallback_error": error,
                        }
                    )
                    diagnostics["attempts"].append(
                        {
                            "label": name,
                            "transport": "structured",
                            "input_characters": input_size,
                            "output_characters": 0,
                            "duration_seconds": round(elapsed, 3),
                            "success": False,
                            "error": error,
                        }
                    )
                    logger.warning("%s structured output failed: %s", name, exc)
            else:
                diagnostics.setdefault(
                    "fallback_reason",
                    "structured_output_unavailable",
                )
            diagnostics["fallback_attempts"] += 1
            diagnostics["fallback_used"] = True
            started_at = time.perf_counter()
            response = llm.invoke(prompt_value)
            output = str(getattr(response, "content", response))
            elapsed = time.perf_counter() - started_at
            model_duration += elapsed
            diagnostics["attempts"].append(
                {
                    "label": name,
                    "transport": "free_text",
                    "input_characters": input_size,
                    "output_characters": len(output),
                    "duration_seconds": round(elapsed, 3),
                    "success": True,
                }
            )
            return output, {}

        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        position_context = render_position_context(state.get("position_context"))
        researchability = (state.get("researchability_assessment") or {}).get("markdown", "")
        falsification_audit = (state.get("falsification_audit") or {}).get("markdown", "")
        analysis_date = str(state.get("trade_date") or date.today().isoformat())

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Account Position Context (user supplied for this run only):**
{position_context}

**Risk Analysts Debate History:**
{history}

**AI Researchability Assessment:**
{researchability}

**Independent Falsification Audit:**
{falsification_audit}

**Authoritative Calendar Context:**
- Analysis date: {analysis_date}
- Calendar dates must use ISO YYYY-MM-DD and must not contain a written weekday.
- The application computes and displays weekdays from the ISO date.
- Do not invent a date for an event whose date is not supported by the evidence;
  emit it as an event-triggered review node instead.

---

Be decisive and ground every conclusion in specific evidence from the analysts.
Assess data, thesis, and execution confidence independently; one high confidence
dimension must never inflate another.
Return at most three machine-verifiable market-price predictions. Each must use
Above or Below, a positive target price, and a horizon of exactly 5 or 20
trading days. Omit predictions that cannot be verified from closing prices.
For free-text fallback, include exactly these fields with Low/Medium/High values:
**Data Confidence**, **Data Confidence Reason**, **Thesis Confidence**,
**Thesis Confidence Reason**, **Execution Confidence**, and
**Execution Confidence Reason**.
Also include exactly these four action fields: **Next Action** (exactly one of
Buy / Add / Hold / Reduce / Exit / Wait), **Execution Condition**,
**Risk Boundary**, and **Review Trigger**. Each condition must be concise,
observable, and directly supported by the current analysis. For a held position,
use Add / Hold / Reduce / Exit; for a flat or unknown position, use Buy / Wait.
Return one to six Review Nodes. Date nodes require an exact supported ISO date;
event nodes require an observable event and no date. Never put a weekday name in
the node action or event. For free-text fallback, render each node under
"## Review Nodes" as `- [date][review] YYYY-MM-DD: action` or
`- [event][review] event name: action` (execution/risk may replace review).
Every explicit calendar date mentioned anywhere in Execution Condition, Risk
Boundary, or Review Trigger must have a matching date Review Node; do not leave a
deadline embedded only in prose.
Use cost basis only for risk management; never anchor the decision on breaking even.
Never describe a market low, technical level, prior-report price, or proposed entry
as the user's cost basis. If you mention the user's cost or P/L, copy only the exact
values in Account Position Context and label them explicitly.
Do not output a specific trade quantity or target portfolio percentage.{get_report_instructions()}{json_mode_instruction}"""

        final_trade_decision, decision_metadata = invoke(prompt, "Portfolio Manager")
        snapshot = state.get("current_market_snapshot") or {}
        conflicts = find_current_price_conflicts(final_trade_decision, snapshot)
        diagnostics["correction_attempted"] = bool(conflicts)
        if conflicts:
            correction = (
                f"{prompt}\n\n"
                "VALIDATION FAILURE: Your previous draft described "
                f"{conflicts} as the current price, which conflicts with the authoritative "
                "snapshot below. Recalculate every price-dependent conclusion and produce "
                "a fully corrected final decision. Do not mention the rejected draft.\n\n"
                f"{render_current_market_context(snapshot)}"
            )
            final_trade_decision, decision_metadata = invoke(
                correction, "Portfolio Manager correction"
            )
            remaining = find_current_price_conflicts(final_trade_decision, snapshot)
            if remaining:
                raise ValueError(
                    "Portfolio decision rejected: current-price claims "
                    f"{remaining} conflict with verified close {snapshot.get('close')}"
                )

        diagnostics.update(
            {
                "structured_success": not any(
                    attempt["transport"] == "structured" and not attempt["success"]
                    for attempt in diagnostics["attempts"]
                )
                and bool(diagnostics["structured_attempts"]),
                "structured_duration_seconds": round(
                    sum(
                        attempt["duration_seconds"]
                        for attempt in diagnostics["attempts"]
                        if attempt["transport"] == "structured"
                    ),
                    3,
                ),
                "fallback_duration_seconds": round(
                    sum(
                        attempt["duration_seconds"]
                        for attempt in diagnostics["attempts"]
                        if attempt["transport"] == "free_text"
                    ),
                    3,
                ),
                "model_attempts": len(diagnostics["attempts"]),
                "output_characters": len(final_trade_decision),
                "total_model_duration_seconds": round(model_duration, 3),
            }
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "portfolio_decision_metadata": decision_metadata,
            "portfolio_manager_diagnostics": diagnostics,
            "stage_replay_contexts": append_stage_replay_context(
                state,
                "portfolio",
                {"risk_debate_state": risk_debate_state},
            ),
        }

    return portfolio_manager_node
