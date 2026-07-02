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

from fxxkstock.agents.schemas import PortfolioDecision, render_pm_decision
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.structured import bind_structured
from fxxkstock.agents.utils.position import render_position_context
from fxxkstock.dataflows.market_data_validator import (
    find_current_price_conflicts,
    render_current_market_context,
)

logger = logging.getLogger(__name__)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        def invoke(prompt_value, name):
            if structured_llm is not None:
                try:
                    result = structured_llm.invoke(prompt_value)
                    if result is None:
                        raise ValueError("structured output returned no result")
                    return render_pm_decision(result), result.model_dump(mode="json")
                except Exception as exc:
                    logger.warning("%s structured output failed: %s", name, exc)
            response = llm.invoke(prompt_value)
            return str(getattr(response, "content", response)), {}

        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        position_context = render_position_context(state.get("position_context"))
        researchability = (state.get("researchability_assessment") or {}).get(
            "markdown", ""
        )
        falsification_audit = (state.get("falsification_audit") or {}).get(
            "markdown", ""
        )

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
Use cost basis only for risk management; never anchor the decision on breaking even.
Never describe a market low, technical level, prior-report price, or proposed entry
as the user's cost basis. If you mention the user's cost or P/L, copy only the exact
values in Account Position Context and label them explicitly.
Do not output a specific trade quantity or target portfolio percentage.{get_report_instructions()}"""

        final_trade_decision, decision_metadata = invoke(prompt, "Portfolio Manager")
        snapshot = state.get("current_market_snapshot") or {}
        conflicts = find_current_price_conflicts(final_trade_decision, snapshot)
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
        }

    return portfolio_manager_node
