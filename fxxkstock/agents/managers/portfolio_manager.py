"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from fxxkstock.agents.schemas import PortfolioDecision, render_pm_decision
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from fxxkstock.agents.utils.position import render_position_context
from fxxkstock.dataflows.market_data_validator import (
    find_current_price_conflicts,
    render_current_market_context,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
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
For free-text fallback, include exactly these fields with Low/Medium/High values:
**Data Confidence**, **Data Confidence Reason**, **Thesis Confidence**,
**Thesis Confidence Reason**, **Execution Confidence**, and
**Execution Confidence Reason**.
Use cost basis only for risk management; never anchor the decision on breaking even.
Never describe a market low, technical level, prior-report price, or proposed entry
as the user's cost basis. If you mention the user's cost or P/L, copy only the exact
values in Account Position Context and label them explicitly.
Do not output a specific trade quantity or target portfolio percentage.{get_report_instructions()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )
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
            final_trade_decision = invoke_structured_or_freetext(
                structured_llm,
                llm,
                correction,
                render_pm_decision,
                "Portfolio Manager correction",
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
        }

    return portfolio_manager_node
