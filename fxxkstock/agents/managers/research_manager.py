"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from fxxkstock.agents.schemas import ResearchPlan, render_research_plan
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]
        researchability = (state.get("researchability_assessment") or {}).get(
            "markdown", ""
        )
        evidence_ledger = (state.get("evidence_ledger") or {}).get("markdown", "")
        blind_bull = state.get("blind_bull_argument", "")
        blind_bear = state.get("blind_bear_argument", "")

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Debate History:**
{history}

**Independent Blind Arguments:**
Bull: {blind_bull}

Bear: {blind_bear}

**Evidence Ledger (cite E IDs; identify omitted evidence explicitly):**
{evidence_ledger}

**AI Researchability Assessment:**
{researchability}

Calibrate certainty to the assessment, but do not treat sparse information as
automatic bearish evidence. Explicitly distinguish conclusions both sides
reached independently from consensus or concessions formed after cross-examination.""" + get_report_instructions()

        diagnostics = {
            "input_characters": {
                "prompt": len(prompt),
                "instrument_context": len(instrument_context),
                "debate_history": len(history),
                "blind_bull": len(blind_bull),
                "blind_bear": len(blind_bear),
                "evidence_ledger": len(evidence_ledger),
                "researchability": len(researchability),
            }
        }
        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
            diagnostics=diagnostics,
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "research_manager_diagnostics": diagnostics,
        }

    return research_manager_node
