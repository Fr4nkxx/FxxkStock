"""Independent bull/bear first-pass arguments with strict prompt isolation."""

from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)


def _shared_evidence(state) -> str:
    return f"""Market report:
{state.get("market_report", "")}

Sentiment report:
{state.get("sentiment_report", "")}

News report:
{state.get("news_report", "")}

Fundamentals report:
{state.get("fundamentals_report", "")}

Evidence ledger (cite E IDs where available):
{(state.get("evidence_ledger") or {}).get("markdown", "")}

Researchability assessment:
{(state.get("researchability_assessment") or {}).get("markdown", "")}"""


def create_blind_bull_researcher(llm):
    def blind_bull_node(state) -> dict:
        prompt = f"""You are the Blind Bull Analyst. Build the strongest
independent investment case from the supplied evidence. You have not seen and
must not speculate about any Bear conclusion. Separate ledger-backed claims
from omissions or new inferences.

{get_instrument_context_from_state(state)}

{_shared_evidence(state)}
""" + get_report_instructions()
        response = llm.invoke(prompt)
        argument = f"Blind Bull Analyst: {response.content}"
        debate = dict(state.get("investment_debate_state") or {})
        debate.update({
            "bull_history": argument,
            "history": argument,
            "current_response": "",
            "count": 0,
        })
        return {
            "blind_bull_argument": argument,
            "investment_debate_state": debate,
        }

    return blind_bull_node


def create_blind_bear_researcher(llm):
    def blind_bear_node(state) -> dict:
        # Deliberately do not read blind_bull_argument or debate history here.
        prompt = f"""You are the Blind Bear Analyst. Build the strongest
independent case against investment from the supplied evidence. You have not
seen and must not speculate about any Bull conclusion. Do not treat missing
information as automatically bearish. Separate ledger-backed claims from
omissions or new inferences.

{get_instrument_context_from_state(state)}

{_shared_evidence(state)}
""" + get_report_instructions()
        response = llm.invoke(prompt)
        argument = f"Blind Bear Analyst: {response.content}"
        debate = dict(state.get("investment_debate_state") or {})
        blind_bull = state.get("blind_bull_argument", "")
        debate.update({
            "bear_history": argument,
            "history": "\n\n".join(item for item in (blind_bull, argument) if item),
            "current_response": "",
            "count": 0,
        })
        return {
            "blind_bear_argument": argument,
            "investment_debate_state": debate,
        }

    return blind_bear_node
