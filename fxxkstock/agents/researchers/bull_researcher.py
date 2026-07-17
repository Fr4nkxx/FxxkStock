from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.diagnostics import (
    append_stage_replay_context,
    invoke_plain_with_diagnostics,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        researchability = (state.get("researchability_assessment") or {}).get("markdown", "")
        evidence_ledger = (state.get("evidence_ledger") or {}).get("markdown", "")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        prompt = f"""You are a Bull Analyst advocating for investing in the {target_label}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
AI researchability assessment (do not overstate evidence beyond these limits):
{researchability}
Evidence ledger (cite E IDs for ledger claims; label any omitted evidence):
{evidence_ledger}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
""" + get_report_instructions()

        sequence = investment_debate_state["count"] // 2 + 1
        response, diagnostics = invoke_plain_with_diagnostics(
            llm,
            prompt,
            "Bull Researcher",
            input_characters={
                "instrument_context": len(instrument_context),
                "debate_history": len(history),
                "market_report": len(market_research_report),
                "sentiment_report": len(sentiment_report),
                "news_report": len(news_report),
                "fundamentals_report": len(fundamentals_report),
                "researchability": len(researchability),
                "evidence_ledger": len(evidence_ledger),
            },
            sequence=sequence,
        )

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "bull_researcher_diagnostics": diagnostics,
            "stage_replay_contexts": append_stage_replay_context(
                state,
                "bull",
                {"investment_debate_state": investment_debate_state},
            ),
        }

    return bull_node
