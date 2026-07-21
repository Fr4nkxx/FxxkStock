"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from fxxkstock.agents.schemas import TraderProposal, render_trader_proposal
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_report_instructions,
)
from fxxkstock.agents.utils.diagnostics import append_stage_replay_context
from fxxkstock.agents.utils.position import render_position_context
from fxxkstock.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        position_context = render_position_context(state.get("position_context"))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    " Apply the supplied account-position semantics exactly, but do not output a "
                    "specific trade quantity or target portfolio percentage. Never describe a "
                    "market low, technical level, prior-report price, or proposed entry as the "
                    "user's cost basis. When discussing the user's cost or P/L, use only the "
                    "explicit Account Position Context values." + get_report_instructions()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Account Position Context:\n{position_context}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        diagnostics = {
            "agent": "Trader",
            "sequence": 1,
            "input_characters": {
                "prompt": sum(len(message["content"]) for message in messages),
                "instrument_context": len(instrument_context),
                "investment_plan": len(investment_plan),
                "position_context": len(position_context),
            },
        }
        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
            diagnostics=diagnostics,
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
            "trader_diagnostics": diagnostics,
            "stage_replay_contexts": append_stage_replay_context(
                state,
                "trader",
                {
                    "investment_plan": investment_plan,
                    "position_context": state.get("position_context") or {},
                },
            ),
        }

    return functools.partial(trader_node, name="Trader")
