# FxxKStock/graph/propagation.py

from typing import Any

from fxxkstock.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)
from fxxkstock.agents.utils.position import build_position_context


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        prior_analysis_context: str = "",
        prior_reports: dict[str, Any] | None = None,
        current_market_snapshot: dict[str, Any] | None = None,
        prior_market_snapshot: dict[str, Any] | None = None,
        analysis_mode: str = "full",
        initial_reports: dict[str, str] | None = None,
        position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``FxxKStockGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.
        """
        reports = initial_reports or {}
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "prior_analysis_context": prior_analysis_context,
            "prior_reports": prior_reports or {},
            "current_market_snapshot": current_market_snapshot or {},
            "prior_market_snapshot": prior_market_snapshot or {},
            "analysis_mode": analysis_mode,
            "position_context": build_position_context(
                position, current_market_snapshot
            ),
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": reports.get("market_report", ""),
            "fundamentals_report": reports.get("fundamentals_report", ""),
            "sentiment_report": reports.get("sentiment_report", ""),
            "news_report": reports.get("news_report", ""),
            "researchability_assessment": {},
            "initial_investment_plan": "",
            "falsification_audit": {},
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
