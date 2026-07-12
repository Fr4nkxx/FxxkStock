# FxxKStock/graph/setup.py

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from fxxkstock.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_blind_bear_researcher,
    create_blind_bull_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_falsification_auditor,
    create_evidence_ledger_builder,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_research_manager_revision,
    create_researchability_assessor,
    create_sentiment_analyst,
    create_trader,
)
from fxxkstock.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic
from .parallel_analysts import create_parallel_initial_analysts_node


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        *,
        parallel_initial_analysts: bool = False,
        parallel_initial_analyst_workers: int = 4,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.parallel_initial_analysts = parallel_initial_analysts
        self.parallel_initial_analyst_workers = parallel_initial_analyst_workers

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        blind_bull_node = create_blind_bull_researcher(self.quick_thinking_llm)
        blind_bear_node = create_blind_bear_researcher(self.quick_thinking_llm)
        evidence_ledger_node = create_evidence_ledger_builder(
            self.quick_thinking_llm
        )
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        researchability_node = create_researchability_assessor(
            self.quick_thinking_llm
        )
        falsification_node = create_falsification_auditor(self.deep_thinking_llm)
        research_revision_node = create_research_manager_revision(
            self.deep_thinking_llm
        )
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        analyst_nodes = {
            spec.key: analyst_factories[spec.key]()
            for spec in plan.specs
        }

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_nodes[spec.key])
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Blind Bull", blind_bull_node)
        workflow.add_node("Blind Bear", blind_bear_node)
        workflow.add_node("Evidence Ledger Builder", evidence_ledger_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Researchability Assessor", researchability_node)
        workflow.add_node("Falsification Auditor", falsification_node)
        workflow.add_node("Research Manager Revision", research_revision_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        if self.parallel_initial_analysts and len(plan.specs) > 1:
            workflow.add_node(
                "Parallel Initial Analysts",
                create_parallel_initial_analysts_node(
                    plan,
                    analyst_nodes,
                    self.tool_nodes,
                    max_workers=self.parallel_initial_analyst_workers,
                ),
            )
            workflow.add_edge(START, "Parallel Initial Analysts")
            workflow.add_edge("Parallel Initial Analysts", "Evidence Ledger Builder")
        else:
            # Start with the first analyst
            workflow.add_edge(START, plan.specs[0].agent_node)

            # Connect analysts in sequence
            for i, spec in enumerate(plan.specs):
                current_analyst = spec.agent_node
                current_tools = spec.tool_node
                current_clear = spec.clear_node

                # Add conditional edges for current analyst
                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)

                # Connect to next analyst or the pre-debate researchability gate.
                if i < len(plan.specs) - 1:
                    workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
                else:
                    workflow.add_edge(current_clear, "Evidence Ledger Builder")

        # Add remaining edges
        workflow.add_edge("Evidence Ledger Builder", "Researchability Assessor")
        workflow.add_edge("Researchability Assessor", "Blind Bull")
        workflow.add_edge("Blind Bull", "Blind Bear")
        workflow.add_edge("Blind Bear", "Bull Researcher")
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Falsification Auditor")
        workflow.add_conditional_edges(
            "Falsification Auditor",
            self.conditional_logic.should_revise_research,
            {
                "Research Manager Revision": "Research Manager Revision",
                "Trader": "Trader",
            },
        )
        workflow.add_edge("Research Manager Revision", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
