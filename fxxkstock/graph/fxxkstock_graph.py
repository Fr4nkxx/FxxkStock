# FxxKStock/graph/fxxkstock_graph.py

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from fxxkstock.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from fxxkstock.agents.utils.memory import TradingMemoryLog
from fxxkstock.agents.utils.calibration import CalibrationStore
from fxxkstock.agents.utils.ticker_memory import TickerMemoryStore
from fxxkstock.dataflows.config import get_config, set_config
from fxxkstock.dataflows.market_data_validator import (
    build_current_market_snapshot_data,
)
from fxxkstock.dataflows.utils import safe_ticker_component
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.llm_clients import create_llm_client
from fxxkstock.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)


class FxxKStockGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.callbacks = callbacks or []
        self.selected_analysts = tuple(selected_analysts)

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)
        self.ticker_memory = TickerMemoryStore(self.config)
        self.calibration_store = CalibrationStore(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            parallel_initial_analysts=bool(
                self.config.get("parallel_initial_analysts", False)
            ),
            parallel_initial_analyst_workers=int(
                self.config.get("parallel_initial_analyst_workers", 4)
            ),
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(self.selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None
        self._prepared_run = None
        self._chrome_manager = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a FXXKSTOCK_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Deterministic verification snapshot (bound to the analyst
                    # LLM and required by its prompt; must be executable here or
                    # the call fails and the model reports it "unavailable").
                    get_verified_market_snapshot,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_macro_indicators,
                    get_prediction_markets,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        from fxxkstock.dataflows.symbol_utils import normalize_symbol

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            stock = yf.Ticker(normalize_symbol(ticker)).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def _fetch_calibration_outcome(
        self, ticker: str, trade_date: str, horizon: int, benchmark: str
    ) -> dict[str, Any] | None:
        """Return an exact-horizon close and returns, or None until it is available."""
        from fxxkstock.dataflows.symbol_utils import normalize_symbol

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=horizon * 2 + 10)
            stock = yf.Ticker(normalize_symbol(ticker)).history(
                start=trade_date, end=end.strftime("%Y-%m-%d")
            )
            bench = yf.Ticker(benchmark).history(
                start=trade_date, end=end.strftime("%Y-%m-%d")
            )
            if len(stock) <= horizon:
                return None
            initial = float(stock["Close"].iloc[0])
            actual = float(stock["Close"].iloc[horizon])
            raw = actual / initial - 1
            alpha = None
            benchmark_return = None
            if len(bench) > horizon:
                benchmark_return = float(
                    bench["Close"].iloc[horizon] / bench["Close"].iloc[0] - 1
                )
                alpha = raw - benchmark_return
            return {
                "actual_price": actual,
                "raw_return": raw,
                "benchmark_return": benchmark_return,
                "alpha_return": alpha,
                "actual_holding_days": horizon,
            }
        except Exception as exc:
            logger.warning("Calibration outcome unavailable for %s: %s", ticker, exc)
            return None

    def resolve_instrument_context(
        self,
        ticker: str,
        asset_type: str = "stock",
        trade_date: str | None = None,
    ) -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). CN-region tickers get
        authoritative Chinese names from domestic sources; non-CNY quote
        currencies get an FX line for CNY reporting.
        """
        from fxxkstock.dataflows.currency_utils import detect_source_currency, get_fx_to_cny
        from fxxkstock.dataflows.market_utils import (
            detect_market_region,
            get_security_cn_name,
            is_cn_region,
        )

        identity = dict(resolve_instrument_identity(ticker))
        market_region = detect_market_region(ticker, identity)

        if is_cn_region(market_region):
            cn_name = get_security_cn_name(ticker, market_region)
            if cn_name:
                identity["company_name"] = cn_name

        source_ccy = detect_source_currency(ticker, identity, market_region)
        if source_ccy:
            identity["currency"] = source_ccy

        if trade_date and source_ccy and source_ccy != "CNY":
            fx_rate = get_fx_to_cny(source_ccy, trade_date)
            if fx_rate is not None:
                identity["fx_rate"] = f"{fx_rate:.4f}"
                identity["fx_as_of"] = trade_date

        return build_instrument_context(ticker, asset_type, identity)

    def inject_market_region(self, company_name: str) -> str:
        """Detect CN/US market region for *company_name* and update global config.

        Both ``propagate()`` (via ``_run_graph``) and the interactive CLI must
        call this before analysts run so vendor routing and CN prefetch paths
        see the correct ``market_region``.
        """
        from fxxkstock.dataflows.market_utils import detect_market_region

        identity = resolve_instrument_identity(company_name)
        market_region = detect_market_region(company_name, identity)
        set_config({"market_region": market_region})
        self.config = get_config()
        return market_region

    def prepare_run(
        self,
        company_name: str,
        trade_date,
        asset_type: str = "stock",
        analysis_mode: str = "auto",
        browser_status_callback=None,
        position: dict | None = None,
    ) -> dict[str, Any]:
        """Load ticker memory, select refresh modules, and build initial state."""
        if analysis_mode not in {"auto", "full"}:
            raise ValueError("analysis_mode must be 'auto' or 'full'")

        self.ticker = company_name
        self._resolve_pending_entries(company_name)
        self.calibration_store.resolve_pending(
            company_name, self._fetch_calibration_outcome
        )
        snapshot = self.ticker_memory.load(company_name)
        reuse_fundamentals = (
            analysis_mode == "auto"
            and "fundamentals" in self.selected_analysts
            and self.ticker_memory.fundamentals_fresh(snapshot, str(trade_date))
        )
        active_analysts = tuple(
            key for key in self.selected_analysts
            if not (key == "fundamentals" and reuse_fundamentals)
        )
        if not active_analysts:
            active_analysts = ("market",)

        self.workflow = self.graph_setup.setup_graph(active_analysts)
        self.graph = self.workflow.compile()

        reports = (snapshot or {}).get("reports", {})
        initial_reports = {}
        if reuse_fundamentals:
            initial_reports["fundamentals_report"] = reports.get("fundamentals_report", "")

        market_region = self.inject_market_region(company_name)
        browser_status = None
        if (
            self.config.get("cn_browser_enabled", True)
            and self.config.get("cn_browser_auto_start", True)
        ):
            from fxxkstock.dataflows.chrome_manager import ChromeManager
            from fxxkstock.dataflows.market_utils import is_cn_region

            if is_cn_region(market_region):
                manager = ChromeManager(self.config)
                self._chrome_manager = manager
                if not manager.is_cdp_available():
                    starting = {
                        **manager.status(),
                        "state": "starting",
                        "message": f"Starting Google Chrome for {manager.platform}",
                    }
                    if browser_status_callback:
                        browser_status_callback(starting)
                browser_status = manager.ensure_running()
                if browser_status_callback:
                    browser_status_callback(browser_status)
                if browser_status["state"] == "failed_fallback":
                    logger.warning(
                        "Chrome auto-start failed; browser vendors will fall back to HTTP: %s",
                        browser_status.get("message", "unknown error"),
                    )
        try:
            current_market_snapshot = build_current_market_snapshot_data(
                company_name,
                str(trade_date),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Current market snapshot unavailable for %s: %s", company_name, exc)
            current_market_snapshot = {
                "ticker": company_name.upper(),
                "requested_date": str(trade_date),
                "error": str(exc),
            }

        initial_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=self.memory_log.get_past_context(company_name),
            instrument_context=self.resolve_instrument_context(
                company_name, asset_type, trade_date=str(trade_date)
            ),
            prior_analysis_context=self.ticker_memory.prior_context(snapshot),
            prior_reports=reports,
            current_market_snapshot=current_market_snapshot,
            prior_market_snapshot=(snapshot or {}).get("market_snapshot") or {},
            analysis_mode="incremental" if reuse_fundamentals else "full",
            initial_reports=initial_reports,
            position=position,
        )
        prepared = {
            "snapshot": snapshot,
            "initial_state": initial_state,
            "active_analysts": list(active_analysts),
            "reuse": ["fundamentals"] if reuse_fundamentals else [],
            "refresh": list(active_analysts),
            "analysis_mode": "incremental" if reuse_fundamentals else "full",
            "browser_status": browser_status,
        }
        self._prepared_run = prepared
        return prepared

    def close_managed_browser(self) -> dict[str, Any] | None:
        """Release and optionally close Chrome started for this graph run."""
        manager = self._chrome_manager
        self._chrome_manager = None
        if manager is None or not self.config.get("cn_browser_auto_close", True):
            return None
        return manager.close_managed()

    def finalize_run(
        self,
        company_name: str,
        trade_date,
        final_state: dict[str, Any],
        *,
        log_state: bool = True,
    ) -> dict[str, Any]:
        """Persist successful run state to both decision and ticker memory."""
        self.curr_state = final_state
        if log_state:
            self._log_state(trade_date, final_state)
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=str(trade_date),
            final_trade_decision=final_state["final_trade_decision"],
        )
        previous = (self._prepared_run or {}).get("snapshot")
        snapshot = self.ticker_memory.update_from_state(
            company_name,
            str(trade_date),
            final_state,
            previous,
            refreshed=(self._prepared_run or {}).get("refresh"),
        )
        decision = final_state.get("portfolio_decision_metadata") or {}
        if not decision:
            markdown = final_state.get("final_trade_decision", "")
            field = lambda label: (
                re.search(
                    rf"\*\*{re.escape(label)}\*\*:\s*([^\n]+)",
                    markdown,
                    re.IGNORECASE,
                )
            )
            rating_match = field("Rating")
            if rating_match:
                decision = {
                    "rating": rating_match.group(1).strip().capitalize(),
                    "data_confidence": (
                        field("Data Confidence").group(1).strip().capitalize()
                        if field("Data Confidence") else None
                    ),
                    "thesis_confidence": (
                        field("Thesis Confidence").group(1).strip().capitalize()
                        if field("Thesis Confidence") else None
                    ),
                    "execution_confidence": (
                        field("Execution Confidence").group(1).strip().capitalize()
                        if field("Execution Confidence") else None
                    ),
                    "predictions": [],
                }
        current_snapshot = final_state.get("current_market_snapshot") or {}
        initial_close = current_snapshot.get("close")
        if decision and initial_close is not None:
            self.calibration_store.record(
                company_name,
                str(trade_date),
                decision,
                float(initial_close),
                self._resolve_benchmark(company_name),
            )
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
        return snapshot

    def propagate(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        analysis_mode: str = "auto",
        position: dict | None = None,
    ):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.prepare_run(
            company_name,
            trade_date,
            asset_type=asset_type,
            analysis_mode=analysis_mode,
            position=position,
        )

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date, asset_type=asset_type)
        finally:
            self.close_managed_browser()
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock"):
        """Execute the graph and write the resulting state to disk and memory log."""
        prepared = self._prepared_run or self.prepare_run(
            company_name, trade_date, asset_type=asset_type
        )
        init_agent_state = prepared["initial_state"]
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            last_printed = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk["messages"]:
                    msg = chunk["messages"][-1]
                    # Nodes after the trader don't append to messages, so the
                    # same trailing message repeats across chunks. Print it only
                    # when it changes (#1027); the trace/state merge is unchanged.
                    signature = (type(msg).__name__, getattr(msg, "content", None))
                    if signature != last_printed:
                        msg.pretty_print()
                        last_printed = signature
                    trace.append(chunk)
            # Streamed chunks are per-node deltas. Merge them so the returned
            # state matches what graph.invoke() yields in the non-debug path.
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        self.finalize_run(company_name, trade_date, final_state)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "researchability_assessment": final_state.get(
                "researchability_assessment", {}
            ),
            "evidence_ledger": final_state.get("evidence_ledger", {}),
            "blind_bull_argument": final_state.get("blind_bull_argument", ""),
            "blind_bear_argument": final_state.get("blind_bear_argument", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "initial_investment_plan": final_state.get(
                "initial_investment_plan", ""
            ),
            "falsification_audit": final_state.get("falsification_audit", {}),
            "final_trade_decision": final_state["final_trade_decision"],
            "portfolio_decision_metadata": final_state.get(
                "portfolio_decision_metadata", {}
            ),
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "FxxKStockStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
