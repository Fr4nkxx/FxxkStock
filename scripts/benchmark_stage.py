"""Dry-run or execute one expensive stage from a saved report directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fxxkstock.agents.utils.structured import summarize_diagnostic_error
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.diagnostics.stage_replay import (
    SUPPORTED_REPLAY_STAGES,
    load_stage_replay,
    run_stage_replay,
)
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph
from fxxkstock.llm_clients.capabilities import (
    resolve_falsification_structured_method,
)


def _console_safe(text: str, encoding: str | None = None) -> str:
    """Return text that can be written to the active Windows console."""
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="backslashreplace").decode(
        target_encoding
    )


def _print_json(value: dict) -> None:
    print(_console_safe(json.dumps(value, ensure_ascii=False, indent=2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=SUPPORTED_REPLAY_STAGES,
        default="falsification",
    )
    parser.add_argument(
        "--invocation",
        type=int,
        default=-1,
        help="Captured stage invocation to replay; -1 selects the latest.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the model call. Without this flag, only validate inputs.",
    )
    parser.add_argument("--provider", help="Override the configured LLM provider")
    parser.add_argument("--model", help="Override the model used by the selected stage")
    parser.add_argument("--backend-url", help="Override the configured backend URL")
    parser.add_argument(
        "--structured-method",
        choices=("function_calling", "json_mode", "json_schema"),
        help=(
            "Override structured output only for this replay. Omit it to "
            "resolve the configured method exactly as a normal analysis does."
        ),
    )
    parser.add_argument("--show-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay = load_stage_replay(
        args.report_dir,
        args.stage,
        invocation=args.invocation,
    )
    config = DEFAULT_CONFIG.copy()
    if args.provider:
        config["llm_provider"] = args.provider
    if args.model:
        config["deep_think_llm"] = args.model
        config["quick_think_llm"] = args.model
    if args.backend_url:
        config["backend_url"] = args.backend_url

    summary = {
        "stage": args.stage,
        "report_dir": str(replay.report_dir),
        "ticker": replay.ticker,
        "context": "exact" if replay.exact_context else "reconstructed",
        "warnings": list(replay.warnings),
        "component_characters": replay.component_sizes(),
    }
    effective_structured_method = args.structured_method
    if args.stage == "falsification":
        requested_method = args.structured_method or config.get(
            "falsification_structured_method"
        )
        if effective_structured_method is None:
            effective_structured_method = resolve_falsification_structured_method(
                requested=str(requested_method or "provider_default"),
                provider=str(config.get("llm_provider") or ""),
                model=str(config.get("deep_think_llm") or ""),
                backend_url=config.get("backend_url"),
            )
        summary["structured_method_requested"] = str(
            requested_method or "provider_default"
        )
        summary["structured_method"] = (
            effective_structured_method or "provider_default"
        )
    elif args.stage == "portfolio":
        summary["structured_method"] = (
            effective_structured_method or "provider_default"
        )
    if not args.execute:
        summary["mode"] = "dry-run"
        _print_json(summary)
        return 0

    graph = FxxKStockGraph(
        selected_analysts=("market",),
        config=config,
    )
    try:
        stage_llm = (
            graph.deep_thinking_llm
            if args.stage in {"falsification", "portfolio"}
            else graph.quick_thinking_llm
        )
        result = run_stage_replay(
            replay,
            stage_llm,
            structured_method=effective_structured_method,
        )
    except Exception as exc:
        summary.update(
            {
                "mode": "execute",
                "provider": config["llm_provider"],
                "model": config["deep_think_llm"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": summarize_diagnostic_error(exc),
            }
        )
        _print_json(summary)
        return 2
    diagnostic_fields = {
        "evidence": "evidence_ledger_builder_diagnostics",
        "bull": "bull_researcher_diagnostics",
        "bear": "bear_researcher_diagnostics",
        "falsification": "falsification_auditor_diagnostics",
        "trader": "trader_diagnostics",
        "aggressive": "aggressive_analyst_diagnostics",
        "conservative": "conservative_analyst_diagnostics",
        "neutral": "neutral_analyst_diagnostics",
        "portfolio": "portfolio_manager_diagnostics",
    }
    output_fields = {
        "evidence": "evidence_ledger",
        "bull": "investment_debate_state",
        "bear": "investment_debate_state",
        "falsification": "falsification_audit",
        "trader": "trader_investment_plan",
        "aggressive": "risk_debate_state",
        "conservative": "risk_debate_state",
        "neutral": "risk_debate_state",
        "portfolio": "final_trade_decision",
    }
    output = result.get(output_fields[args.stage])
    summary.update(
        {
            "mode": "execute",
            "provider": config["llm_provider"],
            "model": config["deep_think_llm"],
            "diagnostics": result.get(diagnostic_fields[args.stage], {}),
            "status": (
                output.get("status", "available")
                if isinstance(output, dict)
                else "available"
                if output
                else "unavailable"
            ),
        }
    )
    if args.stage == "falsification":
        summary["requires_revision"] = (output or {}).get("requires_revision")
    _print_json(summary)
    if args.show_output:
        if isinstance(output, dict):
            rendered = (
                output.get("markdown")
                or output.get("history")
                or json.dumps(
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            rendered = str(output or "")
        print(_console_safe("\n" + rendered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
