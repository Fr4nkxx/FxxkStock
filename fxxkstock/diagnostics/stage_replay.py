"""Load saved report artifacts and replay one expensive analysis stage."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fxxkstock.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_evidence_ledger_builder,
    create_neutral_debator,
    create_portfolio_manager,
    create_trader,
)
from fxxkstock.agents.managers.anti_bias import create_falsification_auditor
from fxxkstock.agents.utils.agent_utils import build_instrument_context

_NESTED_RUN_PATTERN = re.compile(r"^\d{8}_\d{6}$")
_LEGACY_RUN_PATTERN = re.compile(r"^(.+)_\d{8}_\d{6}$")


class ReplayInputError(ValueError):
    """Raised when a report directory cannot supply a stage's required input."""


@dataclass(frozen=True)
class StageReplayInput:
    """A reconstructed graph-state slice for one diagnostic stage."""

    report_dir: Path
    ticker: str
    state: dict[str, Any]
    exact_context: bool
    stage: str = "falsification"
    warnings: tuple[str, ...] = ()

    def component_sizes(self) -> dict[str, int]:
        debate = self.state.get("investment_debate_state") or {}
        risk_debate = self.state.get("risk_debate_state") or {}
        return {
            "market_report": len(self.state.get("market_report", "")),
            "sentiment_report": len(self.state.get("sentiment_report", "")),
            "news_report": len(self.state.get("news_report", "")),
            "fundamentals_report": len(self.state.get("fundamentals_report", "")),
            "evidence_ledger": len((self.state.get("evidence_ledger") or {}).get("markdown", "")),
            "researchability": len(
                (self.state.get("researchability_assessment") or {}).get("markdown", "")
            ),
            "debate_history": len(debate.get("history", "")),
            "initial_plan": len(self.state.get("investment_plan", "")),
            "trader_plan": len(self.state.get("trader_investment_plan", "")),
            "risk_debate_history": len(risk_debate.get("history", "")),
            "past_context": len(self.state.get("past_context", "")),
            "instrument_context": len(self.state.get("instrument_context", "")),
        }


def _read_text(path: Path, *, required: bool = True) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if required:
            raise ReplayInputError(f"missing replay input: {path}") from exc
        return ""
    if required and not value:
        raise ReplayInputError(f"empty replay input: {path}")
    return value


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if required:
            raise ReplayInputError(f"invalid replay input: {path}") from exc
        return {}
    if not isinstance(value, dict):
        if required:
            raise ReplayInputError(f"replay input is not an object: {path}")
        return {}
    return value


def _infer_ticker(report_dir: Path, context: dict[str, Any]) -> str:
    configured = str(context.get("ticker") or "").strip().upper()
    if configured:
        return configured
    if _NESTED_RUN_PATTERN.fullmatch(report_dir.name):
        return report_dir.parent.name.upper()
    match = _LEGACY_RUN_PATTERN.fullmatch(report_dir.name)
    if match:
        return match.group(1).upper()
    raise ReplayInputError(f"cannot infer ticker from report directory: {report_dir}")


def _without_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix) :].lstrip()
    return value


def _legacy_debate_state(report_dir: Path) -> dict[str, Any]:
    research_dir = report_dir / "2_research"
    blind_bull = _read_text(research_dir / "blind_bull.md", required=False)
    blind_bear = _read_text(research_dir / "blind_bear.md", required=False)
    bull_history = _read_text(research_dir / "bull.md")
    bear_history = _read_text(research_dir / "bear.md")
    bull_debate = _without_prefix(bull_history, blind_bull)
    bear_debate = _without_prefix(bear_history, blind_bear)
    ordered = [blind_bull, blind_bear, bull_debate, bear_debate]
    history = "\n\n".join(item for item in ordered if item)
    current_response = bear_debate or bull_debate or blind_bear or blind_bull
    return {
        "bull_history": bull_history,
        "bear_history": bear_history,
        "history": history,
        "current_response": current_response,
        "judge_decision": _read_text(research_dir / "manager.md", required=False),
        "count": int(bool(bull_debate)) + int(bool(bear_debate)),
    }


def load_falsification_replay(report_dir: str | Path) -> StageReplayInput:
    """Reconstruct the state consumed by the Falsification Auditor.

    Reports generated after replay-context support preserve the exact debate
    history and instrument context. Older reports remain usable, but their
    per-side debate files cannot preserve multi-round interleaving exactly.
    """

    root = Path(report_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReplayInputError(f"report directory does not exist: {root}")

    audit_dir = root / "6_audit"
    context = _read_json(audit_dir / "replay_context.json", required=False)
    audit = _read_json(audit_dir / "audit.json")
    ticker = _infer_ticker(root, context)
    warnings: list[str] = []

    debate = context.get("investment_debate_state")
    exact_context = isinstance(debate, dict) and bool(debate.get("history"))
    if exact_context:
        debate_state = dict(debate)
    else:
        debate_state = _legacy_debate_state(root)
        warnings.append(
            "report predates replay_context.json; debate history was reconstructed "
            "and may not preserve multi-round interleaving"
        )

    evidence = dict(audit.get("evidence_ledger") or {})
    evidence["markdown"] = _read_text(audit_dir / "evidence_ledger.md")
    researchability = dict(audit.get("researchability") or {})
    researchability["markdown"] = _read_text(audit_dir / "researchability.md")

    instrument_context = str(context.get("instrument_context") or "").strip()
    if not instrument_context:
        instrument_context = build_instrument_context(
            ticker,
            str(context.get("asset_type") or "stock"),
        )

    state = {
        "company_of_interest": str(context.get("company_of_interest") or ticker),
        "asset_type": str(context.get("asset_type") or "stock"),
        "instrument_context": instrument_context,
        "trade_date": str(context.get("trade_date") or ""),
        "analysis_mode": str(context.get("analysis_mode") or "full"),
        "market_report": _read_text(root / "1_analysts" / "market.md"),
        "sentiment_report": _read_text(root / "1_analysts" / "sentiment.md"),
        "news_report": _read_text(root / "1_analysts" / "news.md"),
        "fundamentals_report": _read_text(root / "1_analysts" / "fundamentals.md"),
        "evidence_ledger": evidence,
        "researchability_assessment": researchability,
        "investment_debate_state": debate_state,
        "investment_plan": _read_text(audit_dir / "research_manager_initial.md"),
    }
    return StageReplayInput(
        report_dir=root,
        ticker=ticker,
        state=state,
        exact_context=exact_context,
        warnings=tuple(warnings),
    )


def run_falsification_replay(
    replay: StageReplayInput,
    llm: Any,
    *,
    structured_method: str | None = None,
) -> dict[str, Any]:
    """Invoke only the Falsification Auditor with a saved state slice."""

    started_at = time.perf_counter()
    result = create_falsification_auditor(
        llm,
        structured_method=structured_method,
    )(replay.state)
    diagnostics = result.setdefault("falsification_auditor_diagnostics", {})
    diagnostics["replay_wall_seconds"] = round(time.perf_counter() - started_at, 3)
    diagnostics["replay_context_exact"] = replay.exact_context
    diagnostics["replay_warnings"] = list(replay.warnings)
    return result


SUPPORTED_REPLAY_STAGES = (
    "evidence",
    "bull",
    "bear",
    "falsification",
    "trader",
    "aggressive",
    "conservative",
    "neutral",
    "portfolio",
)

_DIAGNOSTIC_FIELDS = {
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


def _empty_risk_state() -> dict[str, Any]:
    return {
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


def _legacy_risk_state(report_dir: Path) -> dict[str, Any]:
    risk_dir = report_dir / "4_risk"
    aggressive = _read_text(risk_dir / "aggressive.md", required=False)
    conservative = _read_text(risk_dir / "conservative.md", required=False)
    neutral = _read_text(risk_dir / "neutral.md", required=False)
    history = "\n".join(item for item in (aggressive, conservative, neutral) if item)
    state = _empty_risk_state()
    state.update(
        {
            "aggressive_history": aggressive,
            "conservative_history": conservative,
            "neutral_history": neutral,
            "history": history,
            "latest_speaker": "Neutral"
            if neutral
            else "Conservative"
            if conservative
            else "Aggressive",
            "current_aggressive_response": aggressive,
            "current_conservative_response": conservative,
            "current_neutral_response": neutral,
            "count": sum(bool(item) for item in (aggressive, conservative, neutral)),
        }
    )
    return state


def _legacy_stage_state(report_dir: Path, stage: str, state: dict[str, Any]) -> None:
    if stage in {"bull", "bear"}:
        research_dir = report_dir / "2_research"
        blind_bull = _read_text(research_dir / "blind_bull.md", required=False)
        blind_bear = _read_text(research_dir / "blind_bear.md", required=False)
        bull_history = _read_text(research_dir / "bull.md", required=False)
        bull_argument = _without_prefix(bull_history, blind_bull)
        debate = {
            "bull_history": blind_bull,
            "bear_history": blind_bear,
            "history": "\n\n".join(item for item in (blind_bull, blind_bear) if item),
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        }
        if stage == "bear":
            debate.update(
                {
                    "bull_history": bull_history,
                    "history": "\n".join(
                        item for item in (debate["history"], bull_argument) if item
                    ),
                    "current_response": bull_argument,
                    "count": 1,
                }
            )
        state["investment_debate_state"] = debate
        return

    if stage in {"aggressive", "conservative", "neutral"}:
        final_risk = _legacy_risk_state(report_dir)
        risk = _empty_risk_state()
        aggressive = final_risk["aggressive_history"]
        conservative = final_risk["conservative_history"]
        if stage in {"conservative", "neutral"} and aggressive:
            risk.update(
                {
                    "aggressive_history": aggressive,
                    "history": aggressive,
                    "latest_speaker": "Aggressive",
                    "current_aggressive_response": aggressive,
                    "count": 1,
                }
            )
        if stage == "neutral" and conservative:
            risk.update(
                {
                    "conservative_history": conservative,
                    "history": "\n".join((aggressive, conservative)),
                    "latest_speaker": "Conservative",
                    "current_conservative_response": conservative,
                    "count": 2,
                }
            )
        state["risk_debate_state"] = risk


def load_stage_replay(
    report_dir: str | Path,
    stage: str,
    *,
    invocation: int = -1,
) -> StageReplayInput:
    """Load a saved state slice for one supported model stage."""
    normalized = stage.strip().lower()
    if normalized not in SUPPORTED_REPLAY_STAGES:
        raise ReplayInputError(f"unsupported replay stage: {stage}")
    if normalized == "falsification":
        replay = load_falsification_replay(report_dir)
        return StageReplayInput(
            report_dir=replay.report_dir,
            ticker=replay.ticker,
            state=replay.state,
            exact_context=replay.exact_context,
            stage=normalized,
            warnings=replay.warnings,
        )

    base = load_falsification_replay(report_dir)
    root = base.report_dir
    audit_dir = root / "6_audit"
    context = _read_json(audit_dir / "replay_context.json", required=False)
    audit = _read_json(audit_dir / "audit.json")
    state = dict(base.state)
    state.update(
        {
            "blind_bull_argument": str(
                context.get("blind_bull_argument")
                or _read_text(root / "2_research" / "blind_bull.md", required=False)
            ),
            "blind_bear_argument": str(
                context.get("blind_bear_argument")
                or _read_text(root / "2_research" / "blind_bear.md", required=False)
            ),
            "investment_plan": str(
                context.get("investment_plan") or _read_text(root / "2_research" / "manager.md")
            ),
            "trader_investment_plan": str(
                context.get("trader_investment_plan")
                or _read_text(root / "3_trading" / "trader.md", required=False)
            ),
            "position_context": dict(context.get("position_context") or {}),
            "past_context": str(context.get("past_context") or ""),
            "current_market_snapshot": dict(context.get("current_market_snapshot") or {}),
            "falsification_audit": dict(
                context.get("falsification_audit") or audit.get("falsification") or {}
            ),
            "risk_debate_state": dict(context.get("risk_debate_state") or _legacy_risk_state(root)),
            "stage_replay_contexts": dict(context.get("stage_replay_contexts") or {}),
        }
    )
    falsification_markdown = _read_text(
        audit_dir / "falsification.md",
        required=False,
    )
    if falsification_markdown:
        state["falsification_audit"]["markdown"] = falsification_markdown

    warnings: list[str] = []
    snapshots = state["stage_replay_contexts"].get(normalized) or []
    exact_context = False
    if snapshots:
        try:
            snapshot = snapshots[invocation]
        except IndexError as exc:
            raise ReplayInputError(
                f"stage {normalized} has {len(snapshots)} captured invocation(s); "
                f"cannot select {invocation}"
            ) from exc
        if not isinstance(snapshot, dict):
            raise ReplayInputError(f"invalid captured context for stage {normalized}")
        state.update(snapshot)
        exact_context = True
    elif normalized == "evidence":
        exact_context = bool(context.get("instrument_context"))
        if not exact_context:
            warnings.append("instrument identity was reconstructed from the ticker")
    elif normalized in {"trader", "portfolio"} and int(context.get("version") or 0) >= 2:
        exact_context = normalized == "trader" or bool(context.get("risk_debate_state"))
    else:
        _legacy_stage_state(root, normalized, state)
        warnings.append(
            f"report predates exact {normalized} replay capture; "
            "input was reconstructed from saved artifacts"
        )

    return StageReplayInput(
        report_dir=root,
        ticker=base.ticker,
        state=state,
        exact_context=exact_context,
        stage=normalized,
        warnings=tuple(warnings),
    )


def run_stage_replay(
    replay: StageReplayInput,
    llm: Any,
    *,
    structured_method: str | None = None,
) -> dict[str, Any]:
    """Invoke exactly one saved stage with no graph routing or report writes."""
    if replay.stage == "falsification":
        return run_falsification_replay(
            replay,
            llm,
            structured_method=structured_method,
        )
    if replay.stage == "portfolio":
        started_at = time.perf_counter()
        result = create_portfolio_manager(
            llm,
            structured_method=structured_method,
        )(replay.state)
        diagnostics = result.setdefault("portfolio_manager_diagnostics", {})
        diagnostics["replay_wall_seconds"] = round(
            time.perf_counter() - started_at,
            3,
        )
        diagnostics["replay_context_exact"] = replay.exact_context
        diagnostics["replay_warnings"] = list(replay.warnings)
        return result

    factories = {
        "evidence": create_evidence_ledger_builder,
        "bull": create_bull_researcher,
        "bear": create_bear_researcher,
        "trader": create_trader,
        "aggressive": create_aggressive_debator,
        "conservative": create_conservative_debator,
        "neutral": create_neutral_debator,
    }
    started_at = time.perf_counter()
    result = factories[replay.stage](llm)(replay.state)
    diagnostics = result.setdefault(_DIAGNOSTIC_FIELDS[replay.stage], {})
    diagnostics["replay_wall_seconds"] = round(time.perf_counter() - started_at, 3)
    diagnostics["replay_context_exact"] = replay.exact_context
    diagnostics["replay_warnings"] = list(replay.warnings)
    return result
