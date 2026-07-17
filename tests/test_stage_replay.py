"""Saved-report input reconstruction for targeted stage diagnostics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from fxxkstock.diagnostics import stage_replay as stage_replay_module
from fxxkstock.diagnostics.stage_replay import (
    ReplayInputError,
    load_falsification_replay,
    load_stage_replay,
    run_stage_replay,
)
from scripts.benchmark_stage import _console_safe


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.unit
def test_benchmark_output_is_safe_for_windows_gbk_console():
    assert _console_safe("价格 ¥28.56", "gbk") == "价格 \\xa528.56"


def _report_tree(tmp_path, *, exact_context: bool):
    report_dir = tmp_path / "600353.SS" / "20260713_105958"
    for filename, text in {
        "market.md": "Market report",
        "sentiment.md": "Sentiment report",
        "news.md": "News report",
        "fundamentals.md": "Fundamentals report",
    }.items():
        _write(report_dir / "1_analysts" / filename, text)
    _write(report_dir / "2_research" / "blind_bull.md", "Blind Bull: first")
    _write(report_dir / "2_research" / "blind_bear.md", "Blind Bear: first")
    _write(
        report_dir / "2_research" / "bull.md",
        "Blind Bull: first\nBull Analyst: response",
    )
    _write(
        report_dir / "2_research" / "bear.md",
        "Blind Bear: first\nBear Analyst: response",
    )
    _write(report_dir / "2_research" / "manager.md", "Manager plan")
    _write(report_dir / "3_trading" / "trader.md", "Trader plan")
    _write(report_dir / "4_risk" / "aggressive.md", "Aggressive Analyst: first")
    _write(report_dir / "4_risk" / "conservative.md", "Conservative Analyst: first")
    _write(report_dir / "4_risk" / "neutral.md", "Neutral Analyst: first")
    _write(report_dir / "6_audit" / "evidence_ledger.md", "Evidence E01")
    _write(report_dir / "6_audit" / "researchability.md", "Grade B")
    _write(
        report_dir / "6_audit" / "research_manager_initial.md",
        "Initial plan",
    )
    _write(
        report_dir / "6_audit" / "audit.json",
        json.dumps(
            {
                "evidence_ledger": {"status": "available"},
                "researchability": {"status": "available"},
            }
        ),
    )
    if exact_context:
        _write(
            report_dir / "6_audit" / "replay_context.json",
            json.dumps(
                {
                    "version": 2,
                    "ticker": "600353.SS",
                    "company_of_interest": "600353.SS",
                    "asset_type": "stock",
                    "instrument_context": "Exact instrument context",
                    "trade_date": "2026-07-13",
                    "analysis_mode": "full",
                    "investment_debate_state": {
                        "history": "Exact interleaved debate",
                        "bull_history": "Bull history",
                        "bear_history": "Bear history",
                        "current_response": "Bear response",
                        "judge_decision": "Manager plan",
                        "count": 2,
                    },
                    "risk_debate_state": {
                        "history": "Exact risk debate",
                        "aggressive_history": "Aggressive history",
                        "conservative_history": "Conservative history",
                        "neutral_history": "Neutral history",
                        "latest_speaker": "Neutral",
                        "current_aggressive_response": "Aggressive response",
                        "current_conservative_response": "Conservative response",
                        "current_neutral_response": "Neutral response",
                        "judge_decision": "Portfolio decision",
                        "count": 3,
                    },
                    "investment_plan": "Manager plan",
                    "trader_investment_plan": "Trader plan",
                    "position_context": {"status": "flat"},
                    "current_market_snapshot": {"close": 42.0},
                    "falsification_audit": {"markdown": "Audit"},
                    "stage_replay_contexts": {
                        "bull": [
                            {
                                "investment_debate_state": {
                                    "history": "Blind debate",
                                    "bull_history": "Blind Bull",
                                    "bear_history": "Blind Bear",
                                    "current_response": "",
                                    "judge_decision": "",
                                    "count": 0,
                                }
                            }
                        ],
                        "bear": [
                            {
                                "investment_debate_state": {
                                    "history": "Bull spoke",
                                    "bull_history": "Bull spoke",
                                    "bear_history": "Blind Bear",
                                    "current_response": "Bull Analyst: case",
                                    "judge_decision": "",
                                    "count": 1,
                                }
                            }
                        ],
                        "portfolio": [
                            {
                                "risk_debate_state": {
                                    "history": "Exact pre-portfolio risk debate",
                                    "aggressive_history": "Aggressive history",
                                    "conservative_history": "Conservative history",
                                    "neutral_history": "Neutral history",
                                    "latest_speaker": "Neutral",
                                    "current_aggressive_response": "Aggressive response",
                                    "current_conservative_response": "Conservative response",
                                    "current_neutral_response": "Neutral response",
                                    "judge_decision": "",
                                    "count": 3,
                                }
                            }
                        ],
                    },
                }
            ),
        )
    return report_dir


@pytest.mark.unit
def test_load_falsification_replay_prefers_exact_context(tmp_path):
    replay = load_falsification_replay(_report_tree(tmp_path, exact_context=True))

    assert replay.ticker == "600353.SS"
    assert replay.exact_context is True
    assert replay.warnings == ()
    assert replay.state["instrument_context"] == "Exact instrument context"
    assert replay.state["investment_debate_state"]["history"] == "Exact interleaved debate"
    assert replay.state["evidence_ledger"]["markdown"] == "Evidence E01"
    assert replay.component_sizes()["initial_plan"] == len("Initial plan")


@pytest.mark.unit
def test_load_falsification_replay_reconstructs_legacy_report(tmp_path):
    replay = load_falsification_replay(_report_tree(tmp_path, exact_context=False))

    history = replay.state["investment_debate_state"]["history"]
    assert replay.exact_context is False
    assert replay.warnings
    assert history.index("Blind Bull") < history.index("Blind Bear")
    assert history.index("Bull Analyst") < history.index("Bear Analyst")
    assert replay.state["instrument_context"].startswith("The instrument to analyze is `600353.SS`")


@pytest.mark.unit
def test_load_falsification_replay_rejects_incomplete_report(tmp_path):
    report_dir = tmp_path / "600353.SS" / "20260713_105958"
    report_dir.mkdir(parents=True)

    with pytest.raises(ReplayInputError, match="invalid replay input"):
        load_falsification_replay(report_dir)


@pytest.mark.unit
def test_load_stage_replay_uses_exact_pre_call_snapshot(tmp_path):
    report_dir = _report_tree(tmp_path, exact_context=True)

    bear = load_stage_replay(report_dir, "bear")
    portfolio = load_stage_replay(report_dir, "portfolio")

    assert bear.exact_context is True
    assert bear.state["investment_debate_state"]["history"] == "Bull spoke"
    assert portfolio.exact_context is True
    assert portfolio.state["risk_debate_state"]["history"] == "Exact pre-portfolio risk debate"
    assert portfolio.state["position_context"] == {"status": "flat"}


@pytest.mark.unit
def test_load_stage_replay_reconstructs_old_stage_input_with_warning(tmp_path):
    replay = load_stage_replay(
        _report_tree(tmp_path, exact_context=False),
        "neutral",
    )

    assert replay.exact_context is False
    assert replay.warnings
    assert replay.state["risk_debate_state"]["count"] == 2
    assert "Aggressive Analyst" in replay.state["risk_debate_state"]["history"]


@pytest.mark.unit
def test_run_plain_stage_replay_returns_model_diagnostics(tmp_path):
    replay = load_stage_replay(
        _report_tree(tmp_path, exact_context=True),
        "bull",
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Replay bull case")

    result = run_stage_replay(replay, llm)

    diagnostics = result["bull_researcher_diagnostics"]
    assert diagnostics["model_attempts"] == 1
    assert diagnostics["input_characters"]["prompt"] > 0
    assert diagnostics["output_characters"] == len("Replay bull case")
    assert diagnostics["replay_context_exact"] is True
    llm.invoke.assert_called_once()


@pytest.mark.unit
def test_run_portfolio_replay_forwards_structured_method(tmp_path, monkeypatch):
    replay = load_stage_replay(
        _report_tree(tmp_path, exact_context=True),
        "portfolio",
    )
    captured = {}

    def factory(llm, *, structured_method=None):
        captured["llm"] = llm
        captured["structured_method"] = structured_method

        def node(state):
            captured["state"] = state
            return {
                "final_trade_decision": "**Rating**: Hold",
                "portfolio_manager_diagnostics": {},
            }

        return node

    monkeypatch.setattr(stage_replay_module, "create_portfolio_manager", factory)
    llm = MagicMock()

    result = run_stage_replay(replay, llm, structured_method="json_mode")

    assert result["final_trade_decision"] == "**Rating**: Hold"
    assert captured["llm"] is llm
    assert captured["structured_method"] == "json_mode"
    assert captured["state"]["risk_debate_state"]["history"] == (
        "Exact pre-portfolio risk debate"
    )
    assert result["portfolio_manager_diagnostics"]["replay_context_exact"] is True
