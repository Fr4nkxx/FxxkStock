"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

import json
import re
from types import SimpleNamespace

import pytest

from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph
from fxxkstock.reporting import render_account_position_section, write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_write_report_tree_persists_calendar_nodes(tmp_path):
    state = _state()
    state["trade_date"] = "2026-07-11"
    state["portfolio_decision_metadata"] = {
        "rating": "Overweight",
        "next_action": "Add",
        "execution_condition": "Break above 1.10 on volume.",
        "risk_boundary": "Close below 1.00.",
        "review_nodes": [{
            "node_type": "review",
            "trigger_type": "date",
            "calendar_date": "2026-07-13",
            "event": None,
            "action": "Run FC01 test.",
        }],
    }

    write_report_tree(state, "159819.SZ", tmp_path)
    payload = json.loads((tmp_path / "calendar_nodes.json").read_text())

    assert payload["ticker"] == "159819.SZ"
    assert payload["analysis_date"] == "2026-07-11"
    assert payload["nodes"][0]["calendar_date"] == "2026-07-13"
    assert {item["node_type"] for item in payload["nodes"]} == {
        "review", "execution", "risk"
    }


@pytest.mark.unit
def test_write_report_tree_persists_anti_bias_audit(tmp_path):
    state = _state()
    state.update(
        {
            "researchability_assessment": {
                "status": "available",
                "information_grade": "B",
                "markdown": "**Information Grade**: B",
            },
            "falsification_audit": {
                "status": "available",
                "requires_revision": False,
                "falsification_triggers": ["Revenue misses"],
                "markdown": "**Requires Revision**: No",
            },
            "evidence_ledger": {
                "status": "available",
                "claims": [{"claim_id": "E01", "claim": "Revenue grew"}],
                "markdown": "# Evidence Ledger\n\nE01 Revenue grew",
            },
            "blind_bull_argument": "Blind bull E01.",
            "blind_bear_argument": "Blind bear E01.",
            "initial_investment_plan": "INITIAL PLAN",
            "final_trade_decision": (
                "**Data Confidence**: High\n"
                "**Data Confidence Reason**: Fresh.\n"
                "**Thesis Confidence**: Medium\n"
                "**Thesis Confidence Reason**: Some uncertainty.\n"
                "**Execution Confidence**: Low\n"
                "**Execution Confidence Reason**: No trigger."
            ),
        }
    )
    out = write_report_tree(state, "AAPL", tmp_path)
    payload = json.loads((tmp_path / "6_audit" / "audit.json").read_text())
    assert payload["researchability"]["information_grade"] == "B"
    assert payload["confidence"]["execution"]["level"] == "Low"
    assert payload["evidence_ledger"]["claims"][0]["claim_id"] == "E01"
    assert (tmp_path / "6_audit" / "evidence_ledger.json").is_file()
    assert (tmp_path / "2_research" / "blind_bull.md").is_file()
    assert (tmp_path / "2_research" / "blind_bear.md").is_file()
    assert (tmp_path / "6_audit" / "research_manager_initial.md").read_text() == "INITIAL PLAN"
    assert "Anti-Bias Audit" in out.read_text()


@pytest.mark.unit
def test_report_has_deterministic_account_position_section(tmp_path):
    state = _state()
    state["position_context"] = {
        "status": "held",
        "quantity": 3800,
        "average_cost": 1.932,
        "current_price": 2.228,
        "currency": "CNY",
        "cost_basis": 7341.6,
        "market_value": 8466.4,
        "unrealized_pnl": 1124.8,
        "unrealized_return_pct": 15.3209,
    }

    report = write_report_tree(state, "159819.SZ", tmp_path).read_text()

    assert "本次账户持仓 / Account Position" in report
    assert "| 持股数量 | 3,800 |" in report
    assert "| 真实平均成本 | 1.932 CNY |" in report
    assert "| 浮动盈亏 | 1,124.80 CNY |" in report
    assert "| 浮动收益率 | 15.32% |" in report
    assert "市场低点、技术位和历史报告价格均不是用户成本" in report


@pytest.mark.unit
def test_unknown_position_is_omitted_from_report():
    assert render_account_position_section({"status": "unknown"}) == ""


@pytest.mark.unit
def test_cost_only_position_report_omits_quantity_and_money_values():
    report = render_account_position_section({
        "status": "held",
        "quantity": None,
        "average_cost": 1.932,
        "current_price": 2.228,
        "currency": "CNY",
        "unrealized_return_pct": 15.3209,
    })

    assert "持股数量" not in report
    assert "当前市值" not in report
    assert "浮动盈亏" not in report
    assert "| 真实平均成本 | 1.932 CNY |" in report
    assert "| 浮动收益率 | 15.32% |" in report


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = FxxKStockGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = FxxKStockGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "AAPL"
    assert out.parent.parent.parent.name == "reports"
    assert re.fullmatch(r"\d{8}_\d{6}", out.parent.name)
