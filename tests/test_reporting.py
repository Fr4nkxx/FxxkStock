"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

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
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")
