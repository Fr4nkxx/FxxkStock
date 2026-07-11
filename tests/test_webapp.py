"""Tests for FxxKStock web visualization layer."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webapp.runner import MODE_DEPTH, RunParams, RunState, build_run_config, run_analysis
from webapp.server import RUNS, RunRequest, app


@pytest.mark.unit
def test_build_run_config_mode_mapping():
    for mode, depth in MODE_DEPTH.items():
        cfg = build_run_config(
            RunParams(
                ticker="600353.SS",
                provider="deepseek",
                quick_model="deepseek-v4-flash",
                deep_model="deepseek-v4-pro",
                mode=mode,
            )
        )
        assert cfg["max_debate_rounds"] == depth
        assert cfg["max_risk_discuss_rounds"] == depth
        assert cfg["llm_provider"] == "deepseek"
        assert cfg["output_language"] == "Chinese"


@pytest.mark.unit
def test_api_models_lists_providers():
    client = TestClient(app)
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "openai" in data["provider_list"]
    assert "quick" in data["providers"]["openai"]
    assert "deep" in data["providers"]["openai"]


@pytest.mark.unit
def test_calendar_page_and_home_navigation_are_available():
    client = TestClient(app)
    response = client.get("/calendar")
    assert response.status_code == 200
    assert "节点日历" in response.text
    home = client.get("/")
    assert home.status_code == 200
    assert 'href="/calendar"' in home.text
    assert 'id="prevMonthBtn"' in response.text
    assert 'id="nextMonthBtn"' in response.text
    assert 'id="selectedDateNodes"' in response.text
    assert "function renderMonth" in response.text
    assert "function renderSelectedDate" in response.text
    assert 'fetch("/api/calendar/nodes")' in response.text
    assert "function renderWeekAndConditions" in response.text
    assert "data-delete-ticker" in home.text
    assert "data-delete-report" in home.text
    assert "function deleteStock" in home.text
    assert "function deleteReport" in home.text
    assert ".run-fill {" in home.text
    assert "display: block;" in home.text
    assert "transition: width .28s ease;" in home.text
    assert "贵州茅台" not in response.text
    assert "五粮液" not in response.text


@pytest.mark.unit
def test_run_events_and_report_flow(tmp_path):
    client = TestClient(app)

    fake_chunks = [
        {
            "sender": "Market Analyst",
            "messages": [],
            "market_report": "Market analysis done.",
        },
        {
            "sender": "Portfolio Manager",
            "messages": [],
            "final_trade_decision": "FINAL TRANSACTION PROPOSAL: **HOLD**",
        },
    ]

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            self.propagator = MagicMock()
            self.propagator.create_initial_state.return_value = {"messages": []}
            self.propagator.get_graph_args.return_value = {}
            self.graph = MagicMock()
            self.graph.stream.return_value = iter(fake_chunks)

        def prepare_run(
            self,
            ticker,
            trade_date,
            asset_type="stock",
            analysis_mode="auto",
            browser_status_callback=None,
            position=None,
        ):
            return {
                "snapshot": None,
                "initial_state": {"messages": []},
                "active_analysts": ["market", "social", "news", "fundamentals"],
                "reuse": [],
                "refresh": ["market", "social", "news", "fundamentals"],
                "analysis_mode": "full",
            }

        def finalize_run(self, ticker, trade_date, final_state, log_state=True):
            return {}

        def process_signal(self, signal):
            return "HOLD"

    def fake_write_report_tree(final_state, ticker, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        report = save_path / "complete_report.md"
        report.write_text("# Test Report\n\nHOLD", encoding="utf-8")
        return report

    captured_config = {}

    def capture_graph(*args, **kwargs):
        captured_config["config"] = kwargs.get("config")
        return FakeGraph()

    with (
        patch("webapp.runner.FxxKStockGraph", side_effect=capture_graph),
        patch("webapp.runner.write_report_tree", side_effect=fake_write_report_tree),
    ):
        res = client.post(
            "/api/run",
            json={
                "ticker": "600353.SS",
                "provider": "deepseek",
                "quick_model": "deepseek-v4-flash",
                "deep_model": "deepseek-v4-pro",
                "mode": "medium",
            },
        )
        assert res.status_code == 200
        run_id = res.json()["run_id"]

        assert captured_config["config"]["max_debate_rounds"] == 3
        assert captured_config["config"]["max_risk_discuss_rounds"] == 3

        # 等待后台线程完成
        deadline = time.time() + 5
        while time.time() < deadline:
            state = RUNS.get(run_id)
            if state and state.status in ("done", "error"):
                break
            time.sleep(0.05)
        assert RUNS[run_id].status == "done"

        events: list[dict] = []
        with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
            for line in stream.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                events.append(payload)
                if payload.get("type") in ("done", "error"):
                    break

        types = {e.get("type") for e in events}
        assert "report_section" in types or "message" in types or "status" in types
        assert "done" in types

        report_res = client.get(f"/api/runs/{run_id}/report")
        assert report_res.status_code == 200
        report_data = report_res.json()
        assert report_data["available"] is True
        assert "Test Report" in report_data["markdown"]
        assert report_data["decision"] == "HOLD"


@pytest.mark.unit
def test_memory_status_api():
    client = TestClient(app)
    with patch("webapp.server.TickerMemoryStore.status", return_value={
        "ticker": "600353.SS",
        "has_memory": True,
        "analysis_count": 2,
        "last_analysis_date": "2026-06-27",
        "updated_at": "2026-06-27T10:00:00Z",
        "reuse": ["fundamentals"],
        "refresh": ["market", "social", "news"],
    }):
        res = client.get("/api/memory/600353.SS?trade_date=2026-06-28")
    assert res.status_code == 200
    assert res.json()["reuse"] == ["fundamentals"]


@pytest.mark.unit
def test_run_request_position_is_optional_and_validated():
    request = RunRequest(
        ticker="159516.SZ",
        quick_model="quick",
        deep_model="deep",
    )
    assert request.position is None

    held = RunRequest(
        ticker="159516.SZ",
        quick_model="quick",
        deep_model="deep",
        position={"status": "held", "average_cost": 1.72},
    )
    assert held.position.quantity is None

    with pytest.raises(Exception):
        RunRequest(
            ticker="159516.SZ",
            quick_model="quick",
            deep_model="deep",
            position={"status": "held", "quantity": 0, "average_cost": 1.72},
        )


@pytest.mark.unit
def test_browser_status_api():
    client = TestClient(app)
    with patch("webapp.server.ChromeManager.status", return_value={
        "available": True,
        "platform": "macos",
        "managed": True,
        "managed_platform": "macos",
        "cdp_url": "http://127.0.0.1:9222",
    }):
        res = client.get("/api/browser/status?platform=macos")
    assert res.status_code == 200
    assert res.json()["managed"] is True


@pytest.mark.unit
def test_settings_api_does_not_return_key_values():
    client = TestClient(app)
    with (
        patch("webapp.server.get_general_settings", return_value={"llm_provider": "deepseek"}),
        patch(
            "webapp.server.get_api_key_status",
            return_value=[
                {
                    "key": "DEEPSEEK_API_KEY",
                    "providers": ["deepseek"],
                    "configured": True,
                }
            ],
        ),
    ):
        response = client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_keys"][0]["configured"] is True
    assert "value" not in payload["api_keys"][0]


@pytest.mark.unit
def test_unknown_login_site_is_rejected():
    client = TestClient(app)
    response = client.post("/api/browser/open-login-site/not-allowed")
    assert response.status_code == 404


@pytest.mark.unit
def test_report_unavailable_while_running():
    client = TestClient(app)
    run_id = "pending-run"
    RUNS[run_id] = RunState(run_id=run_id, ticker="600353.SS", status="running")

    res = client.get(f"/api/runs/{run_id}/report")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["markdown"] == ""

    del RUNS[run_id]


@pytest.mark.unit
def test_run_analysis_emits_error_event():
    state = RunState(run_id="err-run", ticker="600353.SS")
    params = RunParams(
        ticker="600353.SS",
        provider="deepseek",
        quick_model="deepseek-v4-flash",
        deep_model="deepseek-v4-pro",
    )

    with patch("webapp.runner.FxxKStockGraph", side_effect=RuntimeError("boom")):
        run_analysis(state, params)

    events: list[dict] = []
    while True:
        try:
            events.append(state.event_queue.get_nowait())
        except queue.Empty:
            break

    assert state.status == "error"
    assert any(e.get("type") == "error" for e in events)


@pytest.mark.unit
def test_list_historical_reports(tmp_path):
    report_dir = tmp_path / "600353.SS_20260627_141703"
    report_dir.mkdir()
    (report_dir / "complete_report.md").write_text(
        "# Trading Analysis Report: 600353.SS\n\nFINAL TRANSACTION PROPOSAL: **SELL**",
        encoding="utf-8",
    )

    from webapp.history import get_historical_report, list_historical_reports

    items = list_historical_reports(tmp_path)
    assert len(items) == 1
    assert items[0]["ticker"] == "600353.SS"
    assert items[0]["decision"] == "SELL"

    detail = get_historical_report("600353.SS_20260627_141703", tmp_path)
    assert detail["available"] is True
    assert "600353.SS" in detail["markdown"]
    assert detail["audit"] == {}
    assert detail["sections"]["audit"] == ""


@pytest.mark.unit
def test_nested_report_layout_is_listed_and_read(tmp_path):
    report_dir = tmp_path / "600353.SS" / "20260628_091530"
    report_dir.mkdir(parents=True)
    (report_dir / "complete_report.md").write_text(
        "# Trading Analysis Report: 600353.SS\n\nFINAL TRANSACTION PROPOSAL: **HOLD**",
        encoding="utf-8",
    )

    from webapp.history import get_historical_report, list_historical_reports

    items = list_historical_reports(tmp_path)
    assert len(items) == 1
    assert items[0]["id"] == "600353.SS/20260628_091530"
    assert items[0]["ticker"] == "600353.SS"
    assert items[0]["created_at"] == "2026-06-28T09:15:30"

    detail = get_historical_report("600353.SS/20260628_091530", tmp_path)
    assert detail["available"] is True
    assert detail["ticker"] == "600353.SS"


@pytest.mark.unit
def test_delete_historical_report_and_prune_ticker_directory(tmp_path):
    report_dir = tmp_path / "600353.SS" / "20260628_091530"
    report_dir.mkdir(parents=True)
    (report_dir / "complete_report.md").write_text("# Report", encoding="utf-8")

    from webapp.history import delete_historical_report

    result = delete_historical_report("600353.SS/20260628_091530", tmp_path)
    assert result == {"deleted": True, "report_id": "600353.SS/20260628_091530"}
    assert not report_dir.exists()
    assert not (tmp_path / "600353.SS").exists()


@pytest.mark.unit
def test_delete_stock_reports_removes_nested_and_legacy_layouts(tmp_path):
    nested = tmp_path / "600353.SS" / "20260628_091530"
    legacy = tmp_path / "600353.SS_20260627_141703"
    other = tmp_path / "159819.SZ" / "20260711_120000"
    for report_dir in (nested, legacy, other):
        report_dir.mkdir(parents=True)
        (report_dir / "complete_report.md").write_text("# Report", encoding="utf-8")

    from webapp.history import delete_stock_reports

    result = delete_stock_reports("600353.SS", tmp_path)
    assert result["reports_deleted"] == 2
    assert not nested.exists()
    assert not legacy.exists()
    assert other.exists()


@pytest.mark.unit
def test_legacy_review_trigger_is_backfilled_into_calendar_nodes(tmp_path):
    report_dir = tmp_path / "159819.SZ" / "20260711_120000"
    portfolio = report_dir / "5_portfolio"
    portfolio.mkdir(parents=True)
    (report_dir / "complete_report.md").write_text("# Report 159819.SZ", encoding="utf-8")
    (portfolio / "decision.md").write_text(
        "**Review Trigger**: 07-13（周一）收盘：执行FC01证伪测试。"
        "07-17（周四）WAIC大会后收盘：评估核心催化剂。"
        "下一融资余额报告发布日：验证融资盘行为方向。\n\n"
        "**Execution Condition**: 价格回踩至¥2.08-2.12区间。若07-16收盘前未触发则放弃加仓。\n\n"
        "**Risk Boundary**: 收盘跌破风险位。",
        encoding="utf-8",
    )

    from webapp.history import list_calendar_nodes

    nodes = list_calendar_nodes(tmp_path)
    dated = [item for item in nodes if item.get("trigger_type") == "date"]
    events = [item for item in nodes if item.get("trigger_type") == "event"]
    assert sorted(item["calendar_date"] for item in dated) == [
        "2026-07-13", "2026-07-16", "2026-07-17"
    ]
    assert all(item["calendar_date"] != "2026-02-08" for item in dated)
    deadline = next(item for item in dated if item["calendar_date"] == "2026-07-16")
    assert deadline["node_type"] == "execution"
    assert "未触发则放弃加仓" in deadline["action"]
    assert all("周" not in item["action"] for item in dated)
    assert events[0]["event"] == "下一融资余额报告发布日"
    assert events[0]["action"] == "验证融资盘行为方向"
    assert {item["node_type"] for item in nodes} >= {"review", "execution", "risk"}


@pytest.mark.unit
def test_historical_report_decision_prefers_portfolio_decision(tmp_path):
    report_dir = tmp_path / "159516.SZ_20260706_113237"
    decision_dir = report_dir / "5_portfolio"
    decision_dir.mkdir(parents=True)
    (report_dir / "complete_report.md").write_text(
        "# Trading Analysis Report: 159516.SZ\n\n"
        "## Audit\n\n"
        "**建议**：下修评级至**Sell**。\n\n"
        "## Portfolio Manager Decision\n\n"
        "**Rating**: Hold",
        encoding="utf-8",
    )
    (decision_dir / "decision.md").write_text(
        "**Rating**: Hold\n\n"
        "**Executive Summary**: 账户FLAT空仓不变，继续场外观望。",
        encoding="utf-8",
    )

    from webapp.history import get_historical_report, list_historical_reports

    items = list_historical_reports(tmp_path)
    assert items[0]["decision"] == "HOLD"

    detail = get_historical_report("159516.SZ_20260706_113237", tmp_path)
    assert detail["decision"] == "HOLD"
    assert "账户FLAT空仓不变" in detail["sections"]["summary"]


@pytest.mark.unit
def test_api_report_history(tmp_path):
    report_dir = tmp_path / "600353.SS_20260627_141703"
    report_dir.mkdir()
    (report_dir / "complete_report.md").write_text("# Report\n\nHOLD", encoding="utf-8")

    with patch("webapp.server.list_historical_reports", return_value=[{
        "id": "600353.SS_20260627_141703",
        "ticker": "600353.SS",
        "created_at": "2026-06-27T14:17:03",
        "title": "Report",
        "decision": "HOLD",
        "report_dir": str(report_dir),
        "modified_at": 1,
    }]):
        client = TestClient(app)
        res = client.get("/api/reports/history")
        assert res.status_code == 200
        assert len(res.json()["reports"]) == 1

    with patch("webapp.server.get_historical_report", return_value={
        "id": "600353.SS_20260627_141703",
        "available": True,
        "ticker": "600353.SS",
        "markdown": "# Report\n\nHOLD",
        "decision": "HOLD",
    }):
        detail = client.get("/api/reports/history/600353.SS_20260627_141703")
        assert detail.status_code == 200
        assert detail.json()["markdown"].startswith("# Report")

    with patch("webapp.server.get_historical_report", return_value={
        "id": "600353.SS/20260628_091530",
        "available": True,
        "ticker": "600353.SS",
        "markdown": "# Nested Report\n\nHOLD",
        "decision": "HOLD",
    }) as nested_loader:
        detail = client.get("/api/reports/history/600353.SS/20260628_091530")
        assert detail.status_code == 200
        nested_loader.assert_called_once_with("600353.SS/20260628_091530")

    with patch("webapp.server.delete_historical_report", return_value={
        "deleted": True,
        "report_id": "600353.SS/20260628_091530",
    }) as report_deleter:
        deleted = client.delete("/api/reports/history/600353.SS/20260628_091530")
        assert deleted.status_code == 200
        report_deleter.assert_called_once_with("600353.SS/20260628_091530")

    RUNS.clear()
    with patch("webapp.server.delete_stock_reports", return_value={
        "deleted": True,
        "ticker": "600353.SS",
        "reports_deleted": 2,
    }) as stock_deleter:
        deleted = client.delete("/api/stocks/600353.SS")
        assert deleted.status_code == 200
        assert deleted.json()["reports_deleted"] == 2
        stock_deleter.assert_called_once_with("600353.SS")

    with patch("webapp.server.list_calendar_nodes", return_value=[{
        "id": "159819.SZ/20260711_120000#0",
        "ticker": "159819.SZ",
        "node_type": "review",
        "trigger_type": "date",
        "calendar_date": "2026-07-13",
        "action": "执行FC01证伪测试",
    }]):
        nodes = client.get("/api/calendar/nodes")
        assert nodes.status_code == 200
        assert nodes.json()["nodes"][0]["calendar_date"] == "2026-07-13"


@pytest.mark.unit
def test_stock_overview_groups_reports_and_calculates_change(tmp_path):
    report_dir = tmp_path / "600353.SS_20260627_141703"
    report_dir.mkdir()
    (report_dir / "complete_report.md").write_text(
        "# Trading Analysis Report: 600353.SS\n\nFINAL TRANSACTION PROPOSAL: **HOLD**",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {"Close": [10.0, 10.5], "Volume": [100, 120]},
        index=pd.to_datetime(["2026-06-26", "2026-06-27"]),
    )
    from webapp.history import get_stock_overview

    with (
        patch("webapp.history.resolve_instrument_identity", return_value={
            "company_name": "Test Company",
            "exchange": "SSE",
            "currency": "CNY",
        }),
        patch("webapp.history.detect_market_region", return_value="default"),
        patch("webapp.history.yf.Ticker") as ticker,
    ):
        ticker.return_value.history.return_value = frame
        result = get_stock_overview("600353.SS", tmp_path)

    assert result["name"] == "Test Company"
    assert result["quote"]["last_price"] == 10.5
    assert result["quote"]["change_percent"] == pytest.approx(5.0)
    assert len(result["reports"]) == 1


@pytest.mark.unit
def test_report_name_extraction_prefers_ticker_parenthesized_name():
    from webapp.history import _extract_instrument_name

    markdown = """
# Trading Analysis Report: 000725.SZ
# 深度技术分析报告：000725.SZ（京东方A）
> **总体判断**：京东方A（000725.SZ）处于强趋势中。
"""
    assert _extract_instrument_name(markdown, "000725.SZ") == "京东方A"


@pytest.mark.unit
def test_analysis_progress_is_rendered_inside_stock_card_without_main_run_panel():
    html = (Path(__file__).parents[1] / "webapp" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'class="analysis-strip"' not in html
    assert "run-track" in html
    assert "run-fill" in html
    assert 'id="stageLine"' not in html
    assert 'id="runStatus"' not in html
    assert 'id="eventList"' not in html
    assert "group.activeRun?.percent" in html
    assert 'id="debugBtn"' in html


@pytest.mark.unit
def test_decision_summary_is_compact_and_clamped():
    html = (Path(__file__).parents[1] / "webapp" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "-webkit-line-clamp: 6" in html
    assert "function truncateText" in html
    assert "function firstMeaningfulParagraph" in html
    assert "summaryText(summary, data.core_insights)" in html


@pytest.mark.unit
def test_next_actions_use_four_fixed_portfolio_manager_fields():
    html = (Path(__file__).parents[1] / "webapp" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "function renderNextActions" in html
    assert 'parseField(markdown, "Next Action")' in html
    assert 'parseField(markdown, "Execution Condition")' in html
    assert 'parseField(markdown, "Risk Boundary")' in html
    assert 'parseField(markdown, "Review Trigger")' in html
    assert "renderCorePoints" not in html


@pytest.mark.unit
def test_stock_overview_uses_daily_bar_for_quote_cards(tmp_path):
    intraday = pd.DataFrame(
        {
            "Open": [183.0, 192.5],
            "High": [190.0, 193.2],
            "Low": [176.08, 192.5],
            "Close": [190.0, 193.2],
            "Volume": [800_000, 110_900],
        },
        index=pd.to_datetime(["2026-07-01 09:35", "2026-07-01 14:55"]),
    )
    daily = pd.DataFrame(
        {
            "Open": [180.0, 183.0],
            "High": [195.0, 193.4],
            "Low": [178.0, 176.08],
            "Close": [192.5, 193.4],
            "Volume": [1_000_000, 910_900],
        },
        index=pd.to_datetime(["2026-06-30", "2026-07-01"]),
    )
    from webapp import history

    history._overview_cache.clear()
    with (
        patch("webapp.history.resolve_instrument_identity", return_value={
            "company_name": "利通电子",
            "exchange": "SHH",
            "currency": "CNY",
        }),
        patch("webapp.history.detect_market_region", return_value="default"),
        patch("webapp.history.yf.Ticker") as ticker,
    ):
        ticker.return_value.history.side_effect = [intraday, daily]
        ticker.return_value.info = {}
        result = history.get_stock_overview("603629.SS", tmp_path)

    quote = result["quote"]
    assert quote["open"] == 183.0
    assert quote["high"] == 193.4
    assert quote["low"] == 176.08
    assert quote["last_price"] == 193.4
    assert quote["previous_close"] == 192.5
