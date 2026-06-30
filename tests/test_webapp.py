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
        position={"status": "held", "quantity": 1000, "average_cost": 1.72},
    )
    assert held.position.quantity == 1000

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
