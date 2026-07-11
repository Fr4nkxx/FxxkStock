from __future__ import annotations

import json
from pathlib import Path

from fxxkstock.agents.utils.ticker_memory import TickerMemoryStore


def make_store(tmp_path: Path, ttl: int = 30) -> TickerMemoryStore:
    return TickerMemoryStore(
        {
            "ticker_memory_dir": str(tmp_path / "memory" / "tickers"),
            "reports_dir": str(tmp_path / "reports"),
            "ticker_memory_fundamentals_ttl_days": ttl,
        }
    )


def test_first_save_and_ticker_isolation(tmp_path):
    store = make_store(tmp_path)
    state = {
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "final_trade_decision": "HOLD",
        "current_market_snapshot": {
            "close": 10.5,
            "currency": "CNY",
            "latest_trading_date": "2026-06-27",
        },
    }
    snapshot = store.update_from_state("600353.SS", "2026-06-28", state)

    assert snapshot["analysis_count"] == 1
    assert store.load("600353.SS")["reports"]["fundamentals_report"] == "fundamentals"
    assert store.load("600353.SS")["market_snapshot"]["close"] == 10.5
    assert store.load("000001.SZ", import_reports=False) is None
    assert not store.path_for("600353.SS").with_suffix(".tmp").exists()


def test_position_context_is_not_stored_as_ticker_memory_field(tmp_path):
    store = make_store(tmp_path)
    snapshot = store.update_from_state(
        "600353.SS",
        "2026-06-30",
        {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "fundamentals_report": "fundamentals",
            "final_trade_decision": "HOLD",
            "position_context": {
                "status": "held",
                "quantity": 3800,
                "average_cost": 1.932,
            },
        },
    )

    assert "position_context" not in snapshot
    assert "position_context" not in json.loads(
        store.path_for("600353.SS").read_text(encoding="utf-8")
    )


def test_incremental_update_keeps_reused_fundamentals(tmp_path):
    store = make_store(tmp_path)
    previous = store.update_from_state(
        "600353.SS",
        "2026-06-01",
        {
            "market_report": "old market",
            "sentiment_report": "old sentiment",
            "news_report": "old news",
            "fundamentals_report": "cached fundamentals",
            "final_trade_decision": "HOLD",
        },
    )
    current = store.update_from_state(
        "600353.SS",
        "2026-06-10",
        {
            "market_report": "new market",
            "sentiment_report": "new sentiment",
            "news_report": "new news",
            "fundamentals_report": "cached fundamentals",
            "final_trade_decision": "BUY",
        },
        previous,
    )

    assert current["analysis_count"] == 2
    assert current["fundamentals_as_of"] == "2026-06-01"
    assert current["reports"]["fundamentals_report"] == "cached fundamentals"


def test_fundamentals_ttl(tmp_path):
    store = make_store(tmp_path, ttl=30)
    snapshot = {
        "reports": {"fundamentals_report": "fundamentals"},
        "fundamentals_as_of": "2026-06-01",
    }
    assert store.fundamentals_fresh(snapshot, "2026-06-30")
    assert not store.fundamentals_fresh(snapshot, "2026-07-02")


def test_prior_context_marks_old_prices_as_historical(tmp_path):
    store = make_store(tmp_path)
    context = store.prior_context({
        "last_analysis_date": "2026-06-28",
        "market_snapshot": {
            "close": 2.05,
            "currency": "CNY",
            "latest_trading_date": "2026-06-27",
        },
        "reports": {
            "final_trade_decision": "当前价格¥2.05，建议减仓",
        },
    })

    assert "HISTORICAL SAME-TICKER MEMORY" in context
    assert "Previous verified close: 2.05 CNY on 2026-06-27" in context
    assert "<historical_decision>" in context


def test_invalid_json_is_ignored(tmp_path):
    store = make_store(tmp_path)
    path = store.path_for("600353.SS")
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    assert store.load("600353.SS") is None


def test_imports_latest_legacy_report(tmp_path):
    store = make_store(tmp_path)
    report = tmp_path / "reports" / "600353.SS_20260627_141703"
    (report / "1_analysts").mkdir(parents=True)
    (report / "5_portfolio").mkdir(parents=True)
    (report / "1_analysts" / "fundamentals.md").write_text("legacy fundamentals")
    (report / "1_analysts" / "market.md").write_text("legacy market")
    (report / "5_portfolio" / "decision.md").write_text("legacy HOLD")

    snapshot = store.load("600353.SS")

    assert snapshot["last_analysis_date"] == "2026-06-27"
    assert snapshot["reports"]["fundamentals_report"] == "legacy fundamentals"
    saved = json.loads(store.path_for("600353.SS").read_text())
    assert saved["imported_from"].endswith("600353.SS_20260627_141703")


def test_imports_latest_nested_report(tmp_path):
    store = make_store(tmp_path)
    report = tmp_path / "reports" / "600353.SS" / "20260628_091530"
    (report / "1_analysts").mkdir(parents=True)
    (report / "5_portfolio").mkdir(parents=True)
    (report / "1_analysts" / "fundamentals.md").write_text("nested fundamentals")
    (report / "5_portfolio" / "decision.md").write_text("nested HOLD")

    snapshot = store.load("600353.SS")

    assert snapshot["last_analysis_date"] == "2026-06-28"
    assert snapshot["reports"]["fundamentals_report"] == "nested fundamentals"
    assert snapshot["imported_from"].endswith("600353.SS/20260628_091530")
