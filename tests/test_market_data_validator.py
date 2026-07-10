"""Tests for the deterministic market-data verification snapshot (#830/#881)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import fxxkstock.dataflows.market_data_validator as validator


@pytest.fixture(autouse=True)
def _cny_fx_context(monkeypatch):
    """默认按 CNY 标的测试，避免单测依赖外网汇率。"""
    monkeypatch.setattr(
        validator,
        "get_instrument_fx_context",
        lambda s, d: ("CNY", 1.0),
    )
    monkeypatch.setattr(validator, "fetch_latest_market_quote", lambda s: None)


def _sample_ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-05-20")
    closes = [100 + i for i in range(len(dates))]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i for i in range(len(dates))],
    })


@pytest.mark.unit
class TestVerifiedSnapshot:
    def test_excludes_future_rows(self, monkeypatch):
        data = pd.concat([
            _sample_ohlcv(),
            pd.DataFrame({"Date": [pd.Timestamp("2026-06-01")], "Open": [999.0],
                          "High": [999.0], "Low": [999.0], "Close": [999.0], "Volume": [999]}),
        ], ignore_index=True)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)

        snap = validator.build_verified_market_snapshot("COF", "2026-05-13")
        assert "Verified market data snapshot for COF" in snap
        assert "Requested analysis date: 2026-05-13" in snap
        assert "Latest trading row used: 2026-05-13" in snap
        assert "999.00" not in snap          # future row excluded
        assert "999 CNY" not in snap
        assert "boll_lb" in snap             # indicators present

    def test_uses_previous_trading_day_when_date_is_weekend(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        # 2026-05-16 is a Saturday; latest row should be Fri 2026-05-15
        snap = validator.build_verified_market_snapshot("COF", "2026-05-16")
        assert "Latest trading row used: 2026-05-15" in snap
        assert "Recent verified closes" in snap

    def test_raises_when_no_rows_on_or_before_date(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2020-01-01")

    def test_raises_on_empty_data(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2026-05-13")

    def test_look_back_window_capped_at_30(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        snap = validator.build_verified_market_snapshot("COF", "2026-05-20", look_back_days=999)
        # last-N closes table has at most 30 data rows
        close_rows = [ln for ln in snap.splitlines() if ln.startswith("| 2026-")]
        assert 0 < len(close_rows) <= 30

    def test_structured_snapshot_ignores_partial_latest_row(self, monkeypatch):
        data = pd.concat([
            _sample_ohlcv(),
            pd.DataFrame({
                "Date": [pd.Timestamp("2026-05-21")],
                "Open": [None],
                "High": [None],
                "Low": [None],
                "Close": [None],
                "Volume": [123],
            }),
        ], ignore_index=True)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)

        snapshot = validator.build_current_market_snapshot_data("COF", "2026-05-21")

        assert snapshot["latest_trading_date"] == "2026-05-20"
        assert snapshot["close"] == _sample_ohlcv()["Close"].iloc[-1]
        assert snapshot["price_basis"] == "latest_complete_ohlcv"

    def test_structured_snapshot_prefers_latest_quote_for_current_date(self, monkeypatch):
        today = date.today().isoformat()
        data = pd.DataFrame({
            "Date": [pd.Timestamp(today)],
            "Open": [99.0],
            "High": [101.0],
            "Low": [98.0],
            "Close": [100.0],
            "Volume": [1_000_000],
        })
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)
        monkeypatch.setattr(
            validator,
            "fetch_latest_market_quote",
            lambda s: {
                "source": "test",
                "symbol": s,
                "currency": "CNY",
                "last_price": 102.25,
                "open": 100.5,
                "high": 103.0,
                "low": 99.8,
                "previous_close": 100.0,
                "as_of": f"{today}T10:32:00",
            },
        )

        snapshot = validator.build_current_market_snapshot_data("COF", today)

        assert snapshot["price_basis"] == "latest_quote"
        assert snapshot["close"] == 102.25
        assert snapshot["latest_complete_ohlcv_close"] == 100.0
        assert snapshot["latest_quote_source"] == "test"

    def test_verified_snapshot_documents_latest_quote_when_available(self, monkeypatch):
        today = date.today().isoformat()
        data = pd.DataFrame({
            "Date": [pd.Timestamp(today)],
            "Open": [99.0],
            "High": [101.0],
            "Low": [98.0],
            "Close": [100.0],
            "Volume": [1_000_000],
        })
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)
        monkeypatch.setattr(
            validator,
            "fetch_latest_market_quote",
            lambda s: {
                "source": "test",
                "currency": "CNY",
                "last_price": 102.25,
                "as_of": f"{today}T10:32:00",
            },
        )

        snap = validator.build_verified_market_snapshot("COF", today)

        assert "Current latest quote" in snap
        assert "102.25 CNY" in snap
        assert "OHLCV row and indicators below remain based on the latest complete trading row" in snap

    def test_detects_only_conflicting_current_price_claims(self):
        snapshot = {"close": 2.16}
        assert validator.find_current_price_conflicts("当前价格为¥2.05", snapshot) == [2.05]
        assert validator.find_current_price_conflicts("当前价格：2.16 CNY", snapshot) == []
        assert validator.find_current_price_conflicts("历史价格为¥2.05", snapshot) == []


@pytest.mark.unit
class TestTool:
    def test_tool_delegates_to_builder(self, monkeypatch):
        from fxxkstock.agents.utils.market_data_validation_tools import (
            get_verified_market_snapshot,
        )
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        out = get_verified_market_snapshot.invoke(
            {"symbol": "COF", "curr_date": "2026-05-20"}
        )
        assert "Verified market data snapshot for COF" in out
