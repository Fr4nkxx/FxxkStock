"""yfinance treats ``end`` as exclusive; we must request one extra day so the
requested end_date (and the current day) is actually included.

Regressions for #986 (current-day OHLCV excluded) and #987 (requested end_date
row omitted).
"""
import pandas as pd
import pytest

import fxxkstock.dataflows.stockstats_utils as su
import fxxkstock.dataflows.y_finance as yfin
from fxxkstock.dataflows.config import set_config


@pytest.mark.unit
def test_get_yfin_requests_inclusive_end(monkeypatch):
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, start, end):
            captured["start"] = start
            captured["end"] = end
            idx = pd.to_datetime(["2025-05-08", "2025-05-09"])
            return pd.DataFrame(
                {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                 "Close": [1.0, 2.0], "Volume": [1, 2]},
                index=idx,
            )

    monkeypatch.setattr(yfin.yf, "Ticker", FakeTicker)
    out = yfin.get_YFin_data_online("AAPL", "2025-05-01", "2025-05-09")

    # end is requested one day past end_date so 2025-05-09 is included (#987).
    assert captured["end"] == "2025-05-10"
    # Header still reflects the requested range, not the internal +1 day.
    assert "to 2025-05-09" in out


@pytest.mark.unit
def test_get_yfin_drops_incomplete_requested_date(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, start, end):
            return pd.DataFrame(
                {
                    "Open": [2.21, None],
                    "High": [2.21, None],
                    "Low": [2.13, None],
                    "Close": [2.16, None],
                    "Volume": [301_996_607, 312_864_807],
                },
                index=pd.to_datetime(["2026-06-26", "2026-06-29"]),
            )

    monkeypatch.setattr(yfin.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(
        yfin,
        "get_instrument_fx_context",
        lambda symbol, as_of: ("CNY", 1.0),
    )

    out = yfin.get_YFin_data_online("513100.SS", "2026-06-01", "2026-06-29")

    assert "2026-06-29,,,,," not in out
    assert "latest complete OHLCV row is 2026-06-26" in out
    assert "Do not infer a holiday" in out


@pytest.mark.unit
def test_load_ohlcv_requests_inclusive_end(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    captured = {}

    def fake_download(symbol, start, end, **kwargs):
        captured["end"] = end
        idx = pd.to_datetime([pd.Timestamp.today().normalize()])
        return pd.DataFrame(
            {"Open": [100.0], "High": [100.0], "Low": [100.0],
             "Close": [100.0], "Volume": [1]},
            index=idx,
        )

    monkeypatch.setattr(su.yf, "download", fake_download)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    su.load_ohlcv("AAPL", today)

    expected_end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert captured["end"] == expected_end  # tomorrow -> today's row included (#986)
