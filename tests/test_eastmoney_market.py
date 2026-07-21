from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.symbol_utils import NoMarketDataError


@pytest.mark.unit
def test_parse_eastmoney_klines_to_ohlcv():
    from fxxkstock.dataflows.eastmoney_market import _parse_klines

    frame = _parse_klines([
        "2026-07-06,53.42,54.19,56.66,53.00,482928,2661689093.46,6.92,2.44,1.29,8.65"
    ])

    assert list(frame[["Open", "High", "Low", "Close", "Volume"]].iloc[0]) == [
        53.42,
        56.66,
        53.0,
        54.19,
        482928,
    ]


@pytest.mark.unit
def test_load_ohlcv_falls_back_to_eastmoney_for_cn_symbol(monkeypatch, tmp_path):
    from fxxkstock.dataflows import eastmoney_market, stockstats_utils

    set_config({
        "data_cache_dir": str(tmp_path),
        "market_region": "cn_a",
        "cn_market_data_source": "eastmoney",
    })
    today = date.today().isoformat()

    monkeypatch.setattr(
        stockstats_utils.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        eastmoney_market,
        "load_eastmoney_ohlcv",
        lambda symbol, curr_date: pd.DataFrame({
            "Date": [pd.Timestamp(today)],
            "Open": [53.42],
            "High": [56.66],
            "Low": [53.0],
            "Close": [54.19],
            "Volume": [482928],
        }),
    )

    data = stockstats_utils.load_ohlcv("002364.SZ", today)

    assert not data.empty
    assert data["Close"].iloc[-1] == 54.19


@pytest.mark.unit
def test_load_ohlcv_prefers_eastmoney_for_cn_symbol(monkeypatch, tmp_path):
    from fxxkstock.dataflows import eastmoney_market, stockstats_utils

    set_config({
        "data_cache_dir": str(tmp_path),
        "market_region": "cn_a",
        "cn_market_data_source": "eastmoney",
    })
    today = date.today().isoformat()

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("A-share OHLCV should prefer Eastmoney before yfinance")

    monkeypatch.setattr(stockstats_utils.yf, "download", fail_yfinance)
    monkeypatch.setattr(
        eastmoney_market,
        "load_eastmoney_ohlcv",
        lambda symbol, curr_date: pd.DataFrame({
            "Date": [pd.Timestamp(today)],
            "Open": [53.42],
            "High": [56.66],
            "Low": [53.0],
            "Close": [54.19],
            "Volume": [482928],
        }),
    )

    data = stockstats_utils.load_ohlcv("002364.SZ", today)

    assert data["Close"].iloc[-1] == 54.19


@pytest.mark.unit
def test_load_ohlcv_preserves_yfinance_path_by_default(monkeypatch, tmp_path):
    from fxxkstock.dataflows import eastmoney_market, stockstats_utils

    set_config({
        "data_cache_dir": str(tmp_path),
        "market_region": "cn_a",
        "cn_market_data_source": "yfinance",
    })
    today = date.today().isoformat()

    monkeypatch.setattr(
        stockstats_utils.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    def fail_eastmoney(symbol, curr_date):
        raise AssertionError("Eastmoney should be opt-in for CN OHLCV")

    monkeypatch.setattr(eastmoney_market, "load_eastmoney_ohlcv", fail_eastmoney)

    with pytest.raises(NoMarketDataError, match="Yahoo Finance returned no rows"):
        stockstats_utils.load_ohlcv("002364.SZ", today)


@pytest.mark.unit
def test_eastmoney_refresh_failure_promotes_recent_cache(monkeypatch, tmp_path):
    from fxxkstock.dataflows import eastmoney_market

    set_config({"data_cache_dir": str(tmp_path)})
    today = date.today()
    previous_day = today - timedelta(days=1)
    cache_dir = tmp_path / "eastmoney_ohlcv"
    cache_dir.mkdir()
    legacy_cache = cache_dir / (
        "1.600353-EM-data-2021-07-13-2026-07-13.csv"
    )
    pd.DataFrame(
        {
            "Date": [previous_day.isoformat()],
            "Open": [43.04],
            "High": [45.23],
            "Low": [41.2],
            "Close": [41.27],
            "Volume": [227080],
        }
    ).to_csv(legacy_cache, index=False)

    calls = 0

    def fail_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise NoMarketDataError("600353.SS", "1.600353", "proxy unavailable")

    monkeypatch.setattr(
        eastmoney_market,
        "_download_eastmoney_ohlcv",
        fail_refresh,
    )

    data = eastmoney_market.load_eastmoney_ohlcv(
        "600353.SS",
        today.isoformat(),
    )

    assert calls == 1
    assert data["Close"].iloc[-1] == 41.27
    start = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    current_cache = eastmoney_market._eastmoney_cache_path(
        "600353.SS",
        start,
        pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    assert os.path.exists(current_cache)

    monkeypatch.setattr(
        eastmoney_market,
        "_download_eastmoney_ohlcv",
        lambda *args, **kwargs: pytest.fail("promoted cache should avoid a retry"),
    )
    second = eastmoney_market.load_eastmoney_ohlcv(
        "600353.SS",
        today.isoformat(),
    )
    assert second["Close"].iloc[-1] == 41.27
