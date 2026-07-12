from __future__ import annotations

from datetime import date

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
