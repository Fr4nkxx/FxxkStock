"""Tests for CN-market vendor chain prepending in route_to_vendor."""

import copy
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.interface import route_to_vendor


@pytest.mark.unit
def test_cn_region_prepends_eastmoney_for_get_news():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    call_order = []

    def fake_eastmoney(ticker, start, end):
        call_order.append("eastmoney")
        raise Exception("eastmoney failed")

    def fake_yfinance(ticker, start, end):
        call_order.append("yfinance")
        return f"news for {ticker}"

    with patch.dict(
        "fxxkstock.dataflows.interface.VENDOR_METHODS",
        {
            "get_news": {
                "eastmoney": fake_eastmoney,
                "yfinance": fake_yfinance,
            }
        },
    ):
        out = route_to_vendor("get_news", "600519.SS", "2025-05-25", "2025-06-05")

    assert call_order == ["eastmoney", "yfinance"]
    assert "600519.SS" in out


@pytest.mark.unit
def test_default_region_skips_eastmoney():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "default"})

    call_order = []

    def fake_eastmoney(ticker, start, end):
        call_order.append("eastmoney")
        return "cn news"

    def fake_yfinance(ticker, start, end):
        call_order.append("yfinance")
        return "us news"

    with patch.dict(
        "fxxkstock.dataflows.interface.VENDOR_METHODS",
        {
            "get_news": {
                "eastmoney": fake_eastmoney,
                "yfinance": fake_yfinance,
            }
        },
    ):
        out = route_to_vendor("get_news", "AAPL", "2025-05-25", "2025-06-05")

    assert call_order == ["yfinance"]
    assert out == "us news"


@pytest.mark.unit
def test_eastmoney_no_data_falls_back():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    from fxxkstock.dataflows.errors import NoMarketDataError

    def fake_eastmoney(ticker, start, end):
        raise NoMarketDataError(ticker, ticker, "empty")

    def fake_yfinance(ticker, start, end):
        return "fallback news"

    with patch.dict(
        "fxxkstock.dataflows.interface.VENDOR_METHODS",
        {
            "get_news": {
                "eastmoney": fake_eastmoney,
                "yfinance": fake_yfinance,
            }
        },
    ):
        out = route_to_vendor("get_news", "600519.SS", "2025-05-25", "2025-06-05")

    assert out == "fallback news"
