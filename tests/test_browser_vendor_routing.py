"""Tests for browser vendor prepending and fallback in route_to_vendor."""

import copy
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.errors import BrowserUnavailableError, NoMarketDataError
from fxxkstock.dataflows.interface import _build_vendor_chain, route_to_vendor


@pytest.mark.unit
def test_cn_region_prepends_browser_for_get_news():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_browser_enabled": True})

    chain = _build_vendor_chain(
        "get_news",
        explicit=[],
        all_available=["browser", "eastmoney", "yfinance"],
    )
    assert chain[0] == "browser"
    assert "eastmoney" in chain


@pytest.mark.unit
def test_cn_browser_disabled_skips_browser():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_browser_enabled": False})

    chain = _build_vendor_chain(
        "get_news",
        explicit=[],
        all_available=["browser", "eastmoney", "yfinance"],
    )
    assert chain[0] == "eastmoney"
    assert "browser" not in chain


@pytest.mark.unit
def test_browser_unavailable_falls_back_to_eastmoney():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_browser_enabled": True})

    call_order = []

    def fake_browser(ticker, start, end):
        call_order.append("browser")
        raise BrowserUnavailableError("CDP not reachable")

    def fake_eastmoney(ticker, start, end):
        call_order.append("eastmoney")
        return f"news for {ticker}"

    with patch.dict(
        "fxxkstock.dataflows.interface.VENDOR_METHODS",
        {
            "get_news": {
                "browser": fake_browser,
                "eastmoney": fake_eastmoney,
                "yfinance": lambda *a, **k: "yf",
            }
        },
    ):
        out = route_to_vendor("get_news", "600519.SS", "2025-05-25", "2025-06-05")

    assert call_order == ["browser", "eastmoney"]
    assert out == "news for 600519.SS"


@pytest.mark.unit
def test_browser_no_data_falls_back():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    def fake_browser(ticker, start, end):
        raise NoMarketDataError(ticker, ticker, "empty")

    def fake_eastmoney(ticker, start, end):
        return "fallback news"

    with patch.dict(
        "fxxkstock.dataflows.interface.VENDOR_METHODS",
        {
            "get_news": {
                "browser": fake_browser,
                "eastmoney": fake_eastmoney,
            }
        },
    ):
        out = route_to_vendor("get_news", "600519.SS", "2025-05-25", "2025-06-05")

    assert out == "fallback news"


@pytest.mark.unit
def test_guba_dispatch_falls_back_to_http(monkeypatch):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_browser_enabled": True})

    from fxxkstock.dataflows import eastmoney_guba

    def fake_browser_guba(ticker, limit=None):
        return "<no browser guba posts found for TEST>"

    def fake_json(code, limit):
        return [{"title": "HTTP post", "created": "?", "read_count": None, "comment_count": None, "source": "json"}]

    monkeypatch.setattr(
        "fxxkstock.dataflows.eastmoney_browser.fetch_browser_guba",
        fake_browser_guba,
    )
    monkeypatch.setattr(eastmoney_guba, "_fetch_guba_json", fake_json)

    out = eastmoney_guba.fetch_eastmoney_guba("600519.SS", limit=5)
    assert "HTTP post" in out
