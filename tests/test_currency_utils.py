"""Tests for currency detection and FX conversion."""

import copy
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.currency_utils import (
    clear_fx_cache,
    convert_to_cny,
    detect_source_currency,
    format_fx_header_line,
    format_money_cny,
    get_fx_to_cny,
    get_instrument_fx_context,
    scale_price_for_display,
)


@pytest.fixture(autouse=True)
def _reset_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    clear_fx_cache()
    yield
    clear_fx_cache()


@pytest.mark.unit
def test_detect_source_currency_from_identity():
    assert detect_source_currency("NVDA", {"currency": "USD"}) == "USD"


@pytest.mark.unit
def test_detect_source_currency_suffix_fallback():
    assert detect_source_currency("600519.SS", {}) == "CNY"
    assert detect_source_currency("0700.HK", {}) == "HKD"
    assert detect_source_currency("NVDA", {}) == "USD"


@pytest.mark.unit
def test_get_fx_to_cny_cny_is_one():
    assert get_fx_to_cny("CNY", "2026-06-26") == 1.0


@pytest.mark.unit
def test_get_fx_to_cny_override_priority():
    set_config({"fx_rate_override": {"USD": 7.25}})
    assert get_fx_to_cny("USD", "2026-06-26") == 7.25


@pytest.mark.unit
def test_get_fx_to_cny_yfinance_cached():
    with patch(
        "fxxkstock.dataflows.currency_utils.yf_retry",
        return_value=__import__("pandas").DataFrame(
            {"Close": [7.2]},
            index=[__import__("pandas").Timestamp("2026-06-25")],
        ),
    ):
        rate = get_fx_to_cny("USD", "2026-06-26")
    assert rate == 7.2


@pytest.mark.unit
def test_get_fx_to_cny_fail_open_returns_none():
    with patch(
        "fxxkstock.dataflows.currency_utils.yf_retry",
        side_effect=RuntimeError("network"),
    ):
        assert get_fx_to_cny("USD", "2026-06-26") is None


@pytest.mark.unit
def test_convert_and_format_money_cny():
    assert convert_to_cny(100, 7.2) == 720.0
    assert format_money_cny(720.123) == "720.12 CNY"


@pytest.mark.unit
def test_format_fx_header_line():
    line = format_fx_header_line("USD", 7.23, "2026-06-26")
    assert "1 USD = 7.2300 CNY" in line
    assert "USDCNY=X" in line


@pytest.mark.unit
def test_scale_price_for_display():
    assert scale_price_for_display(10.0, 7.0, "USD") == "70.00 CNY"
    assert scale_price_for_display(10.0, None, "CNY") == "10.00"
    assert scale_price_for_display(72.5, None, "USD") == "72.50"


@pytest.mark.unit
def test_get_instrument_fx_context_cny():
    with patch(
        "fxxkstock.dataflows.currency_utils._quick_identity_for_currency",
        side_effect=AssertionError("A-share CNY detection should not call yfinance"),
    ):
        ccy, rate = get_instrument_fx_context("600519.SS", "2026-06-26")
    assert ccy == "CNY"
    assert rate == 1.0
