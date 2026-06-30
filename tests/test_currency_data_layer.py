"""Tests for CNY display in yfinance fundamentals and verified snapshot."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import fxxkstock.dataflows.market_data_validator as validator
from fxxkstock.dataflows.y_finance import get_fundamentals


@pytest.mark.unit
def test_get_fundamentals_converts_monetary_fields_to_cny():
    fake_info = {
        "longName": "NVIDIA Corporation",
        "sector": "Technology",
        "marketCap": 1000,
        "trailingPE": 30.5,
        "fiftyTwoWeekHigh": 10.0,
        "trailingEps": 2.0,
    }
    with (
        patch("fxxkstock.dataflows.y_finance.normalize_symbol", return_value="NVDA"),
        patch("fxxkstock.dataflows.y_finance.yf.Ticker") as mock_ticker,
        patch("fxxkstock.dataflows.y_finance.yf_retry", side_effect=lambda f: f()),
        patch(
            "fxxkstock.dataflows.y_finance.get_instrument_fx_context",
            return_value=("USD", 7.0),
        ),
    ):
        mock_ticker.return_value.info = fake_info
        out = get_fundamentals("NVDA", curr_date="2026-06-26")

    assert "FX: 1 USD = 7.0000 CNY" in out
    assert "Market Cap: 7000.00 CNY" in out
    assert "52 Week High: 70.00 CNY" in out
    assert "EPS (TTM): 14.00 CNY" in out
    assert "PE Ratio (TTM): 30.5" in out


@pytest.mark.unit
def test_get_fundamentals_uses_cn_name_for_a_share():
    fake_info = {
        "longName": "Jiangsu Lettall Electronic Co.,Ltd",
        "sector": "Technology",
        "marketCap": 100,
        "trailingPE": 30.5,
    }
    with (
        patch(
            "fxxkstock.dataflows.y_finance.normalize_symbol",
            return_value="603629.SS",
        ),
        patch("fxxkstock.dataflows.y_finance.yf.Ticker") as mock_ticker,
        patch("fxxkstock.dataflows.y_finance.yf_retry", side_effect=lambda f: f()),
        patch(
            "fxxkstock.dataflows.y_finance.get_instrument_fx_context",
            return_value=("CNY", 1.0),
        ),
        patch(
            "fxxkstock.dataflows.y_finance.detect_market_region",
            return_value="cn_a",
        ),
        patch(
            "fxxkstock.dataflows.y_finance.get_security_cn_name",
            return_value="利通电子",
        ),
    ):
        mock_ticker.return_value.info = fake_info
        out = get_fundamentals("603629.SS", curr_date="2026-06-26")

    assert "Name: 利通电子" in out
    assert "Lettall" not in out


@pytest.mark.unit
def test_verified_snapshot_displays_cny_prices(monkeypatch):
    dates = pd.bdate_range("2026-05-01", "2026-05-20")
    closes = [100 + i for i in range(len(dates))]
    data = pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * len(dates),
    })
    monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)
    monkeypatch.setattr(
        validator,
        "get_instrument_fx_context",
        lambda s, d: ("USD", 7.0),
    )

    snap = validator.build_verified_market_snapshot("NVDA", "2026-05-20")
    assert "Display currency: CNY" in snap
    assert "FX: 1 USD = 7.0000 CNY" in snap
    # last close 100 USD -> 700 CNY
    assert "700.00 CNY" in snap
    assert "rsi" in snap.lower() or "rsi" in snap
