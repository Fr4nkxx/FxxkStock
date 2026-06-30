"""Tests for yfinance insider and financial statement CNY conversion."""

import copy
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.currency_utils import (
    convert_financial_frame,
    convert_insider_frame,
    is_share_count_row,
)
from fxxkstock.dataflows.errors import NoMarketDataError
from fxxkstock.dataflows.y_finance import (
    get_balance_sheet,
    get_insider_transactions,
)


@pytest.fixture(autouse=True)
def _reset_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    yield


@pytest.mark.unit
def test_is_share_count_row():
    assert is_share_count_row("Ordinary Shares Number")
    assert is_share_count_row("Basic Average Shares")
    assert not is_share_count_row("Total Revenue")


@pytest.mark.unit
def test_convert_insider_frame_value_only():
    df = pd.DataFrame({
        "Shares": [1000, 2000],
        "Value": [10.0, 20.0],
        "Price": [0.01, 0.01],
    })
    out = convert_insider_frame(df, 7.0, "USD")
    assert out["Shares"].tolist() == [1000, 2000]
    assert out["Value"].tolist() == [70.0, 140.0]
    assert out["Price"].tolist() == [0.07, 0.07]


@pytest.mark.unit
def test_convert_financial_frame_skips_share_rows():
    df = pd.DataFrame(
        {"2025-12-31": [1_000_000.0, 500.0]},
        index=["Total Revenue", "Ordinary Shares Number"],
    )
    out = convert_financial_frame(df, 7.0)
    assert out.loc["Total Revenue", "2025-12-31"] == 7_000_000.0
    assert out.loc["Ordinary Shares Number", "2025-12-31"] == 500.0


@pytest.mark.unit
def test_get_insider_transactions_raises_for_cn_region():
    set_config({"market_region": "cn_a"})
    with pytest.raises(NoMarketDataError, match="yfinance insider disabled"):
        get_insider_transactions("603629.SS")


@pytest.mark.unit
def test_get_insider_transactions_converts_usd_value():
    insider = pd.DataFrame({
        "Insider": ["Alice"],
        "Shares": [1000],
        "Value": [24.74],
    })
    with (
        patch("fxxkstock.dataflows.y_finance.normalize_symbol", return_value="NVDA"),
        patch("fxxkstock.dataflows.y_finance.yf.Ticker") as mock_ticker,
        patch("fxxkstock.dataflows.y_finance.yf_retry", side_effect=lambda f: f()),
        patch(
            "fxxkstock.dataflows.y_finance.get_instrument_fx_context",
            return_value=("USD", 7.0),
        ),
        patch("fxxkstock.dataflows.y_finance.get_config") as mock_cfg,
    ):
        mock_cfg.return_value = {"market_region": "default"}
        mock_ticker.return_value.insider_transactions = insider
        out = get_insider_transactions("NVDA")

    assert "FX: 1 USD = 7.0000 CNY" in out
    assert "1000" in out
    assert "173.18" in out or "173.180000" in out


@pytest.mark.unit
def test_get_balance_sheet_usd_converts_amounts_not_shares():
    dates = pd.to_datetime(["2025-12-31"])
    bs = pd.DataFrame(
        {
            dates[0]: [1_000_000.0, 500.0],
        },
        index=["Total Assets", "Ordinary Shares Number"],
    )
    with (
        patch("fxxkstock.dataflows.y_finance.normalize_symbol", return_value="NVDA"),
        patch("fxxkstock.dataflows.y_finance.yf.Ticker") as mock_ticker,
        patch("fxxkstock.dataflows.y_finance.yf_retry", side_effect=lambda f: f()),
        patch(
            "fxxkstock.dataflows.y_finance.filter_financials_by_date",
            side_effect=lambda d, c: d,
        ),
        patch(
            "fxxkstock.dataflows.y_finance.get_fx_to_cny",
            return_value=7.0,
        ),
        patch("fxxkstock.dataflows.y_finance.get_config") as mock_cfg,
    ):
        mock_cfg.return_value = {"market_region": "default"}
        mock_ticker.return_value.quarterly_balance_sheet = bs
        mock_ticker.return_value.info = {"financialCurrency": "USD"}
        out = get_balance_sheet("NVDA", freq="quarterly", curr_date="2026-06-26")

    assert "FX: 1 USD = 7.0000 CNY" in out
    parsed = pd.read_csv(StringIO(out.split("\n\n", 1)[1]), index_col=0)
    assert parsed.loc["Total Assets"].iloc[0] == 7_000_000.0
    assert parsed.loc["Ordinary Shares Number"].iloc[0] == 500.0


@pytest.mark.unit
def test_get_balance_sheet_cny_label_only():
    dates = pd.to_datetime(["2025-12-31"])
    bs = pd.DataFrame(
        {dates[0]: [1_000_000.0]},
        index=["Total Assets"],
    )
    with (
        patch(
            "fxxkstock.dataflows.y_finance.normalize_symbol",
            return_value="603629.SS",
        ),
        patch("fxxkstock.dataflows.y_finance.yf.Ticker") as mock_ticker,
        patch("fxxkstock.dataflows.y_finance.yf_retry", side_effect=lambda f: f()),
        patch(
            "fxxkstock.dataflows.y_finance.filter_financials_by_date",
            side_effect=lambda d, c: d,
        ),
        patch("fxxkstock.dataflows.y_finance.get_config") as mock_cfg,
    ):
        mock_cfg.return_value = {"market_region": "cn_a"}
        mock_ticker.return_value.quarterly_balance_sheet = bs
        mock_ticker.return_value.info = {"financialCurrency": "CNY"}
        out = get_balance_sheet("603629.SS", freq="quarterly", curr_date="2026-06-26")

    assert "Currency: CNY" in out
    assert "FX: 1" not in out
    parsed = pd.read_csv(StringIO(out.split("\n\n", 1)[1]), index_col=0)
    assert parsed.loc["Total Assets"].iloc[0] == 1_000_000.0
