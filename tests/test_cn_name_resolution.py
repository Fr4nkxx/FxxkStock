"""Tests for domestic Chinese company name resolution."""

import json
from unittest.mock import patch

import pytest

from fxxkstock.dataflows.cninfo import get_cn_name
from fxxkstock.dataflows.market_utils import get_security_cn_name
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph


@pytest.mark.unit
def test_get_cn_name_from_cache(tmp_path):
    cache = tmp_path / "cninfo" / "cn_name_map.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"600519": "贵州茅台"}), encoding="utf-8")

    with patch("fxxkstock.dataflows.cninfo.get_config") as mock_cfg:
        mock_cfg.return_value = {
            "data_cache_dir": str(tmp_path),
            "cninfo_cache_ttl_hours": 24,
        }
        assert get_cn_name("600519") == "贵州茅台"


@pytest.mark.unit
def test_get_security_cn_name_a_share_eastmoney_first():
    with patch(
        "fxxkstock.dataflows.market_utils._fetch_eastmoney_cn_name",
        return_value="利通电子",
    ) as mock_em:
        name = get_security_cn_name("603629.SS", "cn_a")
    mock_em.assert_called_once_with("1.603629")
    assert name == "利通电子"


@pytest.mark.unit
def test_get_security_cn_name_a_share_falls_back_to_cninfo():
    with (
        patch(
            "fxxkstock.dataflows.market_utils._fetch_eastmoney_cn_name",
            return_value=None,
        ),
        patch(
            "fxxkstock.dataflows.cninfo.get_cn_name",
            return_value="利通电子",
        ),
    ):
        name = get_security_cn_name("603629.SS", "cn_a")
    assert name == "利通电子"


@pytest.mark.unit
def test_get_security_cn_name_adr_eastmoney():
    with patch(
        "fxxkstock.dataflows.market_utils._fetch_eastmoney_cn_name",
        return_value="阿里巴巴",
    ):
        name = get_security_cn_name("BABA", "cn_adr")
    assert name == "阿里巴巴"


@pytest.mark.unit
def test_resolve_instrument_context_uses_cn_name_and_fx():
    graph = FxxKStockGraph(config={"cn_data_enabled": True})
    with (
        patch(
            "fxxkstock.graph.fxxkstock_graph.resolve_instrument_identity",
            return_value={
                "company_name": "Wrong English Name",
                "currency": "CNY",
                "sector": "Tech",
            },
        ),
        patch(
            "fxxkstock.dataflows.market_utils.detect_market_region",
            return_value="cn_a",
        ),
        patch(
            "fxxkstock.dataflows.market_utils.get_security_cn_name",
            return_value="利通电子",
        ),
    ):
        ctx = graph.resolve_instrument_context("603629.SS", trade_date="2026-06-26")

    assert "Company: 利通电子" in ctx
    assert "Wrong English Name" not in ctx
    assert "never translate, romanize, or rewrite it" in ctx
    assert "Source data quote currency" not in ctx


@pytest.mark.unit
def test_resolve_instrument_context_usd_fx_line():
    graph = FxxKStockGraph()
    with (
        patch(
            "fxxkstock.graph.fxxkstock_graph.resolve_instrument_identity",
            return_value={"company_name": "NVIDIA", "currency": "USD"},
        ),
        patch(
            "fxxkstock.dataflows.market_utils.detect_market_region",
            return_value="default",
        ),
        patch(
            "fxxkstock.dataflows.currency_utils.get_fx_to_cny",
            return_value=7.25,
        ),
    ):
        ctx = graph.resolve_instrument_context("NVDA", trade_date="2026-06-26")

    assert "Source data quote currency: USD" in ctx
    assert "1 USD = 7.2500 CNY" in ctx
