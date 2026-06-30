"""Tests for market region detection and symbol conversion."""

import copy
import unittest

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.market_utils import (
    detect_market_region,
    is_cn_region,
    to_cninfo_stock_code,
    to_eastmoney_symbol,
    to_guba_code,
)


@pytest.mark.unit
class MarketUtilsTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_a_share_suffix(self):
        assert detect_market_region("600519.SS") == "cn_a"
        assert detect_market_region("000001.SZ") == "cn_a"

    def test_hk_suffix(self):
        assert detect_market_region("0700.HK") == "cn_hk"

    def test_adr_list(self):
        assert detect_market_region("BABA") == "cn_adr"
        assert detect_market_region("AAPL") == "default"

    def test_identity_country_china(self):
        identity = {"country": "China", "exchange": "NYQ"}
        assert detect_market_region("BABA", identity) == "cn_adr"

    def test_identity_hk_exchange(self):
        identity = {"exchange": "HKG"}
        assert detect_market_region("0700", identity) == "cn_hk"

    def test_cn_data_disabled(self):
        set_config({"cn_data_enabled": False})
        assert detect_market_region("600519.SS") == "default"

    def test_is_cn_region(self):
        assert is_cn_region("cn_a") is True
        assert is_cn_region("default") is False

    def test_to_eastmoney_symbol_sh(self):
        em, bare = to_eastmoney_symbol("600519.SS", "cn_a")
        assert em == "1.600519"
        assert bare == "600519"

    def test_to_eastmoney_symbol_sz(self):
        em, bare = to_eastmoney_symbol("000001.SZ", "cn_a")
        assert em == "0.000001"

    def test_to_eastmoney_symbol_hk(self):
        em, bare = to_eastmoney_symbol("0700.HK", "cn_hk")
        assert em.startswith("116.")
        assert bare == "0700"

    def test_to_guba_code(self):
        assert to_guba_code("600519.SS", "cn_a") == "600519"
        assert to_guba_code("0700.HK", "cn_hk") == "hk00700"

    def test_to_cninfo_stock_code(self):
        assert to_cninfo_stock_code("600519.SS") == "600519"
        with pytest.raises(ValueError):
            to_cninfo_stock_code("AAPL")
