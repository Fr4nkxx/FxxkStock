"""Tests for per-run market_region injection on graph entry points."""

import copy

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import get_config, set_config
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph


@pytest.mark.unit
def test_inject_market_region_sets_cn_a_for_sse_ticker():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    graph = FxxKStockGraph.__new__(FxxKStockGraph)
    graph.config = get_config()

    region = graph.inject_market_region("603678.SS")

    assert region == "cn_a"
    assert get_config()["market_region"] == "cn_a"
    assert graph.config["market_region"] == "cn_a"


@pytest.mark.unit
def test_inject_market_region_leaves_us_default():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    graph = FxxKStockGraph.__new__(FxxKStockGraph)
    graph.config = get_config()

    region = graph.inject_market_region("AAPL")

    assert region == "default"
    assert get_config()["market_region"] == "default"
