"""Tests for East Money guba fetcher."""

import copy
import json
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.eastmoney_guba import fetch_eastmoney_guba


@pytest.mark.unit
def test_fetch_guba_json_success():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_guba_post_limit": 5})

    payload = json.dumps(
        {
            "re": [
                {
                    "post_title": "茅台还能买吗",
                    "post_id": "1234567890",
                    "post_publish_time": "2025-06-01",
                    "post_click_count": 100,
                    "post_comment_count": 20,
                }
            ]
        }
    ).encode()

    with patch(
        "fxxkstock.dataflows.eastmoney_guba._http_get",
        return_value=payload,
    ):
        out = fetch_eastmoney_guba("600519.SS")

    assert "茅台还能买吗" in out
    assert "reads=100" in out
    assert "Link: https://guba.eastmoney.com/news," in out


@pytest.mark.unit
def test_fetch_guba_adr_unavailable():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_adr"})
    out = fetch_eastmoney_guba("BABA")
    assert "unavailable" in out
