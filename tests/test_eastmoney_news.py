"""Tests for East Money news fetcher."""

import copy
import json
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.eastmoney_news import (
    _parse_jsonp,
    _parse_pub_date,
    get_eastmoney_global_news,
    get_eastmoney_news,
)
from fxxkstock.dataflows.errors import NoMarketDataError


@pytest.mark.unit
def test_parse_jsonp():
    raw = 'jQuery({"result": {"cmsArticleWebOld": []}})'
    data = _parse_jsonp(raw)
    assert "result" in data


@pytest.mark.unit
def test_parse_pub_date_string():
    dt = _parse_pub_date("2025-06-01 10:00:00")
    assert dt is not None
    assert dt.year == 2025


@pytest.mark.unit
def test_get_eastmoney_news_formats_output():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    fake_jsonp = json.dumps(
        {
            "result": {
                "cmsArticleWebOld": [
                    {
                        "title": "茅台业绩超预期",
                        "content": "摘要内容",
                        "mediaName": "证券时报",
                        "url": "https://example.com/1",
                        "date": "2025-06-01 09:00:00",
                    }
                ]
            }
        }
    )

    with patch(
        "fxxkstock.dataflows.eastmoney_news._stock_announcement_news",
        return_value=[],
    ), patch(
        "fxxkstock.dataflows.eastmoney_news._search_news",
        return_value=[
            {
                "title": "茅台业绩超预期",
                "summary": "摘要内容",
                "publisher": "证券时报",
                "link": "https://example.com/1",
                "pub_date": _parse_pub_date("2025-06-01 09:00:00"),
            }
        ],
    ):
        out = get_eastmoney_news("600519.SS", "2025-05-25", "2025-06-05")

    assert "茅台业绩超预期" in out
    assert "East Money" in out


@pytest.mark.unit
def test_get_eastmoney_news_raises_no_data():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    with patch(
        "fxxkstock.dataflows.eastmoney_news._fetch_stock_articles",
        return_value=[],
    ):
        with pytest.raises(NoMarketDataError):
            get_eastmoney_news("600519.SS", "2025-05-25", "2025-06-05")


@pytest.mark.unit
def test_get_eastmoney_global_news_cn_queries():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    with patch(
        "fxxkstock.dataflows.eastmoney_news._search_news",
        return_value=[
            {
                "title": "央行降息",
                "summary": "",
                "publisher": "新华社",
                "link": "",
                "pub_date": _parse_pub_date("2025-06-01 08:00:00"),
            }
        ],
    ):
        out = get_eastmoney_global_news("2025-06-05", look_back_days=7, limit=5)

    assert "央行降息" in out
    assert "CN Global" in out


@pytest.mark.unit
def test_eastmoney_news_excludes_future_articles():
    """Look-ahead safety: future-dated East Money articles must not leak into backtests."""
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    from fxxkstock.dataflows.eastmoney_news import _parse_pub_date

    future_art = {
        "title": "FUTURE EVENT",
        "summary": "",
        "publisher": "Test",
        "link": "",
        "pub_date": _parse_pub_date("2025-06-20 10:00:00"),
    }
    past_art = {
        "title": "PAST EVENT",
        "summary": "",
        "publisher": "Test",
        "link": "",
        "pub_date": _parse_pub_date("2025-06-01 10:00:00"),
    }

    with patch(
        "fxxkstock.dataflows.eastmoney_news._fetch_stock_articles",
        return_value=[future_art, past_art],
    ):
        out = get_eastmoney_news("600519.SS", "2025-05-25", "2025-06-05")

    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out
