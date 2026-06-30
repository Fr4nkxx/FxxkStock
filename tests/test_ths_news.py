"""Tests for browser-backed Tonghuashun ticker news."""

import copy
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.ths_news import (
    get_browser_ths_news,
    parse_ths_news_html,
)


_SAMPLE_HTML = """
<html><body>
  <a href="https://news.10jqka.com.cn/20260629/c678901234.shtml">
    公司发布最新经营公告
  </a>
  <a href="https://news.10jqka.com.cn/20260620/c678901235.shtml">
    较早的一条公司新闻
  </a>
  <a href="https://example.com/not-news">无关链接</a>
</body></html>
"""


@pytest.mark.unit
def test_parse_ths_news_html_extracts_dated_articles():
    articles = parse_ths_news_html(_SAMPLE_HTML)

    assert len(articles) == 2
    assert articles[0]["title"] == "公司发布最新经营公告"
    assert articles[0]["pub_date"].strftime("%Y-%m-%d") == "2026-06-29"


@pytest.mark.unit
def test_get_browser_ths_news_filters_requested_window():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    with patch(
        "fxxkstock.dataflows.ths_news.render_html",
        return_value=_SAMPLE_HTML,
    ):
        output = get_browser_ths_news(
            "159516.SZ",
            "2026-06-25",
            "2026-06-30",
        )

    assert "公司发布最新经营公告" in output
    assert "较早的一条公司新闻" not in output
    assert "Tonghuashun Browser" in output
