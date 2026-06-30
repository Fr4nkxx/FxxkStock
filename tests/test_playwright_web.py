"""Tests for browser/CDP fetchers (mocked render_html, no real Chrome)."""

import copy
from datetime import datetime
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.eastmoney_browser import (
    _parse_guba_from_html,
    _parse_news_from_html,
    fetch_browser_guba,
    get_browser_global_news,
    get_browser_news,
)
from fxxkstock.dataflows.errors import NoMarketDataError


_SAMPLE_NEWS_HTML = """
<html><body>
  <div class="news-item">
    <a href="https://finance.eastmoney.com/a/abc.html" title="Moutai earnings beat">Moutai earnings beat</a>
    <span class="time">2025-06-01 10:00:00</span>
    <p class="content">Summary line</p>
  </div>
  <div class="news-item">
    <a href="https://finance.eastmoney.com/a/old.html" title="Old headline">Old headline</a>
    <span class="time">2024-01-01 08:00:00</span>
  </div>
</body></html>
"""

_CURRENT_EASTMONEY_NEWS_HTML = """
<html><body>
  <div class="news_item">
    <div class="news_item_t">
      <a href="https://finance.eastmoney.com/a/current.html">
        央行降息观察
      </a>
    </div>
    <div class="news_item_c">
      <span>政策观察摘要</span>
      <span>东方财富网</span>
    </div>
    <div class="news_item_time">2026-06-29 15:39:00 -</div>
  </div>
</body></html>
"""

_SAMPLE_GUBA_HTML = """
<html><body>
  <div class="articleh">
    <span class="l3"><a>Bullish on this stock</a></span>
    <span class="l5">06-01 12:00</span>
    <span class="l1">100</span>
    <span class="l2">5</span>
  </div>
</body></html>
"""

_CURRENT_GUBA_HTML = """
<html><body><script>
var article_list={"re":[{
  "post_title":"太极实业还能继续持有吗",
  "post_content":"今天放量上涨，短期仍需注意追高风险。",
  "post_publish_time":"2026-06-29 10:00:00",
  "post_last_time":"2026-06-29 12:00:00",
  "post_click_count":1200,
  "post_comment_count":18,
  "post_id":1730000001,
  "post_guba":{"stockbar_external_code":"600667"},
  "reply_list":[{"reply_text":"看多但不追高"}]
}]};
</script></body></html>
"""

_CURRENT_XUEQIU_HTML = """
<html><body>
  <article class="timeline__item">
    <a class="user-name" href="/123">雪球用户甲</a>
    <a class="date-and-source" href="/123/456">30分钟前<span>· 来自Android</span></a>
    <div class="timeline__item__content">
      <div class="content content--description">
        <a href="/S/SH600667">$太极实业(SH600667)$</a>
        高位放量，需要警惕追高风险。
      </div>
    </div>
    <div class="timeline__item__ft"><span>讨论 12</span><span>赞 8</span></div>
  </article>
  <article class="timeline__item">
    <div class="timeline__item__content">
      <div class="content content--description">
        <a href="/S/SH600667">$太极实业(SH600667)$</a>
      </div>
    </div>
  </article>
</body></html>
"""


@pytest.mark.unit
def test_parse_news_from_html_extracts_articles():
    articles = _parse_news_from_html(_SAMPLE_NEWS_HTML)
    titles = [a["title"] for a in articles]
    assert "Moutai earnings beat" in titles
    assert articles[0]["pub_date"] == datetime(2025, 6, 1, 10, 0, 0)


@pytest.mark.unit
def test_parse_current_eastmoney_news_item_markup():
    articles = _parse_news_from_html(_CURRENT_EASTMONEY_NEWS_HTML)

    assert len(articles) == 1
    assert articles[0]["title"] == "央行降息观察"
    assert articles[0]["summary"] == "政策观察摘要 东方财富网"
    assert articles[0]["pub_date"] == datetime(2026, 6, 29, 15, 39, 0)


@pytest.mark.unit
def test_get_browser_news_filters_look_ahead_window():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    with patch(
        "fxxkstock.dataflows.eastmoney_browser.render_html",
        return_value=_SAMPLE_NEWS_HTML,
    ):
        out = get_browser_news("600519.SS", "2025-05-25", "2025-06-05")

    assert "Moutai earnings beat" in out
    assert "Old headline" not in out
    assert "Browser/CDP" in out


@pytest.mark.unit
def test_get_browser_news_raises_no_data_when_empty():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    with patch(
        "fxxkstock.dataflows.eastmoney_browser.render_html",
        return_value="<html><body></body></html>",
    ):
        with pytest.raises(NoMarketDataError):
            get_browser_news("600519.SS", "2025-05-25", "2025-06-05")


@pytest.mark.unit
def test_get_browser_global_news_with_mock():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config(
        {
            "market_region": "cn_a",
            "cn_global_news_queries": ["央行"],
            "cn_global_news_article_limit": 5,
        }
    )
    with patch(
        "fxxkstock.dataflows.eastmoney_browser.render_html",
        return_value=_SAMPLE_NEWS_HTML,
    ):
        out = get_browser_global_news("2025-06-05", look_back_days=7, limit=5)

    assert "Moutai earnings beat" in out
    assert "Browser/CDP" in out


@pytest.mark.unit
def test_parse_guba_from_html():
    posts = _parse_guba_from_html(_SAMPLE_GUBA_HTML)
    assert len(posts) == 1
    assert posts[0]["title"] == "Bullish on this stock"
    assert posts[0]["read_count"] == "100"


@pytest.mark.unit
def test_parse_current_eastmoney_embedded_guba_list():
    posts = _parse_guba_from_html(_CURRENT_GUBA_HTML)

    assert len(posts) == 1
    assert posts[0]["title"] == "太极实业还能继续持有吗"
    assert posts[0]["read_count"] == 1200
    assert "今天放量上涨" in posts[0]["summary"]
    assert "看多但不追高" in posts[0]["summary"]
    assert posts[0]["link"].endswith("1730000001.html")


@pytest.mark.unit
def test_parse_current_xueqiu_content_and_ignore_tag_only_rows():
    posts = _parse_guba_from_html(_CURRENT_XUEQIU_HTML)

    assert len(posts) == 1
    assert "高位放量，需要警惕追高风险" in posts[0]["summary"]
    assert posts[0]["author"] == "雪球用户甲"
    assert posts[0]["comment_count"] == 12
    assert posts[0]["link"] == "https://xueqiu.com/123/456"


@pytest.mark.unit
def test_fetch_browser_guba_returns_formatted_block():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_browser_guba_source": "eastmoney"})
    with patch(
        "fxxkstock.dataflows.eastmoney_browser.render_html",
        return_value=_SAMPLE_GUBA_HTML,
    ):
        out = fetch_browser_guba("600519.SS", limit=5)

    assert "Bullish on this stock" in out
    assert "East Money Guba" in out
