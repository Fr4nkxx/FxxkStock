"""Tests for the production NGA Great Times sentiment source."""

from datetime import datetime
from unittest.mock import patch

from fxxkstock.dataflows.nga_sentiment import (
    fetch_nga_sentiment,
    filter_recent_threads,
    parse_nga_replies,
    parse_nga_threads,
)


SEARCH_HTML = """
<table>
  <tr>
    <td><a class="topic" href="/read.php?tid=12345">太极实业讨论楼</a></td>
    <td><a class="replydate" title="26-06-29 12:00">昨天</a></td>
  </tr>
</table>
"""

THREAD_HTML = """
<table>
  <tr id="post1strow0">
    <td id="postauthor0">楼主甲</td>
    <td>
      <span id="postdate0">2026-06-29 10:00</span>
      <div id="postcontent0">短期走势偏强，但不建议追高。</div>
    </td>
  </tr>
  <tr id="post1strow1">
    <td id="postauthor1">用户乙</td>
    <td>
      <span id="postdate1">2026-06-29 10:15</span>
      <div id="postcontent1">需要继续观察成交量。</div>
    </td>
  </tr>
</table>
"""


def test_parse_threads_and_recent_filter():
    threads = parse_nga_threads(SEARCH_HTML)

    assert threads[0]["title"] == "太极实业讨论楼"
    assert filter_recent_threads(
        threads,
        lookback_days=30,
        as_of=datetime(2026, 6, 30),
    ) == threads


def test_parse_replies_reads_actual_floor_content():
    replies = parse_nga_replies(THREAD_HTML, limit=20)

    assert len(replies) == 2
    assert replies[0]["author"] == "楼主甲"
    assert replies[1]["content"] == "需要继续观察成交量。"


def test_fetch_nga_sentiment_formats_replies():
    with (
        patch(
            "fxxkstock.dataflows.nga_sentiment.get_security_cn_name",
            return_value="太极实业",
        ),
        patch(
            "fxxkstock.dataflows.nga_sentiment._resolve_industry",
            return_value="半导体",
        ),
        patch(
            "fxxkstock.dataflows.nga_sentiment._render_nga_html",
            side_effect=[SEARCH_HTML, SEARCH_HTML, THREAD_HTML, THREAD_HTML],
        ),
        patch(
            "fxxkstock.dataflows.nga_sentiment.get_config",
            return_value={
                "market_region": "cn_a",
                "cn_nga_lookback_days": 30,
                "cn_nga_min_stock_threads": 3,
                "cn_nga_thread_limit": 1,
                "cn_nga_reply_limit": 20,
                "cn_guba_post_limit": 15,
            },
        ),
    ):
        result = fetch_nga_sentiment("600667.SS", as_of_date="2026-06-30")

    assert "NGA 大时代" in result
    assert "太极实业讨论楼" in result
    assert "短期走势偏强" in result
    assert "需要继续观察成交量" in result


def test_fetch_nga_sentiment_missing_chinese_name_is_normal_empty_result():
    with (
        patch(
            "fxxkstock.dataflows.nga_sentiment.get_security_cn_name",
            return_value=None,
        ),
        patch(
            "fxxkstock.dataflows.nga_sentiment.get_config",
            return_value={"market_region": "cn_a"},
        ),
    ):
        result = fetch_nga_sentiment("159819.SZ")

    assert result.startswith("<no nga posts:")
