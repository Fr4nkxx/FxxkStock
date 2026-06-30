"""Tests for the experimental NGA Great Times browser diagnostic."""

from datetime import datetime

from scripts.diagnose_data_sources import (
    filter_recent_nga_threads,
    infer_nga_industry_term,
    nga_latest_page_url,
    nga_search_url,
    parse_nga_replies,
    parse_nga_thread_candidates,
)


def test_parse_nga_classic_topic_links():
    html = """
    <table>
      <tbody class="topicrow">
        <tr><td><a class="topic" href="/read.php?tid=12345">科技板块下周怎么看</a></td></tr>
        <tr><td><a href="read.php?tid=67890">人工智能板块是否还能继续持有</a></td></tr>
      </tbody>
    </table>
    """

    threads = parse_nga_thread_candidates(html)

    assert len(threads) == 2
    assert threads[0]["title"] == "科技板块下周怎么看"
    assert threads[0]["link"] == "https://bbs.nga.cn/read.php?tid=12345"


def test_parse_nga_deduplicates_repeated_links():
    html = """
    <a class="topic" href="/read.php?tid=12345">同一个主题帖子</a>
    <a class="topic" href="/read.php?tid=12345">同一个主题帖子</a>
    """

    assert len(parse_nga_thread_candidates(html)) == 1


def test_parse_nga_ignores_reply_links_and_unavailable_topics():
    html = """
    <a class="topic" href="/read.php?tid=12345">人工智能行业怎么看</a>
    <a class="replydate" href="/read.php?tid=12345&page=e">前天 20:23</a>
    <a class="topic" href="/read.php?tid=67890">帖子发布或回复时间超过限制</a>
    <a class="replies" href="/read.php?tid=12345">12</a>
    """

    threads = parse_nga_thread_candidates(html)

    assert len(threads) == 1
    assert threads[0]["title"] == "人工智能行业怎么看"


def test_nga_search_url_uses_chinese_keyword():
    url = nga_search_url(
        "https://bbs.nga.cn/thread.php?fid=706",
        "人工智能ETF易方达",
    )

    assert "fid=706" in url
    assert "key=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BDETF%E6%98%93%E6%96%B9%E8%BE%BE" in url
    assert "159819" not in url


def test_industry_term_prefers_chinese_name_then_identity():
    assert infer_nga_industry_term("人工智能ETF易方达") == "人工智能"
    assert infer_nga_industry_term(
        "利通电子",
        {"industry": "Electronic Components"},
    ) == "电子"


def test_parse_nga_thread_replies_with_author_and_date():
    html = """
    <table>
      <tr id="post1strow0">
        <td id="postauthor0"><a>楼主甲</a></td>
        <td>
          <span id="postdate0">2026-06-29 10:00</span>
          <h3 id="postsubject0">太极实业讨论</h3>
          <div id="postcontent0">主帖认为短期走势偏强。</div>
        </td>
      </tr>
      <tr id="post1strow1">
        <td id="postauthor1"><a>用户乙</a></td>
        <td>
          <span id="postdate1">2026-06-29 10:15</span>
          <div id="postcontent1">回帖认为仍需关注成交量。</div>
        </td>
      </tr>
    </table>
    """

    replies = parse_nga_replies(html)

    assert len(replies) == 2
    assert replies[0]["floor"] == 0
    assert replies[0]["author"] == "楼主甲"
    assert replies[1]["content"] == "回帖认为仍需关注成交量。"


def test_nga_latest_page_url_preserves_thread_id():
    url = nga_latest_page_url("https://bbs.nga.cn/read.php?tid=12345")

    assert "tid=12345" in url
    assert "page=e" in url


def test_filter_recent_nga_threads_uses_last_reply_time():
    threads = [
        {"title": "近期主题", "last_reply": "26-06-29 12:00"},
        {"title": "过期主题", "last_reply": "26-05-01 12:00"},
        {"title": "时间未知", "last_reply": ""},
    ]

    recent = filter_recent_nga_threads(
        threads,
        30,
        now=datetime(2026, 6, 30, 12, 0),
    )

    assert [thread["title"] for thread in recent] == ["近期主题"]
