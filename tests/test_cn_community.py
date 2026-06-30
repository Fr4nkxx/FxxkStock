"""Tests for CN retail community aggregation (East Money / THS / Taoguba)."""

import copy
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.agents.analysts.sentiment_analyst import _build_cn_system_message
from fxxkstock.dataflows.cn_community import (
    fetch_cn_community,
    fetch_taoguba,
    fetch_ths_guba,
    fetch_xueqiu_community,
    to_taoguba_symbol,
)
from fxxkstock.dataflows.config import set_config

THS_HTML = """
<html><body>
<table class="m_table ggtable">
  <tr><th>标题</th><th>作者</th><th>回复/点击</th><th>最后回复</th></tr>
  <tr>
    <td><a href="https://t.10jqka.com.cn/lgt/post/detail/abc" title="算力还能追吗">算力还能追吗</a></td>
    <td>散户甲</td>
    <td>8/602</td>
    <td>06-26 17:00</td>
  </tr>
</table>
</body></html>
"""

TAOGUBA_HTML = """
<html><body>
<div class="articleh normal_post">
  <span class="l5 a5">2026-06-26 17:13</span>
  <span class="l3 a3"><a href="/a/12345" title="利通电子还能格局吗">利通电子还能格局吗</a></span>
  <span class="l1 a1">206</span>
  <span class="l2 a2">0</span>
</div>
</body></html>
"""


@pytest.mark.unit
def test_to_taoguba_symbol_sh_sz():
    assert to_taoguba_symbol("603629.SS", "cn_a") == "sh603629"
    assert to_taoguba_symbol("000001.SZ", "cn_a") == "sz000001"
    assert to_taoguba_symbol("BABA", "cn_adr") == "BABA"


@pytest.mark.unit
def test_fetch_ths_guba_parses_browser_html():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_guba_post_limit": 5, "cn_browser_enabled": True})

    with patch(
        "fxxkstock.dataflows.cn_community.render_html",
        return_value=THS_HTML,
    ):
        out = fetch_ths_guba("603629.SS")

    assert "同花顺股吧" in out
    assert "算力还能追吗" in out
    assert "reads=602" in out
    assert "comments=8" in out


@pytest.mark.unit
def test_fetch_taoguba_parses_browser_html():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a", "cn_guba_post_limit": 5, "cn_browser_enabled": True})

    with patch(
        "fxxkstock.dataflows.cn_community.render_html",
        return_value=TAOGUBA_HTML,
    ):
        out = fetch_taoguba("603629.SS")

    assert "淘股吧" in out
    assert "利通电子还能格局吗" in out
    assert "reads=206" in out
    assert "Link: https://www.tgb.cn/a/12345" in out


@pytest.mark.unit
def test_fetch_xueqiu_does_not_fall_back_to_eastmoney():
    with patch(
        "fxxkstock.dataflows.eastmoney_browser.fetch_browser_guba",
        return_value="Xueqiu Community — 1 recent posts\n  看多但不追高",
    ) as fetcher:
        out = fetch_xueqiu_community("600667.SS", limit=5)

    assert "看多但不追高" in out
    fetcher.assert_called_once_with(
        "600667.SS",
        limit=5,
        source="xueqiu",
        fallback_to_eastmoney=False,
    )


@pytest.mark.unit
def test_fetch_cn_community_best_effort_partial_success():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config(
        {
            "market_region": "cn_a",
            "cn_guba_post_limit": 5,
            "cn_community_sources": ["eastmoney", "ths", "taoguba"],
            "cn_http_inter_request_delay": 0,
            "cn_browser_enabled": False,
        }
    )

    eastmoney_block = "East Money Guba — 2 recent posts for 603629.SS\n  [?] 东财测试帖"

    with (
        patch(
            "fxxkstock.dataflows.cn_community.fetch_eastmoney_guba",
            return_value=eastmoney_block,
        ),
        patch(
            "fxxkstock.dataflows.cn_community.fetch_ths_guba",
            side_effect=RuntimeError("ths blocked"),
        ),
        patch(
            "fxxkstock.dataflows.cn_community.fetch_taoguba",
            return_value="淘股吧 (Taoguba) — 1 recent posts for 603629.SS\n  [?] 淘股吧测试帖",
        ),
    ):
        out = fetch_cn_community("603629.SS")

    assert "CN Retail Community" in out
    assert "2 source(s) succeeded" in out
    assert "东财测试帖" in out
    assert "淘股吧测试帖" in out
    assert "ths blocked" not in out


@pytest.mark.unit
def test_fetch_cn_community_all_failed_returns_placeholder():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config(
        {
            "market_region": "cn_a",
            "cn_community_sources": ["eastmoney", "ths"],
            "cn_http_inter_request_delay": 0,
        }
    )

    with (
        patch(
            "fxxkstock.dataflows.cn_community.fetch_eastmoney_guba",
            return_value="<no guba posts found for 603629.SS>",
        ),
        patch(
            "fxxkstock.dataflows.cn_community.fetch_ths_guba",
            side_effect=RuntimeError("ths failed"),
        ),
    ):
        out = fetch_cn_community("603629.SS")

    assert out == "<no retail community posts found for 603629.SS>"


@pytest.mark.unit
def test_build_cn_system_message_includes_attribution_rules():
    msg = _build_cn_system_message(
        ticker="603629.SS",
        start_date="2026-06-19",
        end_date="2026-06-26",
        news_block="news",
        community_block="community",
        official_block="",
    )

    assert "东财股吧 + 雪球 + NGA 大时代" in msg
    assert "actual opening posts and replies" in msg
    assert "换手率" in msg
    assert "龙虎榜" in msg
    assert "资金流向" in msg
    assert "do **not** treat these as retail sentiment" in msg
