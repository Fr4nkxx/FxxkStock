"""东方财富股吧抓取 — 免 API Key，JSON 接口 + HTML fallback。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from parsel import Selector

from .config import get_config
from .market_utils import detect_market_region, to_guba_code

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
_GUBA_API = "https://gbapi.eastmoney.com/webarticlelist/api/Article/ArticleList"
_GUBA_LIST_URL = "https://guba.eastmoney.com/list,{code},99_{page}.html"
_GUBA_POST_URL = "https://guba.eastmoney.com/news,{code},{post_id}.html"


def _guba_post_link(code: str, post_id) -> str:
    """Build East Money guba post URL when post_id is available."""
    if post_id is None or post_id == "":
        return ""
    pid = str(post_id).strip()
    if not pid.isdigit():
        return ""
    return _GUBA_POST_URL.format(code=code, post_id=pid)


def _format_guba_post_line(p: dict) -> str:
    """Format one guba post with optional Link line."""
    meta = p.get("created", "?")
    read_c = p.get("read_count")
    comment_c = p.get("comment_count")
    if read_c is not None and comment_c is not None:
        meta += f" · reads={read_c} · comments={comment_c}"
    line = f"  [{meta}] {p['title']}"
    link = (p.get("link") or "").strip()
    if link:
        line += f"\n    Link: {link}"
    summary = " ".join(str(p.get("summary") or "").split())
    if summary:
        line += f"\n    Content: {summary}"
    return line


def _http_get(url: str, timeout: float = 12.0) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "application/json,text/html,*/*",
            "Referer": "https://guba.eastmoney.com/",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_guba_json(code: str, limit: int) -> list[dict]:
    """Fetch guba posts via frontend JSON API."""
    url = f"{_GUBA_API}?code={code}&ps={limit}&p=1&sorttype=1"
    raw = _http_get(url).decode("utf-8", errors="replace")
    data = json.loads(raw)
    re_list = data.get("re") or data.get("result") or []
    posts = []
    for item in re_list:
        title = (item.get("post_title") or item.get("title") or "").strip()
        if not title:
            continue
        post_id = item.get("post_id") or item.get("post_source_id")
        link = _guba_post_link(code, post_id)
        posts.append(
            {
                "title": title,
                "created": item.get("post_publish_time") or item.get("post_last_time") or "?",
                "read_count": item.get("post_click_count"),
                "comment_count": item.get("post_comment_count"),
                "source": "json",
                "link": link,
            }
        )
    return posts


def _fetch_guba_html(code: str, limit: int) -> list[dict]:
    """HTML fallback when JSON API is unavailable."""
    url = _GUBA_LIST_URL.format(code=code, page=1)
    html = _http_get(url).decode("utf-8", errors="replace")
    sel = Selector(text=html)
    posts = []
    for row in sel.css("div.articleh")[:limit]:
        title = (row.css("span.l3 a::text").get() or "").strip()
        created = (row.css("span.l5::text").get() or "?").strip()
        read_count = (row.css("span.l1::text").get() or "").strip()
        comment_count = (row.css("span.l2::text").get() or "").strip()
        href = (row.css("span.l3 a::attr(href)").get() or "").strip()
        link = href
        if link and link.startswith("//"):
            link = "https:" + link
        elif link and not link.startswith("http"):
            link = f"https://guba.eastmoney.com{link}" if link.startswith("/") else ""
        if title:
            posts.append(
                {
                    "title": title,
                    "created": created,
                    "read_count": read_count or None,
                    "comment_count": comment_count or None,
                    "source": "html",
                    "link": link,
                }
            )
    return posts


def fetch_eastmoney_guba(ticker: str, limit: int | None = None) -> str:
    """Fetch recent East Money guba posts for ``ticker`` as formatted plaintext.

    Returns a placeholder string on failure — callers never need to catch.
    When ``cn_browser_enabled`` is on, tries browser/CDP first, then HTTP fallback.
    """
    config = get_config()
    if config.get("cn_browser_enabled", True):
        try:
            from .eastmoney_browser import fetch_browser_guba

            result = fetch_browser_guba(ticker, limit=limit)
            if result and not result.startswith("<no browser guba"):
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Browser guba dispatch failed for %s: %s", ticker, exc)

    if limit is None:
        limit = config.get("cn_guba_post_limit", 15)
    region = config.get("market_region") or detect_market_region(ticker)

    if region == "cn_adr":
        return (
            f"<guba unavailable: no dedicated East Money forum for ADR {ticker.upper()}; "
            f"use news headlines only>"
        )

    code = to_guba_code(ticker, region)
    posts: list[dict] = []

    for fetcher in (_fetch_guba_json, _fetch_guba_html):
        try:
            posts = fetcher(code, limit)
            if posts:
                break
        except HTTPError as exc:
            logger.warning("Guba fetch HTTP error for %s (%s): %s", ticker, code, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Guba fetch failed for %s (%s): %s", ticker, code, exc)

    if not posts:
        return f"<no guba posts found for {ticker.upper()} (code={code})>"

    via = posts[0].get("source", "json")
    header = f"East Money Guba — {len(posts)} recent posts for {ticker.upper()}"
    if via == "html":
        header += " (via HTML fallback)"
    lines = [header]
    for p in posts:
        lines.append(_format_guba_post_line(p))
    return "\n".join(lines)
