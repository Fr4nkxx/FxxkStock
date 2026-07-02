"""国内散户社区聚合 — 东财股吧 / 同花顺股吧 / 淘股吧。"""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from parsel import Selector

from .config import get_config
from .eastmoney_guba import _format_guba_post_line, fetch_eastmoney_guba
from .errors import BrowserUnavailableError
from .market_utils import detect_market_region, to_eastmoney_symbol
from .nga_sentiment import fetch_nga_sentiment
from .playwright_web import render_html

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_FAILURE_PREFIXES = (
    "<guba unavailable",
    "<no guba",
    "<no browser guba",
    "<no retail community",
    "<no nga",
)


def to_taoguba_symbol(ticker: str, region: str) -> str:
    """将 ticker 转为淘股吧 quotes 路径段，如 sh603629 / sz000001 / BABA。"""
    upper = ticker.upper().strip()
    _, bare = to_eastmoney_symbol(ticker, region)

    if region == "cn_adr":
        return upper.split(".")[0]

    if region == "cn_hk" or upper.endswith(".HK"):
        hk = bare.zfill(5) if bare.isdigit() else bare
        return f"hk{hk}"

    if bare.startswith(("6", "9")) or upper.endswith(".SS"):
        return f"sh{bare}"
    return f"sz{bare}"


def _ths_stock_url(ticker: str, region: str) -> str:
    _, bare = to_eastmoney_symbol(ticker, region)
    return f"https://stockpage.10jqka.com.cn/{bare}/"


def _taoguba_stock_url(ticker: str, region: str) -> str:
    symbol = to_taoguba_symbol(ticker, region)
    return f"https://www.tgb.cn/quotes/{symbol}"


def _http_get(url: str, *, referer: str, timeout: float = 12.0) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/json,*/*",
            "Referer": referer,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _walk_json_for_posts(node, depth: int = 0) -> list[dict]:
    """从内嵌 JSON 中递归提取疑似帖子节点。"""
    if depth > 8:
        return []
    found: list[dict] = []
    if isinstance(node, dict):
        title = (
            node.get("post_title")
            or node.get("title")
            or node.get("subject")
            or node.get("content")
        )
        if isinstance(title, str) and title.strip():
            created = (
                node.get("post_publish_time")
                or node.get("post_last_time")
                or node.get("ctime")
                or node.get("created_at")
                or node.get("time")
                or "?"
            )
            link = node.get("url") or node.get("link") or node.get("post_url") or ""
            read_count = node.get("post_click_count") or node.get("read_count") or node.get("click")
            comment_count = (
                node.get("post_comment_count") or node.get("comment_count") or node.get("reply_count")
            )
            found.append(
                {
                    "title": title.strip()[:300],
                    "created": str(created).strip() if created is not None else "?",
                    "read_count": read_count,
                    "comment_count": comment_count,
                    "link": str(link).strip() if link else "",
                    "source": "json",
                }
            )
        for value in node.values():
            found.extend(_walk_json_for_posts(value, depth + 1))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_json_for_posts(item, depth + 1))
    return found


def _extract_embedded_json_posts(html: str) -> list[dict]:
    posts: list[dict] = []
    seen: set[str] = set()
    for pattern in (
        r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
        r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;",
    ):
        for match in re.finditer(pattern, html, re.DOTALL):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            for post in _walk_json_for_posts(data):
                title = post.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    posts.append(post)
    return posts


def _normalize_link(link: str, base: str) -> str:
    link = (link or "").strip()
    if not link:
        return ""
    if link.startswith("//"):
        return "https:" + link
    if link.startswith("/"):
        return base.rstrip("/") + link
    if not link.startswith("http"):
        return ""
    return link


def _parse_ths_from_html(html: str) -> list[dict]:
    """解析同花顺个股页股吧/讨论列表。"""
    posts: list[dict] = []
    seen: set[str] = set()
    sel = Selector(text=html)

    row_selectors = (
        "table.m_table tr",
        "table.ggtable tr",
        "div.guba_list tr",
        "ul.guba-list li",
        "#column .post_list li",
    )
    for row_sel in row_selectors:
        for row in sel.css(row_sel):
            title = (
                (row.css("a::attr(title)").get() or "").strip()
                or (row.css("a::text").get() or "").strip()
            )
            if not title or len(title) < 2 or title in seen:
                continue
            href = (row.css("a::attr(href)").get() or "").strip()
            created = (
                row.css("td:last-child::text").get()
                or row.css("span.time::text").get()
                or row.css(".time::text").get()
                or "?"
            )
            meta = (row.css("td:nth-child(3)::text").get() or "").strip()
            read_count = comment_count = None
            if "/" in meta:
                parts = [p.strip() for p in meta.split("/", 1)]
                if len(parts) == 2:
                    comment_count, read_count = parts[0], parts[1]
            seen.add(title)
            posts.append(
                {
                    "title": title,
                    "created": str(created).strip(),
                    "read_count": read_count,
                    "comment_count": comment_count,
                    "link": _normalize_link(href, "https://t.10jqka.com.cn"),
                    "source": "ths",
                }
            )

    for post in _extract_embedded_json_posts(html):
        title = post.get("title", "")
        if title and title not in seen:
            seen.add(title)
            post["source"] = "ths"
            posts.append(post)

    return posts


def _parse_taoguba_from_html(html: str) -> list[dict]:
    """解析淘股吧个股讨论页。"""
    posts: list[dict] = []
    seen: set[str] = set()
    sel = Selector(text=html)
    base = "https://www.tgb.cn"

    row_selectors = (
        "div.articleh",
        "div.datelist div",
        "div.community-content div[class*='item']",
        "div[class*='topic-list'] div[class*='item']",
        "li[class*='topic']",
    )
    for row_sel in row_selectors:
        for row in sel.css(row_sel):
            title = (
                (row.css("a::attr(title)").get() or "").strip()
                or (row.css("a::text").get() or "").strip()
                or (row.css("h3::text").get() or "").strip()
            )
            if not title or len(title) < 2 or title in seen:
                continue
            href = (row.css("a::attr(href)").get() or "").strip()
            created = (
                row.css("span.l5::text").get()
                or row.css(".time::text").get()
                or row.css("time::text").get()
                or "?"
            )
            read_raw = row.css("span.l1::text").get() or ""
            comment_raw = row.css("span.l2::text").get() or ""
            if not read_raw:
                read_match = re.search(r"浏览\((\d+)\)", row.get() or "")
                read_raw = read_match.group(1) if read_match else None
            if not comment_raw:
                comment_match = re.search(r"评论\((\d+)\)", row.get() or "")
                comment_raw = comment_match.group(1) if comment_match else None
            seen.add(title)
            posts.append(
                {
                    "title": title,
                    "created": str(created).strip(),
                    "read_count": read_raw or None,
                    "comment_count": comment_raw or None,
                    "link": _normalize_link(href, base),
                    "source": "taoguba",
                }
            )

    # 文本正则兜底：适配 tgb.cn 渲染后的主帖/跟帖块
    for match in re.finditer(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(?:发布主帖|跟帖回复)\s*(?:【摘要】)?(.{4,200}?)"
        r"(?:\s*赞\(\d+\)|\s*浏览\(\d+\)|\s*评论\(\d+\)|\s*来自)",
        html,
        re.DOTALL,
    ):
        created, title = match.group(1), match.group(2).strip()
        title = re.sub(r"\s+", " ", title)
        if title in seen:
            continue
        seen.add(title)
        posts.append(
            {
                "title": title,
                "created": created,
                "read_count": None,
                "comment_count": None,
                "link": "",
                "source": "taoguba",
            }
        )

    return posts


def _fetch_page_posts(url: str, *, referer: str, parser) -> list[dict]:
    """浏览器优先抓取页面并解析帖子；失败时尝试 HTTP。"""
    config = get_config()
    errors: list[str] = []

    if config.get("cn_browser_enabled", True):
        try:
            html = render_html(url, wait_selector="body")
            posts = parser(html)
            if posts:
                return posts
        except BrowserUnavailableError as exc:
            errors.append(f"browser: {exc}")
            logger.warning("Browser fetch unavailable for %s: %s", url, exc)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"browser: {exc}")
            logger.warning("Browser fetch failed for %s: %s", url, exc)

    try:
        raw = _http_get(url, referer=referer)
        posts = parser(raw.decode("utf-8", errors="replace"))
        if posts:
            return posts
    except HTTPError as exc:
        errors.append(f"http: {exc}")
        logger.warning("HTTP fetch error for %s: %s", url, exc)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"http: {exc}")
        logger.warning("HTTP fetch failed for %s: %s", url, exc)

    if errors:
        raise RuntimeError("; ".join(errors))
    return []


def _format_source_block(label: str, posts: list[dict], ticker: str, via: str) -> str:
    header = f"{label} — {len(posts)} recent posts for {ticker.upper()}"
    if via != "json":
        header += f" (via {via})"
    lines = [header]
    for post in posts:
        lines.append(_format_guba_post_line(post))
    return "\n".join(lines)


def fetch_ths_guba(ticker: str, limit: int | None = None) -> str:
    """抓取同花顺个股讨论帖，失败时抛异常供聚合层吞掉。"""
    config = get_config()
    if limit is None:
        limit = config.get("cn_guba_post_limit", 15)
    region = config.get("market_region") or detect_market_region(ticker)
    if region == "cn_adr":
        raise RuntimeError(f"no THS forum for ADR {ticker.upper()}")

    url = _ths_stock_url(ticker, region)
    posts = _fetch_page_posts(url, referer="https://stockpage.10jqka.com.cn/", parser=_parse_ths_from_html)
    if not posts:
        raise RuntimeError(f"no THS posts parsed for {ticker.upper()}")
    via = posts[0].get("source", "browser")
    return _format_source_block("同花顺股吧 (THS)", posts[:limit], ticker, via)


def fetch_taoguba(ticker: str, limit: int | None = None) -> str:
    """抓取淘股吧个股讨论帖，失败时抛异常供聚合层吞掉。"""
    config = get_config()
    if limit is None:
        limit = config.get("cn_guba_post_limit", 15)
    region = config.get("market_region") or detect_market_region(ticker)

    url = _taoguba_stock_url(ticker, region)
    posts = _fetch_page_posts(url, referer="https://www.tgb.cn/", parser=_parse_taoguba_from_html)
    if not posts:
        raise RuntimeError(f"no Taoguba posts parsed for {ticker.upper()}")
    via = posts[0].get("source", "browser")
    return _format_source_block("淘股吧 (Taoguba)", posts[:limit], ticker, via)


def fetch_xueqiu_community(ticker: str, limit: int | None = None) -> str:
    """通过浏览器抓取雪球个股讨论，不混入东方财富回退结果。"""
    from .eastmoney_browser import fetch_browser_guba

    result = fetch_browser_guba(
        ticker,
        limit=limit,
        source="xueqiu",
        fallback_to_eastmoney=False,
    )
    if not result or _is_failure_placeholder(result):
        raise RuntimeError(f"no Xueqiu posts parsed for {ticker.upper()}")
    return result


def _is_failure_placeholder(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _FAILURE_PREFIXES)


def fetch_cn_community(
    ticker: str,
    limit: int | None = None,
    *,
    as_of_date: str | None = None,
) -> str:
    """聚合国内散户社区帖子，各源独立尽力抓取。"""
    config = get_config()
    if limit is None:
        limit = config.get("cn_guba_post_limit", 15)
    region = config.get("market_region") or detect_market_region(ticker)

    if region == "cn_adr":
        return (
            f"<guba unavailable: no dedicated CN retail forum for ADR {ticker.upper()}; "
            f"use news headlines only>"
        )

    source_handlers = {
        "eastmoney": fetch_eastmoney_guba,
        "ths": fetch_ths_guba,
        "taoguba": fetch_taoguba,
        "xueqiu": fetch_xueqiu_community,
        "nga": fetch_nga_sentiment,
    }
    sources = config.get("cn_community_sources", ["eastmoney", "xueqiu", "nga"])
    delay = float(config.get("cn_http_inter_request_delay", 0.5))

    sections: list[str] = []
    for index, source in enumerate(sources):
        if index > 0 and delay > 0:
            time.sleep(delay)
        handler = source_handlers.get(source)
        if handler is None:
            logger.warning("Unknown cn_community source %r — skipped", source)
            continue
        try:
            if source == "nga":
                result = handler(ticker, limit=limit, as_of_date=as_of_date)
            else:
                result = handler(ticker, limit=limit)
            if result and not _is_failure_placeholder(result):
                sections.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CN community source %s failed for %s: %s", source, ticker, exc)

    if not sections:
        return f"<no retail community posts found for {ticker.upper()}>"

    header = (
        f"CN Retail Community — aggregated posts for {ticker.upper()} "
        f"({len(sections)} source(s) succeeded)"
    )
    return header + "\n\n" + "\n\n".join(sections)
