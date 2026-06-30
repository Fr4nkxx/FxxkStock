"""浏览器(CDP)版国内软信息抓取 — 东方财富 / 巨潮 / 雪球。"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from urllib.parse import quote

from dateutil.relativedelta import relativedelta
from parsel import Selector

from .config import get_config
from .cninfo import _format_announcements, _query_announcements, _resolve_org_id
from .errors import BrowserUnavailableError, NoMarketDataError, VendorRateLimitError
from .market_utils import detect_market_region, to_cninfo_stock_code, to_eastmoney_symbol, to_guba_code
from .eastmoney_guba import _format_guba_post_line, _guba_post_link
from .news_utils import format_article_block, in_news_window
from .playwright_web import render_html

logger = logging.getLogger(__name__)

_INSIDER_CATEGORIES = (
    "category_jjgg;category_jjgg_szsh;category_dshgg_szsh;"
    "category_sdgg_szsh;category_rcjy_szsh;"
)


def _parse_pub_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = value / 1000 if value > 1e12 else value
        with contextlib.suppress(ValueError, OSError, TypeError):
            return datetime.fromtimestamp(ts)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%m-%d %H:%M",
            "%m-%d",
        ):
            with contextlib.suppress(ValueError):
                dt = datetime.strptime(text[:19], fmt)
                if fmt == "%m-%d":
                    dt = dt.replace(year=datetime.now().year)
                elif fmt == "%m-%d %H:%M":
                    dt = dt.replace(year=datetime.now().year)
                return dt
    return None


def _to_xueqiu_symbol(ticker: str, region: str) -> str:
    """A/H 股代码转雪球 symbol，如 SH600519 / SZ000001 / HK00700。"""
    em_code, bare = to_eastmoney_symbol(ticker, region)
    if region == "cn_hk" or ticker.upper().endswith(".HK"):
        hk = bare.zfill(5) if bare.isdigit() else bare.upper()
        return f"HK{hk}"
    if em_code.startswith("1."):
        return f"SH{bare}"
    if em_code.startswith("0."):
        return f"SZ{bare}"
    return bare.upper()


def _stock_news_url(ticker: str, region: str) -> str:
    _, bare = to_eastmoney_symbol(ticker, region)
    return f"https://so.eastmoney.com/news/s?keyword={quote(bare)}"


def _global_news_search_url(query: str) -> str:
    return f"https://so.eastmoney.com/news/s?keyword={quote(query)}"


def _cninfo_disclosure_url(stock_code: str) -> str | None:
    org_id = _resolve_org_id(stock_code)
    if not org_id:
        return None
    return (
        "https://www.cninfo.com.cn/new/disclosure/stock?"
        f"stockCode={stock_code}&orgId={org_id}"
    )


def _guba_url(ticker: str, region: str) -> str:
    code = to_guba_code(ticker, region)
    return f"https://guba.eastmoney.com/list,{code}.html"


def _xueqiu_url(ticker: str, region: str) -> str:
    symbol = _to_xueqiu_symbol(ticker, region)
    return f"https://xueqiu.com/S/{symbol}"


def _extract_embedded_json(html: str) -> list[dict]:
    """尝试从页面内嵌 JSON 提取新闻/帖子列表。"""
    articles: list[dict] = []
    for pattern in (
        r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
        r"__NEXT_DATA__\s*=\s*(\{.*?\})\s*</script>",
    ):
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            continue
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            payload = json.loads(match.group(1))
            articles.extend(_walk_json_for_articles(payload))
    return articles


def _walk_json_for_articles(node, depth: int = 0) -> list[dict]:
    """递归扫描 JSON 树中的新闻形字段。"""
    if depth > 8:
        return []
    found: list[dict] = []
    if isinstance(node, dict):
        title = node.get("title") or node.get("post_title") or node.get("notice_title")
        if title and isinstance(title, str):
            link = (
                node.get("url")
                or node.get("article_url")
                or node.get("target")
                or node.get("notice_url")
                or ""
            )
            pub = _parse_pub_date(
                node.get("date")
                or node.get("publishTime")
                or node.get("notice_date")
                or node.get("display_time")
                or node.get("post_publish_time")
                or node.get("created_at")
            )
            summary = node.get("summary") or node.get("content") or node.get("digest") or ""
            publisher = (
                node.get("mediaName")
                or node.get("source")
                or node.get("publisher")
                or "browser"
            )
            found.append(
                {
                    "title": title.strip(),
                    "summary": str(summary)[:500] if summary else "",
                    "publisher": str(publisher),
                    "link": str(link) if link else "",
                    "pub_date": pub,
                }
            )
        for value in node.values():
            found.extend(_walk_json_for_articles(value, depth + 1))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_json_for_articles(item, depth + 1))
    return found


def _parse_news_from_html(html: str) -> list[dict]:
    """从东财搜索/新闻列表 DOM 解析文章。"""
    articles = _extract_embedded_json(html)
    seen = {a["title"] for a in articles if a.get("title")}

    sel = Selector(text=html)
    row_selectors = (
        "div.news-item",
        "div.news_item",
        "li.result-item",
        "div.search-item",
        "ul.news_list li",
        "div.txt-cont",
        "div.articleh",
    )
    for row_sel in row_selectors:
        for row in sel.css(row_sel):
            current_news_title = " ".join(
                row.css(".news_item_t a").xpath(".//text()").getall()
            ).strip()
            title = (
                (row.css("a::attr(title)").get() or "").strip()
                or current_news_title
                or (row.css("a::text").get() or "").strip()
                or (row.css("span.l3 a::text").get() or "").strip()
            )
            title = re.sub(r"\s+", " ", title)
            if not title or title in seen:
                continue
            link = (row.css("a::attr(href)").get() or "").strip()
            if link and link.startswith("//"):
                link = "https:" + link
            date_raw = (
                row.css(".time::text").get()
                or row.css("span.time::text").get()
                or row.css(".news_item_time::text").get()
                or row.css(".date::text").get()
                or row.css("span.l5::text").get()
            )
            summary = (
                row.css(".content::text").get()
                or row.css(".news_item_c span:last-child::text").get()
                or row.css("p::text").get()
                or ""
            ).strip()
            articles.append(
                {
                    "title": title,
                    "summary": summary,
                    "publisher": "东方财富",
                    "link": link,
                    "pub_date": _parse_pub_date(date_raw),
                }
            )
            seen.add(title)
    return articles


def _parse_guba_from_html(html: str) -> list[dict]:
    """从股吧 HTML 或雪球讨论页解析帖子。"""
    posts: list[dict] = []
    seen: set[str] = set()
    sel = Selector(text=html)

    # Current East Money pages expose the complete list as `article_list`.
    marker = re.search(r"\barticle_list\s*=\s*", html)
    if marker:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            payload, _ = json.JSONDecoder().raw_decode(html[marker.end():])
            for item in payload.get("re") or []:
                title = str(item.get("post_title") or "").strip()
                content = " ".join(
                    Selector(text=str(item.get("post_content") or ""))
                    .xpath("//text()")
                    .getall()
                )
                content = " ".join(content.split())
                if not title:
                    title = content[:120]
                if not title or title in seen:
                    continue
                seen.add(title)
                replies = []
                for reply in (item.get("reply_list") or [])[:2]:
                    text = " ".join(str(reply.get("reply_text") or "").split())
                    if text:
                        replies.append(text[:160])
                summary = content[:500]
                if replies:
                    summary += " | 热门回复: " + "；".join(replies)
                posts.append(
                    {
                        "title": title,
                        "created": (
                            item.get("post_last_time")
                            or item.get("post_publish_time")
                            or "?"
                        ),
                        "read_count": item.get("post_click_count"),
                        "comment_count": item.get("post_comment_count"),
                        "source": "eastmoney",
                        "link": _guba_post_link(
                            str(
                                (item.get("post_guba") or {}).get("stockbar_external_code")
                                or (item.get("post_guba") or {}).get("stockbar_code")
                                or ""
                            ),
                            item.get("post_id"),
                        ),
                        "summary": summary,
                    }
                )

    # 东财股吧列表
    for row in sel.css("div.articleh"):
        title = (row.css("span.l3 a::text").get() or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        href = (row.css("span.l3 a::attr(href)").get() or "").strip()
        link = href
        if link and link.startswith("//"):
            link = "https:" + link
        elif link and not link.startswith("http"):
            link = f"https://guba.eastmoney.com{link}" if link.startswith("/") else ""
        posts.append(
            {
                "title": title,
                "created": (row.css("span.l5::text").get() or "?").strip(),
                "read_count": (row.css("span.l1::text").get() or "").strip() or None,
                "comment_count": (row.css("span.l2::text").get() or "").strip() or None,
                "source": "eastmoney",
                "link": link,
            }
        )

    # 雪球 timeline：正文在 content--description 中，第一个链接通常只是股票标签。
    xueqiu_seen: set[str] = set()
    for row in sel.css(
        ".timeline__item, div.status-card, div.feed__item"
    ):
        content_node = row.css(
            ".timeline__item__content .content--description, "
            ".timeline__item__content .content"
        )
        body = " ".join(content_node.xpath(".//text()").getall())
        if not body:
            body = " ".join(
                row.css(".status-content, .feed-content").xpath(".//text()").getall()
            )
        body = re.sub(r"\s+", " ", body).strip()
        body = re.sub(r"^(?:收起|展开)\s*", "", body)
        opinion = re.sub(r"\$[^$]+\$", "", body)
        opinion = re.sub(r"\s+", " ", opinion).strip()
        if len(opinion) < 4:
            continue
        status_link = (
            row.css("a.date-and-source::attr(href)").get()
            or row.css("time").xpath("ancestor::a[1]/@href").get()
            or ""
        ).strip()
        if status_link.startswith("/"):
            status_link = f"https://xueqiu.com{status_link}"
        dedupe_key = status_link or opinion
        if dedupe_key in xueqiu_seen:
            continue
        xueqiu_seen.add(dedupe_key)
        created = " ".join(
            row.css("a.date-and-source").xpath(".//text()").getall()
        ).strip() or (
            row.css(".timeline__item__time::text").get()
            or row.css("span.time::text").get()
            or row.css("time::attr(datetime)").get()
            or "?"
        )
        footer = " ".join(
            row.css(".timeline__item__ft").xpath(".//text()").getall()
        )
        comment_match = re.search(r"讨论\s*(\d+)", footer)
        like_match = re.search(r"赞\s*(\d+)", footer)
        author = " ".join(row.css("a.user-name").xpath(".//text()").getall()).strip()
        posts.append(
            {
                "title": body[:160] + ("..." if len(body) > 160 else ""),
                "created": str(created).strip(),
                "read_count": None,
                "comment_count": int(comment_match.group(1)) if comment_match else None,
                "like_count": int(like_match.group(1)) if like_match else None,
                "source": "xueqiu",
                "link": status_link,
                "summary": body[:700],
                "author": author,
            }
        )

    if not posts:
        for item in _extract_embedded_json(html):
            title = item.get("title", "")
            if title:
                posts.append(
                    {
                        "title": title,
                        "created": "?",
                        "read_count": None,
                        "comment_count": None,
                        "source": "json",
                    }
                )
    return posts


def _format_guba_block(posts: list[dict], ticker: str, via: str) -> str:
    label = "East Money Guba" if via == "eastmoney" else "Xueqiu Community"
    header = f"{label} — {len(posts)} recent posts for {ticker.upper()} (browser)"
    lines = [header]
    for p in posts:
        lines.append(_format_guba_post_line(p))
    return "\n".join(lines)


def _fetch_stock_articles_browser(ticker: str, region: str, limit: int) -> list[dict]:
    articles: list[dict] = []
    seen: set[str] = set()
    urls = [
        _stock_news_url(ticker, region),
        f"https://data.eastmoney.com/notices/stock/{to_eastmoney_symbol(ticker, region)[1]}.html",
    ]
    delay = get_config().get("cn_http_inter_request_delay", 0.5)
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(delay)
        try:
            html = render_html(url, wait_selector="body")
            batch = _parse_news_from_html(html)
        except (BrowserUnavailableError, VendorRateLimitError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("Browser news fetch failed for %s at %s: %s", ticker, url, exc)
            continue
        for art in batch:
            title = art.get("title", "")
            if title and title not in seen:
                seen.add(title)
                articles.append(art)
        if len(articles) >= limit:
            break
    return articles[:limit]


def get_browser_news(ticker: str, start_date: str, end_date: str) -> str:
    """通过浏览器渲染东财新闻页获取个股新闻 (vendor entry for get_news)。"""
    config = get_config()
    limit = config.get("cn_news_article_limit", config["news_article_limit"])
    region = config.get("market_region") or detect_market_region(ticker)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    try:
        articles = _fetch_stock_articles_browser(ticker, region, limit)
    except VendorRateLimitError:
        raise
    except BrowserUnavailableError:
        raise
    except Exception as exc:
        logger.warning("Browser news failed for %s: %s", ticker, exc)
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    if not articles:
        raise NoMarketDataError(
            ticker, ticker, f"no browser news between {start_date} and {end_date}"
        )

    news_str = ""
    kept = 0
    for art in articles:
        if not in_news_window(art["pub_date"], start_dt, end_dt):
            continue
        news_str += format_article_block(
            art["title"], art["publisher"], art.get("summary", ""), art.get("link", "")
        )
        kept += 1

    if kept == 0:
        raise NoMarketDataError(
            ticker, ticker, f"no browser news in window {start_date} to {end_date}"
        )

    return f"## {ticker} News (Browser/CDP), from {start_date} to {end_date}:\n\n{news_str}"


def get_browser_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """通过浏览器搜索东财宏观新闻 (vendor entry for get_global_news)。"""
    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("cn_global_news_article_limit", config["global_news_article_limit"])

    region = config.get("market_region", "default")
    queries = (
        config.get("cn_global_news_queries", [])
        if region != "default"
        else config["global_news_queries"]
    )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    all_articles: list[dict] = []
    seen: set[str] = set()
    delay = config.get("cn_http_inter_request_delay", 0.5)

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(delay)
        try:
            html = render_html(_global_news_search_url(query), wait_selector="body")
            batch = _parse_news_from_html(html)
        except VendorRateLimitError:
            raise
        except BrowserUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Browser global search failed for %r: %s", query, exc)
            continue
        for art in batch:
            title = art.get("title", "")
            if title and title not in seen:
                seen.add(title)
                all_articles.append(art)
        if len(all_articles) >= limit:
            break

    if not all_articles:
        raise NoMarketDataError("GLOBAL", "GLOBAL", f"no browser global news for {curr_date}")

    news_str = ""
    kept = 0
    for art in all_articles[:limit]:
        if not in_news_window(art["pub_date"], start_dt, curr_dt):
            continue
        news_str += format_article_block(
            art["title"], art["publisher"], art.get("summary", ""), art.get("link", "")
        )
        kept += 1

    if kept == 0:
        raise NoMarketDataError(
            "GLOBAL", "GLOBAL", f"no global news between {start_date} and {curr_date}"
        )

    label = "CN Global Market News" if region != "default" else "Global Market News"
    return f"## {label} (Browser/CDP), from {start_date} to {curr_date}:\n\n{news_str}"


def get_browser_insider(ticker: str) -> str:
    """通过浏览器 + 巨潮 API 获取 A 股减持/股东变动公告。"""
    config = get_config()
    region = config.get("market_region", "default")
    if region not in ("cn_a", "default"):
        raise NoMarketDataError(
            ticker, ticker, "browser CNINFO insider data only available for A-shares"
        )

    try:
        stock_code = to_cninfo_stock_code(ticker)
    except ValueError as exc:
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=90)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    anns: list[dict] = []
    try:
        # 优先 HTTP API（orgId 缓存），浏览器仅作披露页补充
        anns = _query_announcements(
            stock_code, start_date, end_date, category=_INSIDER_CATEGORIES, limit=30
        )
        if not anns:
            url = _cninfo_disclosure_url(stock_code)
            if url:
                html = render_html(url, wait_selector="body")
                sel = Selector(text=html)
                for row in sel.css(
                    "table.el-table__body tr, .announcement-item, .search-result-item"
                ):
                    title = (
                        (row.css("a::text").get() or "").strip()
                        or (row.css(".title::text").get() or "").strip()
                    )
                    if not title:
                        continue
                    link = (row.css("a::attr(href)").get() or "").strip()
                    if link and not link.startswith("http"):
                        link = f"https://www.cninfo.com.cn{link}"
                    date_raw = (
                        row.css("td:nth-child(3)::text").get()
                        or row.css(".date::text").get()
                    )
                    ts = None
                    parsed = _parse_pub_date(date_raw)
                    if parsed:
                        ts = int(parsed.timestamp() * 1000)
                    anns.append(
                        {
                            "announcementTitle": title,
                            "announcementTime": ts,
                            "adjunctUrl": link,
                        }
                    )
    except VendorRateLimitError:
        raise
    except BrowserUnavailableError:
        raise
    except Exception as exc:
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    if not anns:
        raise NoMarketDataError(ticker, ticker, "no browser/CNINFO insider announcements")

    header = f"## {ticker} Insider/Shareholder Transactions (Browser/CDP), last 90 days:\n\n"
    body = _format_announcements(
        anns, f"CNINFO shareholder change announcements for {ticker.upper()}"
    )
    return header + body


def fetch_browser_guba(
    ticker: str,
    limit: int | None = None,
    *,
    source: str | None = None,
    fallback_to_eastmoney: bool = True,
) -> str:
    """通过浏览器抓取雪球/股吧讨论 — 失败返回占位串，不抛异常。"""
    config = get_config()
    if limit is None:
        limit = config.get("cn_guba_post_limit", 15)
    region = config.get("market_region") or detect_market_region(ticker)

    if region == "cn_adr":
        return (
            f"<guba unavailable: no dedicated forum for ADR {ticker.upper()}; "
            f"use news headlines only>"
        )

    source = source or config.get("cn_browser_guba_source", "eastmoney")
    urls: list[tuple[str, str]] = []
    if source == "xueqiu":
        urls.append(("xueqiu", _xueqiu_url(ticker, region)))
    if source != "xueqiu" or fallback_to_eastmoney:
        urls.append(("eastmoney", _guba_url(ticker, region)))

    posts: list[dict] = []
    for via, url in urls:
        try:
            html = render_html(url, wait_selector="body")
            posts = _parse_guba_from_html(html)[:limit]
            if posts:
                return _format_guba_block(posts, ticker, via)
        except BrowserUnavailableError as exc:
            logger.warning("Browser guba unavailable for %s: %s", ticker, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Browser guba fetch failed for %s (%s): %s", ticker, via, exc)
            continue

    return f"<no browser guba posts found for {ticker.upper()}>"
