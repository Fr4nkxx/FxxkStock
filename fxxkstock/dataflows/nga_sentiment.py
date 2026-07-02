"""NGA 大时代散户情绪数据源。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from parsel import Selector

from .config import get_config
from .errors import BrowserUnavailableError
from .market_utils import (
    detect_market_region,
    get_cn_etf_metadata,
    get_security_cn_name,
    is_cn_etf,
)

NGA_GREAT_TIMES_URL = "https://bbs.nga.cn/thread.php?fid=706"

_INDUSTRY_NAME_HINTS = (
    "人工智能", "半导体", "芯片", "电子", "机器人", "新能源", "光伏",
    "军工", "医药", "医疗", "证券", "银行", "保险", "房地产", "汽车",
    "消费", "白酒", "黄金", "有色", "煤炭", "石油", "化工", "传媒",
    "游戏", "通信", "算力", "软件", "互联网",
)
_ENGLISH_INDUSTRY_MAP = {
    "artificial intelligence": "人工智能",
    "semiconductor": "半导体",
    "electronic": "电子",
    "software": "软件",
    "internet": "互联网",
    "communication": "通信",
    "telecom": "通信",
    "bank": "银行",
    "insurance": "保险",
    "real estate": "房地产",
    "auto": "汽车",
    "vehicle": "汽车",
    "health": "医药",
    "biotech": "医药",
    "pharma": "医药",
    "energy": "能源",
    "oil": "石油",
    "coal": "煤炭",
    "metal": "有色",
    "gold": "黄金",
    "consumer": "消费",
    "media": "传媒",
    "gaming": "游戏",
}


def nga_search_url(keyword: str, base_url: str = NGA_GREAT_TIMES_URL) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["key"] = keyword
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def nga_latest_page_url(thread_url: str) -> str:
    parts = urlsplit(thread_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["page"] = "e"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def parse_nga_threads(html: str) -> list[dict]:
    selector = Selector(text=html)
    threads: list[dict] = []
    seen: set[str] = set()
    unavailable = {"帖子发布或回复时间超过限制", "帐号权限不足", "主题不存在", "主题已被删除"}
    for anchor in selector.css(
        "a.topic, a[class*='topic'], a[href*='read.php?tid='], a[href*='/read.php?tid=']"
    ):
        href = (anchor.css("::attr(href)").get() or "").strip()
        title = " ".join(" ".join(anchor.xpath(".//text()").getall()).split())
        if not href or len(title) < 4 or title in unavailable:
            continue
        if href.startswith("//"):
            href = f"https:{href}"
        elif href.startswith("/"):
            href = f"https://bbs.nga.cn{href}"
        elif not href.startswith("http"):
            href = f"https://bbs.nga.cn/{href.lstrip('/')}"
        params = dict(parse_qsl(urlsplit(href).query, keep_blank_values=True))
        if "tid" not in params or "page" in params or href in seen:
            continue
        seen.add(href)
        row = anchor.xpath("ancestor::tr[1]")
        last_reply = ""
        if row:
            last_reply = (
                row.css("a.replydate::attr(title)").get()
                or row.css("a.replydate::text").get()
                or ""
            ).strip()
        threads.append({"title": title, "link": href, "last_reply": last_reply})
    return threads


def parse_nga_replies(
    html: str,
    *,
    limit: int,
    content_limit: int = 800,
) -> list[dict]:
    selector = Selector(text=html)
    replies: list[dict] = []
    for content in selector.css("[id^='postcontent']"):
        content_id = content.css("::attr(id)").get() or ""
        match = re.fullmatch(r"postcontent(\d+)", content_id)
        if not match:
            continue
        floor = int(match.group(1))
        body = " ".join(" ".join(content.xpath(".//text()").getall()).split())
        if not body:
            continue
        author = " ".join(
            " ".join(selector.css(f"#postauthor{floor}").xpath(".//text()").getall()).split()
        )
        posted_at = " ".join(
            " ".join(selector.css(f"#postdate{floor}").xpath(".//text()").getall()).split()
        )
        replies.append(
            {
                "floor": floor,
                "author": author,
                "date": posted_at,
                "content": body[:content_limit],
            }
        )
        if len(replies) >= limit:
            break
    return replies


def filter_recent_threads(
    threads: list[dict],
    *,
    lookback_days: int,
    as_of: datetime,
) -> list[dict]:
    cutoff = as_of - timedelta(days=lookback_days)
    recent: list[dict] = []
    for thread in threads:
        try:
            replied_at = datetime.strptime(thread.get("last_reply", ""), "%y-%m-%d %H:%M")
        except (TypeError, ValueError):
            continue
        if cutoff <= replied_at <= as_of + timedelta(days=1):
            recent.append(thread)
    return recent


def infer_industry_term(chinese_name: str, identity: dict | None = None) -> str | None:
    for hint in _INDUSTRY_NAME_HINTS:
        if hint in chinese_name:
            return hint
    classification = " ".join(
        str((identity or {}).get(key) or "") for key in ("industry", "sector")
    ).casefold()
    for english, chinese in _ENGLISH_INDUSTRY_MAP.items():
        if english in classification:
            return chinese
    return None


def _render_nga_html(url: str) -> str:
    """Render one NGA page, handling its timed welcome page and closing the tab."""
    config = get_config()
    cdp_url = config.get("cn_browser_cdp_url", "http://127.0.0.1:9222")
    timeout_ms = int(config.get("cn_browser_nav_timeout_ms", 20000))
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserUnavailableError("playwright package is not installed") from exc

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url, timeout=timeout_ms)
            if not browser.contexts:
                raise BrowserUnavailableError(f"no browser context at {cdp_url}")
            page = browser.contexts[0].new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                if "欢迎访问NGA玩家社区" in page.title():
                    page.wait_for_timeout(1700)
                    jump = page.locator("#jump1")
                    if jump.count():
                        jump.click(force=True)
                        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                html = page.content()
            finally:
                page.close()
    except PlaywrightError as exc:
        raise BrowserUnavailableError(f"NGA browser navigation failed: {exc}") from exc

    compact = " ".join(Selector(text=html).xpath("//body//text()").getall())
    if any(token in compact.lower() for token in ("验证码", "访问过于频繁", "access denied")):
        raise BrowserUnavailableError("NGA blocked or captcha page")
    return html


def _search_threads(keyword: str, *, lookback_days: int, as_of: datetime) -> list[dict]:
    html = _render_nga_html(nga_search_url(keyword))
    return filter_recent_threads(
        parse_nga_threads(html),
        lookback_days=lookback_days,
        as_of=as_of,
    )


def _resolve_industry(ticker: str, chinese_name: str) -> str | None:
    direct = infer_industry_term(chinese_name)
    if direct:
        return direct
    try:
        from fxxkstock.agents.utils.agent_utils import resolve_instrument_identity

        return infer_industry_term(chinese_name, resolve_instrument_identity(ticker))
    except Exception:  # noqa: BLE001
        return None


def fetch_nga_sentiment(
    ticker: str,
    *,
    as_of_date: str | None = None,
    limit: int | None = None,
) -> str:
    """Fetch recent NGA Great Times threads and their actual replies."""
    config = get_config()
    region = config.get("market_region") or detect_market_region(ticker)
    if region not in {"cn_a", "cn_hk"}:
        raise RuntimeError(f"NGA source is unavailable for {ticker.upper()}")

    etf_metadata = get_cn_etf_metadata(ticker, region) if is_cn_etf(ticker) else {}
    chinese_name = (
        str(etf_metadata.get("fund_name") or "")
        or (get_security_cn_name(ticker, region) or "")
    ).strip()
    if not chinese_name:
        return (
            f"<no nga posts: Chinese security name unavailable for "
            f"{ticker.upper()}>"
        )

    as_of = (
        datetime.strptime(as_of_date, "%Y-%m-%d")
        if as_of_date
        else datetime.now()
    )
    lookback_days = int(config.get("cn_nga_lookback_days", 30))
    min_stock_threads = int(config.get("cn_nga_min_stock_threads", 3))
    thread_limit = int(config.get("cn_nga_thread_limit", 3))
    reply_limit = int(config.get("cn_nga_reply_limit", 20))
    total_limit = int(limit or config.get("cn_guba_post_limit", 15))

    query_results: list[tuple[str, str, list[dict]]] = []
    if etf_metadata:
        aliases = [
            str(item).strip()
            for item in etf_metadata.get("search_aliases", [])
            if str(item).strip()
        ][: int(config.get("cn_nga_etf_query_limit", 4))]
        tracking_index = str(etf_metadata.get("tracking_index") or "")
        for index, keyword in enumerate(aliases):
            scope = (
                "ETF"
                if keyword == chinese_name
                else "跟踪指数"
                if keyword == tracking_index
                else "指数别名"
            )
            query_results.append((
                scope,
                keyword,
                _search_threads(keyword, lookback_days=lookback_days, as_of=as_of),
            ))
    else:
        query_results.append((
            "个股",
            chinese_name,
            _search_threads(chinese_name, lookback_days=lookback_days, as_of=as_of),
        ))

    stock_threads = query_results[0][2] if query_results else []
    industry = _resolve_industry(ticker, chinese_name)
    industry_threads: list[dict] = []
    query_thread_count = sum(len(items) for _, _, items in query_results)
    if (
        not etf_metadata
        and query_thread_count < min_stock_threads
        and industry
        and industry != chinese_name
    ):
        industry_threads = _search_threads(
            industry,
            lookback_days=lookback_days,
            as_of=as_of,
        )

    selected = []
    seen_links: set[str] = set()
    for scope, keyword, threads in query_results:
        for thread in threads[:thread_limit]:
            link = thread.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            selected.append((scope, keyword, thread))
    selected.extend(
        ("行业", industry or "", thread)
        for thread in industry_threads[:thread_limit]
    )

    lines = [
        f"NGA 大时代 — {ticker.upper()}，主关键词“{chinese_name}”",
        (
            f"查询层级 {len(query_results)} 个；"
            f"近期候选主题 {query_thread_count} 个；"
            f"行业补充 {len(industry_threads)} 个"
        ),
    ]
    if etf_metadata:
        lines.extend([
            f"ETF 跟踪指数：{etf_metadata.get('tracking_index') or '未取得'}",
            "情绪范围说明：ETF、跟踪指数和指数别名分别采集；"
            "指数或主题情绪不等同于 ETF 交易情绪。",
        ])
    reply_count = 0
    for scope, keyword, thread in selected:
        if reply_count >= total_limit:
            break
        replies = parse_nga_replies(
            _render_nga_html(nga_latest_page_url(thread["link"])),
            limit=min(reply_limit, total_limit - reply_count),
        )
        if not replies:
            continue
        lines.append(
            f"\n[{scope}主题] {thread['title']} | 关键词={keyword} | Link: {thread['link']}"
        )
        for reply in replies:
            lines.append(
                f"- {reply['date'] or '?'} | {reply['author'] or '匿名'} | "
                f"{reply['floor']}楼: {reply['content']}"
            )
        reply_count += len(replies)

    if reply_count == 0:
        raise RuntimeError(f"no NGA replies found for {ticker.upper()}")
    lines[1] += f"；提取楼层 {reply_count} 条"
    return "\n".join(lines)
