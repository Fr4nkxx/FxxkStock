"""Diagnose CN browser, community, news, and prediction-market data sources."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import ssl
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from parsel import Selector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fxxkstock.dataflows.chrome_manager import ChromeManager
from fxxkstock.dataflows.cn_community import (
    _http_get as community_http_get,
    _parse_taoguba_from_html,
    _parse_ths_from_html,
    _taoguba_stock_url,
    _ths_stock_url,
)
from fxxkstock.dataflows.config import get_config, set_config
from fxxkstock.dataflows.eastmoney_guba import fetch_eastmoney_guba
from fxxkstock.dataflows.eastmoney_browser import (
    _global_news_search_url,
    _parse_news_from_html,
)
from fxxkstock.dataflows.eastmoney_news import _search_news
from fxxkstock.dataflows.market_utils import detect_market_region
from fxxkstock.dataflows.market_utils import get_security_cn_name
from fxxkstock.dataflows.playwright_web import render_html
from fxxkstock.dataflows.polymarket import _request as polymarket_request
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.agents.utils.agent_utils import resolve_instrument_identity

NGA_GREAT_TIMES_URL = "https://bbs.nga.cn/thread.php?fid=706"
_NGA_INDUSTRY_NAME_HINTS = (
    "人工智能", "半导体", "芯片", "电子", "机器人", "新能源", "光伏",
    "军工", "医药", "医疗", "证券", "银行", "保险", "房地产", "汽车",
    "消费", "白酒", "黄金", "有色", "煤炭", "石油", "化工", "传媒",
    "游戏", "通信", "算力", "软件", "互联网",
)
_NGA_ENGLISH_INDUSTRY_MAP = {
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


@dataclass
class Result:
    source: str
    transport: str
    status: str
    duration_ms: int
    detail: str
    items: int | None = None


def classify_error(exc: BaseException) -> str:
    text = str(exc).lower()
    if "login required" in text or "需要登录" in text or "请先登录" in text:
        return "login_required"
    if isinstance(exc, ssl.SSLError) or "unexpected_eof" in text or "ssl" in text:
        return "ssl_error"
    if "captcha" in text or "验证码" in text or "blocked" in text:
        return "blocked"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "websocket" in text or "cdp" in text or "econnrefused" in text:
        return "browser_unavailable"
    if "no " in text and ("data" in text or "posts" in text or "news" in text):
        return "success_empty"
    return "error"


def compact_error(exc: BaseException, limit: int = 280) -> str:
    text = " ".join(str(exc).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text or type(exc).__name__


def run_check(
    source: str,
    transport: str,
    operation: Callable[[], tuple[str, int | None]],
) -> Result:
    started = time.monotonic()
    try:
        detail, items = operation()
        status = "success" if items is None or items > 0 else "success_empty"
    except Exception as exc:  # noqa: BLE001
        status = classify_error(exc)
        detail = compact_error(exc)
        items = None
    return Result(
        source=source,
        transport=transport,
        status=status,
        duration_ms=round((time.monotonic() - started) * 1000),
        detail=detail,
        items=items,
    )


def save_html(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", errors="replace")


def browser_posts_check(
    url: str,
    parser: Callable[[str], list[dict]],
    artifact_path: Path,
) -> tuple[str, int]:
    html = render_html(url, wait_selector="body")
    save_html(artifact_path, html)
    posts = parser(html)
    return f"loaded {len(html)} HTML characters; saved {artifact_path}", len(posts)


def http_posts_check(
    url: str,
    referer: str,
    parser: Callable[[str], list[dict]],
    timeout: float,
) -> tuple[str, int]:
    raw = community_http_get(url, referer=referer, timeout=timeout)
    posts = parser(raw.decode("utf-8", errors="replace"))
    return f"loaded {len(raw)} response bytes", len(posts)


def eastmoney_guba_check(ticker: str) -> tuple[str, int]:
    text = fetch_eastmoney_guba(ticker, limit=5)
    lowered = text.lower()
    if lowered.startswith("<") or "no guba" in lowered:
        return compact_error(RuntimeError(text)), 0
    return text.splitlines()[0], max(0, len(text.splitlines()) - 1)


def news_check(query: str) -> tuple[str, int]:
    articles = _search_news(query, 5)
    return f"query={query}", len(articles)


def browser_news_check(query: str, artifact_path: Path) -> tuple[str, int]:
    html = render_html(_global_news_search_url(query), wait_selector="body")
    save_html(artifact_path, html)
    articles = _parse_news_from_html(html)
    parser_name = "production"
    if not articles:
        articles = parse_eastmoney_news_experimental(html)
        parser_name = "experimental:.news_item"
    return (
        f"query={query}; parser={parser_name}; loaded {len(html)} HTML characters; "
        f"saved {artifact_path}",
        len(articles),
    )


def parse_eastmoney_news_experimental(html: str) -> list[dict]:
    """Test the underscore-based selectors used by East Money's current page."""
    selector = Selector(text=html)
    articles: list[dict] = []
    for row in selector.css("div.news_item"):
        title = " ".join(row.css(".news_item_t a *::text, .news_item_t a::text").getall())
        title = " ".join(title.split())
        href = row.css(".news_item_t a::attr(href)").get()
        if title and href:
            articles.append({"title": title, "link": href})
    return articles


def parse_ths_news_candidates(html: str) -> list[dict]:
    """Count article-like links on the current Next.js stock-news page."""
    selector = Selector(text=html)
    candidates: list[dict] = []
    seen: set[str] = set()
    for anchor in selector.css("a"):
        href = (anchor.css("::attr(href)").get() or "").strip()
        title = " ".join(anchor.xpath(".//text()").getall())
        title = " ".join(title.split())
        if (
            href
            and title
            and len(title) >= 6
            and href not in seen
            and any(token in href.lower() for token in ("news", "article", "/a/"))
        ):
            seen.add(href)
            candidates.append({"title": title, "link": href})
    return candidates


def parse_nga_thread_candidates(html: str) -> list[dict]:
    """Parse thread titles from NGA's classic and current forum markup."""
    selector = Selector(text=html)
    threads: list[dict] = []
    seen: set[str] = set()
    unavailable_titles = {
        "帖子发布或回复时间超过限制",
        "帐号权限不足",
        "主题不存在",
        "主题已被删除",
    }
    selectors = (
        "a.topic",
        "a[class*='topic']",
        "a[href*='read.php?tid=']",
        "a[href*='/read.php?tid=']",
    )
    for anchor in selector.css(", ".join(selectors)):
        href = (anchor.css("::attr(href)").get() or "").strip()
        title = " ".join(anchor.xpath(".//text()").getall())
        title = " ".join(title.split())
        if not href or len(title) < 4 or title in unavailable_titles:
            continue
        if href.startswith("//"):
            href = f"https:{href}"
        elif href.startswith("/"):
            href = f"https://bbs.nga.cn{href}"
        elif not href.startswith("http"):
            href = f"https://bbs.nga.cn/{href.lstrip('/')}"
        parsed = urlsplit(href)
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "tid" not in params or "page" in params:
            continue
        if href in seen:
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
        threads.append(
            {
                "title": title,
                "link": href,
                "last_reply": last_reply,
            }
        )
    return threads


def parse_nga_replies(html: str, limit: int | None = None) -> list[dict]:
    """Parse the opening post and replies from an NGA thread page."""
    selector = Selector(text=html)
    replies: list[dict] = []
    for content in selector.css("[id^='postcontent']"):
        content_id = content.css("::attr(id)").get() or ""
        match = re.fullmatch(r"postcontent(\d+)", content_id)
        if not match:
            continue
        floor = int(match.group(1))
        body = " ".join(content.xpath(".//text()").getall())
        body = " ".join(body.split())
        if not body:
            continue
        author = " ".join(
            selector.css(f"#postauthor{floor}").xpath(".//text()").getall()
        )
        subject = " ".join(
            selector.css(f"#postsubject{floor}").xpath(".//text()").getall()
        )
        posted_at = " ".join(
            selector.css(f"#postdate{floor}").xpath(".//text()").getall()
        )
        replies.append(
            {
                "floor": floor,
                "author": " ".join(author.split()),
                "date": " ".join(posted_at.split()),
                "subject": " ".join(subject.split()),
                "content": body,
            }
        )
        if limit is not None and len(replies) >= limit:
            break
    return replies


def nga_search_url(base_url: str, keyword: str) -> str:
    """Build an NGA in-forum title-search URL without changing its forum id."""
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["key"] = keyword
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def nga_latest_page_url(thread_url: str) -> str:
    """Open the latest page so active threads include their newest replies."""
    parts = urlsplit(thread_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["page"] = "e"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def filter_recent_nga_threads(
    threads: list[dict],
    lookback_days: int,
    *,
    now: datetime | None = None,
) -> list[dict]:
    """Keep threads whose last-reply timestamp falls within the lookback window."""
    cutoff = (now or datetime.now()) - timedelta(days=lookback_days)
    recent: list[dict] = []
    for thread in threads:
        raw = str(thread.get("last_reply") or "").strip()
        try:
            replied_at = datetime.strptime(raw, "%y-%m-%d %H:%M")
        except ValueError:
            continue
        if replied_at >= cutoff:
            recent.append(thread)
    return recent


def infer_nga_industry_term(
    chinese_name: str,
    identity: dict | None = None,
) -> str | None:
    """Resolve a concise Chinese industry keyword for fallback search."""
    for hint in _NGA_INDUSTRY_NAME_HINTS:
        if hint in chinese_name:
            return hint
    metadata = identity or {}
    classification = " ".join(
        str(metadata.get(key) or "") for key in ("industry", "sector")
    ).casefold()
    for english, chinese in _NGA_ENGLISH_INDUSTRY_MAP.items():
        if english in classification:
            return chinese
    return None


def _render_nga_search_html(url: str, *, keep_open: bool = False) -> str:
    """Render NGA and pass its timed welcome/ad interstitial once."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    config = get_config()
    cdp_url = config.get("cn_browser_cdp_url", "http://127.0.0.1:9222")
    timeout_ms = int(config.get("cn_browser_nav_timeout_ms", 20000))
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url, timeout=timeout_ms)
            if not browser.contexts:
                raise RuntimeError(f"no browser context at {cdp_url}")
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
                return page.content()
            finally:
                if not keep_open:
                    page.close()
    except PlaywrightError as exc:
        raise RuntimeError(f"NGA browser navigation failed: {exc}") from exc


def _fetch_nga_search(
    url: str,
    query: str,
    artifact_path: Path,
    *,
    keep_open: bool = False,
) -> tuple[list[dict], int]:
    html = _render_nga_search_html(
        nga_search_url(url, query),
        keep_open=keep_open,
    )
    save_html(artifact_path, html)
    compact = " ".join(Selector(text=html).xpath("//body//text()").getall())
    compact = " ".join(compact.split())
    lowered = compact.lower()
    if any(
        token in lowered
        for token in ("验证码", "访问过于频繁", "禁止访问", "access denied", "403 forbidden")
    ):
        raise RuntimeError(f"blocked or captcha page; saved {artifact_path}")
    if "欢迎访问NGA玩家社区" in compact:
        raise RuntimeError(f"blocked by NGA welcome interstitial; saved {artifact_path}")

    threads = parse_nga_thread_candidates(html)
    if not threads and any(
        token in compact for token in ("需要登录", "请先登录", "登录后", "用户登录")
    ):
        raise RuntimeError(f"login required; saved {artifact_path}")
    return threads, len(html)


def _fetch_nga_thread_replies(
    thread: dict,
    artifact_path: Path,
    *,
    reply_limit: int,
    keep_open: bool = False,
) -> tuple[list[dict], int]:
    html = _render_nga_search_html(
        nga_latest_page_url(thread["link"]),
        keep_open=keep_open,
    )
    save_html(artifact_path, html)
    replies = parse_nga_replies(html, limit=reply_limit)
    return replies, len(html)


def nga_sentiment_search_check(
    *,
    ticker: str,
    region: str,
    url: str,
    stock_query: str | None,
    industry_query: str | None,
    min_stock_posts: int,
    lookback_days: int,
    thread_limit: int,
    reply_limit: int,
    artifact_dir: Path,
    keep_open: bool = False,
) -> tuple[str, int]:
    """Search Chinese stock name first, then industry when mentions are sparse."""
    chinese_name = (stock_query or get_security_cn_name(ticker, region) or "").strip()
    if not chinese_name:
        raise RuntimeError(
            "Chinese security name unavailable; pass --nga-query with the Chinese name"
        )
    if chinese_name == ticker or chinese_name == ticker.split(".")[0]:
        raise RuntimeError("NGA stock query must be a Chinese name, not a ticker code")

    identity: dict = {}
    resolved_industry = (industry_query or "").strip()
    if not resolved_industry:
        resolved_industry = infer_nga_industry_term(chinese_name) or ""
    if not resolved_industry:
        identity = resolve_instrument_identity(ticker)
        resolved_industry = infer_nga_industry_term(chinese_name, identity) or ""

    stock_threads, stock_html_size = _fetch_nga_search(
        url,
        chinese_name,
        artifact_dir / "nga_stock.html",
        keep_open=keep_open,
    )
    recent_stock_threads = filter_recent_nga_threads(stock_threads, lookback_days)
    industry_threads: list[dict] = []
    recent_industry_threads: list[dict] = []
    industry_html_size = 0
    fallback_attempted = (
        len(recent_stock_threads) < min_stock_posts and bool(resolved_industry)
    )
    if fallback_attempted:
        industry_threads, industry_html_size = _fetch_nga_search(
            url,
            resolved_industry,
            artifact_dir / "nga_industry.html",
            keep_open=keep_open,
        )
        recent_industry_threads = filter_recent_nga_threads(
            industry_threads,
            lookback_days,
        )

    inspected: list[dict] = []
    thread_html_size = 0
    scopes = (
        ("stock", recent_stock_threads[:thread_limit]),
        ("industry", recent_industry_threads[:thread_limit]),
    )
    for scope, threads in scopes:
        for index, thread in enumerate(threads, start=1):
            replies, html_size = _fetch_nga_thread_replies(
                thread,
                artifact_dir / f"nga_{scope}_thread_{index}.html",
                reply_limit=reply_limit,
                keep_open=keep_open,
            )
            thread_html_size += html_size
            inspected.append(
                {
                    "scope": scope,
                    "query": chinese_name if scope == "stock" else resolved_industry,
                    "thread": thread,
                    "replies": replies,
                }
            )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    replies_path = artifact_dir / "nga_replies.json"
    replies_path.write_text(
        json.dumps(inspected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reply_count = sum(len(item["replies"]) for item in inspected)
    samples = " | ".join(item["thread"]["title"] for item in inspected[:3])
    return (
        f"stock={chinese_name}:{len(stock_threads)} (recent={len(recent_stock_threads)}); "
        f"industry={resolved_industry or '-'}:{len(industry_threads)} "
        f"(recent={len(recent_industry_threads)}); "
        f"fallback={'yes' if fallback_attempted else 'no'}; "
        f"threads={len(inspected)}; replies={reply_count}; "
        f"loaded={stock_html_size + industry_html_size + thread_html_size} HTML characters; "
        f"sample={samples or '-'}; saved {replies_path}",
        reply_count,
    )


def polymarket_check(topic: str) -> tuple[str, int]:
    payload = polymarket_request(
        "public-search",
        {"q": topic, "limit_per_type": 20},
    )
    events = payload.get("events", [])
    return f"query={topic}", len(events)


def proxy_detail() -> str:
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    active = [f"{key}={os.environ[key]}" for key in keys if os.environ.get(key)]
    return "; ".join(active) if active else "no proxy environment variables"


def print_results(results: list[Result]) -> None:
    widths = {
        "source": max(8, *(len(item.source) for item in results)),
        "transport": max(9, *(len(item.transport) for item in results)),
        "status": max(6, *(len(item.status) for item in results)),
    }
    print(
        f"{'SOURCE':<{widths['source']}}  "
        f"{'TRANSPORT':<{widths['transport']}}  "
        f"{'STATUS':<{widths['status']}}  TIME    ITEMS  DETAIL"
    )
    print("-" * 110)
    for item in results:
        count = "-" if item.items is None else str(item.items)
        print(
            f"{item.source:<{widths['source']}}  "
            f"{item.transport:<{widths['transport']}}  "
            f"{item.status:<{widths['status']}}  "
            f"{item.duration_ms:>5}ms  {count:>5}  {item.detail}"
        )


def finish_results(results: list[Result], json_output: bool) -> int:
    if json_output:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print_results(results)
    failures = {
        "ssl_error",
        "blocked",
        "rate_limited",
        "timeout",
        "browser_unavailable",
        "error",
        "failed_fallback",
        "login_required",
    }
    return 1 if any(item.status in failures for item in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test FxxKStock browser/community/news data sources without running agents."
    )
    parser.add_argument("--ticker", default="159516.SZ")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--platform", choices=("macos", "windows", "ubuntu"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--only-nga",
        action="store_true",
        help="Run only Chrome health and NGA diagnostics.",
    )
    parser.add_argument(
        "--keep-nga-open",
        action="store_true",
        help="Leave NGA search tabs open in the managed Chrome for inspection.",
    )
    parser.add_argument(
        "--nga-url",
        default=NGA_GREAT_TIMES_URL,
        help="NGA Great Times forum URL to test.",
    )
    parser.add_argument(
        "--nga-query",
        default=None,
        help="Chinese stock/security name; auto-resolved when omitted.",
    )
    parser.add_argument(
        "--nga-industry",
        default=None,
        help="Chinese industry fallback keyword; inferred when omitted.",
    )
    parser.add_argument(
        "--nga-min-stock-posts",
        type=int,
        default=3,
        help="Use industry fallback when fewer stock-name threads are found.",
    )
    parser.add_argument(
        "--nga-lookback-days",
        type=int,
        default=30,
        help="Only inspect threads active within this many days.",
    )
    parser.add_argument(
        "--nga-thread-limit",
        type=int,
        default=3,
        help="Maximum recent threads to inspect for each search scope.",
    )
    parser.add_argument(
        "--nga-reply-limit",
        type=int,
        default=20,
        help="Maximum floors to extract from each inspected thread.",
    )
    parser.add_argument(
        "--include-disabled-sources",
        action="store_true",
        help="Also test disabled legacy community sources (THS community and Taoguba).",
    )
    parser.add_argument(
        "--include-http-fallbacks",
        action="store_true",
        help="Also test HTTP fallbacks even when their browser primary succeeds.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="logs/source_diagnostics",
        help="Directory for rendered browser HTML artifacts.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.artifacts_dir) / run_stamp

    # Source libraries log full Playwright/Surge HTML in some exceptions. The
    # diagnostic owns error rendering and always truncates it.
    logging.disable(logging.CRITICAL)

    config = DEFAULT_CONFIG.copy()
    if args.platform:
        config["cn_browser_platform"] = args.platform
    config["cn_browser_nav_timeout_ms"] = int(args.timeout * 1000)
    config["market_region"] = detect_market_region(args.ticker)
    set_config(config)

    results = [
        Result("environment", "process", "info", 0, proxy_detail()),
        Result("artifacts", "filesystem", "info", 0, str(artifact_dir)),
    ]
    manager = ChromeManager(config)
    browser_started = False
    if not args.no_browser:
        status = manager.ensure_running()
        browser_started = status.get("managed", False)
        results.append(
            Result(
                "chrome_cdp",
                "local",
                "success" if status.get("available") else status.get("state", "error"),
                0,
                compact_error(RuntimeError(status.get("message", status.get("state", "")))),
            )
        )

    region = config["market_region"]
    if args.only_nga:
        if args.no_browser:
            results.append(
                Result(
                    "nga_great_times",
                    "browser",
                    "browser_unavailable",
                    0,
                    "--only-nga requires browser access",
                )
            )
        else:
            results.append(run_check(
                "nga_great_times",
                "browser",
                lambda: nga_sentiment_search_check(
                    ticker=args.ticker,
                    region=region,
                    url=args.nga_url,
                    stock_query=args.nga_query,
                    industry_query=args.nga_industry,
                    min_stock_posts=max(1, args.nga_min_stock_posts),
                    lookback_days=max(1, args.nga_lookback_days),
                    thread_limit=max(1, args.nga_thread_limit),
                    reply_limit=max(1, args.nga_reply_limit),
                    artifact_dir=artifact_dir,
                    keep_open=args.keep_nga_open,
                ),
            ))
        if browser_started and not args.keep_nga_open:
            manager.close_managed()
        return finish_results(results, args.json_output)

    ths_url = _ths_stock_url(args.ticker, region)
    ths_news_url = ths_url.rstrip("/") + "/news/"
    taoguba_url = _taoguba_stock_url(args.ticker, region)

    results.append(run_check(
        "eastmoney_guba",
        "http",
        lambda: eastmoney_guba_check(args.ticker),
    ))
    if not args.no_browser:
        results.append(run_check(
            "ths_news",
            "browser",
            lambda: browser_posts_check(
                ths_news_url,
                parse_ths_news_candidates,
                artifact_dir / "ths_news.html",
            ),
        ))
        results.append(run_check(
            "nga_great_times",
            "browser",
            lambda: nga_sentiment_search_check(
                ticker=args.ticker,
                region=region,
                url=args.nga_url,
                stock_query=args.nga_query,
                industry_query=args.nga_industry,
                min_stock_posts=max(1, args.nga_min_stock_posts),
                lookback_days=max(1, args.nga_lookback_days),
                thread_limit=max(1, args.nga_thread_limit),
                reply_limit=max(1, args.nga_reply_limit),
                artifact_dir=artifact_dir,
                keep_open=args.keep_nga_open,
            ),
        ))
    if args.include_disabled_sources:
        if not args.no_browser:
            results.append(run_check(
                "ths",
                "browser",
                lambda: browser_posts_check(
                    ths_url,
                    _parse_ths_from_html,
                    artifact_dir / "ths.html",
                ),
            ))
        results.append(run_check(
            "ths",
            "http",
            lambda: http_posts_check(
                ths_url,
                "https://stockpage.10jqka.com.cn/",
                _parse_ths_from_html,
                args.timeout,
            ),
        ))
        if not args.no_browser:
            results.append(run_check(
                "taoguba",
                "browser",
                lambda: browser_posts_check(
                    taoguba_url,
                    _parse_taoguba_from_html,
                    artifact_dir / "taoguba.html",
                ),
            ))
        results.append(run_check(
            "taoguba",
            "http",
            lambda: http_posts_check(
                taoguba_url,
                "https://www.tgb.cn/",
                _parse_taoguba_from_html,
                args.timeout,
            ),
        ))

    for query in config.get("cn_global_news_queries", []):
        slug = str(abs(hash(query)))
        if not args.no_browser:
            results.append(run_check(
                f"eastmoney_news:{query}",
                "browser",
                lambda query=query, slug=slug: browser_news_check(
                    query,
                    artifact_dir / f"eastmoney_news_{slug}.html",
                ),
            ))
        if args.include_http_fallbacks or args.no_browser:
            results.append(run_check(
                f"eastmoney_news:{query}",
                "http",
                lambda query=query: news_check(query),
            ))
    results.append(run_check(
        "polymarket",
        "http",
        lambda: polymarket_check("Fed rate cut 2026"),
    ))

    if browser_started and not args.keep_nga_open:
        manager.close_managed()

    return finish_results(results, args.json_output)


if __name__ == "__main__":
    raise SystemExit(main())
