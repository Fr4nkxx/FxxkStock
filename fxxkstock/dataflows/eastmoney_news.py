"""东方财富新闻抓取 — 免 API Key，使用网站前端 JSON 接口。"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dateutil.relativedelta import relativedelta

from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .market_utils import detect_market_region, to_eastmoney_symbol
from .news_utils import format_article_block, in_news_window

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
_SEARCH_API = "https://search-api-web.eastmoney.com/search/jsonp"
_STOCK_NEWS_API = "https://np-anotice-stock.eastmoney.com/api/security/ann"


def _http_get(url: str, timeout: float = 12.0) -> str:
    req = Request(url, headers={"User-Agent": _UA, "Accept": "*/*", "Referer": "https://www.eastmoney.com/"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code in (429, 503):
            raise VendorRateLimitError(str(exc)) from exc
        raise


def _parse_jsonp(text: str) -> dict:
    """Strip JSONP wrapper and parse JSON body."""
    text = text.strip()
    match = re.match(r"^[^(]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
    payload = match.group(1) if match else text
    return json.loads(payload)


def _parse_pub_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = value / 1000 if value > 1e12 else value
        with contextlib.suppress(ValueError, OSError, TypeError):
            return datetime.fromtimestamp(ts)
    if isinstance(value, str) and value:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            with contextlib.suppress(ValueError):
                return datetime.strptime(value[:19], fmt)
    return None


def _search_news(keyword: str, limit: int) -> list[dict]:
    """Search East Money news by keyword via frontend JSONP API."""
    param = json.dumps(
        {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "pageIndex": 1,
            "pageSize": limit,
        },
        ensure_ascii=False,
    )
    url = f"{_SEARCH_API}?cb=jQuery&param={quote(param)}"
    raw = _http_get(url)
    data = _parse_jsonp(raw)
    result = data.get("result") or {}
    articles = result.get("cmsArticleWebOld") or []
    parsed = []
    for item in articles:
        parsed.append(
            {
                "title": item.get("title") or "No title",
                "summary": item.get("content") or item.get("summary") or "",
                "publisher": item.get("mediaName") or item.get("source") or "东方财富",
                "link": item.get("url") or item.get("code") or "",
                "pub_date": _parse_pub_date(item.get("date") or item.get("publishTime")),
            }
        )
    return parsed


def _stock_announcement_news(em_code: str, bare_code: str, limit: int) -> list[dict]:
    """Fetch stock-specific news/announcements from np-anotice-stock API."""
    url = (
        f"{_STOCK_NEWS_API}?cb=&page_size={limit}&page_index=1"
        f"&ann_type=A&client_source=web&stock_list={bare_code}&fer_type=0"
    )
    raw = _http_get(url)
    # 响应可能是 JSONP 或纯 JSON
    with contextlib.suppress(json.JSONDecodeError):
        if raw.strip().startswith("{"):
            data = json.loads(raw)
        else:
            data = _parse_jsonp(raw)
        items = data.get("data") or data.get("list") or []
        parsed = []
        for item in items:
            title = item.get("title") or item.get("notice_title") or "No title"
            link = item.get("art_code") or item.get("notice_url") or ""
            if link and not link.startswith("http"):
                link = f"https://data.eastmoney.com/notices/detail/{bare_code}/{link}.html"
            parsed.append(
                {
                    "title": title,
                    "summary": item.get("summary") or "",
                    "publisher": "东方财富公告",
                    "link": link,
                    "pub_date": _parse_pub_date(
                        item.get("notice_date") or item.get("display_time") or item.get("publish_date")
                    ),
                }
            )
        return parsed
    return []


def _fetch_stock_articles(ticker: str, region: str, limit: int) -> list[dict]:
    em_code, bare = to_eastmoney_symbol(ticker, region)
    articles: list[dict] = []
    seen: set[str] = set()

    for fetcher in (
        lambda: _stock_announcement_news(em_code, bare, limit),
        lambda: _search_news(bare, limit),
        lambda: _search_news(ticker.split(".")[0], limit),
    ):
        try:
            batch = fetcher()
        except Exception as exc:  # noqa: BLE001
            logger.debug("East Money news fetcher failed for %s: %s", ticker, exc)
            continue
        for art in batch:
            title = art.get("title", "")
            if title and title not in seen:
                seen.add(title)
                articles.append(art)
        if len(articles) >= limit:
            break
        delay = get_config().get("cn_http_inter_request_delay", 0.5)
        time.sleep(delay)
    return articles[:limit]


def get_eastmoney_news(ticker: str, start_date: str, end_date: str) -> str:
    """Retrieve ticker news from East Money (vendor entry point for get_news)."""
    config = get_config()
    limit = config.get("cn_news_article_limit", config["news_article_limit"])
    region = config.get("market_region") or detect_market_region(ticker)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    try:
        articles = _fetch_stock_articles(ticker, region, limit)
    except VendorRateLimitError:
        raise
    except Exception as exc:
        logger.warning("East Money news failed for %s: %s", ticker, exc)
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    if not articles:
        raise NoMarketDataError(
            ticker, ticker, f"no East Money news between {start_date} and {end_date}"
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
            ticker, ticker, f"no East Money news in window {start_date} to {end_date}"
        )

    return f"## {ticker} News (East Money), from {start_date} to {end_date}:\n\n{news_str}"


def get_eastmoney_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve CN macro/global news from East Money search (vendor entry for get_global_news)."""
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
            batch = _search_news(query, limit)
        except VendorRateLimitError:
            raise
        except Exception as exc:
            logger.warning("East Money global search failed for %r: %s", query, exc)
            continue
        for art in batch:
            title = art.get("title", "")
            if title and title not in seen:
                seen.add(title)
                all_articles.append(art)
        if len(all_articles) >= limit:
            break

    if not all_articles:
        raise NoMarketDataError("GLOBAL", "GLOBAL", f"no East Money global news for {curr_date}")

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
    return f"## {label} (East Money), from {start_date} to {curr_date}:\n\n{news_str}"
