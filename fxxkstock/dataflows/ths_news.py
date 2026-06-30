"""Browser-backed Tonghuashun ticker news for China-market instruments."""

from __future__ import annotations

import re
from datetime import datetime

from parsel import Selector

from .config import get_config
from .market_utils import detect_market_region, to_eastmoney_symbol
from .news_utils import format_article_block, in_news_window
from .playwright_web import render_html
from .symbol_utils import NoMarketDataError


def _ths_news_url(ticker: str, region: str) -> str:
    _, bare = to_eastmoney_symbol(ticker, region)
    return f"https://stockpage.10jqka.com.cn/{bare}/news/"


def parse_ths_news_html(html: str) -> list[dict]:
    """Parse article links from Tonghuashun's current Next.js stock page."""
    selector = Selector(text=html)
    articles: list[dict] = []
    seen: set[str] = set()
    for anchor in selector.css("a"):
        link = (anchor.css("::attr(href)").get() or "").strip()
        match = re.search(r"news\.10jqka\.com\.cn/(\d{8})/c\d+\.shtml", link)
        if not match or link in seen:
            continue
        title = " ".join(anchor.xpath(".//text()").getall())
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) < 6:
            continue
        try:
            published = datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue
        seen.add(link)
        articles.append(
            {
                "title": title,
                "summary": "",
                "publisher": "同花顺",
                "link": link,
                "pub_date": published,
            }
        )
    return articles


def get_browser_ths_news(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int | None = None,
) -> str:
    """Return dated Tonghuashun ticker headlines fetched through local Chrome."""
    config = get_config()
    if limit is None:
        limit = int(config.get("cn_global_news_article_limit", 10))
    region = config.get("market_region") or detect_market_region(ticker)
    html = render_html(_ths_news_url(ticker, region), wait_selector="body")
    articles = parse_ths_news_html(html)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    articles = [
        article
        for article in articles
        if in_news_window(article["pub_date"], start_dt, end_dt)
    ][:limit]
    if not articles:
        raise NoMarketDataError(
            ticker,
            ticker,
            f"no Tonghuashun browser news between {start_date} and {end_date}",
        )
    blocks = [
        format_article_block(
            article["title"],
            article["publisher"],
            article["summary"],
            article["link"],
        )
        for article in articles
    ]
    return (
        f"## {ticker.upper()} News (Tonghuashun Browser), "
        f"from {start_date} to {end_date}:\n\n"
        + "".join(blocks)
    )
