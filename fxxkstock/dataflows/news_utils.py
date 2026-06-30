"""Shared news date-window filtering and article formatting utilities."""

from __future__ import annotations

from datetime import datetime

from dateutil.relativedelta import relativedelta


def in_news_window(pub_date, start_dt: datetime, end_dt: datetime) -> bool:
    """Whether an article belongs in the [start_dt, end_dt] window.

    Dated articles are kept only if they fall in the window. An undated article
    is kept only when the window reaches the present (live run) — in a
    historical/backtest window it's excluded, since we can't prove it isn't
    future news (look-ahead safety, #992/#1007).
    """
    if pub_date is not None:
        naive = pub_date.replace(tzinfo=None) if hasattr(pub_date, "replace") else pub_date
        return start_dt <= naive <= end_dt + relativedelta(days=1)
    return end_dt >= datetime.now() - relativedelta(days=1)


def format_article_block(title: str, publisher: str, summary: str = "", link: str = "") -> str:
    """Format a single news article as markdown-ish plaintext."""
    block = f"### {title} (source: {publisher})\n"
    if summary:
        block += f"{summary}\n"
    if link:
        block += f"Link: {link}\n"
    return block + "\n"
