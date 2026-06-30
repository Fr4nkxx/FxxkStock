"""扫描磁盘上的历史报告目录。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yfinance as yf
import pandas as pd

from fxxkstock.agents.utils.agent_utils import resolve_instrument_identity
from fxxkstock.dataflows.market_utils import (
    detect_market_region,
    get_security_cn_name,
    is_cn_region,
)
from fxxkstock.dataflows.symbol_utils import normalize_symbol
from fxxkstock.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_REPORT_DIR_PATTERN = re.compile(r"^(.+)_(\d{8})_(\d{6})$")
_CORE_INSIGHTS_FILE = "core_insights.json"
_CHART_RANGES = {
    "1d": ("1d", "5m"),
    "5d": ("5d", "15m"),
    "1mo": ("1mo", "1d"),
    "6mo": ("6mo", "1d"),
    "1y": ("1y", "1d"),
}
_OVERVIEW_CACHE_TTL_SECONDS = 60
_overview_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _infer_instrument_type(ticker: str, quote_type: str | None = None) -> str:
    """Return a user-facing type, with deterministic CN code fallbacks."""
    upper = ticker.upper().strip()
    bare = upper.split(".")[0]
    if upper.endswith((".SS", ".SZ")) and re.fullmatch(r"\d{6}", bare):
        if bare.startswith(("51", "56", "58", "15", "16")):
            return "ETF"
        if bare.startswith(("600", "601", "603", "605", "688", "689", "000", "001", "002", "003", "300", "301")):
            return "股票"
    normalized_type = (quote_type or "").strip().upper()
    type_labels = {
        "EQUITY": "股票",
        "ETF": "ETF",
        "MUTUALFUND": "基金",
        "INDEX": "指数",
        "FUTURE": "期货",
        "CRYPTOCURRENCY": "加密货币",
    }
    if normalized_type in type_labels:
        return type_labels[normalized_type]
    return "证券"


def get_reports_root() -> Path:
    """返回 reports 根目录（与 web runner / main.py 默认保存路径一致）。"""
    return Path("reports")


def _parse_report_dir_name(name: str) -> dict[str, str] | None:
    match = _REPORT_DIR_PATTERN.match(name)
    if not match:
        return None
    ticker, ymd, hms = match.groups()
    try:
        created_at = datetime.strptime(f"{ymd}{hms}", "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        created_at = f"{ymd} {hms}"
    return {"ticker": ticker, "created_at": created_at}


def _extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        text = line.strip()
        if text.startswith("# "):
            return text[2:].strip()
    return ""


def _extract_decision(markdown: str) -> str | None:
    upper = markdown.upper()
    for token in ("BUY", "SELL", "HOLD", "OVERWEIGHT", "UNDERWEIGHT"):
        if f"**{token}**" in upper or f"FINAL TRANSACTION PROPOSAL: **{token}**" in upper:
            return token
    return None


def _extract_instrument_name(markdown: str, ticker: str) -> str | None:
    """Best-effort local name extraction without another market-data request."""
    bare = re.escape(ticker.split(".")[0])
    patterns = (
        rf"([^\n|]{{2,100}})\s*[（(]{bare}(?:\.(?:SS|SZ|HK))?[）)]",
        rf"{bare}(?:\.(?:SS|SZ|HK))?\s*[（(]([^）)\n]{{2,80}})[）)]",
        r"(?:标的|公司|基金名称)\s*[：:*]*\s*([^|\n（(]{2,80})",
    )
    rejected = {"分析日期", "最近交易日", "已验证", "股票", "证券", "ETF"}
    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if not match:
            continue
        name = re.sub(r"[*#`]", "", match.group(1))
        name = re.sub(r"^[^\w\u4e00-\u9fff]+", "", name).strip(" ：:|-（(")
        if re.search(r"(?:报告|标的|公司|基金名称)\s*[：:]", name):
            name = re.split(r"[：:]", name)[-1].strip()
        if name and name.upper() != ticker and name not in rejected:
            return name
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_report_sections(report_dir: Path) -> dict[str, str]:
    """Group the saved report tree into the UI's reader tabs."""
    summary = _read_text(report_dir / "5_portfolio" / "decision.md")
    thesis_parts = [
        _read_text(report_dir / "2_research" / "manager.md"),
        _read_text(report_dir / "3_trading" / "trader.md"),
    ]
    risk_parts = [
        _read_text(report_dir / "4_risk" / "aggressive.md"),
        _read_text(report_dir / "4_risk" / "neutral.md"),
        _read_text(report_dir / "4_risk" / "conservative.md"),
    ]
    data_parts = [
        _read_text(report_dir / "1_analysts" / "market.md"),
        _read_text(report_dir / "1_analysts" / "fundamentals.md"),
        _read_text(report_dir / "1_analysts" / "news.md"),
        _read_text(report_dir / "1_analysts" / "sentiment.md"),
    ]
    return {
        "summary": summary,
        "thesis": "\n\n---\n\n".join(part for part in thesis_parts if part),
        "risks": "\n\n---\n\n".join(part for part in risk_parts if part),
        "data": "\n\n---\n\n".join(part for part in data_parts if part),
    }


def read_core_insights(report_dir: Path) -> list[str]:
    """Read post-analysis AI insights; legacy reports intentionally return none."""
    path = report_dir / _CORE_INSIGHTS_FILE
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        logger.warning("Ignoring invalid core insights file: %s", path)
        return []
    insights = payload.get("insights") if isinstance(payload, dict) else None
    if not isinstance(insights, list):
        return []
    return [item.strip() for item in insights if isinstance(item, str) and item.strip()][:6]


def list_historical_reports(reports_root: Path | None = None, limit: int = 100) -> list[dict]:
    """列出 reports/ 下已保存的历史报告，按时间倒序。"""
    root = reports_root or get_reports_root()
    if not root.is_dir():
        return []

    items: list[dict] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        report_file = entry / "complete_report.md"
        if not report_file.is_file():
            continue
        meta = _parse_report_dir_name(entry.name) or {
            "ticker": entry.name,
            "created_at": "",
        }
        try:
            mtime = report_file.stat().st_mtime
        except OSError:
            mtime = 0
        markdown = report_file.read_text(encoding="utf-8", errors="replace")
        items.append(
            {
                "id": entry.name,
                "ticker": meta["ticker"],
                "created_at": meta["created_at"],
                "title": _extract_title(markdown) or meta["ticker"],
                "name": _extract_instrument_name(markdown, meta["ticker"]),
                "decision": _extract_decision(markdown),
                "report_dir": str(entry.resolve()),
                "modified_at": mtime,
            }
        )

    items.sort(key=lambda x: x["modified_at"], reverse=True)
    return items[:limit]


def _validate_report_id(report_id: str) -> None:
    if not report_id or report_id in {".", ".."}:
        raise ValueError("invalid report id")
    if "/" in report_id or "\\" in report_id or ".." in report_id:
        raise ValueError("invalid report id")


def get_historical_report(report_id: str, reports_root: Path | None = None) -> dict:
    """读取单个历史报告。"""
    _validate_report_id(report_id)
    root = reports_root or get_reports_root()
    report_dir = (root / report_id).resolve()
    if not str(report_dir).startswith(str(root.resolve())):
        raise ValueError("invalid report path")
    report_file = report_dir / "complete_report.md"
    if not report_file.is_file():
        raise FileNotFoundError(report_id)

    markdown = report_file.read_text(encoding="utf-8")
    meta = _parse_report_dir_name(report_id) or {"ticker": report_id, "created_at": ""}
    return {
        "id": report_id,
        "available": True,
        "ticker": meta["ticker"],
        "created_at": meta["created_at"],
        "title": _extract_title(markdown) or meta["ticker"],
        "decision": _extract_decision(markdown),
        "markdown": markdown,
        "sections": read_report_sections(report_dir),
        "core_insights": read_core_insights(report_dir),
        "report_dir": str(report_dir),
    }


def get_stock_overview(
    ticker: str,
    reports_root: Path | None = None,
    chart_range: str = "1d",
) -> dict[str, Any]:
    """Return identity, latest market move, and all saved reports for one ticker."""
    ticker = ticker.strip().upper()
    safe_ticker_component(ticker)
    cache_key = (ticker, chart_range)
    cached = _overview_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _OVERVIEW_CACHE_TTL_SECONDS:
        return cached[1]
    reports = [
        item for item in list_historical_reports(reports_root, limit=1000)
        if item["ticker"].upper() == ticker
    ]
    local_name = next((item.get("name") for item in reports if item.get("name")), None)

    region = detect_market_region(ticker)
    identity = {} if is_cn_region(region) else dict(resolve_instrument_identity(ticker))
    region = detect_market_region(ticker, identity)
    if is_cn_region(region) and not local_name:
        cn_name = get_security_cn_name(ticker, region)
        if cn_name:
            identity["company_name"] = cn_name
    elif local_name:
        identity["company_name"] = local_name

    if chart_range not in _CHART_RANGES:
        raise ValueError(f"unsupported chart range: {chart_range}")
    quote = {
        "last_price": None,
        "open": None,
        "high": None,
        "low": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "volume": None,
        "as_of": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "trailing_pe": None,
        "price_to_book": None,
        "market_cap": None,
        "turnover": None,
    }
    details = {
        "quote_type": identity.get("quote_type"),
        "instrument_type": _infer_instrument_type(ticker, identity.get("quote_type")),
        "category": None,
        "fund_family": None,
        "tracking_index": None,
        "inception_date": None,
        "website": None,
    }
    chart: list[dict[str, Any]] = []
    try:
        stock = yf.Ticker(normalize_symbol(ticker))
        period, interval = _CHART_RANGES[chart_range]
        frame = stock.history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
        closes = frame["Close"].dropna()
        if not closes.empty:
            last = float(closes.iloc[-1])
            previous = float(closes.iloc[-2]) if len(closes) > 1 else None
            quote["last_price"] = last
            quote["previous_close"] = previous
            if previous not in (None, 0):
                quote["change"] = last - previous
                quote["change_percent"] = (last - previous) / previous * 100
            if "Volume" in frame and len(frame["Volume"]):
                volume = frame["Volume"].iloc[-1]
                quote["volume"] = int(volume) if volume == volume else None
            latest = frame.iloc[-1]
            for source, target in (("Open", "open"), ("High", "high"), ("Low", "low")):
                value = latest.get(source)
                quote[target] = float(value) if value == value else None
            if quote["last_price"] is not None and quote["volume"] is not None:
                quote["turnover"] = quote["last_price"] * quote["volume"]
            index_value = closes.index[-1]
            quote["as_of"] = (
                index_value.isoformat()
                if hasattr(index_value, "isoformat")
                else str(index_value)
            )
            stride = max(1, len(closes) // 140)
            sampled = closes.iloc[::stride]
            if sampled.index[-1] != closes.index[-1]:
                sampled = pd.concat([sampled, closes.iloc[[-1]]])
            chart = [
                {
                    "time": index.isoformat() if hasattr(index, "isoformat") else str(index),
                    "close": round(float(value), 6),
                }
                for index, value in sampled.items()
            ]
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        for source, target in (
            ("longName", "company_name"),
            ("exchange", "exchange"),
            ("currency", "currency"),
            ("sector", "sector"),
            ("industry", "industry"),
            ("quoteType", "quote_type"),
        ):
            value = info.get(source)
            if value and not identity.get(target):
                identity[target] = value
        for source, target in (
            ("fiftyTwoWeekHigh", "fifty_two_week_high"),
            ("fiftyTwoWeekLow", "fifty_two_week_low"),
            ("trailingPE", "trailing_pe"),
            ("priceToBook", "price_to_book"),
            ("marketCap", "market_cap"),
        ):
            value = info.get(source)
            if isinstance(value, (int, float)):
                quote[target] = value
        details.update({
            "category": info.get("category"),
            "fund_family": info.get("fundFamily"),
            "tracking_index": info.get("indexTracked"),
            "website": info.get("website"),
        })
        inception = info.get("fundInceptionDate")
        if isinstance(inception, (int, float)):
            details["inception_date"] = datetime.fromtimestamp(inception).date().isoformat()
        details["quote_type"] = identity.get("quote_type")
        details["instrument_type"] = _infer_instrument_type(
            ticker,
            identity.get("quote_type"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load stock overview quote for %s: %s", ticker, exc)

    result = {
        "ticker": ticker,
        "name": (
            identity.get("company_name")
            or local_name
            or ticker
        ),
        "market_region": region,
        "exchange": identity.get("exchange"),
        "currency": identity.get("currency"),
        "sector": identity.get("sector"),
        "industry": identity.get("industry"),
        "details": details,
        "chart_range": chart_range,
        "chart": chart,
        "quote": quote,
        "reports": reports,
    }
    _overview_cache[cache_key] = (time.monotonic(), result)
    return result
