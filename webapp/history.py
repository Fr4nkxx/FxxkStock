"""扫描磁盘上的历史报告目录。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from fxxkstock.agents.utils.agent_utils import resolve_instrument_identity
from fxxkstock.dataflows.config import get_config
from fxxkstock.dataflows.latest_quote import fetch_latest_market_quote
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
_REPORT_INDEX_FILE = "index.json"
_REPORT_INDEX_VERSION = 2
_OVERVIEW_CACHE_TTL_SECONDS = 60
_QUOTE_CACHE_TTL_SECONDS = 45
_CHART_CACHE_TTL_SECONDS = 300
_overview_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_chart_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


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


def _normalize_decision_token(value: str | None) -> str | None:
    token = (value or "").strip().strip("*`：:，,。.").upper()
    cn_map = {
        "买入": "BUY",
        "增持": "OVERWEIGHT",
        "持有": "HOLD",
        "观望": "HOLD",
        "减持": "UNDERWEIGHT",
        "卖出": "SELL",
        "等待": "WAIT",
    }
    if token in {"BUY", "SELL", "HOLD", "OVERWEIGHT", "UNDERWEIGHT", "WAIT"}:
        return token
    return cn_map.get((value or "").strip())


def _clean_instrument_name(value: Any, ticker: str) -> str | None:
    """Return a displayable security name, rejecting exchange/category labels."""
    if not isinstance(value, str):
        return None
    name = re.sub(r"[*#`]", "", value)
    name = re.sub(r"^[^\w\u4e00-\u9fff]+", "", name).strip(" ：:|-（(")
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return None
    rejected = {"分析日期", "最近交易日", "已验证", "股票", "证券", "ETF"}
    if name in rejected or name.upper() == ticker.upper():
        return None
    if re.fullmatch(r"[\dA-Za-z._-]+", name):
        return None
    if re.search(r"(?:证券交易所|交易所)$", name):
        return None
    if re.search(r"(?:stock exchange|securities exchange| exchange)$", name, re.I):
        return None
    return name


def _extract_decision(markdown: str) -> str | None:
    patterns = (
        r"\*\*(?:Rating|评级)\*\*\s*[:：]\s*\**"
        r"(Buy|Overweight|Hold|Underweight|Sell|Wait|买入|增持|持有|观望|减持|卖出|等待)\**",
        r"(?:^|\n)\s*(?:Rating|评级)\s*[:：]\s*\**"
        r"(Buy|Overweight|Hold|Underweight|Sell|Wait|买入|增持|持有|观望|减持|卖出|等待)\**",
        r"FINAL TRANSACTION PROPOSAL\s*[:：]\s*\**"
        r"(Buy|Overweight|Hold|Underweight|Sell|Wait|买入|增持|持有|观望|减持|卖出|等待)\**",
        r"(?:最终交易建议|最终裁决)\s*[:：]\s*\**"
        r"(买入|增持|持有|观望|减持|卖出|等待|Buy|Overweight|Hold|Underweight|Sell|Wait)\**",
    )
    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
        decision = _normalize_decision_token(match.group(1) if match else None)
        if decision:
            return decision

    upper = markdown.upper()
    for token in ("BUY", "SELL", "HOLD", "OVERWEIGHT", "UNDERWEIGHT", "WAIT"):
        if f"**{token}**" in upper:
            return token
    return None


def _extract_instrument_name(markdown: str, ticker: str) -> str | None:
    """Best-effort local name extraction without another market-data request."""
    bare = re.escape(ticker.split(".")[0])
    patterns = (
        rf"^#+\s*([^#\n|（(]{{2,80}}?)\s*[（(]\s*{bare}(?:\.(?:SS|SZ|HK))?\s*[）)]",
        rf"(?:报告对象|标的|公司|基金名称)\s*[：:*]*\s*([^|\n（(]{{2,80}}?)\s*[（(]\s*{bare}(?:\.(?:SS|SZ|HK))?\s*[）)]",
        rf"{bare}(?:\.(?:SS|SZ|HK))?\s*[（(]([^）)\n]{{2,80}})[）)]",
        rf"([^\n|]{{2,100}})\s*[（(]{bare}(?:\.(?:SS|SZ|HK))?[）)]",
        r"(?:标的|公司|基金名称)\s*[：:*]*\s*([^|\n（(]{2,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        name = _clean_instrument_name(match.group(1), ticker)
        if not name:
            continue
        if re.search(
            r"(?:总体判断|综合判断|核心判断|报告对象|报告|标的|公司|基金名称)\s*[：:]",
            name,
        ):
            name = re.split(r"[：:]", name)[-1].strip()
        name = _clean_instrument_name(name, ticker)
        if name:
            return name
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _read_decision_source(report_dir: Path, fallback_markdown: str) -> str:
    return _read_text(report_dir / "5_portfolio" / "decision.md") or fallback_markdown


def read_report_sections(report_dir: Path) -> dict[str, str]:
    """Group the saved report tree into the UI's reader tabs."""
    summary = _read_text(report_dir / "5_portfolio" / "decision.md")
    thesis_parts = [
        _read_text(report_dir / "2_research" / "blind_bull.md"),
        _read_text(report_dir / "2_research" / "blind_bear.md"),
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
    audit_parts = [
        _read_text(report_dir / "6_audit" / "evidence_ledger.md"),
        _read_text(report_dir / "6_audit" / "researchability.md"),
        _read_text(report_dir / "6_audit" / "falsification.md"),
    ]
    return {
        "summary": summary,
        "thesis": "\n\n---\n\n".join(part for part in thesis_parts if part),
        "risks": "\n\n---\n\n".join(part for part in risk_parts if part),
        "data": "\n\n---\n\n".join(part for part in data_parts if part),
        "audit": "\n\n---\n\n".join(part for part in audit_parts if part),
        "calibration": "",
    }


def read_audit_metadata(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "6_audit" / "audit.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        logger.warning("Ignoring invalid anti-bias audit file: %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _report_index_path(root: Path) -> Path:
    return root / _REPORT_INDEX_FILE


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime if path.is_file() else 0
    except OSError:
        return 0


def _report_fingerprints(root: Path) -> dict[str, dict[str, float]]:
    fingerprints: dict[str, dict[str, float]] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        report_file = entry / "complete_report.md"
        if not report_file.is_file():
            continue
        fingerprints[entry.name] = {
            "report_modified_at": _safe_mtime(report_file),
            "decision_modified_at": _safe_mtime(
                entry / "5_portfolio" / "decision.md"
            ),
        }
    return fingerprints


def _fingerprints_match(
    expected: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
) -> bool:
    if expected.keys() != actual.keys():
        return False
    for report_id, values in expected.items():
        current = actual.get(report_id) or {}
        if values.get("report_modified_at") != current.get("report_modified_at"):
            return False
        if values.get("decision_modified_at") != current.get("decision_modified_at"):
            return False
    return True


def _read_report_index(
    root: Path,
    fingerprints: dict[str, dict[str, float]],
) -> list[dict] | None:
    path = _report_index_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Ignoring invalid report index %s: %s", path, exc)
        return None
    if payload.get("version") != _REPORT_INDEX_VERSION:
        return None
    indexed_fingerprints = payload.get("fingerprints")
    if not isinstance(indexed_fingerprints, dict):
        return None
    if not _fingerprints_match(indexed_fingerprints, fingerprints):
        return None
    reports = payload.get("reports")
    if not isinstance(reports, list):
        return None
    return [item for item in reports if isinstance(item, dict)]


def _write_report_index(
    root: Path,
    items: list[dict],
    fingerprints: dict[str, dict[str, float]],
) -> None:
    path = _report_index_path(root)
    payload = {
        "version": _REPORT_INDEX_VERSION,
        "generated_at": datetime.now().isoformat(),
        "fingerprints": fingerprints,
        "reports": items,
    }
    try:
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        logger.debug("Could not write report index %s: %s", path, exc)


def _build_report_index_item(
    entry: Path,
    fingerprint: dict[str, float],
) -> dict:
    report_file = entry / "complete_report.md"
    meta = _parse_report_dir_name(entry.name) or {
        "ticker": entry.name,
        "created_at": "",
    }
    markdown = report_file.read_text(encoding="utf-8", errors="replace")
    decision_source = _read_decision_source(entry, markdown)
    return {
        "id": entry.name,
        "ticker": meta["ticker"],
        "created_at": meta["created_at"],
        "title": _extract_title(markdown) or meta["ticker"],
        "name": _extract_instrument_name(markdown, meta["ticker"]),
        "decision": _extract_decision(decision_source),
        "report_dir": str(entry.resolve()),
        "modified_at": fingerprint.get("report_modified_at", 0),
    }


def _rebuild_report_index(
    root: Path,
    fingerprints: dict[str, dict[str, float]],
) -> list[dict]:
    items: list[dict] = []
    for report_id, fingerprint in fingerprints.items():
        try:
            items.append(_build_report_index_item(root / report_id, fingerprint))
        except OSError as exc:
            logger.debug("Skipping report %s during index rebuild: %s", report_id, exc)
    items.sort(key=lambda x: x["modified_at"], reverse=True)
    _write_report_index(root, items, fingerprints)
    return items


def list_historical_reports(
    reports_root: Path | None = None,
    limit: int | None = 100,
) -> list[dict]:
    """List saved reports newest-first, using an index to avoid rereading Markdown."""
    root = reports_root or get_reports_root()
    if not root.is_dir():
        return []

    fingerprints = _report_fingerprints(root)
    indexed = _read_report_index(root, fingerprints)
    items = indexed if indexed is not None else _rebuild_report_index(root, fingerprints)
    items.sort(key=lambda x: x.get("modified_at", 0), reverse=True)
    return items if limit is None else items[:limit]


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
    decision_source = _read_decision_source(report_dir, markdown)
    meta = _parse_report_dir_name(report_id) or {"ticker": report_id, "created_at": ""}
    return {
        "id": report_id,
        "available": True,
        "ticker": meta["ticker"],
        "created_at": meta["created_at"],
        "title": _extract_title(markdown) or meta["ticker"],
        "decision": _extract_decision(decision_source),
        "markdown": markdown,
        "sections": read_report_sections(report_dir),
        "audit": read_audit_metadata(report_dir),
        "core_insights": read_core_insights(report_dir),
        "report_dir": str(report_dir),
    }


def _normalize_requested_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    safe_ticker_component(ticker)
    return ticker


def get_stock_reports(
    ticker: str,
    reports_root: Path | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Return saved report summaries for one ticker without loading full reports."""
    ticker = _normalize_requested_ticker(ticker)
    reports = [
        item for item in list_historical_reports(reports_root, limit=None)
        if item["ticker"].upper() == ticker
    ]
    return reports[:limit]


def _resolve_overview_identity(
    ticker: str,
    reports: list[dict],
) -> tuple[dict[str, Any], str, str | None]:
    local_name = next(
        (
            cleaned
            for item in reports
            if (cleaned := _clean_instrument_name(item.get("name"), ticker))
        ),
        None,
    )
    region = detect_market_region(ticker)
    identity: dict[str, Any] = {}
    if not is_cn_region(region):
        if local_name:
            identity["company_name"] = local_name
        else:
            identity.update(resolve_instrument_identity(ticker))
    region = detect_market_region(ticker, identity)
    if is_cn_region(region):
        if local_name:
            identity["company_name"] = local_name
        else:
            cn_name = get_security_cn_name(ticker, region)
            if cn_name:
                identity["company_name"] = cn_name
    elif local_name:
        identity["company_name"] = local_name
    return identity, region, local_name


def _empty_quote() -> dict[str, Any]:
    return {
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
        "price_basis": "latest_complete_ohlcv",
        "source": "yfinance",
    }


def _overview_details(ticker: str, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "quote_type": identity.get("quote_type"),
        "instrument_type": _infer_instrument_type(ticker, identity.get("quote_type")),
        "category": None,
        "fund_family": None,
        "tracking_index": None,
        "inception_date": None,
        "website": None,
    }


def _apply_latest_quote(
    quote: dict[str, Any],
    identity: dict[str, Any],
    latest_quote: dict[str, Any] | None,
) -> bool:
    if not isinstance(latest_quote, dict) or not latest_quote.get("last_price"):
        return False
    quote["price_basis"] = "latest_quote"
    quote["source"] = latest_quote.get("source") or quote["source"]
    for source, target in (
        ("last_price", "last_price"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("previous_close", "previous_close"),
        ("change", "change"),
        ("change_percent", "change_percent"),
        ("volume", "volume"),
        ("turnover", "turnover"),
        ("fifty_two_week_high", "fifty_two_week_high"),
        ("fifty_two_week_low", "fifty_two_week_low"),
        ("market_cap", "market_cap"),
    ):
        value = latest_quote.get(source)
        if value is not None:
            quote[target] = value
    if latest_quote.get("as_of"):
        quote["as_of"] = latest_quote["as_of"]
    quote_name = _clean_instrument_name(latest_quote.get("name"), str(latest_quote.get("ticker") or ""))
    if not identity.get("company_name") and quote_name:
        identity["company_name"] = quote_name
    if not identity.get("currency") and latest_quote.get("currency"):
        identity["currency"] = latest_quote["currency"]
    if quote["change"] is None and quote["previous_close"] not in (None, 0):
        quote["change"] = quote["last_price"] - quote["previous_close"]
        quote["change_percent"] = quote["change"] / quote["previous_close"] * 100
    return True


def _apply_daily_quote_from_yfinance(
    stock: Any,
    quote: dict[str, Any],
) -> None:
    daily_frame = stock.history(
        period="5d",
        interval="1d",
        auto_adjust=False,
    )
    daily_closes = daily_frame["Close"].dropna()
    if daily_closes.empty:
        return
    latest_index = daily_closes.index[-1]
    latest = daily_frame.loc[latest_index]
    last = float(daily_closes.iloc[-1])
    previous = float(daily_closes.iloc[-2]) if len(daily_closes) > 1 else None
    quote["last_price"] = last
    quote["previous_close"] = previous
    if previous not in (None, 0):
        quote["change"] = last - previous
        quote["change_percent"] = (last - previous) / previous * 100
    for source, target in (
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
    ):
        value = latest.get(source)
        quote[target] = (
            float(value)
            if value is not None and pd.notna(value)
            else None
        )
    volume = latest.get("Volume")
    quote["volume"] = (
        int(volume)
        if volume is not None and pd.notna(volume)
        else None
    )
    if quote["last_price"] is not None and quote["volume"] is not None:
        quote["turnover"] = quote["last_price"] * quote["volume"]
    quote["as_of"] = (
        latest_index.date().isoformat()
        if hasattr(latest_index, "date")
        else str(latest_index)
    )


def _apply_daily_quote_from_eastmoney(
    ticker: str,
    identity: dict[str, Any],
    quote: dict[str, Any],
) -> bool:
    if (
        str(get_config().get("cn_market_data_source", "yfinance")).strip().lower()
        != "eastmoney"
    ):
        return False
    if detect_market_region(ticker) not in {"cn_a", "cn_hk"}:
        return False
    try:
        from fxxkstock.dataflows.eastmoney_market import load_eastmoney_ohlcv

        frame = load_eastmoney_ohlcv(ticker, datetime.now().date().isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load Eastmoney daily quote for %s: %s", ticker, exc)
        return False
    if frame is None or frame.empty or "Close" not in frame.columns:
        return False
    frame = frame.sort_values("Date") if "Date" in frame.columns else frame
    latest = frame.iloc[-1]
    last = latest.get("Close")
    if last is None or pd.isna(last):
        return False
    quote["source"] = "eastmoney"
    quote["price_basis"] = "latest_complete_ohlcv"
    quote["last_price"] = float(last)
    if not identity.get("currency"):
        identity["currency"] = "CNY"
    previous = frame["Close"].dropna().iloc[-2] if len(frame["Close"].dropna()) > 1 else None
    if previous not in (None, 0):
        quote["previous_close"] = float(previous)
        quote["change"] = quote["last_price"] - quote["previous_close"]
        quote["change_percent"] = quote["change"] / quote["previous_close"] * 100
    for source, target in (
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
    ):
        value = latest.get(source)
        quote[target] = (
            float(value)
            if value is not None and pd.notna(value)
            else None
        )
    volume = latest.get("Volume")
    quote["volume"] = (
        int(volume)
        if volume is not None and pd.notna(volume)
        else None
    )
    amount = latest.get("Amount")
    if amount is not None and pd.notna(amount):
        quote["turnover"] = float(amount)
    elif quote["last_price"] is not None and quote["volume"] is not None:
        quote["turnover"] = quote["last_price"] * quote["volume"]
    if "Date" in frame.columns:
        latest_date = pd.to_datetime(latest.get("Date"), errors="coerce")
        if pd.notna(latest_date):
            quote["as_of"] = latest_date.date().isoformat()
    return True


def _apply_intraday_quote_fallback(
    frame: pd.DataFrame,
    quote: dict[str, Any],
) -> None:
    closes = frame["Close"].dropna()
    if closes.empty:
        return
    quote["last_price"] = float(closes.iloc[-1])
    first_open = frame["Open"].dropna() if "Open" in frame else closes
    highs = frame["High"].dropna() if "High" in frame else closes
    lows = frame["Low"].dropna() if "Low" in frame else closes
    quote["open"] = float(first_open.iloc[0])
    quote["high"] = float(highs.max())
    quote["low"] = float(lows.min())
    if "Volume" in frame:
        quote["volume"] = int(frame["Volume"].fillna(0).sum())
    quote["as_of"] = closes.index[-1].isoformat()


def _apply_yfinance_info(
    stock: Any,
    identity: dict[str, Any],
    details: dict[str, Any],
    quote: dict[str, Any],
    ticker: str,
) -> None:
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    if not isinstance(info, dict):
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
            if target == "company_name":
                value = _clean_instrument_name(value, ticker)
                if not value:
                    continue
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


def _quote_payload(
    ticker: str,
    identity: dict[str, Any],
    region: str,
    local_name: str | None,
    quote: dict[str, Any],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "name": (
            _clean_instrument_name(identity.get("company_name"), ticker)
            or _clean_instrument_name(local_name, ticker)
            or ticker
        ),
        "market_region": region,
        "exchange": identity.get("exchange"),
        "currency": identity.get("currency"),
        "sector": identity.get("sector"),
        "industry": identity.get("industry"),
        "details": details,
        "quote": quote,
    }


def get_stock_quote(
    ticker: str,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    """Return identity and latest quote data without loading chart history."""
    ticker = _normalize_requested_ticker(ticker)
    use_cache = reports_root is None
    cached = _quote_cache.get(ticker) if use_cache else None
    if cached and time.monotonic() - cached[0] < _QUOTE_CACHE_TTL_SECONDS:
        return cached[1]

    reports = get_stock_reports(ticker, reports_root=reports_root, limit=1000)
    identity, region, local_name = _resolve_overview_identity(ticker, reports)
    quote = _empty_quote()
    details = _overview_details(ticker, identity)
    latest_quote_used = False
    if reports_root is None:
        try:
            latest_quote_used = _apply_latest_quote(
                quote,
                identity,
                fetch_latest_market_quote(ticker, region),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load latest quote for %s: %s", ticker, exc)

    if not latest_quote_used:
        eastmoney_daily_used = _apply_daily_quote_from_eastmoney(ticker, identity, quote)
    else:
        eastmoney_daily_used = False

    if not latest_quote_used and not eastmoney_daily_used:
        try:
            stock = yf.Ticker(normalize_symbol(ticker))
            _apply_daily_quote_from_yfinance(stock, quote)
            _apply_yfinance_info(stock, identity, details, quote, ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load stock quote for %s: %s", ticker, exc)

    result = _quote_payload(ticker, identity, region, local_name, quote, details)
    if use_cache:
        _quote_cache[ticker] = (time.monotonic(), result)
    return result


def _sample_chart_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    closes = frame["Close"].dropna()
    if closes.empty:
        return []
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame.loc[closes.index, "Date"], errors="coerce")
        if dates.notna().any():
            closes = closes.copy()
            closes.index = dates
    stride = max(1, len(closes) // 140)
    sampled = closes.iloc[::stride]
    if sampled.index[-1] != closes.index[-1]:
        sampled = pd.concat([sampled, closes.iloc[[-1]]])
    return [
        {
            "time": index.isoformat() if hasattr(index, "isoformat") else str(index),
            "close": round(float(value), 6),
        }
        for index, value in sampled.items()
    ]


def _eastmoney_chart_frame(ticker: str, chart_range: str) -> pd.DataFrame | None:
    if (
        str(get_config().get("cn_market_data_source", "yfinance")).strip().lower()
        != "eastmoney"
    ):
        return None
    region = detect_market_region(ticker)
    if region not in {"cn_a", "cn_hk"}:
        return None
    try:
        from fxxkstock.dataflows.eastmoney_market import load_eastmoney_ohlcv

        frame = load_eastmoney_ohlcv(ticker, datetime.now().date().isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load Eastmoney chart for %s: %s", ticker, exc)
        return None
    if frame is None or frame.empty:
        return None
    frame = frame.sort_values("Date")
    row_counts = {
        "1d": 2,
        "5d": 5,
        "1mo": 22,
        "6mo": 126,
        "1y": 252,
    }
    return frame.tail(row_counts.get(chart_range, 22))


def get_stock_chart(ticker: str, chart_range: str = "1d") -> dict[str, Any]:
    """Return chart points only; this may be slower than the quote endpoint."""
    ticker = _normalize_requested_ticker(ticker)
    if chart_range not in _CHART_RANGES:
        raise ValueError(f"unsupported chart range: {chart_range}")
    cache_key = (ticker, chart_range)
    cached = _chart_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _CHART_CACHE_TTL_SECONDS:
        return cached[1]

    chart: list[dict[str, Any]] = []
    source = "yfinance"
    eastmoney_frame = _eastmoney_chart_frame(ticker, chart_range)
    if eastmoney_frame is not None:
        chart = _sample_chart_points(eastmoney_frame)
        source = "eastmoney"
        result = {
            "ticker": ticker,
            "chart_range": chart_range,
            "chart": chart,
            "source": source,
        }
        _chart_cache[cache_key] = (time.monotonic(), result)
        return result

    try:
        stock = yf.Ticker(normalize_symbol(ticker))
        period, interval = _CHART_RANGES[chart_range]
        frame = stock.history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
        chart = _sample_chart_points(frame)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load stock chart for %s: %s", ticker, exc)

    result = {
        "ticker": ticker,
        "chart_range": chart_range,
        "chart": chart,
        "source": source,
    }
    _chart_cache[cache_key] = (time.monotonic(), result)
    return result


def get_stock_overview(
    ticker: str,
    reports_root: Path | None = None,
    chart_range: str = "1d",
) -> dict[str, Any]:
    """Return the legacy combined overview payload."""
    ticker = _normalize_requested_ticker(ticker)
    cache_key = (ticker, chart_range)
    use_cache = reports_root is None
    cached = _overview_cache.get(cache_key) if use_cache else None
    if cached and time.monotonic() - cached[0] < _OVERVIEW_CACHE_TTL_SECONDS:
        return cached[1]

    reports = get_stock_reports(ticker, reports_root=reports_root, limit=1000)
    chart_payload = get_stock_chart(ticker, chart_range)
    quote_payload = get_stock_quote(ticker, reports_root=reports_root)
    result = {
        **quote_payload,
        "chart_range": chart_payload["chart_range"],
        "chart": chart_payload["chart"],
        "reports": reports,
    }
    if use_cache:
        _overview_cache[cache_key] = (time.monotonic(), result)
    return result
