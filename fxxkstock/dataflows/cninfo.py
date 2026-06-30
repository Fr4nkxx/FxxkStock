"""巨潮资讯 CNINFO 公告抓取 — 免 API Key，A 股法定披露。"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .market_utils import to_cninfo_stock_code

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
_TOP_SEARCH_URL = "https://www.cninfo.com.cn/new/information/topSearch/query"
_STOCK_LIST_URLS = (
    "https://www.cninfo.com.cn/new/data/szse_stock.json",
    "https://www.cninfo.com.cn/new/data/sse_stock.json",
)

# 减持 / 股东变动类公告
_INSIDER_CATEGORIES = (
    "category_jjgg;category_jjgg_szsh;category_dshgg_szsh;"
    "category_sdgg_szsh;category_rcjy_szsh;"
)


def _http_post(url: str, data: dict, timeout: float = 15.0) -> dict:
    body = urlencode(data).encode("utf-8")
    headers = {
        "User-Agent": _UA,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.cninfo.com.cn",
        "Referer": "https://www.cninfo.com.cn/",
    }
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (429, 503):
            raise VendorRateLimitError(str(exc)) from exc
        raise


def _http_get_json(url: str, timeout: float = 15.0) -> dict | list:
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_path(name: str) -> Path:
    config = get_config()
    cache_dir = Path(config["data_cache_dir"]) / "cninfo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _parse_stock_list_item(item: dict) -> tuple[str, str | None, str | None]:
    """从 CNINFO 股票列表项解析 (code, org_id, cn_name)。"""
    code = str(item.get("code", "")).zfill(6)
    org_id = item.get("orgId") or item.get("orgid")
    # zwjc = 中文简称；部分列表用 secName / name
    cn_name = (
        (item.get("zwjc") or item.get("secName") or item.get("name") or "")
        .strip()
        or None
    )
    return code, org_id, cn_name


def _load_orgid_map(force_refresh: bool = False) -> dict[str, str]:
    """Load secCode -> orgId mapping from cache or CNINFO stock lists."""
    cache_file = _cache_path("orgid_map.json")
    ttl_hours = get_config().get("cninfo_cache_ttl_hours", 24)
    if not force_refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl_hours * 3600:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)

    mapping: dict[str, str] = {}
    name_mapping: dict[str, str] = {}
    for url in _STOCK_LIST_URLS:
        try:
            data = _http_get_json(url)
            stock_list = data.get("stockList") if isinstance(data, dict) else data
            if not isinstance(stock_list, list):
                continue
            for item in stock_list:
                code, org_id, cn_name = _parse_stock_list_item(item)
                if code and org_id:
                    mapping[code] = org_id
                if code and cn_name:
                    name_mapping[code] = cn_name
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load CNINFO stock list from %s: %s", url, exc)

    if mapping:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
    if name_mapping:
        name_cache = _cache_path("cn_name_map.json")
        with open(name_cache, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, ensure_ascii=False)
    return mapping


def _load_cn_name_map(force_refresh: bool = False) -> dict[str, str]:
    """Load secCode -> 中文简称 mapping from cache or CNINFO stock lists."""
    cache_file = _cache_path("cn_name_map.json")
    ttl_hours = get_config().get("cninfo_cache_ttl_hours", 24)
    if not force_refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl_hours * 3600:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)

    # cn_name_map 缺失时强制刷新，避免 orgid 旧缓存短路导致名称映射永不生成
    refresh = force_refresh or not cache_file.exists()
    _load_orgid_map(force_refresh=refresh)
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_cn_name(stock_code: str) -> str | None:
    """返回 A 股 6 位代码对应的中文简称（CNINFO zwjc）。"""
    code = str(stock_code).zfill(6)
    mapping = _load_cn_name_map()
    name = mapping.get(code)
    if name:
        return name
    # topSearch fallback
    try:
        url = f"{_TOP_SEARCH_URL}?keyWord={code}"
        data = _http_get_json(url)
        if isinstance(data, list) and data:
            for key in ("zwjc", "secName", "name", "orgName"):
                val = (data[0].get(key) or "").strip()
                if val:
                    return val
    except Exception as exc:  # noqa: BLE001
        logger.debug("CNINFO topSearch name lookup failed for %s: %s", code, exc)
    return None


def _resolve_org_id(stock_code: str) -> str | None:
    mapping = _load_orgid_map()
    org_id = mapping.get(stock_code)
    if org_id:
        return org_id
    # topSearch fallback
    try:
        url = f"{_TOP_SEARCH_URL}?keyWord={stock_code}"
        data = _http_get_json(url)
        if isinstance(data, list) and data:
            return data[0].get("orgId")
    except Exception as exc:  # noqa: BLE001
        logger.debug("CNINFO topSearch failed for %s: %s", stock_code, exc)
    return None


def _resolve_org_id_via_browser(stock_code: str) -> str | None:
    """Resolve ETF/fund orgId through CNINFO's search UI and cache it."""
    cache_file = _cache_path("fund_orgid_map.json")
    mapping: dict[str, str] = {}
    if cache_file.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            mapping = json.loads(cache_file.read_text(encoding="utf-8"))
    if mapping.get(stock_code):
        return mapping[stock_code]
    if not get_config().get("cn_browser_enabled", True):
        return None

    config = get_config()
    cdp_url = config.get("cn_browser_cdp_url", "http://127.0.0.1:9222")
    timeout_ms = int(config.get("cn_browser_nav_timeout_ms", 20000))
    page = None
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url, timeout=timeout_ms)
            if not browser.contexts:
                return None
            page = browser.contexts[0].new_page()
            page.goto(
                "https://www.cninfo.com.cn/new/index",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            inputs = page.locator(
                "input[placeholder*='代码'], input[placeholder*='简称'], "
                "input[placeholder*='拼音']"
            )
            target = None
            for index in range(inputs.count()):
                candidate = inputs.nth(index)
                if candidate.is_visible():
                    target = candidate
                    break
            if target is None:
                return None
            target.fill(stock_code)
            page.wait_for_timeout(1500)
            result = page.locator(
                f"a:has-text('{stock_code}'), "
                f"[class*='search'] li:has-text('{stock_code}'), "
                f"[class*='suggest'] li:has-text('{stock_code}')"
            ).first
            if not result.count():
                return None
            result.click()
            page.wait_for_timeout(1500)
            org_id = (parse_qs(urlparse(page.url).query).get("orgId") or [""])[0]
            if not org_id:
                return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNINFO browser orgId lookup failed for %s: %s", stock_code, exc)
        return None
    finally:
        if page is not None:
            with contextlib.suppress(Exception):
                page.close()

    mapping[stock_code] = org_id
    temp_file = cache_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_file, cache_file)
    return org_id


def _plate_info(stock_code: str) -> tuple[str, str, str]:
    """Return (column, plate, exchange_prefix) for CNINFO query."""
    if stock_code.startswith(("6", "9")):
        return "sse", "sh", "gssh"
    return "szse", "sz", "gssz"


def _normalize_org_id(org_id: str, prefix: str) -> str:
    """Normalize legacy security IDs without altering numeric institution IDs."""
    value = str(org_id).strip()
    if value.startswith(("gssh", "gssz")):
        return value
    if value.startswith("jjjl"):
        return value
    # Newer CNINFO records may use a 10-digit institution ID directly.
    if value.isdigit() and len(value) >= 10:
        return value
    return f"{prefix}{value}"


def _query_announcements(
    stock_code: str,
    start_date: str,
    end_date: str,
    category: str = "",
    limit: int = 20,
) -> list[dict]:
    org_id = _resolve_org_id(stock_code) or _resolve_org_id_via_browser(stock_code)
    if not org_id:
        return []

    column, plate, prefix = _plate_info(stock_code)
    normalized_org_id = _normalize_org_id(org_id, prefix)
    if normalized_org_id.startswith("jjjl"):
        column, plate = "fund", ""
    stock_param = f"{stock_code},{normalized_org_id}"

    payload = {
        "stock": stock_param,
        "tabName": "fulltext",
        "pageSize": str(limit),
        "pageNum": "1",
        "column": column,
        "category": category,
        "plate": plate,
        "seDate": f"{start_date}~{end_date}",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    data = _http_post(_QUERY_URL, payload)
    return data.get("announcements") or []


def _format_announcements(announcements: list[dict], header: str) -> str:
    if not announcements:
        return f"<no CNINFO announcements found: {header}>"
    lines = [header]
    for ann in announcements:
        title = (ann.get("announcementTitle") or "").strip()
        ts = ann.get("announcementTime")
        date_str = "?"
        if ts:
            with contextlib.suppress(ValueError, OSError, TypeError):
                date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        url = ann.get("adjunctUrl") or ""
        if url and not url.startswith("http"):
            url = f"https://static.cninfo.com.cn/{url.lstrip('/')}"
        lines.append(f"  [{date_str}] {title}" + (f"\n    Link: {url}" if url else ""))
    return "\n".join(lines)


def fetch_cninfo_announcements(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int | None = None,
) -> str:
    """Fetch recent CNINFO announcements for sentiment analyst (A-share only).

    Graceful degradation — returns placeholder on failure.
    """
    config = get_config()
    if limit is None:
        limit = config.get("cn_news_article_limit", 20)
    try:
        stock_code = to_cninfo_stock_code(ticker)
    except ValueError:
        return f"<CNINFO unavailable: {ticker} is not an A-share ticker>"

    try:
        anns = _query_announcements(stock_code, start_date, end_date, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNINFO announcement fetch failed for %s: %s", ticker, exc)
        return f"<CNINFO unavailable: {type(exc).__name__}>"

    header = (
        f"CNINFO Official Announcements — {len(anns)} items for "
        f"{ticker.upper()} ({start_date} to {end_date})"
    )
    return _format_announcements(anns, header)


def get_cninfo_insider(ticker: str) -> str:
    """Vendor entry for get_insider_transactions — A-share 减持/股东变动公告."""
    config = get_config()
    region = config.get("market_region", "default")
    if region not in ("cn_a", "default"):
        raise NoMarketDataError(
            ticker, ticker, "CNINFO insider data only available for A-shares"
        )

    try:
        stock_code = to_cninfo_stock_code(ticker)
    except ValueError as exc:
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=90)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    try:
        anns = _query_announcements(
            stock_code, start_date, end_date, category=_INSIDER_CATEGORIES, limit=30
        )
    except VendorRateLimitError:
        raise
    except Exception as exc:
        raise NoMarketDataError(ticker, ticker, str(exc)) from exc

    if not anns:
        raise NoMarketDataError(ticker, ticker, "no CNINFO insider/减持 announcements")

    header = f"## {ticker} Insider/Shareholder Transactions (CNINFO), last 90 days:\n\n"
    body = _format_announcements(
        anns, f"CNINFO shareholder change announcements for {ticker.upper()}"
    )
    return header + body
