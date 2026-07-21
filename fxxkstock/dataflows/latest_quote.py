"""Best-effort latest quote retrieval for current, non-backtest analysis."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests
import yfinance as yf

from .market_utils import detect_market_region, is_cn_region, to_eastmoney_symbol
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_EM_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EM_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_LATEST_QUOTE_CACHE_TTL_SECONDS = 30
_latest_quote_cache: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_float(value: Any) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number > 0 else None


def _fast_get(mapping: Any, key: str) -> Any:
    try:
        if hasattr(mapping, "get"):
            return mapping.get(key)
        return mapping[key]
    except Exception:  # noqa: BLE001
        return None


def _parse_epoch(value: Any) -> str | None:
    number = _finite_float(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _parse_eastmoney_time(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text in {"0", "-"}:
        return None
    for fmt, length in (("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12), ("%Y%m%d", 8)):
        if len(text) >= length and text[:length].isdigit():
            try:
                return datetime.strptime(text[:length], fmt).isoformat()
            except ValueError:
                continue
    return None


def _eastmoney_price(data: dict[str, Any], field: str) -> float | None:
    raw = _positive_float(data.get(field))
    if raw is None:
        return None
    precision = int(_finite_float(data.get("f59")) or 2)
    if raw >= 10:
        return round(raw / (10 ** precision), precision)
    return raw


def _eastmoney_signed_price(data: dict[str, Any], field: str) -> float | None:
    raw = _finite_float(data.get(field))
    if raw is None:
        return None
    precision = int(_finite_float(data.get("f59")) or 2)
    if abs(raw) >= 10:
        return round(raw / (10 ** precision), precision)
    return raw


def fetch_eastmoney_latest_quote(
    ticker: str,
    region: str | None = None,
    timeout: float = 2.5,
) -> dict[str, Any] | None:
    """Return latest quote fields from Eastmoney for CN A/HK symbols."""
    market_region = region or detect_market_region(ticker)
    if not is_cn_region(market_region) or market_region == "cn_adr":
        return None
    secid, bare = to_eastmoney_symbol(ticker, market_region)
    params = urlencode(
        {
            "secid": secid,
            "fields": ",".join(
                [
                    "f43",  # latest price
                    "f44",  # high
                    "f45",  # low
                    "f46",  # open
                    "f48",  # turnover
                    "f57",  # code
                    "f58",  # name
                    "f59",  # price precision
                    "f60",  # previous close
                    "f86",  # latest quote time
                    "f169",  # price change
                    "f170",  # percent change, scaled by 100
                ]
            ),
        }
    )
    url = f"{_EM_QUOTE_URL}?{params}"
    headers = {
        "User-Agent": _EM_UA,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://quote.eastmoney.com/",
    }
    try:
        last_exc: Exception | None = None
        # Latest quote is best-effort and has a daily OHLCV fallback. Avoid the
        # direct no-proxy path here because it can hang for many seconds on
        # some Windows/proxy networks even when requests' read timeout is low.
        for trust_env in (True,):
            try:
                session = requests.Session()
                session.trust_env = trust_env
                response = session.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        else:
            raise last_exc or RuntimeError("Eastmoney latest quote request failed")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Eastmoney latest quote failed for %s: %s", ticker, exc)
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    last_price = _eastmoney_price(data, "f43")
    if last_price is None:
        return None
    change_percent = _finite_float(data.get("f170"))
    if change_percent is not None:
        change_percent /= 100
    return {
        "ticker": ticker.upper(),
        "source": "eastmoney",
        "symbol": str(data.get("f57") or bare),
        "name": data.get("f58"),
        "currency": "CNY",
        "last_price": last_price,
        "open": _eastmoney_price(data, "f46"),
        "high": _eastmoney_price(data, "f44"),
        "low": _eastmoney_price(data, "f45"),
        "previous_close": _eastmoney_price(data, "f60"),
        "change": _eastmoney_signed_price(data, "f169"),
        "change_percent": change_percent,
        "turnover": _finite_float(data.get("f48")),
        "as_of": _parse_eastmoney_time(data.get("f86")),
    }


def fetch_yfinance_latest_quote(ticker: str) -> dict[str, Any] | None:
    """Return latest quote fields from yfinance when fast quote data exists."""
    canonical = normalize_symbol(ticker)
    try:
        stock = yf.Ticker(canonical)
        fast = getattr(stock, "fast_info", None) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance latest quote failed for %s: %s", ticker, exc)
        return None

    last_price = _positive_float(
        _fast_get(fast, "last_price")
        or _fast_get(fast, "lastPrice")
        or _fast_get(fast, "regular_market_price")
    )
    if last_price is None:
        return None
    previous_close = _positive_float(
        _fast_get(fast, "previous_close") or _fast_get(fast, "previousClose")
    )
    change = last_price - previous_close if previous_close else None
    change_percent = change / previous_close * 100 if previous_close else None
    as_of = None
    info: dict[str, Any] = {}
    try:
        info = getattr(stock, "info", None) or {}
    except Exception:  # noqa: BLE001
        info = {}
    as_of = _parse_epoch(
        info.get("regularMarketTime")
        or info.get("postMarketTime")
        or info.get("preMarketTime")
    )
    return {
        "ticker": ticker.upper(),
        "source": "yfinance",
        "symbol": canonical,
        "currency": _fast_get(fast, "currency") or info.get("currency"),
        "last_price": last_price,
        "open": _positive_float(_fast_get(fast, "open")),
        "high": _positive_float(_fast_get(fast, "day_high")),
        "low": _positive_float(_fast_get(fast, "day_low")),
        "previous_close": previous_close,
        "change": change,
        "change_percent": change_percent,
        "volume": _finite_float(_fast_get(fast, "last_volume")),
        "market_cap": _finite_float(_fast_get(fast, "market_cap")),
        "fifty_two_week_high": _finite_float(_fast_get(fast, "year_high")),
        "fifty_two_week_low": _finite_float(_fast_get(fast, "year_low")),
        "as_of": as_of,
    }


def fetch_latest_market_quote(
    ticker: str,
    region: str | None = None,
) -> dict[str, Any] | None:
    """Best-effort latest quote. Returns None so callers can fall back cleanly."""
    market_region = region or detect_market_region(ticker)
    cache_key = (ticker.upper().strip(), market_region)
    cached = _latest_quote_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _LATEST_QUOTE_CACHE_TTL_SECONDS:
        return cached[1]
    if market_region in {"cn_a", "cn_hk"}:
        quote = fetch_eastmoney_latest_quote(ticker, market_region)
    else:
        quote = fetch_yfinance_latest_quote(ticker)
    _latest_quote_cache[cache_key] = (time.monotonic(), quote)
    return quote
