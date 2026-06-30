"""Market region detection and China-market symbol conversion."""

from __future__ import annotations

import logging
import re
from typing import Mapping

from .config import get_config

logger = logging.getLogger(__name__)

# 常见在美上市中概 ADR — 快速命中表；主判仍靠 yfinance country
_DEFAULT_CN_ADR_TICKERS = frozenset(
    {
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME", "VIPS",
        "WB", "YUMC", "ZTO", "BEKE", "FUTU", "TAL", "EDU", "NTES", "TCOM",
    }
)

_CN_COUNTRY_VALUES = frozenset({"china", "hong kong", "hong kong sar china"})
_SHANGHAI_EXCHANGE_HINTS = ("shanghai", "sse", "shh")
_SHENZHEN_EXCHANGE_HINTS = ("shenzhen", "szse", "shz")
_HK_EXCHANGE_HINTS = ("hong kong", "hkex", "hkg")


def _suffix_region(ticker: str) -> str | None:
    """Return region from exchange suffix, or None if not CN-related."""
    upper = ticker.upper().strip()
    if upper.endswith(".SS") or upper.endswith(".SZ"):
        return "cn_a"
    if upper.endswith(".HK"):
        return "cn_hk"
    # 6 位纯数字 A 股代码（无后缀）
    bare = upper.split(".")[0]
    if re.fullmatch(r"\d{6}", bare):
        if bare.startswith(("6", "9")):
            return "cn_a"  # 沪市
        if bare.startswith(("0", "3")):
            return "cn_a"  # 深市
    return None


def _identity_region(identity: Mapping[str, str] | None) -> str | None:
    """Infer CN region from yfinance identity metadata."""
    if not identity:
        return None
    country = (identity.get("country") or "").strip().lower()
    exchange = (identity.get("exchange") or "").strip().lower()
    if not country and not exchange:
        return None
    if any(h in exchange for h in _HK_EXCHANGE_HINTS):
        return "cn_hk"
    if any(h in exchange for h in _SHANGHAI_EXCHANGE_HINTS + _SHENZHEN_EXCHANGE_HINTS):
        return "cn_a"
    if country in _CN_COUNTRY_VALUES:
        # country=China 但无明确交易所 — 视为 ADR/中概
        return "cn_adr"
    return None


def _adr_region(ticker: str) -> str | None:
    """Fast-path ADR list lookup."""
    config = get_config()
    adr_set = set(config.get("cn_adr_tickers") or ())
    adr_set |= _DEFAULT_CN_ADR_TICKERS
    bare = ticker.upper().strip().split(".")[0]
    if bare in adr_set:
        return "cn_adr"
    return None


def detect_market_region(ticker: str, identity: Mapping[str, str] | None = None) -> str:
    """Detect market region for a ticker.

    Returns one of: ``cn_a``, ``cn_hk``, ``cn_adr``, ``default``.

    Priority:
      1. yfinance identity (country / exchange) — covers ADR and ambiguous tickers
      2. Exchange suffix / 6-digit code rules
      3. ADR hard-coded list
    """
    config = get_config()
    if not config.get("cn_data_enabled", True):
        return "default"

    from_identity = _identity_region(identity)
    if from_identity:
        return from_identity

    from_suffix = _suffix_region(ticker)
    if from_suffix:
        return from_suffix

    from_adr = _adr_region(ticker)
    if from_adr:
        return from_adr

    return "default"


def is_cn_region(region: str) -> bool:
    """True when region is any China-related market."""
    return region in {"cn_a", "cn_hk", "cn_adr"}


def to_eastmoney_symbol(ticker: str, region: str) -> tuple[str, str]:
    """Convert ticker to East Money symbol format.

    Returns ``(em_code, bare_code)`` where ``em_code`` is like ``1.600519``
    (market prefix + code) and ``bare_code`` is the 6-digit (or HK) code.
    """
    upper = ticker.upper().strip()
    bare = upper.split(".")[0]

    if region == "cn_hk" or upper.endswith(".HK"):
        code = bare.lstrip("0") or bare
        # 港股东财 market id = 116
        hk = bare.zfill(5) if bare.isdigit() else bare
        return f"116.{hk}", hk

    if upper.endswith(".SS") or (region == "cn_a" and bare.startswith(("6", "9"))):
        return f"1.{bare}", bare

    if upper.endswith(".SZ") or (region == "cn_a" and bare.startswith(("0", "3"))):
        return f"0.{bare}", bare

    if re.fullmatch(r"\d{6}", bare):
        prefix = "1" if bare.startswith(("6", "9")) else "0"
        return f"{prefix}.{bare}", bare

    # ADR / 搜索用原 ticker
    return upper, upper


def to_guba_code(ticker: str, region: str) -> str:
    """Return the guba.eastmoney.com stock code for list/API calls."""
    _, bare = to_eastmoney_symbol(ticker, region)
    if region == "cn_hk":
        return f"hk{bare.zfill(5)}" if bare.isdigit() else f"hk{bare.lower()}"
    return bare


def to_cninfo_stock_code(ticker: str) -> str:
    """Return 6-digit A-share code for CNINFO queries."""
    upper = ticker.upper().strip()
    bare = upper.split(".")[0]
    if re.fullmatch(r"\d{6}", bare):
        return bare
    raise ValueError(f"Not an A-share ticker: {ticker!r}")


_EM_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EM_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _fetch_eastmoney_cn_name(secid: str) -> str | None:
    """从东财行情接口取中文简称（f58）。"""
    import json
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    params = urlencode({"secid": secid, "fields": "f58"})
    url = f"{_EM_QUOTE_URL}?{params}"
    req = Request(url, headers={"User-Agent": _EM_UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        name = (payload.get("data") or {}).get("f58")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Eastmoney name lookup failed for secid=%s: %s", secid, exc)
    return None


def get_security_cn_name(ticker: str, region: str) -> str | None:
    """返回国内源权威中文简称；失败时返回 None（调用方回退 yfinance）。"""
    if not is_cn_region(region):
        return None

    if region == "cn_a":
        em_code, _ = to_eastmoney_symbol(ticker, "cn_a")
        name = _fetch_eastmoney_cn_name(em_code)
        if name:
            return name
        try:
            from .cninfo import get_cn_name

            return get_cn_name(to_cninfo_stock_code(ticker))
        except ValueError:
            return None

    # cn_hk / cn_adr：东财行情
    upper = ticker.upper().strip()
    bare = upper.split(".")[0]
    if region == "cn_hk" or upper.endswith(".HK"):
        em_code, _ = to_eastmoney_symbol(ticker, "cn_hk")
        return _fetch_eastmoney_cn_name(em_code)

    if region == "cn_adr":
        # 美股 ADR 在东财 market id = 105
        return _fetch_eastmoney_cn_name(f"105.{bare}")

    return None
