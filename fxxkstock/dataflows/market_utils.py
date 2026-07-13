"""Market region detection and China-market symbol conversion."""

from __future__ import annotations

import logging
import json
import re
from pathlib import Path
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
        # 港股东财 market id = 116
        hk = bare.zfill(5) if bare.isdigit() else bare
        return f"116.{hk}", bare

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
_EM_FUND_NAME_URL = "https://fundgz.1234567.com.cn/js/{code}.js"
_EM_FUND_DETAIL_URL = "https://fund.eastmoney.com/{code}.html"
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


def _contains_chinese(value: str | None) -> bool:
    return bool(value and re.search(r"[\u4e00-\u9fff]", value))


def _is_cn_etf_code(code: str) -> bool:
    return bool(
        re.fullmatch(r"\d{6}", code)
        and code.startswith(("15", "16", "51", "56", "58"))
    )


def is_cn_etf(ticker: str) -> bool:
    return _is_cn_etf_code(ticker.upper().strip().split(".")[0])


def _fund_name_cache_path() -> Path:
    return Path(get_config()["data_cache_dir"]) / "security_names" / "cn_funds.json"


def _load_cached_fund_cn_name(code: str) -> str | None:
    path = _fund_name_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    name = payload.get(code) if isinstance(payload, dict) else None
    return name.strip() if isinstance(name, str) and _contains_chinese(name) else None


def _cache_fund_cn_name(code: str, name: str) -> None:
    if not _contains_chinese(name):
        return
    path = _fund_name_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[code] = name.strip()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        logger.debug("Could not cache Chinese fund name for %s: %s", code, exc)


def _fetch_eastmoney_fund_cn_name(code: str) -> str | None:
    """Resolve an ETF/fund's official Chinese name from Eastmoney fund data."""
    from urllib.request import Request, urlopen

    url = _EM_FUND_NAME_URL.format(code=code)
    req = Request(
        url,
        headers={
            "User-Agent": _EM_UA,
            "Accept": "application/javascript,text/javascript,*/*",
            "Referer": f"https://fund.eastmoney.com/{code}.html",
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        match = re.search(r"jsonpgz\((\{.*\})\)\s*;?", text, re.DOTALL)
        if not match:
            return None
        payload = json.loads(match.group(1))
        name = payload.get("name")
        if isinstance(name, str) and _contains_chinese(name):
            return name.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Eastmoney fund name lookup failed for code=%s: %s", code, exc)
    return None


def _parse_eastmoney_etf_metadata(html: str, code: str) -> dict[str, object]:
    """Parse official fund name and tracking target from an Eastmoney fund page."""
    from parsel import Selector

    selector = Selector(text=html)
    title = " ".join((selector.css("title::text").get() or "").split())
    visible = " ".join(
        " ".join(selector.xpath("//body//text()").getall()).split()
    )
    name = ""
    title_match = re.match(r"(.+?)\s*[（(]\s*" + re.escape(code), title)
    if title_match:
        name = title_match.group(1).strip()
    tracking_match = re.search(
        r"跟踪标的\s*[：:]\s*([^|｜\s]{2,50}?(?:指数|Index))(?=\s|[|｜])",
        visible,
        re.IGNORECASE,
    )
    tracking_index = tracking_match.group(1).strip() if tracking_match else None
    return {
        "fund_name": name if _contains_chinese(name) else None,
        "tracking_index": tracking_index,
    }


def _etf_metadata_cache_path() -> Path:
    return Path(get_config()["data_cache_dir"]) / "security_names" / "cn_etfs.json"


def _load_etf_metadata(code: str) -> dict[str, object]:
    try:
        payload = json.loads(_etf_metadata_cache_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}
    item = payload.get(code) if isinstance(payload, dict) else None
    return item if isinstance(item, dict) else {}


def _cache_etf_metadata(code: str, metadata: dict[str, object]) -> None:
    path = _etf_metadata_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[code] = metadata
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        logger.debug("Could not cache ETF metadata for %s: %s", code, exc)


def _fetch_eastmoney_etf_metadata(code: str) -> dict[str, object]:
    from urllib.request import Request, urlopen

    url = _EM_FUND_DETAIL_URL.format(code=code)
    req = Request(url, headers={"User-Agent": _EM_UA, "Accept": "text/html,*/*"})
    try:
        with urlopen(req, timeout=12) as response:
            html = response.read().decode("utf-8", errors="replace")
        return _parse_eastmoney_etf_metadata(html, code)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Eastmoney ETF metadata lookup failed for %s: %s", code, exc)
        return {}


def _etf_search_aliases(name: str, tracking_index: str | None) -> list[str]:
    candidates = [name, tracking_index or ""]
    source = f"{name} {tracking_index or ''}"
    if "纳斯达克100" in source or "纳斯达克 100" in source:
        candidates.extend(["纳斯达克100", "纳指100", "纳指"])
    if tracking_index:
        simplified = re.sub(r"^(中证|国证|上证|深证)", "", tracking_index)
        simplified = re.sub(r"(主题)?指数$", "", simplified).strip()
        if len(simplified) >= 2:
            candidates.append(simplified)
    aliases: list[str] = []
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if clean and clean not in aliases:
            aliases.append(clean)
    return aliases[:5]


def get_cn_etf_metadata(ticker: str, region: str) -> dict[str, object]:
    """Return source-labelled ETF metadata; never infer an official index with an LLM."""
    code = ticker.upper().strip().split(".")[0]
    if region != "cn_a" or not _is_cn_etf_code(code):
        return {}
    cached = _load_etf_metadata(code)
    fetched = {} if cached.get("tracking_index") else _fetch_eastmoney_etf_metadata(code)
    fund_name = (
        fetched.get("fund_name")
        or cached.get("fund_name")
        or get_security_cn_name(ticker, region)
    )
    tracking_index = fetched.get("tracking_index") or cached.get("tracking_index")
    metadata = {
        "fund_name": fund_name,
        "tracking_index": tracking_index,
        "search_aliases": _etf_search_aliases(str(fund_name or ""), str(tracking_index or "") or None),
        "source": "Eastmoney fund profile" if fetched else cached.get("source", "local cache"),
    }
    if fund_name or tracking_index:
        _cache_etf_metadata(code, metadata)
    return metadata


def get_security_cn_name(ticker: str, region: str) -> str | None:
    """返回国内源权威中文简称；失败时返回 None（调用方回退 yfinance）。"""
    if not is_cn_region(region):
        return None

    if region == "cn_a":
        em_code, bare = to_eastmoney_symbol(ticker, "cn_a")
        name = _fetch_eastmoney_cn_name(em_code)
        if _contains_chinese(name):
            return name
        if _is_cn_etf_code(bare):
            cached_metadata = _load_etf_metadata(bare)
            metadata_name = cached_metadata.get("fund_name")
            cached = _load_cached_fund_cn_name(bare)
            if not cached and isinstance(metadata_name, str) and _contains_chinese(metadata_name):
                cached = metadata_name.strip()
                _cache_fund_cn_name(bare, cached)
            if cached:
                return cached
            fund_name = _fetch_eastmoney_fund_cn_name(bare)
            if fund_name:
                _cache_fund_cn_name(bare, fund_name)
                return fund_name
            # The lightweight fund quote endpoint is occasionally unavailable
            # for exchange-traded funds. Fall back to the fund profile, which is
            # also the authoritative source used by get_cn_etf_metadata().
            metadata = _fetch_eastmoney_etf_metadata(bare)
            profile_name = metadata.get("fund_name")
            if isinstance(profile_name, str) and _contains_chinese(profile_name):
                profile_name = profile_name.strip()
                _cache_fund_cn_name(bare, profile_name)
                merged = {**cached_metadata, **metadata, "fund_name": profile_name}
                _cache_etf_metadata(bare, merged)
                return profile_name
        try:
            from .cninfo import get_cn_name

            cninfo_name = get_cn_name(to_cninfo_stock_code(ticker))
            return cninfo_name if _contains_chinese(cninfo_name) else None
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
