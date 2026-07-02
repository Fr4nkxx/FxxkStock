"""币种检测与外币→CNY 换算工具。

yfinance 等国外数据源以标的原币种报价（USD/HKD/CNY 等）。本模块在展示层
将非 CNY 金额统一换算为 CNY，并生成可核验的汇率说明行。
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta
from typing import Any, Mapping

import pandas as pd
import yfinance as yf

from .config import get_config
from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)

# 需要按汇率线性缩放的价格量纲指标（展示层，不重算）
_PRICE_SCALE_INDICATORS = frozenset({
    "close_10_ema", "close_50_sma", "close_200_sma",
    "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
})

# get_fundamentals 中需要换算为 CNY 的金额字段
# EPS 等每股价格也应按汇率换算
FUNDAMENTAL_MONETARY_FIELDS = frozenset({
    "marketCap", "totalRevenue", "grossProfits", "ebitda",
    "netIncomeToCommon", "freeCashflow",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "fiftyDayAverage", "twoHundredDayAverage", "bookValue",
    "trailingEps", "forwardEps",
})

# 财报行名含以下子串时视为股数，不做金额换算
SHARE_COUNT_ROW_HINTS = ("Shares", "Share Issued", "Share Number")

# insider 表中金额量纲列（股数列 Shares 不换算）
INSIDER_MONETARY_COLUMNS = frozenset({"Value", "Price"})


def detect_source_currency(
    ticker: str,
    identity: Mapping[str, str] | None = None,
    market_region: str | None = None,
) -> str:
    """推断标的报价币种：优先 yfinance identity，再按后缀/region 回退。"""
    if identity:
        ccy = (identity.get("currency") or "").strip().upper()
        if ccy:
            return ccy

    upper = ticker.upper().strip()
    if upper.endswith((".SS", ".SZ")):
        return "CNY"
    if upper.endswith(".HK"):
        return "HKD"
    if market_region in ("cn_a",):
        return "CNY"
    if market_region in ("cn_hk",):
        return "HKD"
    return "USD"


def _fx_pair_symbol(ccy: str) -> str:
    """yfinance 外汇对符号，如 USD -> USDCNY=X。"""
    return f"{ccy.upper()}CNY=X"


@functools.lru_cache(maxsize=64)
def _fetch_fx_rate_from_yfinance(ccy: str, as_of: str) -> float | None:
    """按 as_of 日期取 ccy/CNY 最近收盘汇率；失败返回 None。"""
    if ccy.upper() == "CNY":
        return 1.0
    pair = _fx_pair_symbol(ccy)
    try:
        end_dt = datetime.strptime(as_of, "%Y-%m-%d") + timedelta(days=1)
        start_dt = end_dt - timedelta(days=14)
        hist = yf_retry(
            lambda: yf.Ticker(pair).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
            )
        )
        if hist is None or hist.empty:
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        cutoff = pd.Timestamp(as_of)
        eligible = hist[hist.index <= cutoff]
        if eligible.empty:
            eligible = hist
        rate = float(eligible["Close"].iloc[-1])
        return rate if rate > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("FX lookup failed for %s as of %s: %s", pair, as_of, exc)
        return None


def get_fx_to_cny(ccy: str, as_of: str) -> float | None:
    """返回 1 单位 ccy 兑多少 CNY；override 优先，CNY 返回 1.0，失败 None。"""
    ccy = ccy.upper().strip()
    if ccy == "CNY":
        return 1.0

    config = get_config()
    if not config.get("fx_convert_enabled", True):
        return None

    overrides = config.get("fx_rate_override") or {}
    if isinstance(overrides, dict) and ccy in overrides:
        try:
            rate = float(overrides[ccy])
            return rate if rate > 0 else None
        except (TypeError, ValueError):
            pass

    return _fetch_fx_rate_from_yfinance(ccy, as_of)


def convert_to_cny(value: float | int, rate: float) -> float:
    """将原币金额按汇率换算为 CNY。"""
    return float(value) * rate


def format_money_cny(value: float | int, *, decimals: int = 2) -> str:
    """格式化为 CNY 展示字符串。"""
    return f"{float(value):.{decimals}f} CNY"


def format_fx_header_line(
    source_ccy: str,
    rate: float | None,
    as_of: str,
    *,
    converted: bool = True,
) -> str:
    """生成数据 header 中的汇率说明行。"""
    source_ccy = source_ccy.upper()
    if source_ccy == "CNY":
        return "# Currency: CNY (no conversion needed)"
    if rate is None:
        return (
            f"# Currency: {source_ccy} (FX to CNY unavailable — "
            f"do not treat {source_ccy} amounts as CNY)"
        )
    pair = _fx_pair_symbol(source_ccy)
    if converted:
        return (
            f"# FX: 1 {source_ccy} = {rate:.4f} CNY "
            f"(yfinance {pair}, as of {as_of}); amounts below shown in CNY"
        )
    return (
        f"# FX: 1 {source_ccy} = {rate:.4f} CNY "
        f"(yfinance {pair}, as of {as_of}); OHLCV rows kept in {source_ccy}"
    )


def scale_price_for_display(value: Any, rate: float | None, source_ccy: str) -> str:
    """将价格量纲数值缩放为 CNY 展示；无量纲或汇率不可用时原样返回。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if source_ccy == "CNY" or rate is None:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    try:
        return format_money_cny(convert_to_cny(float(value), rate))
    except (TypeError, ValueError):
        return str(value)


def should_scale_indicator(name: str) -> bool:
    """指标名是否属于价格量纲（展示层可按汇率缩放）。"""
    return name in _PRICE_SCALE_INDICATORS


def clear_fx_cache() -> None:
    """测试用：清空汇率缓存。"""
    _fetch_fx_rate_from_yfinance.cache_clear()


def _quick_identity_for_currency(ticker: str) -> dict[str, str]:
    """轻量读取 yfinance currency，避免 dataflows→agents 循环依赖。"""
    try:
        from .symbol_utils import normalize_symbol

        info = yf.Ticker(normalize_symbol(ticker)).info or {}
        ccy = info.get("currency")
        if isinstance(ccy, str) and ccy.strip():
            return {"currency": ccy.strip().upper()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Quick currency lookup failed for %s: %s", ticker, exc)
    return {}


def get_instrument_fx_context(ticker: str, as_of: str) -> tuple[str, float | None]:
    """返回 (source_currency, fx_to_cny)；CNY 标的 fx 为 1.0。"""
    from .market_utils import detect_market_region

    identity = _quick_identity_for_currency(ticker)
    region = detect_market_region(ticker, identity)
    source = detect_source_currency(ticker, identity, region)
    if source == "CNY":
        return "CNY", 1.0
    rate = get_fx_to_cny(source, as_of)
    return source, rate


def is_share_count_row(row_label: Any) -> bool:
    """财报行是否为股数（非金额）。"""
    label = str(row_label)
    return any(hint in label for hint in SHARE_COUNT_ROW_HINTS)


def detect_financial_currency(
    info: Mapping[str, Any] | None,
    ticker: str,
    market_region: str | None = None,
) -> str:
    """从 yfinance info 取财报币种；缺省回退 detect_source_currency。"""
    if info:
        for key in ("financialCurrency", "currency"):
            val = info.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().upper()
    return detect_source_currency(ticker, market_region=market_region)


def convert_financial_frame(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """将财报 DataFrame 中非股数行的数值列按汇率换算为 CNY。"""
    out = df.copy()
    for idx in out.index:
        if is_share_count_row(idx):
            continue
        row = pd.to_numeric(out.loc[idx], errors="coerce")
        out.loc[idx] = row * rate
    return out


def convert_insider_frame(
    df: pd.DataFrame,
    rate: float | None,
    source_ccy: str,
) -> pd.DataFrame:
    """将 insider 表中 Value/Price 等金额列换算为 CNY；Shares 等股数列不变。"""
    if source_ccy == "CNY" or rate is None:
        return df
    out = df.copy()
    for col in out.columns:
        col_name = str(col)
        if col_name in INSIDER_MONETARY_COLUMNS or col_name.lower() in {
            "value",
            "price",
        }:
            out[col] = (pd.to_numeric(out[col], errors="coerce") * rate).round(6)
    return out
