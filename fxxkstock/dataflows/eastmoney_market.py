"""Eastmoney OHLCV and indicator provider for CN market symbols."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Annotated, Any
from urllib.parse import urlencode

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError
from .market_utils import detect_market_region, is_cn_region, to_eastmoney_symbol
from .stockstats_utils import _assert_ohlcv_not_stale, _clean_dataframe
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

_EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_EM_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_EM_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _eastmoney_region(ticker: str) -> str:
    region = detect_market_region(ticker)
    if region == "cn_adr" or not is_cn_region(region):
        raise NoMarketDataError(
            ticker,
            ticker,
            "Eastmoney market data only supports CN A/HK symbols",
        )
    return region


def _parse_kline_row(row: str) -> dict[str, Any] | None:
    parts = row.split(",")
    if len(parts) < 6:
        return None
    date_text = parts[0].strip()
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return None
    open_, close, high, low, volume = (_finite_float(item) for item in parts[1:6])
    if close is None:
        return None
    return {
        "Date": date_text,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Amount": _finite_float(parts[6]) if len(parts) > 6 else None,
        "PctChange": _finite_float(parts[8]) if len(parts) > 8 else None,
        "TurnoverRate": _finite_float(parts[10]) if len(parts) > 10 else None,
    }


def _parse_klines(klines: list[str]) -> pd.DataFrame:
    rows = [parsed for item in klines if (parsed := _parse_kline_row(item))]
    if not rows:
        return pd.DataFrame()
    return _clean_dataframe(pd.DataFrame(rows))


def _download_eastmoney_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    timeout: int = 10,
    max_retries: int = 2,
) -> pd.DataFrame:
    region = _eastmoney_region(symbol)
    secid, _ = to_eastmoney_symbol(symbol, region)
    params = urlencode(
        {
            "secid": secid,
            "fields1": _EM_FIELDS1,
            "fields2": _EM_FIELDS2,
            "klt": "101",
            "fqt": "1",
            "beg": pd.Timestamp(start_date).strftime("%Y%m%d"),
            "end": pd.Timestamp(end_date).strftime("%Y%m%d"),
        }
    )
    url = f"{_EM_KLINE_URL}?{params}"
    headers = {
            "User-Agent": _EM_UA,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://quote.eastmoney.com/",
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        for trust_env in (False, True):
            try:
                session = requests.Session()
                session.trust_env = trust_env
                response = session.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        else:
            if attempt < max_retries:
                delay = 0.8 * (attempt + 1)
                logger.warning(
                    "Eastmoney kline request failed for %s, retrying in %.1fs: %s",
                    symbol,
                    delay,
                    last_exc,
                )
                time.sleep(delay)
                continue
            raise NoMarketDataError(
                symbol,
                secid,
                f"Eastmoney kline request failed: {last_exc}",
            ) from last_exc
        break

    data = payload.get("data") if isinstance(payload, dict) else None
    klines = data.get("klines") if isinstance(data, dict) else None
    if not isinstance(klines, list) or not klines:
        raise NoMarketDataError(symbol, secid, "Eastmoney returned no kline rows")
    frame = _parse_klines([item for item in klines if isinstance(item, str)])
    if frame.empty or "Close" not in frame.columns:
        raise NoMarketDataError(symbol, secid, "Eastmoney returned no complete OHLCV rows")
    return frame


def _eastmoney_cache_path(symbol: str, start_date: str, end_date: str) -> str:
    region = _eastmoney_region(symbol)
    secid, _ = to_eastmoney_symbol(symbol, region)
    cache_dir = os.path.join(get_config()["data_cache_dir"], "eastmoney_ohlcv")
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = safe_ticker_component(secid)
    return os.path.join(
        cache_dir,
        f"{safe_symbol}-EM-data-{start_date}-{end_date}.csv",
    )


def load_eastmoney_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load five years of Eastmoney daily OHLCV up to today, then filter to curr_date."""
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")
    data_file = _eastmoney_cache_path(symbol, start_str, end_str)

    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        downloaded = _download_eastmoney_ohlcv(symbol, start_str, end_str)
        tmp_file = f"{data_file}.tmp"
        downloaded.to_csv(tmp_file, index=False, encoding="utf-8")
        os.replace(tmp_file, data_file)
        data = downloaded

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol)
    return data


def get_eastmoney_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return Eastmoney daily OHLCV data as CSV text for CN symbols."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    data = load_eastmoney_ohlcv(symbol, end_date)
    data = data[
        (data["Date"] >= pd.Timestamp(start_date))
        & (data["Date"] <= pd.Timestamp(end_date))
    ]
    if data.empty:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"Eastmoney returned no complete rows between {start_date} and {end_date}",
        )
    _assert_ohlcv_not_stale(data, end_date, symbol, symbol)
    latest_valid_date = pd.to_datetime(data["Date"], errors="coerce").max().strftime(
        "%Y-%m-%d"
    )
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += "# Source: Eastmoney daily kline, forward-adjusted (qfq)\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if latest_valid_date != end_date:
        header += (
            "# Data quality warning: the vendor returned no complete row for "
            f"the requested date; latest complete OHLCV row is {latest_valid_date}. "
            "Do not infer a holiday from this warning alone.\n"
        )
    header += "\n"
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    optional = [col for col in ("Amount", "PctChange", "TurnoverRate") if col in data]
    return header + data[columns + optional].to_csv(index=False)


def _eastmoney_indicator_values(
    symbol: str,
    indicator: str,
    curr_date: str,
) -> dict[str, str]:
    data = load_eastmoney_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]
    values: dict[str, str] = {}
    for _, row in df.iterrows():
        value = row[indicator]
        values[row["Date"]] = "N/A" if pd.isna(value) else str(value)
    return values


def get_eastmoney_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Return stockstats indicator values computed from Eastmoney OHLCV."""
    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)
    indicator_data = _eastmoney_indicator_values(symbol, indicator, curr_date)

    lines: list[str] = []
    current_dt = curr_date_dt
    while current_dt >= before:
        date_str = current_dt.strftime("%Y-%m-%d")
        lines.append(
            f"{date_str}: {indicator_data.get(date_str, 'N/A: No valid vendor row for this date (non-trading day or delayed/incomplete market data)')}"
        )
        current_dt = current_dt - relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + "\n".join(lines)
        + "\n\nSource: Eastmoney daily kline, forward-adjusted (qfq)."
    )
