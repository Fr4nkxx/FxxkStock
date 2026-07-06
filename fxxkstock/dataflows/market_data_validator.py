"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import re
from typing import Any

import pandas as pd
from stockstats import wrap

from fxxkstock.dataflows.latest_quote import fetch_latest_market_quote
from fxxkstock.dataflows.stockstats_utils import load_ohlcv
from fxxkstock.dataflows.currency_utils import (
    format_fx_header_line,
    get_instrument_fx_context,
    scale_price_for_display,
    should_scale_indicator,
)

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _is_current_analysis_date(curr_date: str) -> bool:
    try:
        requested = date.fromisoformat(str(curr_date))
    except ValueError:
        return False
    return requested >= date.today()


def _latest_quote_for_current_date(symbol: str, curr_date: str) -> dict[str, Any] | None:
    if not _is_current_analysis_date(curr_date):
        return None
    try:
        return fetch_latest_market_quote(symbol)
    except Exception:  # noqa: BLE001
        return None


def _usable_latest_quote(quote: dict[str, Any] | None) -> bool:
    if not isinstance(quote, dict):
        return False
    try:
        price = float(quote.get("last_price"))
    except (TypeError, ValueError):
        return False
    return pd.notna(price) and price > 0


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def build_current_market_snapshot_data(
    symbol: str,
    curr_date: str,
) -> dict[str, Any]:
    """Build the compact, machine-readable source of truth for this run."""
    source_ccy, fx_rate = get_instrument_fx_context(symbol, curr_date)
    df = _verified_rows(symbol, curr_date)
    latest = df.iloc[-1]
    rate = fx_rate if source_ccy != "CNY" else 1.0

    def displayed(field: str) -> float | None:
        value = latest.get(field)
        if value is None or pd.isna(value) or rate is None:
            return None
        return round(float(value) * float(rate), 6)

    latest_complete_date = pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d")
    latest_complete_close = displayed("Close")
    snapshot = {
        "ticker": symbol.upper(),
        "requested_date": str(curr_date),
        "latest_trading_date": latest_complete_date,
        "latest_complete_ohlcv_date": latest_complete_date,
        "latest_complete_ohlcv_close": latest_complete_close,
        "price_basis": "latest_complete_ohlcv",
        "currency": "CNY" if rate is not None else source_ccy,
        "source_currency": source_ccy,
        "fx_rate_to_cny": rate,
        "open": displayed("Open"),
        "high": displayed("High"),
        "low": displayed("Low"),
        "close": latest_complete_close,
        "volume": (
            int(latest["Volume"])
            if latest.get("Volume") is not None and not pd.isna(latest.get("Volume"))
            else None
        ),
    }
    latest_quote = _latest_quote_for_current_date(symbol, curr_date)
    if _usable_latest_quote(latest_quote):
        quote_currency = latest_quote.get("currency") or snapshot["currency"]
        quote_rate = 1.0 if quote_currency == "CNY" else rate
        if quote_rate is None:
            return snapshot
        latest_price = round(float(latest_quote["last_price"]) * float(quote_rate), 6)
        snapshot.update(
            {
                "price_basis": "latest_quote",
                "latest_trading_date": (
                    str(latest_quote.get("as_of"))[:10]
                    if latest_quote.get("as_of")
                    else latest_complete_date
                ),
                "latest_quote_as_of": latest_quote.get("as_of"),
                "latest_quote_source": latest_quote.get("source"),
                "latest_quote_symbol": latest_quote.get("symbol"),
                "latest_quote_currency": quote_currency,
                "latest_price": latest_price,
                "close": latest_price,
            }
        )
        for source, target in (
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("previous_close", "previous_close"),
        ):
            value = latest_quote.get(source)
            if value is not None and pd.notna(value):
                snapshot[target] = round(float(value) * float(quote_rate), 6)
    return snapshot


def render_current_market_context(snapshot: dict[str, Any]) -> str:
    """Render a compact hard-priority context shared by every decision agent."""
    if snapshot.get("error"):
        return (
            "CURRENT MARKET SNAPSHOT UNAVAILABLE. Do not present any historical "
            f"price as current. Reason: {snapshot['error']}"
        )
    if snapshot.get("price_basis") == "latest_quote":
        return (
            "AUTHORITATIVE CURRENT MARKET SNAPSHOT (highest-priority facts for this run)\n"
            f"- Ticker: {snapshot.get('ticker')}\n"
            f"- Requested analysis date: {snapshot.get('requested_date')}\n"
            f"- Current latest quote: {snapshot.get('close')} {snapshot.get('currency')}\n"
            f"- Latest quote as of: {snapshot.get('latest_quote_as_of') or 'unknown'}\n"
            f"- Latest quote source: {snapshot.get('latest_quote_source') or 'unknown'}\n"
            f"- Latest complete OHLCV row for indicators: "
            f"{snapshot.get('latest_complete_ohlcv_date')} close "
            f"{snapshot.get('latest_complete_ohlcv_close')} {snapshot.get('currency')}\n"
            f"- Current session quote OHLC when available: {snapshot.get('open')} / "
            f"{snapshot.get('high')} / {snapshot.get('low')} / {snapshot.get('close')}\n"
            "Hard rule: use the current latest quote for current-price, position, "
            "entry/exit, and stop-distance decisions. Use the latest complete OHLCV "
            "row for technical indicators unless a tool explicitly provides newer "
            "indicator data. Historical numbers may only be cited with their original "
            "date and must never be described as current."
        )
    return (
        "AUTHORITATIVE CURRENT MARKET SNAPSHOT (highest-priority facts for this run)\n"
        f"- Ticker: {snapshot.get('ticker')}\n"
        f"- Requested analysis date: {snapshot.get('requested_date')}\n"
        f"- Latest valid trading date: {snapshot.get('latest_trading_date')}\n"
        f"- Current verified close: {snapshot.get('close')} {snapshot.get('currency')}\n"
        f"- Verified OHLC: {snapshot.get('open')} / {snapshot.get('high')} / "
        f"{snapshot.get('low')} / {snapshot.get('close')}\n"
        f"- Verified volume: {snapshot.get('volume')}\n"
        "Hard rule: this snapshot overrides every price, date, OHLCV value, and "
        "current-market claim found in historical memory or prior reports. Historical "
        "numbers may only be cited with their original date and must never be described "
        "as current."
    )


_CURRENT_PRICE_PATTERNS = (
    re.compile(
        r"(?:当前(?:验证)?(?:价格|股价|收盘价)|现价|current\s+(?:verified\s+)?(?:price|close))"
        r"\s*(?:为|是|[:：])?\s*(?:¥|￥|CNY\s*)?([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
)


def find_current_price_conflicts(
    text: str,
    snapshot: dict[str, Any],
) -> list[float]:
    """Return explicit current-price claims that conflict with the snapshot."""
    current = snapshot.get("close")
    if current is None:
        return []
    tolerance = max(0.01, abs(float(current)) * 0.002)
    conflicts: list[float] = []
    for pattern in _CURRENT_PRICE_PATTERNS:
        for match in pattern.finditer(text or ""):
            claimed = float(match.group(1))
            if abs(claimed - float(current)) > tolerance:
                conflicts.append(claimed)
    return conflicts


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    source_ccy, fx_rate = get_instrument_fx_context(symbol, curr_date)
    display_rate = fx_rate if source_ccy != "CNY" else None

    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            raw = stock_df.iloc[-1][name]
            if should_scale_indicator(name):
                indicator_values[name] = scale_price_for_display(
                    raw, display_rate, source_ccy
                )
            else:
                indicator_values[name] = _fmt(raw)
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    fx_line = format_fx_header_line(source_ccy, fx_rate, curr_date).lstrip("# ").strip()
    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        f"- Display currency: CNY",
        f"- {fx_line}",
        "- Rows after the requested analysis date are excluded before verification.",
    ]
    latest_quote = _latest_quote_for_current_date(symbol, curr_date)
    if _usable_latest_quote(latest_quote):
        quote_currency = latest_quote.get("currency") or source_ccy
        quote_rate = 1.0 if quote_currency == "CNY" else fx_rate
        if quote_rate is not None:
            latest_price = round(
                float(latest_quote["last_price"]) * float(quote_rate), 6
            )
            lines += [
                "",
                "### Current latest quote",
                "",
                "| Field | Value |",
                "|---|---:|",
                f"| Source | {latest_quote.get('source') or 'unknown'} |",
                f"| As of | {latest_quote.get('as_of') or 'unknown'} |",
                f"| Latest price | {_fmt(latest_price)} CNY |",
                "",
                "Use this latest quote as the current price for action, position, "
                "entry/exit, and stop-distance decisions. The OHLCV row and indicators "
                "below remain based on the latest complete trading row.",
            ]

    lines += [
        "",
        "### Latest verified OHLCV row (CNY display)",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        if field == "Volume":
            lines.append(f"| {field} | {_fmt(latest.get(field))} |")
        else:
            lines.append(
                f"| {field} | {scale_price_for_display(latest.get(field), display_rate, source_ccy)} |"
            )

    lines += ["", "### Verified technical indicators (latest row, CNY where price-scaled)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows, CNY display)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(
            f"| {_fmt(row['Date'])} | {scale_price_for_display(row.get('Close'), display_rate, source_ccy)} |"
        )

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
