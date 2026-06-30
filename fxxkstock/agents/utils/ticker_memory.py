"""Per-ticker analysis snapshots stored inside the current project."""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fxxkstock.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)
_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}

SCHEMA_VERSION = 1
REPORT_FILES = {
    "market_report": Path("1_analysts/market.md"),
    "sentiment_report": Path("1_analysts/sentiment.md"),
    "news_report": Path("1_analysts/news.md"),
    "fundamentals_report": Path("1_analysts/fundamentals.md"),
    "final_trade_decision": Path("5_portfolio/decision.md"),
}


class TickerMemoryStore:
    """Load and atomically persist the latest complete analysis per ticker."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.directory = Path(cfg.get("ticker_memory_dir", "memory/tickers")).expanduser()
        self.reports_dir = Path(cfg.get("reports_dir", "reports")).expanduser()
        self.fundamentals_ttl_days = int(cfg.get("ticker_memory_fundamentals_ttl_days", 30))

    def path_for(self, ticker: str) -> Path:
        return self.directory / f"{safe_ticker_component(ticker.upper())}.json"

    def _lock_for(self, ticker: str) -> threading.RLock:
        key = str(self.path_for(ticker).resolve())
        with _LOCKS_GUARD:
            return _PATH_LOCKS.setdefault(key, threading.RLock())

    def load(self, ticker: str, *, import_reports: bool = True) -> dict[str, Any] | None:
        path = self.path_for(ticker)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("version") != SCHEMA_VERSION or data.get("ticker") != ticker.upper():
                    raise ValueError("unsupported or mismatched ticker memory")
                return data
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring invalid ticker memory %s: %s", path, exc)
                return None
        if import_reports:
            imported = self._import_latest_report(ticker)
            if imported:
                self.save(imported)
            return imported
        return None

    def save(self, snapshot: dict[str, Any]) -> Path:
        ticker = str(snapshot["ticker"]).upper()
        with self._lock_for(ticker):
            path = self.path_for(ticker)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(snapshot)
            payload["version"] = SCHEMA_VERSION
            payload["ticker"] = ticker
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            return path

    def update_from_state(
        self,
        ticker: str,
        trade_date: str,
        final_state: dict[str, Any],
        previous: dict[str, Any] | None = None,
        refreshed: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock_for(ticker):
            disk_previous = self.load(ticker, import_reports=False)
            previous = disk_previous or previous
            reports = {}
            previous_reports = (previous or {}).get("reports", {})
            for key in REPORT_FILES:
                value = final_state.get(key)
                reports[key] = value if isinstance(value, str) and value.strip() else previous_reports.get(key, "")
            fundamentals_refreshed = (
                "fundamentals" in refreshed
                if refreshed is not None
                else bool(
                    final_state.get("fundamentals_report")
                    and final_state.get("fundamentals_report")
                    != previous_reports.get("fundamentals_report")
                )
            )

            snapshot = {
                "version": SCHEMA_VERSION,
                "ticker": ticker.upper(),
                "analysis_count": int((previous or {}).get("analysis_count", 0)) + 1,
                "last_analysis_date": str(trade_date),
                "fundamentals_as_of": (
                    str(trade_date)
                    if fundamentals_refreshed
                    else (previous or {}).get("fundamentals_as_of", str(trade_date))
                ),
                "reports": reports,
                "market_snapshot": final_state.get("current_market_snapshot")
                or (previous or {}).get("market_snapshot")
                or {},
            }
            self.save(snapshot)
            return snapshot

    def fundamentals_fresh(
        self,
        snapshot: dict[str, Any] | None,
        as_of: str | date,
    ) -> bool:
        if not snapshot or not snapshot.get("reports", {}).get("fundamentals_report"):
            return False
        try:
            current = date.fromisoformat(str(as_of))
            stored = date.fromisoformat(snapshot["fundamentals_as_of"])
        except (KeyError, TypeError, ValueError):
            return False
        age = (current - stored).days
        return 0 <= age <= self.fundamentals_ttl_days

    def status(self, ticker: str, as_of: str | date | None = None) -> dict[str, Any]:
        snapshot = self.load(ticker)
        current = as_of or date.today().isoformat()
        has_memory = snapshot is not None
        reuse_fundamentals = self.fundamentals_fresh(snapshot, current)
        return {
            "ticker": ticker.upper(),
            "has_memory": has_memory,
            "analysis_count": int((snapshot or {}).get("analysis_count", 0)),
            "last_analysis_date": (snapshot or {}).get("last_analysis_date"),
            "updated_at": (snapshot or {}).get("updated_at"),
            "reuse": ["fundamentals"] if reuse_fundamentals else [],
            "refresh": ["market", "social", "news"] + ([] if reuse_fundamentals else ["fundamentals"]),
        }

    def prior_context(self, snapshot: dict[str, Any] | None) -> str:
        if not snapshot:
            return ""
        decision = snapshot.get("reports", {}).get("final_trade_decision", "")
        if not decision:
            return ""
        previous_date = snapshot.get("last_analysis_date", "unknown")
        market_snapshot = snapshot.get("market_snapshot") or {}
        historical_close = market_snapshot.get("close")
        historical_price_line = (
            f"Previous verified close: {historical_close} "
            f"{market_snapshot.get('currency', '')} on "
            f"{market_snapshot.get('latest_trading_date', previous_date)}.\n"
            if historical_close is not None
            else ""
        )
        return (
            "HISTORICAL SAME-TICKER MEMORY (lower priority than the current market snapshot).\n"
            f"Previous analysis date: {previous_date}\n"
            f"{historical_price_line}"
            "Every price, indicator, target, stop, and phrase such as 'current price' "
            "inside the block below describes the previous analysis only. Never reuse "
            "it as a current fact. Recalculate any price-dependent conclusion from the "
            "authoritative current market snapshot and explicitly explain material changes.\n"
            "<historical_decision>\n"
            f"{decision}\n"
            "</historical_decision>"
        )

    def _import_latest_report(self, ticker: str) -> dict[str, Any] | None:
        if not self.reports_dir.is_dir():
            return None
        prefix = f"{safe_ticker_component(ticker.upper())}_"
        candidates = [
            path for path in self.reports_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        reports = {}
        for key, relative in REPORT_FILES.items():
            file = latest / relative
            reports[key] = file.read_text(encoding="utf-8", errors="replace") if file.is_file() else ""

        stamp = latest.name[len(prefix):]
        try:
            analysis_date = datetime.strptime(stamp, "%Y%m%d_%H%M%S").date().isoformat()
        except ValueError:
            analysis_date = datetime.fromtimestamp(latest.stat().st_mtime).date().isoformat()
        return {
            "version": SCHEMA_VERSION,
            "ticker": ticker.upper(),
            "analysis_count": len(candidates),
            "last_analysis_date": analysis_date,
            "fundamentals_as_of": analysis_date,
            "updated_at": datetime.fromtimestamp(
                latest.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "reports": reports,
            "imported_from": str(latest),
        }
