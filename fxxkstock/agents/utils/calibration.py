"""Persistent, idempotent calibration records for portfolio predictions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from fxxkstock.dataflows.utils import safe_ticker_component

OutcomeFetcher = Callable[[str, str, int, str], dict[str, Any] | None]


class CalibrationStore:
    def __init__(self, config: dict[str, Any]):
        self.root = Path(config.get("calibration_memory_dir", "memory/calibration"))

    def _path(self, ticker: str) -> Path:
        return self.root / f"{safe_ticker_component(ticker)}.json"

    def load(self, ticker: str) -> dict[str, Any]:
        path = self._path(ticker)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return {"version": 1, "ticker": ticker.upper(), "records": []}
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            return {"version": 1, "ticker": ticker.upper(), "records": []}
        return data

    def _write(self, ticker: str, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(ticker)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def record(
        self,
        ticker: str,
        trade_date: str,
        decision: dict[str, Any],
        initial_close: float,
        benchmark: str,
    ) -> dict[str, Any]:
        signature = json.dumps(decision, sort_keys=True, ensure_ascii=False)
        record_id = hashlib.sha256(
            f"{ticker.upper()}|{trade_date}|{signature}".encode()
        ).hexdigest()[:16]
        data = self.load(ticker)
        if any(item.get("id") == record_id for item in data["records"]):
            return data
        confidence = {
            key: decision.get(f"{key}_confidence")
            for key in ("data", "thesis", "execution")
        }
        predictions = []
        for index, prediction in enumerate(decision.get("predictions") or []):
            predictions.append({
                **prediction,
                "id": f"P{index + 1:02d}",
                "status": "pending",
            })
        record = {
            "id": record_id,
            "trade_date": trade_date,
            "rating": decision.get("rating"),
            "confidence": confidence,
            "benchmark": benchmark,
            "initial_close": float(initial_close),
            "rating_tasks": [
                {"horizon_trading_days": horizon, "status": "pending"}
                for horizon in (5, 20)
            ],
            "predictions": predictions,
        }
        data["records"].append(record)
        self._write(ticker, data)
        return data

    def resolve_pending(
        self, ticker: str, fetcher: OutcomeFetcher
    ) -> dict[str, Any]:
        data = self.load(ticker)
        changed = False
        for record in data["records"]:
            outcomes: dict[int, dict[str, Any] | None] = {}
            benchmark = record.get("benchmark") or "SPY"
            for task in record.get("rating_tasks", []):
                if task.get("status") != "pending":
                    continue
                horizon = int(task["horizon_trading_days"])
                if horizon not in outcomes:
                    outcomes[horizon] = fetcher(
                        ticker, record["trade_date"], horizon, benchmark
                    )
                outcome = outcomes[horizon]
                if not outcome or outcome.get("actual_price") is None:
                    continue
                rating = str(record.get("rating") or "")
                raw = outcome.get("raw_return")
                hit = None
                if rating in {"Buy", "Overweight"}:
                    hit = raw is not None and raw > 0
                elif rating in {"Sell", "Underweight"}:
                    hit = raw is not None and raw < 0
                task.update({**outcome, "status": "resolved", "hit": hit})
                changed = True
            for prediction in record.get("predictions", []):
                if prediction.get("status") != "pending":
                    continue
                horizon = int(prediction["horizon_trading_days"])
                if horizon not in outcomes:
                    outcomes[horizon] = fetcher(
                        ticker, record["trade_date"], horizon, benchmark
                    )
                outcome = outcomes[horizon]
                if not outcome or outcome.get("actual_price") is None:
                    continue
                actual = float(outcome["actual_price"])
                target = float(prediction["target_price"])
                comparison = prediction.get("comparison")
                prediction.update({
                    **outcome,
                    "status": "resolved",
                    "hit": actual > target if comparison == "Above" else actual < target,
                })
                changed = True
        if changed:
            self._write(ticker, data)
        return data

    def query(self, ticker: str) -> dict[str, Any]:
        data = self.load(ticker)
        rating_tasks = [
            {**task, "record_id": record.get("id"), "trade_date": record.get("trade_date"),
             "rating": record.get("rating"), "confidence": record.get("confidence")}
            for record in data["records"] for task in record.get("rating_tasks", [])
        ]
        predictions = [
            {**prediction, "record_id": record.get("id"),
             "trade_date": record.get("trade_date")}
            for record in data["records"] for prediction in record.get("predictions", [])
        ]
        resolved_predictions = [p for p in predictions if p.get("status") == "resolved"]
        confidence_stats = {}
        for level in ("Low", "Medium", "High"):
            bucket = [p for p in resolved_predictions if p.get("confidence") == level]
            confidence_stats[level] = {
                "count": len(bucket),
                "hit_rate": (
                    sum(bool(p.get("hit")) for p in bucket) / len(bucket)
                    if bucket else None
                ),
            }
        rating_stats = {}
        for rating in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            bucket = [
                task for task in rating_tasks
                if task.get("status") == "resolved" and task.get("rating") == rating
            ]
            judged = [task for task in bucket if task.get("hit") is not None]
            rating_stats[rating] = {
                "count": len(bucket),
                "average_return": (
                    sum(task["raw_return"] for task in bucket) / len(bucket)
                    if bucket else None
                ),
                "average_alpha": (
                    sum(task["alpha_return"] for task in bucket if task.get("alpha_return") is not None)
                    / len([task for task in bucket if task.get("alpha_return") is not None])
                    if any(task.get("alpha_return") is not None for task in bucket)
                    else None
                ),
                "hit_rate": (
                    sum(bool(task["hit"]) for task in judged) / len(judged)
                    if judged else None
                ),
            }
        return {
            "ticker": ticker.upper(),
            "pending": [item for item in rating_tasks + predictions if item.get("status") == "pending"],
            "resolved": [item for item in rating_tasks + predictions if item.get("status") == "resolved"],
            "records": data["records"],
            "summary": {
                "prediction_confidence": confidence_stats,
                "ratings": rating_stats,
            },
        }
