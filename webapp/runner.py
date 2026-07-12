"""后台流式执行 FxxKStock 图，并将 chunk 编码为 SSE 事件。"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from fxxkstock.agents.utils.structured import bind_structured
from fxxkstock.dataflows.utils import safe_ticker_component
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph
from fxxkstock.reporting import write_report_tree

logger = logging.getLogger(__name__)

# 回答模式 → 辩论轮数（与 CLI research_depth 一致）
MODE_DEPTH: dict[str, int] = {
    "simple": 1,
    "medium": 3,
    "complex": 5,
}

DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")

ANALYST_REPORT_SECTIONS: dict[str, str] = {
    "market_report": "Market Analyst",
    "sentiment_report": "Sentiment Analyst",
    "news_report": "News Analyst",
    "fundamentals_report": "Fundamentals Analyst",
}

CORE_INSIGHTS_FILE = "core_insights.json"


class CoreInsights(BaseModel):
    """Compact AI synthesis shown in the report's persistent decision panel."""

    insights: list[str] = Field(
        min_length=4,
        max_length=6,
        description=(
            "Four to six self-contained Simplified Chinese conclusions covering "
            "the decision, action plan, strongest evidence, risks, and triggers."
        ),
    )


@dataclass
class RunParams:
    ticker: str
    provider: str
    quick_model: str
    deep_model: str
    mode: str = "simple"
    trade_date: str | None = None
    analysts: list[str] | None = None
    reports_dir: Path | None = None
    analysis_mode: str = "auto"
    chrome_platform: str | None = None
    chrome_auto_start: bool = True
    position: dict[str, Any] | None = None


@dataclass
class RunState:
    run_id: str
    ticker: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None
    final_state: dict[str, Any] | None = None
    report_path: Path | None = None
    decision: str | None = None
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    debug_events: list[dict[str, Any]] = field(default_factory=list)
    debug_log_path: Path | None = None


def build_run_config(params: RunParams) -> dict[str, Any]:
    """根据 Web 表单参数构造运行 config。"""
    depth = MODE_DEPTH.get(params.mode, MODE_DEPTH["simple"])
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = depth
    config["max_risk_discuss_rounds"] = depth
    config["llm_provider"] = params.provider.lower()
    config["quick_think_llm"] = params.quick_model
    config["deep_think_llm"] = params.deep_model
    config["checkpoint_enabled"] = False
    if params.chrome_platform:
        if params.chrome_platform not in {"macos", "windows", "ubuntu"}:
            raise ValueError("unsupported Chrome platform")
        config["cn_browser_platform"] = params.chrome_platform
    config["cn_browser_auto_start"] = params.chrome_auto_start
    return config


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _extract_content_string(content: Any) -> str | None:
    """从 LangChain message content 提取可读文本。"""
    if _is_empty(content):
        return None
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        joined = " ".join(parts)
        return joined or None
    return str(content).strip() or None


def _parse_core_insights_text(content: Any) -> list[str]:
    text = _extract_content_string(content) or ""
    points = [
        re.sub(r"^\s*(?:[-*•]+|\d+[.)、])\s*", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*•]+|\d+[.)、])\s*", line)
    ]
    return [point for point in points if point][:6]


def _generate_core_insights(llm: Any, final_decision: str) -> list[str]:
    """Run one post-analysis synthesis call without changing the main decision."""
    prompt = (
        "你是投资报告总编。请对下面的最终投资决策做大范围综合总结，输出4至6条"
        "可独立阅读的简体中文核心观点。必须覆盖：最终评级与仓位、具体操作和价格位、"
        "最关键的支持证据、主要风险、后续加减仓或止损触发条件。不要复述分析过程，"
        "不要添加报告中没有的数据，每条只表达一个重点。\n\n"
        f"最终投资决策：\n{final_decision[:30000]}"
    )
    structured = bind_structured(llm, CoreInsights, "Core Insights")
    if structured is not None:
        try:
            result = structured.invoke(prompt)
            if isinstance(result, CoreInsights):
                return [point.strip() for point in result.insights if point.strip()][:6]
            if isinstance(result, dict):
                parsed = CoreInsights.model_validate(result)
                return [point.strip() for point in parsed.insights if point.strip()][:6]
        except Exception as exc:
            logger.warning("Core Insights structured summary failed: %s", exc)

    response = llm.invoke(prompt + "\n\n请使用无标题的 Markdown 项目符号列表输出。")
    points = _parse_core_insights_text(getattr(response, "content", response))
    if len(points) < 4:
        raise ValueError("core insight summary did not contain at least four points")
    return points


def _write_core_insights(report_dir: Path, insights: list[str]) -> None:
    payload = {"version": 1, "generated_by": "ai", "insights": insights}
    (report_dir / CORE_INSIGHTS_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _classify_message(message) -> tuple[str, str | None]:
    content = _extract_content_string(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("user", content)
    if isinstance(message, ToolMessage):
        return ("tool_result", content)
    if isinstance(message, AIMessage):
        return ("agent", content)
    return ("system", content)


def _emit(state: RunState, event: dict[str, Any]) -> None:
    if "emitted_at" not in event:
        event = {
            "emitted_at": datetime.now().isoformat(timespec="seconds"),
            **event,
        }
    state.debug_events.append(event)
    state.event_queue.put(event)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _emit_timing(
    state: RunState,
    timings: list[dict[str, Any]],
    *,
    label: str,
    duration_seconds: float,
    elapsed_seconds: float,
    category: str,
) -> None:
    record = {
        "label": label,
        "category": category,
        "duration_seconds": round(duration_seconds, 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    timings.append(record)
    _emit(
        state,
        {
            "type": "timing",
            **record,
            "message": (
                f"{label}: +{_format_duration(duration_seconds)} "
                f"(total {_format_duration(elapsed_seconds)})"
            ),
        },
    )


def _emit_timing_summary(
    state: RunState,
    timings: list[dict[str, Any]],
    *,
    total_seconds: float,
    status: str,
) -> None:
    ranked = sorted(
        timings,
        key=lambda item: float(item.get("duration_seconds") or 0),
        reverse=True,
    )
    slowest = ", ".join(
        f"{item['label']} {_format_duration(float(item.get('duration_seconds') or 0))}"
        for item in ranked[:5]
    )
    message = f"total {_format_duration(total_seconds)}"
    if slowest:
        message += f"; slowest: {slowest}"
    _emit(
        state,
        {
            "type": "timing_summary",
            "status": status,
            "total_seconds": round(total_seconds, 3),
            "timings": timings,
            "message": message,
        },
    )


def _emit_parallel_initial_timings(
    state: RunState,
    chunk: dict[str, Any],
) -> bool:
    timings = chunk.get("parallel_initial_analyst_timings")
    if not isinstance(timings, list) or not timings:
        return False

    total_seconds = chunk.get("parallel_initial_analysts_total_seconds")
    total = float(total_seconds) if isinstance(total_seconds, (int, float)) else None
    if total is not None:
        _emit(
            state,
            {
                "type": "parallel_initial_analysts_summary",
                "total_seconds": round(total, 3),
                "message": (
                    "Parallel Initial Analysts total: "
                    f"{_format_duration(total)}"
                ),
            },
        )

    for item in timings:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("key") or "Initial Analyst")
        duration = float(item.get("duration_seconds") or 0.0)
        tool_rounds = int(item.get("tool_rounds") or 0)
        _emit(
            state,
            {
                "type": "parallel_initial_analyst_timing",
                "label": label,
                "key": item.get("key"),
                "report_key": item.get("report_key"),
                "duration_seconds": round(duration, 3),
                "tool_rounds": tool_rounds,
                "message_count": int(item.get("message_count") or 0),
                "message": (
                    f"{label}: {_format_duration(duration)} "
                    f"({tool_rounds} tool round{'s' if tool_rounds != 1 else ''})"
                ),
            },
        )
    return True


def _emit_agent_model_diagnostics(
    state: RunState,
    chunk: dict[str, Any],
    seen: set[str],
) -> None:
    diagnostic_fields = {
        "research_manager_diagnostics": "Research Manager",
        "falsification_auditor_diagnostics": "Falsification Auditor",
    }
    for field_name, default_label in diagnostic_fields.items():
        payload = chunk.get(field_name)
        if field_name in seen or not isinstance(payload, dict):
            continue
        seen.add(field_name)

        label = str(payload.get("agent") or default_label)
        input_sizes = payload.get("input_characters") or {}
        prompt_characters = int(input_sizes.get("prompt") or 0)
        attempts = int(payload.get("model_attempts") or 0)
        total_duration = float(payload.get("total_model_duration_seconds") or 0.0)
        fallback_used = bool(payload.get("fallback_used"))
        message = (
            f"{label}: input={prompt_characters:,} chars; "
            f"app_attempts={attempts}; model_time={_format_duration(total_duration)}; "
            f"fallback={'yes' if fallback_used else 'no'}"
        )
        if fallback_used and payload.get("fallback_reason"):
            message += f" ({payload['fallback_reason']})"

        _emit(
            state,
            {
                "type": "agent_model_diagnostics",
                "label": label,
                **payload,
                "message": message,
            },
        )


def _redact_debug_value(value: Any) -> Any:
    if isinstance(value, str):
        value = re.sub(
            r"(?i)(api[_-]?key|token|authorization)=([^&\\s]+)",
            r"\1=<redacted>",
            value,
        )
        return value
    if isinstance(value, dict):
        return {key: _redact_debug_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_debug_value(item) for item in value]
    return value


def _write_debug_log(state: RunState) -> None:
    path = Path("logs") / "web_runs" / f"{state.run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_redact_debug_value(event), ensure_ascii=False)
        for event in state.debug_events
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    state.debug_log_path = path


def _read_debug_log_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(_redact_debug_value(payload))
    return events


def _events_match_ticker(events: list[dict[str, Any]], ticker: str) -> bool:
    key = ticker.strip().upper()
    if not key:
        return True
    for event in events:
        if str(event.get("ticker") or "").upper() == key:
            return True
        message = str(event.get("message") or "")
        if key in message.upper():
            return True
    return False


def read_run_debug_log(
    run_id: str,
    state: RunState | None = None,
) -> dict[str, Any]:
    """Read a run's debug events from memory or the persisted jsonl log."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id or ""):
        raise ValueError("invalid run id")

    if state is not None and state.debug_events:
        return {
            "run_id": run_id,
            "available": True,
            "events": [_redact_debug_value(event) for event in state.debug_events],
            "source": "memory",
        }

    path = Path("logs") / "web_runs" / f"{run_id}.jsonl"
    try:
        resolved = path.resolve()
        root = (Path("logs") / "web_runs").resolve()
        if not str(resolved).startswith(str(root)):
            raise ValueError("invalid run id")
        events = _read_debug_log_file(path)
    except FileNotFoundError:
        return {"run_id": run_id, "available": False, "events": [], "source": None}

    return {
        "run_id": run_id,
        "available": bool(events),
        "events": events,
        "source": "file",
    }


def read_latest_run_debug_log(ticker: str | None = None) -> dict[str, Any]:
    """Return the newest persisted debug log, optionally matching a ticker."""
    root = Path("logs") / "web_runs"
    try:
        files = sorted(
            root.glob("*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        files = []

    for path in files[:50]:
        try:
            events = _read_debug_log_file(path)
        except OSError:
            continue
        if not events or (ticker and not _events_match_ticker(events, ticker)):
            continue
        return {
            "run_id": path.stem,
            "available": True,
            "events": events,
            "source": "file",
        }
    return {"run_id": None, "available": False, "events": [], "source": None}


def _message_side(kind: str, message) -> str:
    """AI 分析靠左；人类/工具/带 tool_calls 的 AI 靠右。"""
    if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
        return "left"
    return "right"


def _format_tool_calls(message) -> str | None:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    lines: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name", "tool")
            args = call.get("args", {})
        else:
            name = getattr(call, "name", "tool")
            args = getattr(call, "args", {})
        lines.append(f"**Tool:** `{name}`\n```json\n{args}\n```")
    return "\n\n".join(lines)


def _detect_report_sections(chunk: dict[str, Any], seen: set[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key, label in ANALYST_REPORT_SECTIONS.items():
        if key in chunk and chunk[key] and key not in seen:
            seen.add(key)
            found.append((key, label))

    if chunk.get("trader_investment_plan") and "trader" not in seen:
        seen.add("trader")
        found.append(("trader", "Trader"))

    if chunk.get("researchability_assessment") and "researchability" not in seen:
        seen.add("researchability")
        found.append(("researchability", "Researchability Assessor"))
    if chunk.get("evidence_ledger") and "evidence_ledger" not in seen:
        seen.add("evidence_ledger")
        found.append(("evidence_ledger", "构建证据账本"))
    if chunk.get("blind_bull_argument") and "blind_bull" not in seen:
        seen.add("blind_bull")
        found.append(("blind_bull", "Blind Bull"))
    if chunk.get("blind_bear_argument") and "blind_bear" not in seen:
        seen.add("blind_bear")
        found.append(("blind_bear", "Blind Bear"))

    debate = chunk.get("investment_debate_state") or {}
    blind_bull = str(chunk.get("blind_bull_argument") or "").strip()
    blind_bear = str(chunk.get("blind_bear_argument") or "").strip()
    bull_history = str(debate.get("bull_history") or "").strip()
    bear_history = str(debate.get("bear_history") or "").strip()
    if (
        bull_history
        and bull_history != blind_bull
        and "investment_bull" not in seen
    ):
        seen.add("investment_bull")
        found.append(("investment_bull", "Bull Researcher"))
    if (
        bear_history
        and bear_history != blind_bear
        and "investment_bear" not in seen
    ):
        seen.add("investment_bear")
        found.append(("investment_bear", "Bear Researcher"))
    if debate.get("judge_decision") and "research_manager" not in seen:
        seen.add("research_manager")
        found.append(("research_manager", "Research Manager"))

    if chunk.get("falsification_audit") and "falsification_audit" not in seen:
        seen.add("falsification_audit")
        found.append(("falsification_audit", "Falsification Auditor"))

    if (
        chunk.get("initial_investment_plan")
        and chunk.get("investment_plan")
        and chunk.get("investment_plan") != chunk.get("initial_investment_plan")
        and "research_revision" not in seen
    ):
        seen.add("research_revision")
        found.append(("research_revision", "Research Manager Revision"))

    risk = chunk.get("risk_debate_state") or {}
    if risk.get("aggressive_history") and "risk_aggressive" not in seen:
        seen.add("risk_aggressive")
        found.append(("risk_aggressive", "Aggressive Analyst"))
    if risk.get("conservative_history") and "risk_conservative" not in seen:
        seen.add("risk_conservative")
        found.append(("risk_conservative", "Conservative Analyst"))
    if risk.get("neutral_history") and "risk_neutral" not in seen:
        seen.add("risk_neutral")
        found.append(("risk_neutral", "Neutral Analyst"))
    if risk.get("judge_decision") and "portfolio_manager" not in seen:
        seen.add("portfolio_manager")
        found.append(("portfolio_manager", "Portfolio Manager"))

    if chunk.get("final_trade_decision") and "final_decision" not in seen:
        seen.add("final_decision")
        found.append(("final_decision", "Final Decision"))

    return found


def run_analysis(state: RunState, params: RunParams) -> None:
    """在后台线程中执行图 stream，并向 event_queue 推送事件。"""
    run_started = time.perf_counter()
    last_timing_mark = run_started
    timings: list[dict[str, Any]] = []

    def emit_interval(label: str, category: str) -> None:
        nonlocal last_timing_mark
        now = time.perf_counter()
        _emit_timing(
            state,
            timings,
            label=label,
            duration_seconds=now - last_timing_mark,
            elapsed_seconds=now - run_started,
            category=category,
        )
        last_timing_mark = now

    state.status = "running"
    _emit(state, {"type": "status", "message": f"Starting analysis for {params.ticker}..."})

    analysts = params.analysts or list(DEFAULT_ANALYSTS)
    trade_date = params.trade_date or date.today().isoformat()
    reports_dir = params.reports_dir or Path("reports")
    graph = None

    try:
        config = build_run_config(params)
        state.config = config
        _emit(
            state,
            {
                "type": "run_config",
                "parallel_initial_analysts": bool(
                    config.get("parallel_initial_analysts", False)
                ),
                "parallel_initial_analyst_workers": int(
                    config.get("parallel_initial_analyst_workers", 4)
                ),
                "analysts": analysts,
                "analysis_mode": params.analysis_mode,
                "mode": params.mode,
                "provider": params.provider,
                "quick_model": params.quick_model,
                "deep_model": params.deep_model,
                "message": (
                    "parallel_initial_analysts="
                    f"{bool(config.get('parallel_initial_analysts', False))}; "
                    "workers="
                    f"{int(config.get('parallel_initial_analyst_workers', 4))}"
                ),
            },
        )
        graph = FxxKStockGraph(selected_analysts=analysts, debug=False, config=config)

        prepared = graph.prepare_run(
            params.ticker,
            trade_date,
            asset_type="stock",
            analysis_mode=params.analysis_mode,
            browser_status_callback=lambda payload: _emit(
                state, {"type": "browser_status", **payload}
            ),
            position=params.position,
        )
        init_agent_state = prepared["initial_state"]
        args = graph.propagator.get_graph_args()
        _emit(
            state,
            {
                "type": "memory_loaded",
                "has_memory": prepared["snapshot"] is not None,
                "analysis_mode": prepared["analysis_mode"],
                "reuse": prepared["reuse"],
                "refresh": prepared["refresh"],
                "last_analysis_date": (prepared["snapshot"] or {}).get("last_analysis_date"),
                "analysis_count": int((prepared["snapshot"] or {}).get("analysis_count", 0)),
            },
        )
        emit_interval("prepare_run and memory load", "setup")

        trace: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
        content_signatures: set[tuple[str, str | None]] = set()
        seen_sections: set[str] = set()
        seen_agent_diagnostics: set[str] = set()
        parallel_initial_timings_emitted = False

        for chunk in graph.graph.stream(init_agent_state, **args):
            trace.append(chunk)
            sender = chunk.get("sender") or "Agent"

            if not parallel_initial_timings_emitted:
                parallel_initial_timings_emitted = _emit_parallel_initial_timings(
                    state,
                    chunk,
                )

            for section_key, label in _detect_report_sections(chunk, seen_sections):
                _emit(
                    state,
                    {
                        "type": "report_section",
                        "section": section_key,
                        "label": label,
                    },
                )
                _emit(state, {"type": "status", "message": f"{label} completed"})
                emit_interval(label, "stage")

            _emit_agent_model_diagnostics(
                state,
                chunk,
                seen_agent_diagnostics,
            )

            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in processed_ids:
                        continue
                    processed_ids.add(msg_id)

                kind, content = _classify_message(message)
                side = _message_side(kind, message)

                if kind == "agent" and content and content.strip():
                    signature = ("agent", content.strip())
                    if signature not in content_signatures:
                        content_signatures.add(signature)
                        _emit(
                            state,
                            {
                                "type": "message",
                                "side": side,
                                "agent": sender,
                                "kind": kind,
                                "content": content,
                            },
                        )

                tool_text = _format_tool_calls(message)
                if tool_text:
                    signature = ("tool_call", tool_text)
                    if signature not in content_signatures:
                        content_signatures.add(signature)
                        _emit(
                            state,
                            {
                                "type": "message",
                                "side": "right",
                                "agent": "Tool Call",
                                "kind": "tool_call",
                                "content": tool_text,
                            },
                        )

                if kind in ("tool_result", "user", "system") and content and content.strip():
                    if kind == "user" and content.strip() == "Continue":
                        continue
                    signature = (kind, content.strip())
                    if signature not in content_signatures:
                        content_signatures.add(signature)
                        agent_label = {
                            "tool_result": getattr(message, "name", None) or "Tool Result",
                            "user": "System Prompt",
                            "system": "System",
                        }[kind]
                        _emit(
                            state,
                            {
                                "type": "message",
                                "side": "right",
                                "agent": agent_label,
                                "kind": kind,
                                "content": content,
                            },
                        )

        final_state: dict[str, Any] = {}
        for chunk in trace:
            final_state.update(chunk)
        state.final_state = final_state

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = reports_dir / f"{safe_ticker_component(params.ticker)}_{stamp}"
        report_file = write_report_tree(final_state, params.ticker, save_path)
        state.report_path = report_file.parent
        state.decision = graph.process_signal(final_state.get("final_trade_decision", ""))
        emit_interval("write report files", "postprocess")
        _emit(state, {"type": "status", "message": "Generating core insights"})
        try:
            final_decision = final_state.get("final_trade_decision", "")
            if final_decision and getattr(graph, "quick_thinking_llm", None) is not None:
                insights = _generate_core_insights(
                    graph.quick_thinking_llm,
                    final_decision,
                )
                _write_core_insights(state.report_path, insights)
        except Exception as exc:
            # The post-processing summary is supplementary. A provider failure
            # must not turn an otherwise complete analysis into a failed run.
            logger.warning("Core insight generation failed for %s: %s", params.ticker, exc)
        emit_interval("core insights", "postprocess")
        graph.finalize_run(
            params.ticker,
            trade_date,
            final_state,
            log_state=False,
        )
        emit_interval("finalize memory and calibration", "postprocess")
        state.status = "done"
        _emit_timing_summary(
            state,
            timings,
            total_seconds=time.perf_counter() - run_started,
            status="done",
        )

        _emit(
            state,
            {
                "type": "done",
                "decision": state.decision,
                "report_available": True,
                "report_dir": str(state.report_path),
                "run_id": state.run_id,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Web run failed for %s", params.ticker)
        state.status = "error"
        state.error = str(exc)
        _emit_timing_summary(
            state,
            timings,
            total_seconds=time.perf_counter() - run_started,
            status="error",
        )
        _emit(
            state,
            {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        close_browser = getattr(graph, "close_managed_browser", None)
        if callable(close_browser):
            browser_status = close_browser()
            if browser_status is not None:
                _emit(state, {"type": "browser_status", **browser_status})
        _write_debug_log(state)


def start_run(state: RunState, params: RunParams) -> None:
    """启动后台分析线程。"""
    thread = threading.Thread(
        target=run_analysis,
        args=(state, params),
        name=f"fxxkstock-run-{state.run_id}",
        daemon=True,
    )
    state.thread = thread
    thread.start()
