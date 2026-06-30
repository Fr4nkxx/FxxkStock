"""后台流式执行 FxxKStock 图，并将 chunk 编码为 SSE 事件。"""

from __future__ import annotations

import logging
import json
import queue
import re
import threading
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
    state.debug_events.append(event)
    state.event_queue.put(event)


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

    debate = chunk.get("investment_debate_state") or {}
    if debate.get("bull_history") and "investment_bull" not in seen:
        seen.add("investment_bull")
        found.append(("investment_bull", "Bull Researcher"))
    if debate.get("bear_history") and "investment_bear" not in seen:
        seen.add("investment_bear")
        found.append(("investment_bear", "Bear Researcher"))
    if debate.get("judge_decision") and "research_manager" not in seen:
        seen.add("research_manager")
        found.append(("research_manager", "Research Manager"))

    if chunk.get("trader_investment_plan") and "trader" not in seen:
        seen.add("trader")
        found.append(("trader", "Trader"))

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
    state.status = "running"
    _emit(state, {"type": "status", "message": f"Starting analysis for {params.ticker}..."})

    analysts = params.analysts or list(DEFAULT_ANALYSTS)
    trade_date = params.trade_date or date.today().isoformat()
    reports_dir = params.reports_dir or Path("reports")
    graph = None

    try:
        config = build_run_config(params)
        state.config = config
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

        trace: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
        content_signatures: set[tuple[str, str | None]] = set()
        seen_sections: set[str] = set()

        for chunk in graph.graph.stream(init_agent_state, **args):
            trace.append(chunk)
            sender = chunk.get("sender") or "Agent"

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
        graph.finalize_run(
            params.ticker,
            trade_date,
            final_state,
            log_state=False,
        )
        state.status = "done"

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
        _emit(
            state,
            {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if graph is not None:
            browser_status = graph.close_managed_browser()
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
