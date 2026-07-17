"""FastAPI server for FxxKStock report visualization."""

from __future__ import annotations

import json
import os
import queue
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fxxkstock.agents.utils.calibration import CalibrationStore
from fxxkstock.agents.utils.position import PositionContext
from fxxkstock.agents.utils.ticker_memory import TickerMemoryStore
from fxxkstock.dataflows.chrome_manager import ChromeManager
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.llm_clients.model_catalog import MODEL_OPTIONS

from .history import (
    delete_historical_report,
    delete_stock_reports,
    get_historical_report,
    get_stock_chart,
    get_stock_overview,
    get_stock_quote,
    get_stock_reports,
    list_calendar_nodes,
    list_historical_reports,
    read_audit_metadata,
    read_core_insights,
    read_report_sections,
)
from .runner import (
    RunParams,
    RunState,
    build_run_config,
    read_latest_run_debug_log,
    read_run_debug_log,
    start_run,
)
from .settings_store import (
    delete_api_key,
    get_api_key_status,
    get_general_settings,
    save_api_key,
    save_general_settings,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS: dict[str, RunState] = {}

app = FastAPI(title="FxxKStock Web", version="0.1.0")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RunRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    provider: str = Field(default="deepseek")
    quick_model: str = Field(..., min_length=1)
    deep_model: str = Field(..., min_length=1)
    mode: str = Field(default="simple", pattern="^(simple|medium|complex)$")
    trade_date: str | None = None
    analysts: list[str] | None = None
    analysis_mode: str = Field(default="auto", pattern="^(auto|full)$")
    chrome_platform: str | None = Field(
        default=None, pattern="^(macos|windows|ubuntu)$"
    )
    chrome_auto_start: bool = True
    position: PositionContext | None = None


class GeneralSettingsRequest(BaseModel):
    llm_provider: str = Field(..., min_length=1)
    quick_think_llm: str = Field(..., min_length=1)
    deep_think_llm: str = Field(..., min_length=1)
    backend_url: str | None = None
    output_language: str = Field(default="Chinese", min_length=1)
    web_research_depth: str = Field(
        default="simple", pattern="^(simple|medium|complex)$"
    )
    web_analysis_mode: str = Field(default="auto", pattern="^(auto|full)$")
    parallel_initial_analysts: bool = True
    parallel_blind_researchers: bool = True
    cn_market_data_source: str = Field(
        default="yfinance", pattern="^(yfinance|eastmoney)$"
    )
    news_article_limit: int = Field(default=20, ge=1, le=100)
    global_news_article_limit: int = Field(default=10, ge=1, le=100)
    cn_guba_post_limit: int = Field(default=15, ge=1, le=100)
    ticker_memory_fundamentals_ttl_days: int = Field(default=30, ge=0, le=3650)
    cn_browser_platform: str = Field(
        default="macos", pattern="^(macos|windows|ubuntu)$"
    )
    cn_browser_executable: str | None = None
    cn_browser_profile_dir: str = Field(
        default="./browser_data/chrome-profile", min_length=1
    )
    cn_browser_startup_timeout_seconds: float = Field(default=15, ge=1, le=120)
    cn_browser_auto_start: bool = True
    cn_browser_auto_close: bool = True
    cn_browser_mode: str = Field(
        default="background",
        pattern="^(background|headless|visible)$",
    )


class ApiKeyRequest(BaseModel):
    value: str = Field(..., min_length=1)


LOGIN_SITES = {
    "nga": "https://bbs.nga.cn/",
    "xueqiu": "https://xueqiu.com/",
    "eastmoney": "https://www.eastmoney.com/",
    "ths": "https://www.10jqka.com.cn/",
    "cninfo": "https://www.cninfo.com.cn/",
}


def _require_local_request(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(
            status_code=403,
            detail="Settings and browser controls are only available locally",
        )


def _serialize_model_options() -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for provider, modes in MODEL_OPTIONS.items():
        providers[provider] = {
            "quick": [{"label": label, "value": value} for label, value in modes.get("quick", [])],
            "deep": [{"label": label, "value": value} for label, value in modes.get("deep", [])],
        }
    return {"providers": providers, "provider_list": sorted(providers.keys())}


@app.get("/")
def index() -> FileResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path)


@app.get("/settings")
def settings_page() -> FileResponse:
    html_path = STATIC_DIR / "settings.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="settings.html not found")
    return FileResponse(html_path)


@app.get("/calendar")
def calendar_page() -> FileResponse:
    html_path = STATIC_DIR / "calendar.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="calendar.html not found")
    return FileResponse(html_path)


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return _serialize_model_options()


@app.post("/api/run")
def create_run(body: RunRequest) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    state = RunState(run_id=run_id, ticker=body.ticker.strip())
    RUNS[run_id] = state

    params = RunParams(
        ticker=body.ticker.strip(),
        provider=body.provider,
        quick_model=body.quick_model,
        deep_model=body.deep_model,
        mode=body.mode,
        trade_date=body.trade_date,
        analysts=body.analysts,
        analysis_mode=body.analysis_mode,
        chrome_platform=body.chrome_platform,
        chrome_auto_start=body.chrome_auto_start,
        position=body.position.model_dump() if body.position is not None else None,
    )
    # 预先写入 config 便于测试断言
    state.config = build_run_config(params)
    start_run(state, params)
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    def event_generator():
        while True:
            try:
                event = state.event_queue.get(timeout=15)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                if state.status in ("done", "error"):
                    break
                continue

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str) -> dict[str, Any]:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if state.status != "done" or not state.report_path:
        return {
            "available": False,
            "markdown": "",
            "decision": state.decision,
            "report_dir": str(state.report_path) if state.report_path else None,
        }

    report_file = state.report_path / "complete_report.md"
    markdown = report_file.read_text(encoding="utf-8") if report_file.is_file() else ""
    return {
        "available": bool(markdown),
        "markdown": markdown,
        "sections": read_report_sections(state.report_path),
        "audit": read_audit_metadata(state.report_path),
        "core_insights": read_core_insights(state.report_path),
        "decision": state.decision,
        "report_dir": str(state.report_path),
    }


@app.get("/api/runs/{run_id}/debug-log")
def get_run_debug_log(run_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    try:
        return read_run_debug_log(run_id, RUNS.get(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/debug-log/latest")
def get_latest_run_debug_log(
    request: Request,
    ticker: str | None = None,
) -> dict[str, Any]:
    _require_local_request(request)
    return read_latest_run_debug_log(ticker)


@app.get("/api/runs/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    report_sections = [
        event.get("section") or event.get("label", "")
        for event in state.debug_events
        if event.get("type") == "report_section"
    ]
    latest_status = next(
        (
            event.get("message")
            for event in reversed(state.debug_events)
            if event.get("type") == "status" and event.get("message")
        ),
        None,
    )
    return {
        "run_id": run_id,
        "ticker": state.ticker,
        "started_at": state.started_at,
        "status": state.status,
        "completed_sections": report_sections,
        "status_message": latest_status,
        "decision": state.decision,
        "report_available": state.status == "done" and state.report_path is not None,
        "error": state.error,
    }


@app.get("/api/reports/history")
def list_report_history(limit: int = 100) -> dict[str, Any]:
    items = list_historical_reports(limit=limit)
    return {"reports": items}


@app.delete("/api/stocks/{ticker}")
def delete_stock(ticker: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if any(
        state.ticker.upper() == ticker.strip().upper()
        and state.status in {"queued", "running"}
        for state in RUNS.values()
    ):
        raise HTTPException(status_code=409, detail="该股票正在分析，暂时不能删除")
    try:
        return delete_stock_reports(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/calendar/nodes")
def get_calendar_nodes() -> dict[str, Any]:
    return {"nodes": list_calendar_nodes()}


@app.get("/api/memory/{ticker}")
def get_ticker_memory_status(ticker: str, trade_date: str | None = None) -> dict[str, Any]:
    return TickerMemoryStore(DEFAULT_CONFIG).status(
        ticker.strip(),
        as_of=trade_date,
    )


@app.get("/api/calibration/{ticker}")
def get_calibration(ticker: str) -> dict[str, Any]:
    try:
        return CalibrationStore(DEFAULT_CONFIG).query(ticker.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/settings")
def get_settings(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return {
        "general": get_general_settings(),
        "api_keys": get_api_key_status(),
        "login_sites": list(LOGIN_SITES),
    }


@app.put("/api/settings/general")
def update_general_settings(
    body: GeneralSettingsRequest,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    current_manager = ChromeManager(DEFAULT_CONFIG)
    requested_profile = str(Path(body.cn_browser_profile_dir).expanduser().resolve())
    if (
        current_manager.is_cdp_available()
        and requested_profile != str(current_manager.profile_dir.resolve())
    ):
        raise HTTPException(
            status_code=409,
            detail="Stop Chrome before changing the profile directory",
        )
    try:
        return {"general": save_general_settings(body.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/settings/api-keys/{key}")
def update_api_key(
    key: str,
    body: ApiKeyRequest,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    try:
        return save_api_key(key, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/settings/api-keys/{key}")
def remove_api_key(key: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    try:
        return delete_api_key(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/browser/status")
def get_browser_status(request: Request, platform: str | None = None) -> dict[str, Any]:
    _require_local_request(request)
    config = DEFAULT_CONFIG.copy()
    if platform:
        if platform not in {"macos", "windows", "ubuntu"}:
            raise HTTPException(status_code=400, detail="unsupported Chrome platform")
        config["cn_browser_platform"] = platform
    return ChromeManager(config).status()


@app.post("/api/browser/start")
def start_browser(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    manager = ChromeManager(DEFAULT_CONFIG)
    status = manager.status()
    if status["available"]:
        return {**status, "state": "already_running"}
    result = manager.ensure_running()
    if result.get("state") == "failed_fallback":
        raise HTTPException(status_code=503, detail=result.get("message"))
    return result


@app.post("/api/browser/close")
def close_browser(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if any(state.status == "running" for state in RUNS.values()):
        raise HTTPException(
            status_code=409,
            detail="Chrome cannot be closed while an analysis is running",
        )
    manager = ChromeManager(DEFAULT_CONFIG)
    if not manager.status()["managed"]:
        raise HTTPException(
            status_code=409,
            detail="The running Chrome process is not managed by this server",
        )
    result = manager.close_managed()
    if result.get("state") == "retained":
        raise HTTPException(
            status_code=409,
            detail="Chrome is still in use by another operation",
        )
    return result


@app.post("/api/browser/open-login-site/{site}")
def open_browser_login_site(site: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    url = LOGIN_SITES.get(site)
    if url is None:
        raise HTTPException(status_code=404, detail="Unknown login site")
    manager = ChromeManager(DEFAULT_CONFIG)
    result = manager.open_url(url)
    if result.get("state") != "opened":
        raise HTTPException(
            status_code=409,
            detail=result.get("message", "Chrome is not running"),
        )
    return result


@app.get("/api/stocks/{ticker}/overview")
def get_stock_overview_api(ticker: str, range: str = "1d") -> dict[str, Any]:
    try:
        return get_stock_overview(ticker, chart_range=range)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/stocks/{ticker}/quote")
def get_stock_quote_api(ticker: str) -> dict[str, Any]:
    try:
        return get_stock_quote(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/stocks/{ticker}/chart")
def get_stock_chart_api(ticker: str, range: str = "1d") -> dict[str, Any]:
    try:
        return get_stock_chart(ticker, chart_range=range)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/stocks/{ticker}/reports")
def get_stock_reports_api(ticker: str, limit: int = 1000) -> dict[str, Any]:
    try:
        return {"reports": get_stock_reports(ticker, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/reports/history/{report_id:path}")
def get_report_history_item(report_id: str) -> dict[str, Any]:
    try:
        return get_historical_report(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/reports/history/{report_id:path}")
def delete_report_history_item(
    report_id: str,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    try:
        return delete_historical_report(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    host = os.getenv("FXXKSTOCK_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    uvicorn.run("webapp.server:app", host=host, port=8000, reload=False)


if __name__ == "__main__":
    main()
