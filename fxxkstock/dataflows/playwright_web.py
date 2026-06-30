"""Playwright + CDP 浏览器传输层 — 连接本地真实 Chrome 渲染页面。"""

from __future__ import annotations

import logging
import re
import time
from typing import Literal

from .config import get_config
from .errors import BrowserUnavailableError, VendorRateLimitError

logger = logging.getLogger(__name__)

# 上次 CDP 请求时间戳，用于请求间隔节流
_last_request_at: float = 0.0

WaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]


def _throttle() -> None:
    """两次浏览器导航之间的最小间隔。"""
    global _last_request_at
    delay = float(get_config().get("cn_http_inter_request_delay", 0.5))
    if delay <= 0:
        return
    now = time.monotonic()
    wait = delay - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _classify_navigation_error(exc: Exception) -> Exception:
    """将 Playwright 导航错误映射为 vendor 契约异常。"""
    msg = str(exc).lower()
    if any(token in msg for token in ("429", "503", "rate limit", "too many requests")):
        return VendorRateLimitError(str(exc))
    if any(
        token in msg
        for token in (
            "connect",
            "econnrefused",
            "websocket",
            "cdp",
            "target closed",
            "browser has been closed",
            "timeout",
        )
    ):
        return BrowserUnavailableError(str(exc))
    return BrowserUnavailableError(str(exc))


def _detect_blocked_page(visible_text: str) -> str | None:
    """Classify only user-visible blocker text, not hidden templates/scripts."""
    text = re.sub(r"\s+", " ", visible_text or "").strip()
    lower = text.lower()
    if re.search(r"(访问|请求)过于频繁|请稍后再试|too many requests", text, re.I):
        return "rate_limit"
    if re.search(r"验证码|安全验证|access denied|verify you are human", lower, re.I):
        return "blocked"
    return None


def render_html(
    url: str,
    *,
    wait_until: WaitUntil | None = None,
    wait_selector: str | None = None,
    timeout_ms: int | None = None,
) -> str:
    """通过 CDP 连接本地 Chrome，加载 URL 并返回渲染后的 HTML。

    要求 Chrome 已以 ``--remote-debugging-port`` 启动。
    """
    config = get_config()
    cdp_url = config.get("cn_browser_cdp_url", "http://127.0.0.1:9222")
    if wait_until is None:
        wait_until = config.get("cn_browser_wait_until", "networkidle")
    if timeout_ms is None:
        timeout_ms = int(config.get("cn_browser_nav_timeout_ms", 20000))

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserUnavailableError("playwright package is not installed") from exc

    _throttle()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url, timeout=timeout_ms)
            if not browser.contexts:
                raise BrowserUnavailableError(f"no browser context at {cdp_url}")
            context = browser.contexts[0]
            page = context.new_page()
            html = ""
            visible_text = ""
            try:
                response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                if response is not None and response.status in (429, 503):
                    raise VendorRateLimitError(
                        f"HTTP {response.status} for {url}"
                    )
                if wait_selector:
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
                html = page.content()
                visible_text = page.locator("body").inner_text(timeout=timeout_ms)
            finally:
                page.close()
    except (BrowserUnavailableError, VendorRateLimitError):
        raise
    except PlaywrightError as exc:
        raise _classify_navigation_error(exc) from exc
    except Exception as exc:
        raise _classify_navigation_error(exc) from exc

    if not html or not html.strip():
        raise BrowserUnavailableError(f"empty HTML from {url}")

    # Hidden login/captcha modals and JavaScript error strings are present in
    # normal vendor pages. Inspect rendered, visible body text to avoid treating
    # those templates as an active blocker.
    blocker = _detect_blocked_page(visible_text)
    if blocker:
        if blocker == "rate_limit":
            raise VendorRateLimitError(f"rate-limited page at {url}")
        raise BrowserUnavailableError(f"blocked or captcha page at {url}")

    return html
