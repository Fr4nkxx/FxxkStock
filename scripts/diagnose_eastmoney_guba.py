"""Diagnose East Money Guba transports without running agents."""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fxxkstock.dataflows.chrome_manager import ChromeManager
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.eastmoney_browser import _guba_url, _parse_guba_from_html
from fxxkstock.dataflows.eastmoney_guba import _fetch_guba_html, _fetch_guba_json
from fxxkstock.dataflows.market_utils import detect_market_region, to_guba_code
from fxxkstock.default_config import DEFAULT_CONFIG


def _sample(posts: list[dict]) -> str:
    return " | ".join(post.get("title", "")[:60] for post in posts[:3]) or "-"


def _open_and_parse(url: str, artifact: Path, timeout_ms: int) -> list[dict]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(
            "http://127.0.0.1:9222",
            timeout=timeout_ms,
        )
        if not browser.contexts:
            raise RuntimeError("Chrome has no browser context")
        page = browser.contexts[0].new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1500)
        html = page.content()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(html, encoding="utf-8")
        print(f"browser_page: OPEN, intentionally left running | {page.title()} | {page.url}")
        return _parse_guba_from_html(html)


def _run(label: str, operation) -> None:
    try:
        posts = operation()
        print(f"{label}: success items={len(posts)} sample={_sample(posts)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: ERROR {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test East Money Guba only.")
    parser.add_argument("--ticker", default="600667.SS")
    parser.add_argument("--platform", choices=("macos", "windows", "ubuntu"))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--artifacts-dir", default="logs/source_diagnostics")
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.platform:
        config["cn_browser_platform"] = args.platform
    region = detect_market_region(args.ticker)
    config["market_region"] = region
    set_config(config)

    code = to_guba_code(args.ticker, region)
    url = _guba_url(args.ticker, region)
    artifact = (
        Path(args.artifacts_dir)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
        / "eastmoney_guba.html"
    )
    print(f"ticker: {args.ticker} code={code}")
    print(f"url: {url}")

    _run("json_api", lambda: _fetch_guba_json(code, args.limit))
    _run("http_html", lambda: _fetch_guba_html(code, args.limit))

    manager = ChromeManager(config)
    status = manager.ensure_running()
    print(f"chrome: {status.get('state')} ({status.get('message', '')})")
    if status.get("available"):
        _run(
            "browser_html",
            lambda: _open_and_parse(
                url,
                artifact,
                int(config.get("cn_browser_nav_timeout_ms", 20000)),
            ),
        )
        print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
