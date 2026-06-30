"""Diagnose Xueqiu stock-community parsing and keep the page open."""

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
from fxxkstock.dataflows.eastmoney_browser import _parse_guba_from_html, _xueqiu_url
from fxxkstock.dataflows.market_utils import detect_market_region
from fxxkstock.default_config import DEFAULT_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Xueqiu community only.")
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

    url = _xueqiu_url(args.ticker, region)
    artifact = (
        Path(args.artifacts_dir)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
        / "xueqiu.html"
    )

    manager = ChromeManager(config)
    status = manager.ensure_running()
    print(f"chrome: {status.get('state')} ({status.get('message', '')})")
    print(f"url: {url}")
    if not status.get("available"):
        return 1

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(
                config.get("cn_browser_cdp_url", "http://127.0.0.1:9222"),
                timeout=int(config.get("cn_browser_nav_timeout_ms", 20000)),
            )
            if not browser.contexts:
                raise RuntimeError("Chrome has no browser context")
            page = browser.contexts[0].new_page()
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(config.get("cn_browser_nav_timeout_ms", 20000)),
            )
            page.wait_for_timeout(2000)
            html = page.content()
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(html, encoding="utf-8")
            posts = _parse_guba_from_html(html)[: args.limit]
            print(f"browser_page: OPEN, intentionally left running | {page.title()} | {page.url}")
            print(f"items: {len(posts)}")
            for index, post in enumerate(posts[:5], start=1):
                print(
                    f"{index}. [{post.get('created', '?')}] "
                    f"{post.get('title', '')[:120]}"
                )
            print(f"artifact: {artifact}")
            return 0 if posts else 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
