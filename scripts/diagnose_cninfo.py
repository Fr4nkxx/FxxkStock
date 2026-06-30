"""Diagnose CNINFO announcement requests and keep the disclosure page open."""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fxxkstock.dataflows.chrome_manager import ChromeManager
from fxxkstock.dataflows.cninfo import (
    _QUERY_URL,
    _TOP_SEARCH_URL,
    _http_get_json,
    _http_post,
    _normalize_org_id,
    _plate_info,
    _resolve_org_id,
    _resolve_org_id_via_browser,
    fetch_cninfo_announcements,
)
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.market_utils import to_cninfo_stock_code
from fxxkstock.default_config import DEFAULT_CONFIG


def _payload(
    stock_code: str,
    org_id: str,
    start_date: str,
    end_date: str,
    limit: int,
) -> dict[str, str]:
    column, plate, _ = _plate_info(stock_code)
    if org_id.startswith("jjjl"):
        column, plate = "fund", ""
    return {
        "stock": f"{stock_code},{org_id}",
        "tabName": "fulltext",
        "pageSize": str(limit),
        "pageNum": "1",
        "column": column,
        "category": "",
        "plate": plate,
        "seDate": f"{start_date}~{end_date}",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def _announcement_count(payload: dict) -> tuple[int, dict]:
    response = _http_post(_QUERY_URL, payload)
    return len(response.get("announcements") or []), response


def _open_disclosure_page(stock_code: str, org_id: str, timeout_ms: int) -> str:
    from playwright.sync_api import sync_playwright

    url = (
        "https://www.cninfo.com.cn/new/disclosure/stock?"
        f"stockCode={stock_code}&orgId={org_id}"
    )
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(
            DEFAULT_CONFIG.get("cn_browser_cdp_url", "http://127.0.0.1:9222"),
            timeout=timeout_ms,
        )
        if not browser.contexts:
            raise RuntimeError("Chrome has no browser context")
        page = browser.contexts[0].new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        return f"{page.title()} | {url}"


def _search_disclosure_page(stock_code: str, timeout_ms: int) -> tuple[str, str]:
    """Search CNINFO's UI and return the selected result URL and orgId."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(
            DEFAULT_CONFIG.get("cn_browser_cdp_url", "http://127.0.0.1:9222"),
            timeout=timeout_ms,
        )
        if not browser.contexts:
            raise RuntimeError("Chrome has no browser context")
        page = browser.contexts[0].new_page()
        page.goto(
            "https://www.cninfo.com.cn/new/index",
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        inputs = page.locator(
            "input[placeholder*='代码'], input[placeholder*='简称'], "
            "input[placeholder*='拼音']"
        )
        target = None
        for index in range(inputs.count()):
            candidate = inputs.nth(index)
            if candidate.is_visible():
                target = candidate
                break
        if target is None:
            raise RuntimeError("CNINFO search input not found")
        target.fill(stock_code)
        page.wait_for_timeout(1500)
        result = page.locator(
            f"a:has-text('{stock_code}'), "
            f"[class*='search'] li:has-text('{stock_code}'), "
            f"[class*='suggest'] li:has-text('{stock_code}')"
        ).first
        if not result.count():
            raise RuntimeError(f"no UI search result for {stock_code}")
        result.click()
        page.wait_for_timeout(1500)
        url = page.url
        org_id = (parse_qs(urlparse(url).query).get("orgId") or [""])[0]
        if not org_id:
            raise RuntimeError(f"search result URL has no orgId: {url}")
        return url, org_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CNINFO without running agents.")
    parser.add_argument("--ticker", default="600667.SS")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--org-id", help="Override CNINFO orgId for diagnostics.")
    parser.add_argument("--platform", choices=("macos", "windows", "ubuntu"))
    args = parser.parse_args()

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.platform:
        config["cn_browser_platform"] = args.platform
    config["market_region"] = "cn_a"
    set_config(config)

    stock_code = to_cninfo_stock_code(args.ticker)
    org_id = (
        args.org_id
        or _resolve_org_id(stock_code)
        or _resolve_org_id_via_browser(stock_code)
    )
    if not org_id:
        print(f"org_id:              no stock mapping for {stock_code}")
        for suffix in ("", "&maxNum=10"):
            try:
                top_search = _http_get_json(
                    f"{_TOP_SEARCH_URL}?keyWord={stock_code}{suffix}"
                )
                print(f"top_search{suffix or '_basic'}: {top_search}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"top_search{suffix or '_basic'}: "
                    f"ERROR {type(exc).__name__}: {exc}"
                )
        fund_payload = _payload(
            stock_code,
            "",
            start_date,
            end_date,
            args.limit,
        )
        fund_payload.update(
            {
                "stock": "",
                "column": "fund",
                "plate": "",
                "searchkey": stock_code,
            }
        )
        try:
            count, response = _announcement_count(fund_payload)
            print(
                f"fund_search_payload: {count} announcements; "
                f"total={response.get('totalAnnouncement', '?')}; "
                f"hasMore={response.get('hasMore', '?')}"
            )
            for ann in (response.get("announcements") or [])[: args.limit]:
                print(
                    f"  {ann.get('secCode')} | "
                    f"{ann.get('secName')} | "
                    f"{ann.get('announcementTitle')}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"fund_search_payload: ERROR {type(exc).__name__}: {exc}")
        for label, overrides in (
            ("fund_stock_code", {"stock": stock_code, "searchkey": ""}),
            ("fund_secid_code", {"stock": "", "searchkey": "", "secid": stock_code}),
        ):
            payload = dict(fund_payload)
            payload.update(overrides)
            try:
                count, response = _announcement_count(payload)
                exact = [
                    ann
                    for ann in (response.get("announcements") or [])
                    if str(ann.get("secCode") or "") == stock_code
                ]
                print(
                    f"{label}: {count} returned; exact={len(exact)}; "
                    f"total={response.get('totalAnnouncement', '?')}"
                )
                for ann in exact[: args.limit]:
                    print(f"  EXACT {ann.get('announcementTitle')}")
            except Exception as exc:  # noqa: BLE001
                print(f"{label}: ERROR {type(exc).__name__}: {exc}")
        manager = ChromeManager(config)
        status = manager.ensure_running()
        print(f"chrome: {status.get('state')} ({status.get('message', '')})")
        if status.get("available"):
            try:
                search_url, search_org_id = _search_disclosure_page(
                    stock_code,
                    int(config.get("cn_browser_nav_timeout_ms", 20000)),
                )
                print(f"ui_search: orgId={search_org_id} | {search_url}")
            except Exception as exc:  # noqa: BLE001
                print(f"ui_search: ERROR {type(exc).__name__}: {exc}")
        return 0

    _, _, prefix = _plate_info(stock_code)
    legacy_org_id = f"{prefix}{org_id}"
    normalized_org_id = _normalize_org_id(org_id, prefix)

    print(f"ticker:              {args.ticker}")
    print(f"stock_code:          {stock_code}")
    print(f"cached_org_id:       {org_id}")
    print(f"legacy_broken_id:    {legacy_org_id}")
    print(f"production_org_id:   {normalized_org_id}")
    print(f"date_range:          {start_date} ~ {end_date}")
    try:
        top_search = _http_get_json(f"{_TOP_SEARCH_URL}?keyWord={stock_code}")
        print(f"top_search:          {top_search}")
    except Exception as exc:  # noqa: BLE001
        print(f"top_search:          ERROR {type(exc).__name__}: {exc}")

    for label, candidate in (
        ("legacy_broken_payload", legacy_org_id),
        ("production_payload", normalized_org_id),
        ("raw_cached_payload", org_id),
    ):
        try:
            count, response = _announcement_count(
                _payload(stock_code, candidate, start_date, end_date, args.limit)
            )
            print(
                f"{label}: {count} announcements; "
                f"total={response.get('totalAnnouncement', '?')}; "
                f"hasMore={response.get('hasMore', '?')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: ERROR {type(exc).__name__}: {exc}")

    print("\nproduction_wrapper:")
    print(fetch_cninfo_announcements(args.ticker, start_date, end_date, limit=args.limit))

    manager = ChromeManager(config)
    status = manager.ensure_running()
    print(f"\nchrome: {status.get('state')} ({status.get('message', '')})")
    if status.get("available"):
        try:
            opened = _open_disclosure_page(
                stock_code,
                normalized_org_id,
                int(config.get("cn_browser_nav_timeout_ms", 20000)),
            )
            print(f"browser_page: OPEN, intentionally left running | {opened}")
        except Exception as exc:  # noqa: BLE001
            print(f"browser_page: ERROR {type(exc).__name__}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
