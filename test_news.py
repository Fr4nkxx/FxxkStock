from playwright.sync_api import sync_playwright

cdp = "http://127.0.0.1:9222"
with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp(cdp, timeout=30000)
    print("contexts:", len(browser.contexts))
    ctx = browser.contexts[0]
    print("existing pages:", len(ctx.pages))

    page = ctx.new_page()
    print("new page ok, url:", page.url)

    page.goto(
        "https://so.eastmoney.com/news/s?keyword=603678",
        wait_until="load",
        timeout=30000,
    )
    print("after goto, url:", page.url)
    page.wait_for_timeout(5000)
    print("html len:", len(page.content()))
    page.screenshot(path="em_debug.png", full_page=True)
    print("screenshot saved: em_debug.png")
    # 先不要 page.close()，方便你在浏览器里看