from __future__ import annotations

import ssl

import scripts.diagnose_data_sources as diagnostics
from scripts.diagnose_data_sources import classify_error, compact_error, run_check


def test_error_classification():
    assert classify_error(ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")) == "ssl_error"
    assert classify_error(RuntimeError("blocked or captcha page")) == "blocked"
    assert classify_error(RuntimeError("HTTP 429 rate limit")) == "rate_limited"
    assert classify_error(RuntimeError("WebSocket CDP refused")) == "browser_unavailable"


def test_error_output_is_compact():
    error = RuntimeError("x" * 1000)
    assert len(compact_error(error, limit=80)) == 80
    assert compact_error(error, limit=80).endswith("...")


def test_run_check_records_success_and_failure():
    success = run_check("source", "http", lambda: ("ok", 2))
    failure = run_check(
        "source",
        "http",
        lambda: (_ for _ in ()).throw(RuntimeError("captcha page")),
    )

    assert success.status == "success"
    assert success.items == 2
    assert failure.status == "blocked"
    assert failure.items is None


def test_browser_check_saves_html(monkeypatch, tmp_path):
    monkeypatch.setattr(
        diagnostics,
        "render_html",
        lambda url, wait_selector: "<html><div class='post'>hello</div></html>",
    )
    artifact = tmp_path / "ths.html"

    detail, count = diagnostics.browser_posts_check(
        "https://example.test",
        lambda html: [{"title": "hello"}],
        artifact,
    )

    assert count == 1
    assert artifact.read_text() == "<html><div class='post'>hello</div></html>"
    assert str(artifact) in detail
