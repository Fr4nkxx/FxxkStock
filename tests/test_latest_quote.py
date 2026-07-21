from __future__ import annotations

import pytest


@pytest.mark.unit
def test_fetch_latest_market_quote_uses_short_cache(monkeypatch):
    from fxxkstock.dataflows import latest_quote

    latest_quote._latest_quote_cache.clear()
    calls = []

    def fake_eastmoney(ticker, region=None):
        calls.append((ticker, region))
        return {"ticker": ticker, "source": "eastmoney", "last_price": 54.19}

    monkeypatch.setattr(latest_quote, "fetch_eastmoney_latest_quote", fake_eastmoney)

    first = latest_quote.fetch_latest_market_quote("002364.SZ", "cn_a")
    second = latest_quote.fetch_latest_market_quote("002364.SZ", "cn_a")

    assert first == second
    assert calls == [("002364.SZ", "cn_a")]
