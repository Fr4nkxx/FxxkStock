"""Tests for CN News Analyst pre-fetch of domestic news sources."""

import copy
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import fxxkstock.default_config as default_config
from fxxkstock.agents.analysts import news_analyst as news_analyst_module
from fxxkstock.agents.analysts.news_analyst import create_news_analyst
from fxxkstock.dataflows.config import set_config


@pytest.fixture(autouse=True)
def _clear_cn_prefetch_cache():
    news_analyst_module._cn_prefetch_cache.clear()
    yield
    news_analyst_module._cn_prefetch_cache.clear()


def _make_news_state(ticker: str = "603678.SS") -> dict:
    return {
        "company_of_interest": ticker,
        "trade_date": "2026-06-26",
        "asset_type": "stock",
        "messages": [],
    }


def _mock_news_llm(captured: dict, *, content: str = "CN news report body") -> MagicMock:
    """LLM mock compatible with LangChain LCEL (prompt | llm.bind_tools)."""

    def _capture_and_reply(formatted):
        captured["prompt"] = formatted
        return AIMessage(content=content)

    llm = MagicMock()

    def bind_tools(tools):
        captured["tools"] = [t.name for t in tools]
        return RunnableLambda(_capture_and_reply)

    llm.bind_tools.side_effect = bind_tools
    return llm


@pytest.mark.unit
def test_cn_market_prefetches_news_and_injects_prompt(monkeypatch):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    calls = {"news": 0, "global": 0, "insider": 0}

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_news.func",
        lambda t, s, e: calls.__setitem__("news", calls["news"] + 1) or "STOCK_NEWS_BLOCK",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_global_news.func",
        lambda curr_date, look_back_days=None, limit=None: calls.__setitem__(
            "global", calls["global"] + 1
        )
        or "GLOBAL_NEWS_BLOCK",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_insider_transactions.func",
        lambda t: calls.__setitem__("insider", calls["insider"] + 1) or "INSIDER_BLOCK",
    )

    captured = {}
    analyst = create_news_analyst(_mock_news_llm(captured))
    result = analyst(_make_news_state())

    assert calls == {"news": 1, "global": 1, "insider": 1}
    assert captured["tools"] == ["get_macro_indicators", "get_prediction_markets"]
    prompt_text = str(captured["prompt"])
    assert "STOCK_NEWS_BLOCK" in prompt_text
    assert "GLOBAL_NEWS_BLOCK" in prompt_text
    assert "INSIDER_BLOCK" in prompt_text
    assert "<start_of_news>" in prompt_text
    assert "<start_of_global_news>" in prompt_text
    assert "<start_of_insider>" in prompt_text
    assert result["news_report"] == "CN news report body"


@pytest.mark.unit
def test_cn_hk_skips_insider_prefetch(monkeypatch):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_hk"})

    insider_called = {"v": False}

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_news.func",
        lambda t, s, e: "news",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_global_news.func",
        lambda curr_date, look_back_days=None, limit=None: "global",
    )

    def _insider(t):
        insider_called["v"] = True
        return "insider"

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_insider_transactions.func",
        _insider,
    )

    captured = {}
    create_news_analyst(_mock_news_llm(captured))(_make_news_state("0700.HK"))

    assert insider_called["v"] is False
    assert "<start_of_insider>" not in str(captured["prompt"])


@pytest.mark.unit
def test_us_market_does_not_prefetch(monkeypatch):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "default"})

    def _fail(*args, **kwargs):
        raise AssertionError("prefetch should not run for US/default region")

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_news.func",
        _fail,
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_global_news.func",
        _fail,
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_insider_transactions.func",
        _fail,
    )

    captured = {}
    analyst = create_news_analyst(
        _mock_news_llm(captured, content="US news report")
    )
    result = analyst(_make_news_state("AAPL"))

    assert "get_news" in captured["tools"]
    assert "get_global_news" in captured["tools"]
    assert "get_macro_indicators" in captured["tools"]
    assert "get_prediction_markets" in captured["tools"]
    assert result["news_report"] == "US news report"


@pytest.mark.unit
def test_cn_prefetch_cache_skips_duplicate_fetches_on_reentry(monkeypatch):
    """Tool round re-enters News Analyst — prefetch must not hit vendors again."""
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    calls = {"news": 0, "global": 0}

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_news.func",
        lambda t, s, e: calls.__setitem__("news", calls["news"] + 1) or "NEWS",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_global_news.func",
        lambda curr_date, look_back_days=None, limit=None: calls.__setitem__(
            "global", calls["global"] + 1
        )
        or "GLOBAL",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_insider_transactions.func",
        lambda t: "INSIDER",
    )

    llm = MagicMock()
    invoke_count = {"n": 0}

    def bind_tools(tools):
        def _reply(formatted):
            invoke_count["n"] += 1
            if invoke_count["n"] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_prediction_markets",
                            "args": {"topic": "Fed", "limit": 3},
                            "id": "call_1",
                        }
                    ],
                )
            return AIMessage(content="Final news report")

        return RunnableLambda(_reply)

    llm.bind_tools.side_effect = bind_tools

    analyst = create_news_analyst(llm)
    state = _make_news_state()
    analyst(state)
    from langchain_core.messages import ToolMessage

    state["messages"].append(
        ToolMessage(content="polymarket data", tool_call_id="call_1")
    )
    analyst(state)

    assert calls == {"news": 1, "global": 1}
