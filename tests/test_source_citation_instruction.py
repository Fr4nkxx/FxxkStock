"""Tests for global source-citation prompt instructions."""

import copy
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import fxxkstock.default_config as default_config
from fxxkstock.agents.analysts import news_analyst as news_analyst_module
from fxxkstock.agents.analysts.news_analyst import create_news_analyst
from fxxkstock.agents.utils.agent_utils import (
    get_currency_instruction,
    get_report_instructions,
    get_source_citation_instruction,
)
from fxxkstock.dataflows.config import set_config


@pytest.mark.unit
def test_source_citation_instruction_contains_link_rules():
    text = get_source_citation_instruction()
    assert "Link:" in text
    assert "never invent links" in text
    assert "preserve any links" in text


@pytest.mark.unit
def test_report_instructions_includes_citation():
    text = get_report_instructions()
    assert get_source_citation_instruction().strip() in text


@pytest.mark.unit
def test_currency_instruction_when_report_currency_cny():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    text = get_currency_instruction()
    assert "CNY" in text
    assert "never invent FX" in text or "do not invent FX" in text


@pytest.mark.unit
def test_report_instructions_includes_currency():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    text = get_report_instructions()
    assert get_currency_instruction().strip() in text


@pytest.mark.unit
def test_cn_news_analyst_prompt_includes_citation_rules(monkeypatch):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"market_region": "cn_a"})

    news_analyst_module._cn_prefetch_cache.clear()

    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_news.func",
        lambda t, s, e: "### Headline (source: EM)\nSummary\nLink: https://example.com/a\n\n",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_global_news.func",
        lambda curr_date, look_back_days=None, limit=None: "global",
    )
    monkeypatch.setattr(
        "fxxkstock.agents.analysts.news_analyst.get_insider_transactions.func",
        lambda t: "insider",
    )

    captured = {}

    def bind_tools(tools):
        return RunnableLambda(
            lambda formatted: captured.__setitem__("prompt", formatted)
            or AIMessage(content="report")
        )

    llm = MagicMock()
    llm.bind_tools.side_effect = bind_tools

    create_news_analyst(llm)(
        {
            "company_of_interest": "603678.SS",
            "trade_date": "2026-06-26",
            "asset_type": "stock",
            "messages": [],
        }
    )

    prompt_text = str(captured["prompt"])
    assert "Source citation rules" in prompt_text
    assert "Link: https://..." in prompt_text
    assert "https://example.com/a" in prompt_text
