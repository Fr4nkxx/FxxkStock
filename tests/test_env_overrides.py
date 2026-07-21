"""Tests for FXXKSTOCK_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import fxxkstock.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.5"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.4-mini"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False
    assert dc.DEFAULT_CONFIG["cn_market_data_source"] == "yfinance"
    assert dc.DEFAULT_CONFIG["cn_browser_mode"] == "background"
    assert dc.DEFAULT_CONFIG["parallel_initial_analysts"] is True
    assert dc.DEFAULT_CONFIG["parallel_blind_researchers"] is True
    assert (
        dc.DEFAULT_CONFIG["falsification_structured_method"]
        == "provider_default"
    )


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_LLM_PROVIDER="google",
        FXXKSTOCK_DEEP_THINK_LLM="gemini-3-pro-preview",
        FXXKSTOCK_QUICK_THINK_LLM="gemini-3-flash-preview",
        FXXKSTOCK_LLM_BACKEND_URL="https://example.invalid/v1",
        FXXKSTOCK_OUTPUT_LANGUAGE="Chinese",
        FXXKSTOCK_CHROME_MODE="headless",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"
    assert dc.DEFAULT_CONFIG["cn_browser_mode"] == "headless"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_MAX_DEBATE_ROUNDS="3",
        FXXKSTOCK_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, FXXKSTOCK_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable (non-interactive runs)."""
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_OPENAI_REASONING_EFFORT="high",
        FXXKSTOCK_GOOGLE_THINKING_LEVEL="minimal",
        FXXKSTOCK_ANTHROPIC_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "high"
    assert dc.DEFAULT_CONFIG["google_thinking_level"] == "minimal"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "low"


def test_cn_market_data_source_override(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_CN_MARKET_DATA_SOURCE="eastmoney",
    )
    assert dc.DEFAULT_CONFIG["cn_market_data_source"] == "eastmoney"


def test_parallel_initial_analyst_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_PARALLEL_INITIAL_ANALYSTS="false",
        FXXKSTOCK_PARALLEL_INITIAL_ANALYST_WORKERS="2",
    )
    assert dc.DEFAULT_CONFIG["parallel_initial_analysts"] is False
    assert dc.DEFAULT_CONFIG["parallel_initial_analyst_workers"] == 2


def test_parallel_blind_researcher_override(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_PARALLEL_BLIND_RESEARCHERS="true",
    )
    assert dc.DEFAULT_CONFIG["parallel_blind_researchers"] is True


def test_falsification_structured_method_override(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_FALSIFICATION_STRUCTURED_METHOD="auto",
    )
    assert dc.DEFAULT_CONFIG["falsification_structured_method"] == "auto"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["google_thinking_level"] is None
    assert dc.DEFAULT_CONFIG["anthropic_effort"] is None


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty FXXKSTOCK_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_LLM_PROVIDER="",
        FXXKSTOCK_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("FXXKSTOCK_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="FXXKSTOCK_MAX_DEBATE_ROUNDS"):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("FXXKSTOCK_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("FXXKSTOCK_CHECKPOINT_ENABLED", bad)
    with pytest.raises(ValueError, match="FXXKSTOCK_CHECKPOINT_ENABLED"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("FXXKSTOCK_CHECKPOINT_ENABLED", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        FXXKSTOCK_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG
