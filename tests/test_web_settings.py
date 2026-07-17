from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

from webapp import settings_store  # noqa: E402
from webapp.server import _require_local_request  # noqa: E402


def test_api_key_store_never_returns_secret(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_store, "ENV_PATH", env_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = settings_store.save_api_key("DEEPSEEK_API_KEY", "secret-value")

    assert response == {"key": "DEEPSEEK_API_KEY", "configured": True}
    assert os.environ["DEEPSEEK_API_KEY"] == "secret-value"
    assert "secret-value" not in str(settings_store.get_api_key_status())
    assert "secret-value" in env_path.read_text(encoding="utf-8")

    settings_store.delete_api_key("DEEPSEEK_API_KEY")
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_arbitrary_environment_key_is_rejected():
    with pytest.raises(ValueError):
        settings_store.save_api_key("PATH", "nope")


def test_general_settings_persist_and_update_runtime(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_store, "ENV_PATH", env_path)
    previous = dict(settings_store.DEFAULT_CONFIG)
    try:
        result = settings_store.save_general_settings(
            {
                "web_research_depth": "medium",
                "cn_market_data_source": "eastmoney",
                "cn_guba_post_limit": 25,
                "cn_browser_auto_start": False,
                "cn_browser_mode": "headless",
                "parallel_initial_analysts": False,
                "parallel_blind_researchers": True,
            }
        )
        assert result["web_research_depth"] == "medium"
        assert result["cn_market_data_source"] == "eastmoney"
        assert settings_store.DEFAULT_CONFIG["max_debate_rounds"] == 3
        assert settings_store.DEFAULT_CONFIG["cn_guba_post_limit"] == 25
        assert settings_store.DEFAULT_CONFIG["cn_browser_mode"] == "headless"
        assert settings_store.DEFAULT_CONFIG["parallel_initial_analysts"] is False
        assert settings_store.DEFAULT_CONFIG["parallel_blind_researchers"] is True
        text = env_path.read_text(encoding="utf-8")
        assert "FXXKSTOCK_WEB_RESEARCH_DEPTH='medium'" in text
        assert "FXXKSTOCK_CN_MARKET_DATA_SOURCE='eastmoney'" in text
        assert "FXXKSTOCK_CN_GUBA_POST_LIMIT='25'" in text
        assert "FXXKSTOCK_CHROME_MODE='headless'" in text
        assert "FXXKSTOCK_PARALLEL_INITIAL_ANALYSTS='false'" in text
        assert "FXXKSTOCK_PARALLEL_BLIND_RESEARCHERS='true'" in text
    finally:
        settings_store.DEFAULT_CONFIG.clear()
        settings_store.DEFAULT_CONFIG.update(previous)


def test_settings_access_is_local_only():
    local = MagicMock()
    local.client.host = "127.0.0.1"
    _require_local_request(local)

    remote = MagicMock()
    remote.client.host = "192.168.1.20"
    with pytest.raises(HTTPException) as exc:
        _require_local_request(remote)
    assert exc.value.status_code == 403
