"""Project-local Web settings persisted through the existing .env contract."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from dotenv import set_key, unset_key

from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
_SETTINGS_LOCK = threading.RLock()

API_KEY_PROVIDERS: dict[str, list[str]] = {}
for _provider, _env_name in PROVIDER_API_KEY_ENV.items():
    if _env_name:
        API_KEY_PROVIDERS.setdefault(_env_name, []).append(_provider)
API_KEY_PROVIDERS.setdefault("FRED_API_KEY", []).append("fred")
API_KEY_PROVIDERS.setdefault("ALPHA_VANTAGE_API_KEY", []).append("alpha_vantage")
ALLOWED_API_KEYS = frozenset(API_KEY_PROVIDERS)

GENERAL_ENV_MAP: dict[str, str] = {
    "llm_provider": "FXXKSTOCK_LLM_PROVIDER",
    "quick_think_llm": "FXXKSTOCK_QUICK_THINK_LLM",
    "deep_think_llm": "FXXKSTOCK_DEEP_THINK_LLM",
    "backend_url": "FXXKSTOCK_LLM_BACKEND_URL",
    "output_language": "FXXKSTOCK_OUTPUT_LANGUAGE",
    "web_research_depth": "FXXKSTOCK_WEB_RESEARCH_DEPTH",
    "web_analysis_mode": "FXXKSTOCK_WEB_ANALYSIS_MODE",
    "news_article_limit": "FXXKSTOCK_NEWS_ARTICLE_LIMIT",
    "global_news_article_limit": "FXXKSTOCK_GLOBAL_NEWS_LIMIT",
    "cn_guba_post_limit": "FXXKSTOCK_CN_GUBA_POST_LIMIT",
    "ticker_memory_fundamentals_ttl_days": "FXXKSTOCK_FUNDAMENTALS_TTL_DAYS",
    "cn_browser_platform": "FXXKSTOCK_CHROME_PLATFORM",
    "cn_browser_executable": "FXXKSTOCK_CHROME_EXECUTABLE",
    "cn_browser_profile_dir": "FXXKSTOCK_CHROME_PROFILE_DIR",
    "cn_browser_startup_timeout_seconds": "FXXKSTOCK_CHROME_STARTUP_TIMEOUT",
    "cn_browser_auto_start": "FXXKSTOCK_CHROME_AUTO_START",
    "cn_browser_auto_close": "FXXKSTOCK_CHROME_AUTO_CLOSE",
}

_DEPTH_ROUNDS = {"simple": 1, "medium": 3, "complex": 5}


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def get_general_settings() -> dict[str, Any]:
    return {key: DEFAULT_CONFIG.get(key) for key in GENERAL_ENV_MAP}


def get_api_key_status() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "providers": sorted(API_KEY_PROVIDERS[key]),
            "configured": bool(os.environ.get(key)),
        }
        for key in sorted(ALLOWED_API_KEYS)
    ]


def save_general_settings(values: dict[str, Any]) -> dict[str, Any]:
    unknown = set(values) - set(GENERAL_ENV_MAP)
    if unknown:
        raise ValueError(f"unsupported settings: {', '.join(sorted(unknown))}")

    with _SETTINGS_LOCK:
        ENV_PATH.touch(exist_ok=True)
        for key, value in values.items():
            env_name = GENERAL_ENV_MAP[key]
            if value is None or value == "":
                unset_key(str(ENV_PATH), env_name)
                os.environ.pop(env_name, None)
            else:
                rendered = _env_value(value)
                set_key(str(ENV_PATH), env_name, rendered)
                os.environ[env_name] = rendered
            DEFAULT_CONFIG[key] = value

        depth = values.get("web_research_depth")
        if depth:
            rounds = _DEPTH_ROUNDS[str(depth)]
            for config_key, env_name in (
                ("max_debate_rounds", "FXXKSTOCK_MAX_DEBATE_ROUNDS"),
                ("max_risk_discuss_rounds", "FXXKSTOCK_MAX_RISK_ROUNDS"),
            ):
                set_key(str(ENV_PATH), env_name, str(rounds))
                os.environ[env_name] = str(rounds)
                DEFAULT_CONFIG[config_key] = rounds
    return get_general_settings()


def save_api_key(key: str, value: str) -> dict[str, Any]:
    if key not in ALLOWED_API_KEYS:
        raise ValueError("unsupported API key")
    value = value.strip()
    if not value:
        raise ValueError("API key cannot be empty")
    with _SETTINGS_LOCK:
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), key, value)
        os.environ[key] = value
    return {"key": key, "configured": True}


def delete_api_key(key: str) -> dict[str, Any]:
    if key not in ALLOWED_API_KEYS:
        raise ValueError("unsupported API key")
    with _SETTINGS_LOCK:
        if ENV_PATH.exists():
            unset_key(str(ENV_PATH), key)
        os.environ.pop(key, None)
    return {"key": key, "configured": False}
