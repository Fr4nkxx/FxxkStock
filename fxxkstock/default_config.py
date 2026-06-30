import os
import sys

_FXXKSTOCK_HOME = os.path.join(os.path.expanduser("~"), ".fxxkstock")
_LEGACY_TRADINGAGENTS_HOME = os.path.join(
    os.path.expanduser("~"), ".tradingagents"
)

# Promote legacy variables so existing deployments keep working while every
# internal consumer can use the new FXXKSTOCK_* namespace consistently.
for _legacy_key, _legacy_value in tuple(os.environ.items()):
    if _legacy_key.startswith("TRADINGAGENTS_"):
        _new_key = _legacy_key.replace("TRADINGAGENTS_", "FXXKSTOCK_", 1)
        os.environ.setdefault(_new_key, _legacy_value)

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "FXXKSTOCK_LLM_PROVIDER":         "llm_provider",
    "FXXKSTOCK_DEEP_THINK_LLM":       "deep_think_llm",
    "FXXKSTOCK_QUICK_THINK_LLM":      "quick_think_llm",
    "FXXKSTOCK_LLM_BACKEND_URL":      "backend_url",
    "FXXKSTOCK_OUTPUT_LANGUAGE":      "output_language",
    "FXXKSTOCK_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "FXXKSTOCK_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "FXXKSTOCK_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "FXXKSTOCK_BENCHMARK_TICKER":     "benchmark_ticker",
    "FXXKSTOCK_TEMPERATURE":          "temperature",
    "FXXKSTOCK_CHROME_PLATFORM":      "cn_browser_platform",
    "FXXKSTOCK_CHROME_AUTO_START":    "cn_browser_auto_start",
    "FXXKSTOCK_CHROME_AUTO_CLOSE":    "cn_browser_auto_close",
    "FXXKSTOCK_CHROME_EXECUTABLE":    "cn_browser_executable",
    "FXXKSTOCK_CHROME_PROFILE_DIR":   "cn_browser_profile_dir",
    "FXXKSTOCK_CHROME_STARTUP_TIMEOUT": "cn_browser_startup_timeout_seconds",
    "FXXKSTOCK_NEWS_ARTICLE_LIMIT":   "news_article_limit",
    "FXXKSTOCK_GLOBAL_NEWS_LIMIT":    "global_news_article_limit",
    "FXXKSTOCK_CN_GUBA_POST_LIMIT":   "cn_guba_post_limit",
    "FXXKSTOCK_FUNDAMENTALS_TTL_DAYS": "ticker_memory_fundamentals_ttl_days",
    "FXXKSTOCK_WEB_RESEARCH_DEPTH":   "web_research_depth",
    "FXXKSTOCK_WEB_ANALYSIS_MODE":    "web_analysis_mode",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "FXXKSTOCK_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "FXXKSTOCK_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "FXXKSTOCK_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply FXXKSTOCK_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("FXXKSTOCK_RESULTS_DIR", os.path.join(_FXXKSTOCK_HOME, "logs")),
    "data_cache_dir": os.getenv("FXXKSTOCK_CACHE_DIR", os.path.join(_FXXKSTOCK_HOME, "cache")),
    "memory_log_path": os.getenv("FXXKSTOCK_MEMORY_LOG_PATH", os.path.join("memory", "trading_memory.md")),
    "memory_log_legacy_path": os.path.join(
        _LEGACY_TRADINGAGENTS_HOME, "memory", "trading_memory.md"
    ),
    "ticker_memory_dir": os.getenv("FXXKSTOCK_TICKER_MEMORY_DIR", os.path.join("memory", "tickers")),
    "ticker_memory_fundamentals_ttl_days": 30,
    "reports_dir": os.getenv("FXXKSTOCK_REPORTS_DIR", "reports"),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "Chinese",
    "web_research_depth": "simple",
    "web_analysis_mode": "auto",
    # Report monetary display: convert non-CNY vendor amounts to CNY
    "report_currency": "CNY",
    "fx_convert_enabled": True,
    "fx_rate_override": None,  # e.g. {"USD": 7.23, "HKD": 0.92}
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # China-market data sources (no API key required)
    "cn_data_enabled": True,
    "cn_adr_tickers": [
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME", "VIPS",
        "WB", "YUMC", "ZTO", "BEKE", "FUTU", "TAL", "EDU", "NTES", "TCOM",
    ],
    "cn_news_article_limit": 20,
    "cn_guba_post_limit": 15,
    "cn_community_sources": ["eastmoney", "xueqiu", "nga"],
    "cn_nga_lookback_days": 30,
    "cn_nga_min_stock_threads": 3,
    "cn_nga_thread_limit": 3,
    "cn_nga_reply_limit": 20,
    "cn_ths_news_enabled": True,
    "cn_global_news_article_limit": 10,
    "cn_global_news_queries": [
        "央行 降息 降准 货币政策",
        "A股 财报 业绩 预告",
        "中美 贸易 制裁 地缘政治",
        "港股 恒指 南向资金",
        "原油 大宗商品 供应链",
    ],
    "cninfo_cache_ttl_hours": 24,
    "cn_http_inter_request_delay": 0.5,
    # Browser/CDP soft-info fetch (connect_over_cdp to local Chrome)
    "cn_browser_enabled": True,
    "cn_browser_auto_start": True,
    "cn_browser_auto_close": True,
    "cn_browser_platform": (
        "macos" if sys.platform == "darwin"
        else "windows" if os.name == "nt"
        else "ubuntu"
    ),
    "cn_browser_executable": os.getenv("FXXKSTOCK_CHROME_EXECUTABLE") or None,
    "cn_browser_profile_dir": os.getenv(
        "FXXKSTOCK_CHROME_PROFILE_DIR", os.path.join("browser_data", "chrome-profile")
    ),
    "cn_browser_startup_timeout_seconds": 15.0,
    "cn_browser_cdp_url": "http://127.0.0.1:9222",
    "cn_browser_wait_until": "load",  # networkidle 在东财/SPA 上易 20s 超时
    "cn_browser_nav_timeout_ms": 20000,
    "cn_browser_guba_source": "eastmoney",  # or "xueqiu" with East Money fallback
    # market_region is injected per-run by FxxKStockGraph._run_graph
    "market_region": "default",
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
