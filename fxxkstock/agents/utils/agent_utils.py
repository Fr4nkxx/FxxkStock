import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from fxxkstock.agents.utils.core_stock_tools import get_stock_data
from fxxkstock.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from fxxkstock.agents.utils.macro_data_tools import get_macro_indicators
from fxxkstock.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from fxxkstock.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from fxxkstock.agents.utils.prediction_markets_tools import get_prediction_markets
from fxxkstock.agents.utils.technical_indicators_tools import get_indicators

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "get_currency_instruction",
    "get_source_citation_instruction",
    "get_report_instructions",
    "create_msg_delete",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from fxxkstock.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_source_citation_instruction() -> str:
    """Return mandatory source-citation rules for verifiable reports."""
    return (
        " Source citation rules (mandatory): When citing news, announcements, or"
        " posts from tool/prefetch output that includes a `Link:` line or URL,"
        " include the verifiable link in your report as Markdown `[title](url)`"
        " or `Source: url`, and add a Link column in summary tables."
        " For numeric market/fundamental data without a web article, cite the tool"
        " name, date, and metric (e.g. get_stock_data, 2026-06-26, RSI=72) —"
        " do not fabricate URLs."
        " If a source has no URL, state the source type and title; never invent links."
        " If data is NO_DATA or unavailable, say so explicitly; do not invent events."
        " When quoting upstream analyst reports, preserve any links they included."
    )


def get_currency_instruction() -> str:
    """Return mandatory currency presentation rules for report-producing agents."""
    from fxxkstock.dataflows.config import get_config

    if get_config().get("report_currency", "CNY").upper() != "CNY":
        return ""
    return (
        " Currency rules (mandatory): Present all monetary amounts in CNY (¥)."
        " Use the exchange rate from instrument_context or tool output FX headers;"
        " do not invent FX rates."
        " Never label USD or HKD numeric values as 元 or 人民币."
    )


def get_report_instructions() -> str:
    """Language + currency + source-citation instructions for report-producing agents."""
    return (
        get_language_instruction()
        + get_currency_instruction()
        + get_source_citation_instruction()
    )


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from fxxkstock.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
        ("country", "country"),
        ("currency", "currency"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )
        name = identity.get("company_name") or identity.get("name") if identity else None
        if name and not is_crypto:
            context += (
                " Always use this exact company name verbatim in every report;"
                " never translate, romanize, or rewrite it."
            )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )

    if identity:
        source_ccy = (identity.get("currency") or "").upper()
        fx_rate = identity.get("fx_rate")
        fx_as_of = identity.get("fx_as_of")
        if source_ccy and source_ccy != "CNY":
            if fx_rate and fx_as_of:
                context += (
                    f" Source data quote currency: {source_ccy}."
                    f" Report ALL monetary values in CNY using rate"
                    f" 1 {source_ccy} = {fx_rate} CNY (as of {fx_as_of});"
                    f" never label {source_ccy} amounts as 元/人民币."
                )
            else:
                context += (
                    f" Source data quote currency: {source_ccy}."
                    f" FX to CNY unavailable — do not treat {source_ccy} amounts as CNY/元."
                )

    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``FxxKStockGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        resolved = context
    else:
        resolved = build_instrument_context(
            str(state["company_of_interest"]),
            state.get("asset_type", "stock"),
        )
    prior = state.get("prior_analysis_context")
    if isinstance(prior, str) and prior.strip():
        resolved += f"\n\n{prior}"
    snapshot = state.get("current_market_snapshot")
    if isinstance(snapshot, Mapping) and snapshot:
        from fxxkstock.dataflows.market_data_validator import (
            render_current_market_context,
        )

        resolved += f"\n\n{render_current_market_context(dict(snapshot))}"
        prior_snapshot = state.get("prior_market_snapshot")
        if isinstance(prior_snapshot, Mapping):
            previous_close = prior_snapshot.get("close")
            current_close = snapshot.get("close")
            if previous_close not in (None, 0) and current_close is not None:
                change = float(current_close) - float(previous_close)
                change_percent = change / float(previous_close) * 100
                resolved += (
                    "\n\nDETERMINISTIC CHANGE SINCE PREVIOUS SNAPSHOT\n"
                    f"- Previous verified close: {previous_close} "
                    f"{prior_snapshot.get('currency', '')} on "
                    f"{prior_snapshot.get('latest_trading_date', 'unknown')}\n"
                    f"- Current verified close: {current_close} "
                    f"{snapshot.get('currency', '')} on "
                    f"{snapshot.get('latest_trading_date', 'unknown')}\n"
                    f"- Verified change: {change:+.6g} ({change_percent:+.2f}%)\n"
                    "Any previous trigger, stop-loss conclusion, or position recommendation "
                    "that depended on the old price must be explicitly re-evaluated."
                )
    return resolved


def get_prior_report_from_state(state: Mapping[str, Any], report_key: str) -> str:
    """Return an analyst's previous same-ticker report, if available."""
    reports = state.get("prior_reports")
    if not isinstance(reports, Mapping):
        return ""
    report = reports.get(report_key)
    return report.strip() if isinstance(report, str) else ""


def get_prior_report_instruction(state: Mapping[str, Any], report_key: str) -> str:
    """Build a change-focused instruction for refreshed analyst reports."""
    report = get_prior_report_from_state(state, report_key)
    if not report:
        return ""
    return (
        "\n\nPrevious same-ticker report follows. Use it only as historical context. "
        "Refresh current facts with tools, identify what materially changed, and do not "
        "repeat unchanged detail unless it remains decision-relevant.\n"
        f"<previous_report>\n{report}\n</previous_report>"
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages
