"""News analyst — macro and ticker news for trading decisions.

CN-market instruments pre-fetch domestic news (East Money / browser CDP),
global CN macro headlines, and CNINFO insider filings before the LLM runs,
mirroring the Sentiment Analyst prefetch pattern so A-share runs always
receive domestic sources instead of relying on optional tool calls.
"""

from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from fxxkstock.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_insider_transactions,
    get_prior_report_instruction,
    get_report_instructions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from fxxkstock.dataflows.config import get_config
from fxxkstock.dataflows.market_utils import is_cn_region
from fxxkstock.dataflows.ths_news import get_browser_ths_news

# 单次分析 run 内缓存 CN 预抓取结果；News Analyst 在 tool round 后会再次进入节点，
# 若不缓存会对同一 ticker 重复拉取 browser/HTTP（每次最多数十秒）。
_cn_prefetch_cache: dict[tuple[str, str, str], dict[str, str]] = {}


def _prefetch_cn_news_blocks(
    *,
    ticker: str,
    trade_date: str,
    region: str,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    key = (ticker, trade_date, region)
    cached = _cn_prefetch_cache.get(key)
    if cached is not None:
        return cached

    ticker_news = get_news.func(ticker, start_date, end_date)
    if get_config().get("cn_ths_news_enabled", True):
        try:
            ths_news = get_browser_ths_news(ticker, start_date, end_date)
            ticker_news = f"{ticker_news}\n\n{ths_news}"
        except Exception:
            # Tonghuashun is supplementary; East Money/browser remains primary.
            pass

    cached = {
        "news": ticker_news,
        "global": get_global_news.func(end_date),
        "insider": (
            get_insider_transactions.func(ticker) if region == "cn_a" else ""
        ),
    }
    _cn_prefetch_cache[key] = cached
    return cached


def _lookback_start(trade_date: str, look_back_days: int) -> str:
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    return (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")


def _build_cn_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    asset_label: str,
    news_block: str,
    global_block: str,
    insider_block: str,
) -> str:
    """Assemble the CN-market news-analyst system message with pre-fetched blocks."""
    insider_section = ""
    if insider_block:
        insider_section = f"""
### CNINFO shareholder / insider filings (A-share, recent)
Official regulatory disclosures — highest factual weight for corporate events.

<start_of_insider>
{insider_block}
<end_of_insider>
"""

    return f"""You are a news researcher covering a China-related instrument. Write a comprehensive news and macro report for {ticker} covering {start_date} to {end_date} that is relevant for trading decisions.

## Pre-fetched domestic data (already collected — do NOT claim these sources are missing)

### Ticker news — East Money / Browser (东方财富), {start_date} to {end_date}
{asset_label.capitalize()}-specific headlines and summaries.

<start_of_news>
{news_block}
<end_of_news>

### China macro / market news — East Money search, lookback through {end_date}
Broader domestic macro and market headlines (央行, A股, 财报, 地缘政治, etc.).

<start_of_global_news>
{global_block}
<end_of_global_news>
{insider_section}
## Additional tools (optional supplements)

You may still call:
- get_macro_indicators(indicator, curr_date, look_back_days) for US FRED series (cpi, fed_funds_rate, etc.) — **supplement only**, not a substitute for the China macro block above.
- get_prediction_markets(topic, limit) for market-implied event probabilities.

## Rules

1. **Ground the report in the pre-fetched blocks above.** Cite specific headlines, dates, and events when present.
2. **Do not fabricate news** not supported by the provided data or optional tool output.
3. If a block shows NO_DATA or a placeholder, say so explicitly — do not invent headlines.
4. Tie macro and ticker news to implications for {ticker}.
5. When pre-fetched blocks include `Link: https://...`, you **must** keep that URL in the report for each cited item.
6. Append a Markdown summary table at the end (include a Link column when URLs are available).

{get_report_instructions()}"""


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        config = get_config()
        region = config.get("market_region", "default")
        cn_market = is_cn_region(region)

        if cn_market:
            look_back = config.get("global_news_lookback_days", 7)
            start_date = _lookback_start(current_date, look_back)

            blocks = _prefetch_cn_news_blocks(
                ticker=ticker,
                trade_date=current_date,
                region=region,
                start_date=start_date,
                end_date=current_date,
            )

            system_message = _build_cn_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=current_date,
                asset_label=asset_label,
                news_block=blocks["news"],
                global_block=blocks["global"],
                insider_block=blocks["insider"],
            )
            tools = [get_macro_indicators, get_prediction_markets]
        else:
            tools = [
                get_news,
                get_global_news,
                get_macro_indicators,
                get_prediction_markets,
            ]

            cn_note = ""

            system_message = (
                f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + cn_note
                + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
                + get_report_instructions()
            )

        system_message += get_prior_report_instruction(state, "news_report")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
