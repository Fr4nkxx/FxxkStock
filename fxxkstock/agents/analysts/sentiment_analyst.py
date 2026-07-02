"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  US/default:
    1. News headlines     — Yahoo Finance (institutional framing)
    2. StockTwits messages — retail-trader posts indexed by cashtag
    3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

  CN market (auto-detected):
    1. News headlines     — East Money (via vendor-routed get_news)
    2. CN retail forums   — East Money / Xueqiu / NGA Great Times posts
    3. CNINFO announcements — official A-share disclosures (cn_a only)

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from fxxkstock.agents.schemas import SentimentReport, render_sentiment_report
from fxxkstock.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_prior_report_instruction,
    get_report_instructions,
    get_news,
)
from fxxkstock.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from fxxkstock.dataflows.cninfo import fetch_cninfo_announcements
from fxxkstock.dataflows.config import get_config
from fxxkstock.dataflows.cn_community import fetch_cn_community
from fxxkstock.dataflows.market_utils import is_cn_region
from fxxkstock.dataflows.reddit import fetch_reddit_posts
from fxxkstock.dataflows.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + community/social data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)
        region = get_config().get("market_region", "default")
        cn_market = is_cn_region(region)

        # Pre-fetch sources. Each fetcher degrades gracefully and returns a
        # string (no exceptions surface from here), so the LLM always sees
        # something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)

        if cn_market:
            community_block = fetch_cn_community(ticker, as_of_date=end_date)
            official_block = (
                fetch_cninfo_announcements(ticker, start_date, end_date)
                if region == "cn_a"
                else ""
            )
            system_message = _build_cn_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                community_block=community_block,
                official_block=official_block,
            )
        else:
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)
            system_message = _build_us_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                stocktwits_block=stocktwits_block,
                reddit_block=reddit_block,
            )

        system_message += get_prior_report_instruction(state, "sentiment_report")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}"
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_us_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the US/default sentiment-analyst system message."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count.

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.**
2. **Look for cross-source divergences** between news, StockTwits, and Reddit.
3. **Weight Reddit posts by engagement.**
4. **Distinguish opinion from event.**
5. **Identify recurring narrative themes.**
6. **Be honest about data limits** — flag low sample size in confidence.
7. **Identify catalysts and risks.**
8. **Past sentiment is not predictive.**
9. **Preserve `Link:` URLs** from news/CNINFO blocks when citing headlines or filings.

## Output fields

- **overall_band**: Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish
- **overall_score**: 0 (bearish) to 10 (bullish); 5 is neutral
- **confidence**: low / medium / high
- **narrative**: Full source-by-source breakdown with markdown summary table

{get_report_instructions()}"""


def _build_cn_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    community_block: str,
    official_block: str,
) -> str:
    """Assemble the CN-market sentiment-analyst system message."""
    official_section = ""
    if official_block:
        official_section = f"""
### CNINFO official announcements — regulatory disclosures (past 7 days)
Legally mandated A-share disclosures. Highest factual weight — similar to SEC 8-K filings.

<start_of_cninfo>
{official_block}
<end_of_cninfo>
"""

    return f"""You are a financial market sentiment analyst covering a China-related instrument. Produce a comprehensive sentiment report for {ticker} covering {start_date} to {end_date}, using the pre-fetched sources below.

## Data sources (pre-fetched, in this prompt)

### News headlines — East Money (东方财富), past 7 days
Institutional/media framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### 散户社区 / Retail community posts — 东财股吧 + 雪球 + NGA 大时代
Primary retail-sentiment sources. Each forum is fetched independently; some may be missing if blocked.
NGA data includes actual opening posts and replies, not title-only search results.

<start_of_guba>
{community_block}
<end_of_guba>
{official_section}
## How to analyze this data (best practices)

1. **Weight CNINFO announcements highest** — they are legally mandated official events.
2. **Retail community post volume and engagement** (东财/雪球/NGA 大时代) signal retail attention. For NGA, analyze reply content and recurring opinions; distinguish the opening post from replies.
3. **Turnover rate (换手率)** when mentioned in news/data ≈ retail participation and sentiment intensity; high turnover + hot community discussion = active retail sentiment.
4. **Dragon-tiger board (龙虎榜) and capital flow (资金流向 / 主力净流入出)** ≈ institutional and hot-money behavior — do **not** treat these as retail sentiment; use them as "smart money" signals to contrast with retail mood and spot divergences.
5. **Look for divergences** between official announcements, news framing, retail forums, and institutional/capital-flow signals.
6. **Distinguish opinion from event** — a forum post is opinion; a CNINFO filing is an event.
7. **Be honest about sparse data** — ADR tickers often lack CN forum coverage; flag confidence accordingly.
8. **Past sentiment is not predictive.**
9. **Preserve `Link:` URLs** from news/CNINFO blocks when citing headlines or filings.
10. **For ETFs, keep sentiment scopes separate** — ETF-name discussions describe
trading attention/premium/liquidity, while tracking-index and alias discussions
describe underlying-market direction. Never present index/theme post volume as
direct ETF discussion, and state which scope supports each conclusion.
11. **CNINFO coverage is not existence proof** — for an ETF, an empty or
unindexed CNINFO result means only that this source returned no data. Never infer
that the ETF has no official announcements or that the instrument does not exist.

## Output fields

- **overall_band**: Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish
- **overall_score**: 0 (bearish) to 10 (bullish); 5 is neutral
- **confidence**: low / medium / high
- **narrative**: Full source-by-source breakdown with markdown summary table

{get_report_instructions()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`."""
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
