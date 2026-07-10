from typing import Annotated

from langchain_core.tools import tool

from fxxkstock.dataflows.market_data_validator import build_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
) -> str:
    """Deterministic verification snapshot for exact market-data claims.

    Returns a current latest quote when available for the current analysis date,
    plus the latest complete OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Use the latest quote for current-price and
    action levels; use complete OHLCV rows for indicators and historical claims.
    """
    return build_verified_market_snapshot(symbol, curr_date, look_back_days)
