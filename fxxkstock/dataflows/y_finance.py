from datetime import datetime
from typing import Annotated

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .config import get_config
from .currency_utils import (
    FUNDAMENTAL_MONETARY_FIELDS,
    convert_financial_frame,
    convert_insider_frame,
    convert_to_cny,
    detect_financial_currency,
    format_fx_header_line,
    format_money_cny,
    get_fx_to_cny,
    get_instrument_fx_context,
)
from .market_utils import detect_market_region, get_security_cn_name, is_cn_region
from .stockstats_utils import (
    StockstatsUtils,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
    filter_financials_by_date,
    load_ohlcv,
    yf_retry,
)
from .symbol_utils import NoMarketDataError, normalize_symbol


def _format_financial_statement_output(
    *,
    title: str,
    canonical: str,
    freq: str,
    data: pd.DataFrame,
    ticker: str,
    info: dict | None,
    curr_date: str | None,
) -> str:
    """为财报三表附加币种头，非 CNY 时换算金额行（跳过股数行）。"""
    as_of = curr_date or datetime.now().strftime("%Y-%m-%d")
    region = get_config().get("market_region", "default")
    source_ccy = detect_financial_currency(info, ticker, region)
    fx_rate = get_fx_to_cny(source_ccy, as_of) if source_ccy != "CNY" else 1.0

    display = data
    if source_ccy != "CNY" and fx_rate is not None:
        display = convert_financial_frame(data, fx_rate)

    header = f"# {title} data for {canonical} ({freq})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += format_fx_header_line(
        source_ccy,
        fx_rate if source_ccy != "CNY" else 1.0,
        as_of,
    ) + "\n\n"
    return header + display.to_csv()


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Resolve broker/forex symbols to Yahoo's convention (XAUUSD+ -> GC=F).
    canonical = normalize_symbol(symbol)
    ticker = yf.Ticker(canonical)

    # yfinance treats ``end`` as EXCLUSIVE, so it would drop the requested
    # end_date row (and the current day when end_date is today). Request one day
    # past end_date so the requested range is actually inclusive (#986/#987).
    end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_inclusive))

    # Empty result means the symbol is unknown/delisted. Raise a typed error
    # instead of returning prose: the routing layer turns it into a single
    # unambiguous "no data" signal so the agent never fabricates a price.
    if data.empty:
        raise NoMarketDataError(
            symbol, canonical, f"no rows between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output.
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    raw_dates = pd.to_datetime(data.index, errors="coerce")
    requested_date = pd.Timestamp(end_date)
    raw_requested_row = bool((raw_dates.normalize() == requested_date).any())

    # Yahoo occasionally emits a placeholder containing Date and Volume but
    # no OHLC. Apply the same completeness rule used by the indicator path.
    data = _clean_dataframe(data.reset_index())
    data = data[
        (data["Date"] >= pd.Timestamp(start_date))
        & (data["Date"] <= requested_date)
    ]
    if data.empty:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"no complete OHLCV rows between {start_date} and {end_date}",
        )

    # Reject a stale frame (e.g. a year-old partial response) before it is
    # formatted into the report. Raises NoMarketDataError, which the router
    # turns into one clear unavailable signal (#1021).
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv(index=False)

    # Add header information; note the resolved symbol when it differs so the
    # agent (and user) can see which instrument was actually priced.
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    source_ccy, fx_rate = get_instrument_fx_context(symbol, end_date)
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += format_fx_header_line(source_ccy, fx_rate, end_date, converted=False) + "\n"
    latest_valid_date = pd.to_datetime(data["Date"], errors="coerce").max().strftime(
        "%Y-%m-%d"
    )
    if latest_valid_date != end_date:
        reason = (
            "the vendor returned an incomplete row for the requested date"
            if raw_requested_row
            else "the vendor returned no row for the requested date"
        )
        header += (
            f"# Data quality warning: {reason}; latest complete OHLCV row is "
            f"{latest_valid_date}. Do not infer a holiday from this warning alone.\n"
        )
    header += "\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)

        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')

            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = (
                    "N/A: No valid vendor row for this date "
                    "(non-trading day or delayed/incomplete market data)"
                )

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except NoMarketDataError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator

    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except NoMarketDataError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(canonical)
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            raise NoMarketDataError(ticker, canonical, "no fundamentals returned")

        as_of = curr_date or datetime.now().strftime("%Y-%m-%d")
        source_ccy, fx_rate = get_instrument_fx_context(ticker, as_of)
        region = detect_market_region(ticker)
        cn_name = get_security_cn_name(ticker, region) if is_cn_region(region) else None

        fields = [
            ("Name", "longName", False),
            ("Sector", "sector", False),
            ("Industry", "industry", False),
            ("Market Cap", "marketCap", True),
            ("PE Ratio (TTM)", "trailingPE", False),
            ("Forward PE", "forwardPE", False),
            ("PEG Ratio", "pegRatio", False),
            ("Price to Book", "priceToBook", False),
            ("EPS (TTM)", "trailingEps", True),
            ("Forward EPS", "forwardEps", True),
            ("Dividend Yield", "dividendYield", False),
            ("Beta", "beta", False),
            ("52 Week High", "fiftyTwoWeekHigh", True),
            ("52 Week Low", "fiftyTwoWeekLow", True),
            ("50 Day Average", "fiftyDayAverage", True),
            ("200 Day Average", "twoHundredDayAverage", True),
            ("Revenue (TTM)", "totalRevenue", True),
            ("Gross Profit", "grossProfits", True),
            ("EBITDA", "ebitda", True),
            ("Net Income", "netIncomeToCommon", True),
            ("Profit Margin", "profitMargins", False),
            ("Operating Margin", "operatingMargins", False),
            ("Return on Equity", "returnOnEquity", False),
            ("Return on Assets", "returnOnAssets", False),
            ("Debt to Equity", "debtToEquity", False),
            ("Current Ratio", "currentRatio", False),
            ("Book Value", "bookValue", True),
            ("Free Cash Flow", "freeCashflow", True),
        ]

        lines = []
        for label, key, is_monetary in fields:
            if key == "longName" and cn_name:
                lines.append(f"{label}: {cn_name}")
                continue
            value = info.get(key)
            if value is None:
                continue
            if (
                is_monetary
                and key in FUNDAMENTAL_MONETARY_FIELDS
                and source_ccy != "CNY"
                and fx_rate is not None
            ):
                try:
                    lines.append(
                        f"{label}: {format_money_cny(convert_to_cny(value, fx_rate))}"
                    )
                except (TypeError, ValueError):
                    lines.append(f"{label}: {value}")
            else:
                lines.append(f"{label}: {value}")

        # yfinance returns a stub dict (e.g. {"trailingPegRatio": None}) for
        # unknown symbols, so `info` is truthy but every field is empty. Treat
        # "no usable fields" as no data rather than emitting a bare header the
        # agent might fabricate around.
        if not lines:
            raise NoMarketDataError(ticker, canonical, "no fundamental fields returned")

        header = f"# Company Fundamentals for {canonical}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += format_fx_header_line(source_ccy, fx_rate, as_of) + "\n\n"

        return header + "\n".join(lines)

    except NoMarketDataError:
        raise
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(canonical)

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no balance sheet data")

        info = yf_retry(lambda: ticker_obj.info) or {}
        return _format_financial_statement_output(
            title="Balance Sheet",
            canonical=canonical,
            freq=freq,
            data=data,
            ticker=ticker,
            info=info,
            curr_date=curr_date,
        )

    except NoMarketDataError:
        raise
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(canonical)

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no cash flow data")

        info = yf_retry(lambda: ticker_obj.info) or {}
        return _format_financial_statement_output(
            title="Cash Flow",
            canonical=canonical,
            freq=freq,
            data=data,
            ticker=ticker,
            info=info,
            curr_date=curr_date,
        )

    except NoMarketDataError:
        raise
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(canonical)

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no income statement data")

        info = yf_retry(lambda: ticker_obj.info) or {}
        return _format_financial_statement_output(
            title="Income Statement",
            canonical=canonical,
            freq=freq,
            data=data,
            ticker=ticker,
            info=info,
            curr_date=curr_date,
        )

    except NoMarketDataError:
        raise
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    canonical = normalize_symbol(ticker)
    region = get_config().get("market_region", "default")
    if is_cn_region(region):
        raise NoMarketDataError(
            ticker,
            canonical,
            "yfinance insider disabled for CN region; use domestic CNINFO/browser",
        )

    try:
        ticker_obj = yf.Ticker(canonical)
        data = yf_retry(lambda: ticker_obj.insider_transactions)

        # Empty is normal here (many valid symbols have no insider filings),
        # so report it plainly rather than treating the symbol as invalid.
        if data is None or data.empty:
            return f"No insider transactions reported for symbol '{canonical}'"

        as_of = datetime.now().strftime("%Y-%m-%d")
        source_ccy, fx_rate = get_instrument_fx_context(ticker, as_of)
        display = convert_insider_frame(data, fx_rate, source_ccy)
        csv_string = display.to_csv()

        header = f"# Insider Transactions data for {canonical}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += format_fx_header_line(source_ccy, fx_rate, as_of) + "\n\n"

        return header + csv_string

    except NoMarketDataError:
        raise
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"
