"""Tests for shared news date-window utilities."""

import time
from datetime import datetime

import pytest

from fxxkstock.dataflows.news_utils import in_news_window


def _epoch(date_str):
    return int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert in_news_window(inside, start, end) is True
    assert in_news_window(future, start, end) is False
    assert in_news_window(None, start, end) is False


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    start = datetime.now()
    end = datetime.now()
    assert in_news_window(None, start, end) is True
