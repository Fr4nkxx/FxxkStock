"""Tests for optional per-run account position context."""

import pytest
from pydantic import ValidationError

from fxxkstock.agents.utils.position import (
    PositionContext,
    build_position_context,
    render_position_context,
)
from fxxkstock.graph.propagation import Propagator


@pytest.mark.unit
def test_position_defaults_to_unknown_without_inference():
    context = build_position_context(None, {"close": 10.0})

    assert context == {
        "status": "unknown",
        "quantity": None,
        "average_cost": None,
    }
    assert "not provided" in render_position_context(context)


@pytest.mark.unit
def test_flat_position_discards_stale_values():
    context = build_position_context(
        {"status": "flat", "quantity": -50, "average_cost": 0},
        {"close": 10.0},
    )

    assert context["quantity"] == 0
    assert context["average_cost"] is None
    assert "market_value" not in context


@pytest.mark.unit
def test_held_position_computes_mark_to_market_values():
    context = build_position_context(
        {"status": "held", "quantity": 100, "average_cost": 8.0},
        {"close": 10.0, "currency": "CNY"},
    )

    assert context["cost_basis"] == 800
    assert context["market_value"] == 1000
    assert context["unrealized_pnl"] == 200
    assert context["unrealized_return_pct"] == pytest.approx(25)


@pytest.mark.unit
@pytest.mark.parametrize(
    "position",
    [
        {"status": "held"},
        {"status": "held", "quantity": 0, "average_cost": 8},
        {"status": "held", "quantity": 100, "average_cost": 0},
        {"status": "held", "quantity": -1, "average_cost": 8},
    ],
)
def test_invalid_held_position_is_rejected(position):
    with pytest.raises(ValidationError):
        PositionContext.model_validate(position)


@pytest.mark.unit
def test_initial_state_contains_position_but_not_analysis_reports():
    state = Propagator().create_initial_state(
        "159516.SZ",
        "2026-06-30",
        position={"status": "held", "quantity": 100, "average_cost": 1.5},
        current_market_snapshot={"close": 1.8, "currency": "CNY"},
    )

    assert state["position_context"]["unrealized_pnl"] == pytest.approx(30)
    for report in (
        "market_report",
        "fundamentals_report",
        "sentiment_report",
        "news_report",
    ):
        assert state[report] == ""
