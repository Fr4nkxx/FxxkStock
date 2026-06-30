"""Optional per-run account position context for trading decisions."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PositionContext(BaseModel):
    """User-supplied position for the instrument currently being analysed."""

    status: Literal["unknown", "flat", "held"] = "unknown"
    quantity: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    average_cost: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def normalize_non_held_values(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if value.get("status", "unknown") == "unknown":
                value.update(quantity=None, average_cost=None)
            elif value.get("status") == "flat":
                value.update(quantity=0.0, average_cost=None)
        return value

    @model_validator(mode="after")
    def validate_held_values(self):
        if self.status == "held" and (
            not self.quantity or self.average_cost is None
        ):
            raise ValueError(
                "held position requires quantity and average_cost greater than zero"
            )
        return self


def build_position_context(
    position: PositionContext | dict[str, Any] | None,
    market_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a position and add deterministic mark-to-market values."""
    normalized = (
        position
        if isinstance(position, PositionContext)
        else PositionContext.model_validate(position or {})
    )
    context = normalized.model_dump()
    snapshot = market_snapshot or {}
    close = snapshot.get("close")
    try:
        current_price = float(close)
    except (TypeError, ValueError):
        current_price = None
    if current_price is not None and (
        not math.isfinite(current_price) or current_price <= 0
    ):
        current_price = None

    if normalized.status == "held" and current_price is not None:
        quantity = float(normalized.quantity)
        average_cost = float(normalized.average_cost)
        cost_basis = quantity * average_cost
        market_value = quantity * current_price
        unrealized_pnl = market_value - cost_basis
        context.update(
            {
                "current_price": current_price,
                "currency": snapshot.get("currency"),
                "cost_basis": cost_basis,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return_pct": (current_price / average_cost - 1) * 100,
            }
        )
    return context


def render_position_context(position: dict[str, Any] | None) -> str:
    """Render position facts for Trader and Portfolio Manager prompts."""
    context = position or {"status": "unknown"}
    status = context.get("status", "unknown")
    if status == "unknown":
        return (
            "Current account position: not provided. Do not infer whether the user "
            "is flat or already holds the instrument; give a general direction."
        )
    if status == "flat":
        return (
            "Current account position: FLAT (no position). Interpret Buy as opening "
            "a position, Hold as remaining on the sidelines, and Sell as avoiding entry."
        )

    lines = [
        "Current account position: HELD.",
        f"- Quantity: {context['quantity']}",
        f"- Average cost: {context['average_cost']}",
    ]
    if context.get("current_price") is not None:
        currency = f" {context['currency']}" if context.get("currency") else ""
        lines.extend(
            [
                f"- Authoritative current price: {context['current_price']}{currency}",
                f"- Current market value: {context['market_value']:.2f}{currency}",
                f"- Unrealized P/L: {context['unrealized_pnl']:.2f}{currency}",
                f"- Unrealized return: {context['unrealized_return_pct']:.2f}%",
            ]
        )
    else:
        lines.append("- Current price unavailable; do not estimate market value or P/L.")
    lines.append(
        "Interpret Buy as adding exposure, Hold as maintaining the position, and Sell "
        "as reducing or exiting. Treat cost as risk context only; do not anchor on "
        "breaking even or recommend waiting merely to recover losses."
    )
    return "\n".join(lines)
