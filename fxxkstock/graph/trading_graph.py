"""Backward-compatible import path for the renamed FxxKStock graph."""

from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph

TradingAgentsGraph = FxxKStockGraph

__all__ = ["FxxKStockGraph", "TradingAgentsGraph"]
