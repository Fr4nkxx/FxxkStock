"""Programmatic entry point — run with ``python main.py`` from the repo root.

Uses ``FxxKStockGraph.propagate()`` so ``market_region`` is injected and
CN tickers (e.g. ``603678.SS``) route to domestic news/browser vendors.

Environment variables and ``.env`` are applied via ``DEFAULT_CONFIG``; override
only what you need below or pass CLI flags.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

from fxxkstock.dataflows.utils import safe_ticker_component
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph


def _configure_llm_from_env(config: dict) -> dict:
    """当 .env 仅有 DEEPSEEK_API_KEY 时自动选用 DeepSeek，避免默认走 OpenAI。"""
    # FXXKSTOCK_LLM_PROVIDER 等已在 DEFAULT_CONFIG 构建时通过 env 覆盖
    if config.get("llm_provider") != "openai" or os.environ.get("OPENAI_API_KEY"):
        return config
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return config

    config["llm_provider"] = "deepseek"
    if not config.get("backend_url"):
        config["backend_url"] = "https://api.deepseek.com"
    if not os.environ.get("FXXKSTOCK_QUICK_THINK_LLM"):
        config["quick_think_llm"] = "deepseek-v4-flash"
    if not os.environ.get("FXXKSTOCK_DEEP_THINK_LLM"):
        config["deep_think_llm"] = "deepseek-v4-pro"
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FxxKStock analysis")
    parser.add_argument(
        "ticker",
        nargs="?",
        default="000657.SZ",
        help="Ticker symbol (default: 603678.SS)",
    )
    parser.add_argument(
        "trade_date",
        nargs="?",
        default=date.today().isoformat(),
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--analysts",
        default="market,social,news,fundamentals",
        help="Comma-separated analysts: market,social,news,fundamentals",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Report output language (default: Chinese). "
        "Override with FXXKSTOCK_OUTPUT_LANGUAGE in .env",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Report output directory (default: ./reports)",
    )
    parser.add_argument(
        "--save",
        default=True,
        action="store_true",
        help="Write report tree under --reports-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]

    config = DEFAULT_CONFIG.copy()
    config = _configure_llm_from_env(config)
    # main.py 默认中文；.env 中 FXXKSTOCK_OUTPUT_LANGUAGE 优先
    if not os.environ.get("FXXKSTOCK_OUTPUT_LANGUAGE"):
        config["output_language"] = args.language

    ta = FxxKStockGraph(
        selected_analysts=analysts,
        debug=True,
        config=config,
    )

    final_state, decision = ta.propagate(args.ticker, args.trade_date)
    print(decision)

    if args.save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            args.reports_dir
            / f"{safe_ticker_component(args.ticker)}_{stamp}"
        )
        out = ta.save_reports(final_state, args.ticker, save_path=save_path)
        print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()
