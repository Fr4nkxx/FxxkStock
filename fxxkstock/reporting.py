"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``FxxKStockGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

from datetime import datetime
from pathlib import Path


def _format_position_number(
    value,
    digits: int = 2,
    *,
    trim_trailing_zeros: bool = False,
) -> str:
    number = float(value)
    if trim_trailing_zeros and number.is_integer():
        return f"{number:,.0f}"
    rendered = f"{number:,.{digits}f}"
    return rendered.rstrip("0").rstrip(".") if trim_trailing_zeros else rendered


def render_account_position_section(position: dict | None) -> str:
    """Render user-supplied position facts for the saved run report only."""
    context = position or {}
    status = context.get("status", "unknown")
    if status == "unknown":
        return ""
    if status == "flat":
        return (
            "## 本次账户持仓 / Account Position\n\n"
            "| 项目 | 数值 |\n|---|---:|\n"
            "| 持仓状态 | 空仓 |\n"
        )

    currency = context.get("currency") or ""
    currency_suffix = f" {currency}" if currency else ""
    rows = [
        ("持仓状态", "持有"),
        (
            "持股数量",
            _format_position_number(
                context["quantity"], 6, trim_trailing_zeros=True
            ),
        ),
        (
            "真实平均成本",
            f"{_format_position_number(context['average_cost'], 6, trim_trailing_zeros=True)}{currency_suffix}",
        ),
    ]
    if context.get("current_price") is not None:
        rows.extend(
            [
                (
                    "本次权威现价",
                    f"{_format_position_number(context['current_price'], 6, trim_trailing_zeros=True)}{currency_suffix}",
                ),
                (
                    "持仓成本",
                    f"{_format_position_number(context['cost_basis'])}{currency_suffix}",
                ),
                (
                    "当前市值",
                    f"{_format_position_number(context['market_value'])}{currency_suffix}",
                ),
                (
                    "浮动盈亏",
                    f"{_format_position_number(context['unrealized_pnl'])}{currency_suffix}",
                ),
                (
                    "浮动收益率",
                    f"{float(context['unrealized_return_pct']):.2f}%",
                ),
            ]
        )
    else:
        rows.append(("行情估值", "现价不可用，未计算"))
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return (
        "## 本次账户持仓 / Account Position\n\n"
        "> 以下数据由程序根据本次用户输入和权威行情快照计算；市场低点、"
        "技术位和历史报告价格均不是用户成本。\n\n"
        "| 项目 | 数值 |\n|---|---:|\n"
        f"{table}\n"
    )


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []
    position_section = render_account_position_section(
        final_state.get("position_context")
    )
    if position_section:
        sections.append(position_section)

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
