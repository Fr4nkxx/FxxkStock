"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``FxxKStockGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import json
import re
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
    rows = [("持仓状态", "持有")]
    if context.get("quantity") is not None:
        rows.append((
            "持股数量",
            _format_position_number(context["quantity"], 6, trim_trailing_zeros=True),
        ))
    rows.append(
        (
            "真实平均成本",
            f"{_format_position_number(context['average_cost'], 6, trim_trailing_zeros=True)}{currency_suffix}",
        )
    )
    if context.get("current_price") is not None:
        rows.append((
            "本次权威现价",
            f"{_format_position_number(context['current_price'], 6, trim_trailing_zeros=True)}{currency_suffix}",
        ))
        if context.get("market_value") is not None:
            rows.extend([
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
            ])
        rows.append(("浮动收益率", f"{float(context['unrealized_return_pct']):.2f}%"))
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
        if final_state.get("blind_bull_argument"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "blind_bull.md").write_text(
                final_state["blind_bull_argument"], encoding="utf-8"
            )
            research_parts.append(("Blind Bull", final_state["blind_bull_argument"]))
        if final_state.get("blind_bear_argument"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "blind_bear.md").write_text(
                final_state["blind_bear_argument"], encoding="utf-8"
            )
            research_parts.append(("Blind Bear", final_state["blind_bear_argument"]))
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

    # 6. Anti-bias audit
    researchability = final_state.get("researchability_assessment") or {}
    falsification = final_state.get("falsification_audit") or {}
    evidence_ledger = final_state.get("evidence_ledger") or {}
    if researchability or falsification or evidence_ledger:
        audit_dir = save_path / "6_audit"
        audit_dir.mkdir(exist_ok=True)
        audit_parts = []
        if evidence_ledger.get("markdown"):
            (audit_dir / "evidence_ledger.md").write_text(
                evidence_ledger["markdown"], encoding="utf-8"
            )
            (audit_dir / "evidence_ledger.json").write_text(
                json.dumps(
                    {k: v for k, v in evidence_ledger.items() if k != "markdown"},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            audit_parts.append(evidence_ledger["markdown"])
        if researchability.get("markdown"):
            (audit_dir / "researchability.md").write_text(
                researchability["markdown"], encoding="utf-8"
            )
            audit_parts.append(researchability["markdown"])
        if falsification.get("markdown"):
            (audit_dir / "falsification.md").write_text(
                falsification["markdown"], encoding="utf-8"
            )
            audit_parts.append(falsification["markdown"])
        if final_state.get("initial_investment_plan"):
            (audit_dir / "research_manager_initial.md").write_text(
                final_state["initial_investment_plan"], encoding="utf-8"
            )

        decision_markdown = final_state.get("final_trade_decision", "") or (
            (final_state.get("risk_debate_state") or {}).get("judge_decision", "")
        )

        def confidence(label: str) -> dict[str, str | None]:
            level = re.search(
                rf"\*\*{re.escape(label)} Confidence\*\*:\s*([^\n]+)",
                decision_markdown,
                re.IGNORECASE,
            )
            reason = re.search(
                rf"\*\*{re.escape(label)} Confidence Reason\*\*:\s*([^\n]+)",
                decision_markdown,
                re.IGNORECASE,
            )
            return {
                "level": level.group(1).strip() if level else None,
                "reason": reason.group(1).strip() if reason else None,
            }

        audit_payload = {
            "version": 2,
            "evidence_ledger": {
                key: value for key, value in evidence_ledger.items()
                if key != "markdown"
            },
            "researchability": {
                key: value for key, value in researchability.items()
                if key != "markdown"
            },
            "falsification": {
                key: value for key, value in falsification.items()
                if key != "markdown"
            },
            "confidence": {
                "data": confidence("Data"),
                "thesis": confidence("Thesis"),
                "execution": confidence("Execution"),
            },
        }
        (audit_dir / "audit.json").write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if audit_parts:
            sections.append(
                "## VI. Anti-Bias Audit\n\n" + "\n\n---\n\n".join(audit_parts)
            )

    decision_metadata = final_state.get("portfolio_decision_metadata") or {}
    if decision_metadata:
        calendar_nodes = list(decision_metadata.get("review_nodes") or [])
        if decision_metadata.get("execution_condition"):
            calendar_nodes.append({
                "node_type": "execution",
                "trigger_type": "condition",
                "condition": decision_metadata["execution_condition"],
                "action": decision_metadata.get("next_action") or "Review execution",
            })
        if decision_metadata.get("risk_boundary"):
            calendar_nodes.append({
                "node_type": "risk",
                "trigger_type": "condition",
                "condition": decision_metadata["risk_boundary"],
                "action": decision_metadata["risk_boundary"],
            })
        (save_path / "calendar_nodes.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "ticker": ticker,
                    "analysis_date": final_state.get("trade_date"),
                    "nodes": calendar_nodes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        calibration_dir = save_path / "7_calibration"
        calibration_dir.mkdir(exist_ok=True)
        predictions = decision_metadata.get("predictions") or []
        (calibration_dir / "prediction_snapshot.json").write_text(
            json.dumps(
                {
                    "rating": decision_metadata.get("rating"),
                    "data_confidence": decision_metadata.get("data_confidence"),
                    "thesis_confidence": decision_metadata.get("thesis_confidence"),
                    "execution_confidence": decision_metadata.get("execution_confidence"),
                    "predictions": predictions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if predictions:
            rows = [
                "| Claim | Condition | Horizon | Confidence |",
                "|---|---|---:|---|",
                *[
                    f"| {item['claim']} | {item['comparison']} {item['target_price']} | "
                    f"{item['horizon_trading_days']} | {item['confidence']} |"
                    for item in predictions
                ],
            ]
            sections.append("## VII. Prediction Snapshot\n\n" + "\n".join(rows))

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
