"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Anti-bias audit
# ---------------------------------------------------------------------------


class InformationGrade(str, Enum):
    """How well public evidence supports AI-assisted research."""

    A = "A"
    B = "B"
    C = "C"


class ConfidenceLevel(str, Enum):
    """Deliberately coarse confidence scale to avoid false precision."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class EvidenceType(str, Enum):
    OBSERVED = "observed"
    CALCULATED = "calculated"
    INFERRED = "inferred"
    OPINION = "opinion"


class EvidenceDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    SINGLE_SOURCE = "single_source"
    UNSUPPORTED = "unsupported"
    CONFLICTED = "conflicted"


class EvidenceClaim(BaseModel):
    claim_id: str = Field(description="Stable identifier E01, E02, and so on.")
    claim: str
    type: EvidenceType
    direction: EvidenceDirection
    source_refs: list[str] = Field(default_factory=list)
    data_date: str | None = None
    independent_source_count: int = Field(default=0, ge=0)
    confidence: ConfidenceLevel
    counter_evidence: list[str] = Field(default_factory=list)
    status: EvidenceStatus

    @field_validator("type", mode="before")
    @classmethod
    def normalize_evidence_type(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "inference": EvidenceType.INFERRED.value,
                "calculation": EvidenceType.CALCULATED.value,
                "observation": EvidenceType.OBSERVED.value,
            }
            return aliases.get(normalized, normalized)
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        return value.strip().capitalize() if isinstance(value, str) else value


class EvidenceLedger(BaseModel):
    """A compact, source-aware ledger of the run's decisive evidence."""

    claims: list[EvidenceClaim] = Field(default_factory=list, max_length=20)
    omitted_or_missing_evidence: list[str] = Field(default_factory=list)


def normalize_evidence_ledger(ledger: EvidenceLedger) -> EvidenceLedger:
    """Apply deterministic IDs and prevent unsupported claims being overstated."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    claims: list[EvidenceClaim] = []
    for item in ledger.claims[:20]:
        key = (
            " ".join(item.claim.lower().split()),
            tuple(sorted(ref.lower().strip() for ref in item.source_refs)),
        )
        if key in seen:
            continue
        seen.add(key)
        data = item.model_dump()
        data["claim_id"] = f"E{len(claims) + 1:02d}"
        if item.status == EvidenceStatus.SUPPORTED:
            if not item.source_refs:
                data["status"] = EvidenceStatus.UNSUPPORTED
            elif item.independent_source_count < 2:
                data["status"] = EvidenceStatus.SINGLE_SOURCE
        claims.append(EvidenceClaim.model_validate(data))
    return EvidenceLedger(
        claims=claims,
        omitted_or_missing_evidence=ledger.omitted_or_missing_evidence,
    )


def render_evidence_ledger(ledger: EvidenceLedger) -> str:
    rows = [
        "| ID | Claim | Type | Direction | Status | Confidence | Sources | Date | Counter-evidence |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for item in ledger.claims:
        clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| {item.claim_id} | {clean(item.claim)} | {item.type.value} | "
            f"{item.direction.value} | {item.status.value} | {item.confidence.value} | "
            f"{item.independent_source_count} | {clean(item.data_date or '-')} | "
            f"{clean('; '.join(item.counter_evidence) or '-')} |"
        )
    missing = "\n".join(f"- {item}" for item in ledger.omitted_or_missing_evidence)
    return "\n".join(
        ["# 证据账本 / Evidence Ledger", "", *(rows or ["No claims extracted."]), "",
         "## Missing or Omitted Evidence", missing or "- None identified"]
    )


class ResearchabilityAssessment(BaseModel):
    """Pre-debate assessment of evidence breadth and research limitations."""

    information_grade: InformationGrade = Field(
        description="A = information rich, B = moderate/inference required, C = sparse.",
    )
    source_diversity: ConfidenceLevel = Field(
        description="Diversity and independence of the available evidence sources.",
    )
    consensus_risk: ConfidenceLevel = Field(
        description="Risk that abundant but homogeneous public information anchors the analysis.",
    )
    critical_missing_data: list[str] = Field(
        default_factory=list,
        description="Material missing facts that could change the investment conclusion.",
    )
    inferred_claims: list[str] = Field(
        default_factory=list,
        description="Important claims that are inferred rather than directly observed.",
    )
    research_limitations: list[str] = Field(
        default_factory=list,
        description="Concrete limitations of this AI research run.",
    )
    recommended_posture: str = Field(
        description="How conservatively downstream agents should use the evidence.",
    )

    @field_validator("information_grade", mode="before")
    @classmethod
    def normalize_information_grade(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("source_diversity", "consensus_risk", mode="before")
    @classmethod
    def normalize_confidence_levels(cls, value):
        return value.strip().capitalize() if isinstance(value, str) else value


def render_researchability(assessment: ResearchabilityAssessment) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "- None identified"

    return "\n".join([
        "# 可研究性评估 / Researchability Assessment",
        "",
        f"**Information Grade**: {assessment.information_grade.value}",
        "",
        f"**Source Diversity**: {assessment.source_diversity.value}",
        "",
        f"**Consensus Risk**: {assessment.consensus_risk.value}",
        "",
        "## Critical Missing Data",
        bullets(assessment.critical_missing_data),
        "",
        "## Inferred Claims",
        bullets(assessment.inferred_claims),
        "",
        "## Research Limitations",
        bullets(assessment.research_limitations),
        "",
        f"## Recommended Posture\n{assessment.recommended_posture}",
    ])


class FalsificationAudit(BaseModel):
    """Independent challenge to the Research Manager's initial plan."""

    strongest_counter_thesis: str = Field(
        description="The strongest coherent case against the initial recommendation.",
    )
    conflicting_or_ignored_evidence: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(
        default_factory=list,
        description="Relevant biases such as confirmation, anchoring, recency, or narrative bias.",
    )
    falsification_triggers: list[str] = Field(
        default_factory=list,
        description="Observable future facts that would invalidate the thesis.",
    )
    critical_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Only severe issues: material data conflict, unsupported decisive claim, "
            "ignored strong counterevidence, or asset/strategy mismatch."
        ),
    )
    requires_revision: bool = Field(
        description="True when critical findings require one Research Manager revision.",
    )
    revision_instructions: list[str] = Field(default_factory=list)


def render_falsification_audit(audit: FalsificationAudit) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "- None identified"

    return "\n".join([
        "# 证伪审计 / Falsification Audit",
        "",
        f"**Requires Revision**: {'Yes' if audit.requires_revision else 'No'}",
        "",
        "## Strongest Counter-Thesis",
        audit.strongest_counter_thesis,
        "",
        "## Conflicting or Ignored Evidence",
        bullets(audit.conflicting_or_ignored_evidence),
        "",
        "## Hidden Assumptions",
        bullets(audit.hidden_assumptions),
        "",
        "## Bias Flags",
        bullets(audit.bias_flags),
        "",
        "## Falsification Triggers",
        bullets(audit.falsification_triggers),
        "",
        "## Critical Findings",
        bullets(audit.critical_findings),
        "",
        "## Revision Instructions",
        bullets(audit.revision_instructions),
    ])


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    data_confidence: ConfidenceLevel = Field(
        description="Confidence in data completeness, freshness, and source independence.",
    )
    data_confidence_reason: str = Field(
        description="One sentence explaining the data confidence level.",
    )
    thesis_confidence: ConfidenceLevel = Field(
        description="Confidence that the investment thesis survives the falsification audit.",
    )
    thesis_confidence_reason: str = Field(
        description="One sentence explaining the thesis confidence level.",
    )
    execution_confidence: ConfidenceLevel = Field(
        description="Confidence in the action, price levels, sizing, and risk triggers.",
    )
    execution_confidence_reason: str = Field(
        description="One sentence explaining the execution confidence level.",
    )
    predictions: list["PricePrediction"] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three market-price predictions that can be verified automatically.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator(
        "data_confidence",
        "thesis_confidence",
        "execution_confidence",
        mode="before",
    )
    @classmethod
    def normalize_confidence_levels(cls, value):
        return value.strip().capitalize() if isinstance(value, str) else value


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    parts.extend([
        "",
        f"**Data Confidence**: {decision.data_confidence.value}",
        "",
        f"**Data Confidence Reason**: {decision.data_confidence_reason}",
        "",
        f"**Thesis Confidence**: {decision.thesis_confidence.value}",
        "",
        f"**Thesis Confidence Reason**: {decision.thesis_confidence_reason}",
        "",
        f"**Execution Confidence**: {decision.execution_confidence.value}",
        "",
        f"**Execution Confidence Reason**: {decision.execution_confidence_reason}",
    ])
    if decision.predictions:
        parts.extend(["", "## Machine-Verifiable Predictions"])
        for index, prediction in enumerate(decision.predictions, 1):
            parts.extend([
                "",
                f"### Prediction {index}",
                f"- Claim: {prediction.claim}",
                f"- Condition: {prediction.comparison.value} {prediction.target_price}",
                f"- Horizon: {prediction.horizon_trading_days} trading days",
                f"- Confidence: {prediction.confidence.value}",
                f"- Rationale: {prediction.rationale}",
            ])
    return "\n".join(parts)


class PredictionComparison(str, Enum):
    ABOVE = "Above"
    BELOW = "Below"


class PricePrediction(BaseModel):
    claim: str
    comparison: PredictionComparison
    target_price: float = Field(gt=0)
    horizon_trading_days: Literal[5, 20]
    confidence: ConfidenceLevel
    rationale: str

    @field_validator("comparison", mode="before")
    @classmethod
    def normalize_comparison(cls, value):
        return value.strip().capitalize() if isinstance(value, str) else value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        return value.strip().capitalize() if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, supporting evidence, and verifiable Link when the "
            "input provides a URL (news/CNINFO); for Guba posts without URLs, "
            "state source type and title only — never fabricate links. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )

    @field_validator("overall_band", mode="before")
    @classmethod
    def normalize_overall_band(cls, value):
        """Accept harmless casing differences from structured-output providers."""
        if isinstance(value, str):
            normalized = value.strip().casefold()
            for band in SentimentBand:
                if band.value.casefold() == normalized:
                    return band.value
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        """Normalize confidence casing while preserving validation of bad values."""
        if isinstance(value, str):
            return value.strip().lower()
        return value


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
