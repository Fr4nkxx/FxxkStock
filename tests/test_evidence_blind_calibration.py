from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from fxxkstock.agents.managers.anti_bias import create_evidence_ledger_builder
from fxxkstock.agents.researchers.blind_researchers import (
    create_blind_bear_researcher,
    create_blind_bull_researcher,
)
from fxxkstock.agents.schemas import (
    EvidenceClaim,
    EvidenceLedger,
    PortfolioDecision,
    PricePrediction,
    normalize_evidence_ledger,
)
from fxxkstock.agents.utils.calibration import CalibrationStore


def _state():
    return {
        "company_of_interest": "TEST",
        "asset_type": "stock",
        "market_report": "Market EVIDENCE",
        "sentiment_report": "Sentiment EVIDENCE",
        "news_report": "News EVIDENCE",
        "fundamentals_report": "Fundamental EVIDENCE",
        "evidence_ledger": {"markdown": "E01 supported"},
        "researchability_assessment": {"markdown": "Grade B"},
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "judge_decision": "", "count": 0,
        },
    }


@pytest.mark.unit
def test_evidence_ledger_deduplicates_numbers_and_downgrades_single_source():
    claim = EvidenceClaim(
        claim_id="temp", claim="Revenue grew", type="observed",
        direction="bullish", source_refs=["filing"], independent_source_count=1,
        confidence="high", status="supported",
    )
    ledger = normalize_evidence_ledger(EvidenceLedger(claims=[claim, claim]))
    assert len(ledger.claims) == 1
    assert ledger.claims[0].claim_id == "E01"
    assert ledger.claims[0].status.value == "single_source"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("inference", "inferred"),
        ("calculation", "calculated"),
        ("observation", "observed"),
    ],
)
def test_evidence_type_normalizes_common_model_nouns(raw_type, expected):
    claim = EvidenceClaim(
        claim_id="E01",
        claim="Test claim",
        type=raw_type,
        direction="neutral",
        confidence="Medium",
        status="unsupported",
    )
    assert claim.type.value == expected


@pytest.mark.unit
def test_evidence_builder_fails_open_when_structured_output_unavailable():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    result = create_evidence_ledger_builder(llm)(_state())
    assert result["evidence_ledger"]["status"] == "unavailable"
    assert result["evidence_ledger"]["claims"] == []
    llm.invoke.assert_not_called()


@pytest.mark.unit
def test_blind_bear_prompt_cannot_see_blind_bull_output():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="argument")
    state = _state()
    state["blind_bull_argument"] = "SECRET BULL CONCLUSION"
    create_blind_bear_researcher(llm)(state)
    prompt = llm.invoke.call_args.args[0]
    assert "SECRET BULL CONCLUSION" not in prompt
    assert "E01 supported" in prompt


@pytest.mark.unit
def test_blind_arguments_do_not_increment_cross_examination_count():
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content="bull"), MagicMock(content="bear"),
    ]
    state = _state()
    state.update(create_blind_bull_researcher(llm)(state))
    state.update(create_blind_bear_researcher(llm)(state))
    assert state["investment_debate_state"]["count"] == 0
    assert "Blind Bull Analyst" in state["investment_debate_state"]["history"]
    assert "Blind Bear Analyst" in state["investment_debate_state"]["history"]


@pytest.mark.unit
def test_prediction_schema_limits_horizon_and_count():
    with pytest.raises(ValidationError):
        PricePrediction(
            claim="x", comparison="Above", target_price=1,
            horizon_trading_days=10, confidence="Low", rationale="x",
        )
    base = {
        "rating": "Hold", "executive_summary": "x", "investment_thesis": "x",
        "data_confidence": "Low", "data_confidence_reason": "x",
        "thesis_confidence": "Low", "thesis_confidence_reason": "x",
        "execution_confidence": "Low", "execution_confidence_reason": "x",
        "predictions": [
            {"claim": "x", "comparison": "Above", "target_price": 1,
             "horizon_trading_days": 5, "confidence": "Low", "rationale": "x"}
        ] * 4,
    }
    with pytest.raises(ValidationError):
        PortfolioDecision(**base)


@pytest.mark.unit
def test_calibration_resolution_is_idempotent_and_hold_has_no_hit(tmp_path):
    store = CalibrationStore({"calibration_memory_dir": str(tmp_path)})
    decision = {
        "rating": "Hold", "data_confidence": "High",
        "thesis_confidence": "Medium", "execution_confidence": "Low",
        "predictions": [{
            "claim": "Close above 105", "comparison": "Above",
            "target_price": 105, "horizon_trading_days": 5,
            "confidence": "High", "rationale": "Momentum",
        }],
    }
    store.record("TEST", "2026-01-01", decision, 100, "SPY")
    store.record("TEST", "2026-01-01", decision, 100, "SPY")
    assert len(store.load("TEST")["records"]) == 1

    def fetcher(ticker, date, horizon, benchmark):
        return {
            "actual_price": 110, "raw_return": .1,
            "benchmark_return": .02, "alpha_return": .08,
            "actual_holding_days": horizon,
        }

    store.resolve_pending("TEST", fetcher)
    store.resolve_pending("TEST", fetcher)
    result = store.query("TEST")
    rating = next(item for item in result["resolved"] if "rating" in item)
    prediction = next(item for item in result["resolved"] if "claim" in item)
    assert rating["hit"] is None
    assert prediction["hit"] is True
    assert result["summary"]["prediction_confidence"]["High"]["hit_rate"] == 1
