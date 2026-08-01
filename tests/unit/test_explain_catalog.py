import inspect
import json

import pytest

from buildml import Session
from buildml.explain import (
    CONCEPT_NOTES,
    OPERATION_CATALOG,
    Action,
    ActionPriority,
    BeforeAfterExplanation,
    DecisionOrigin,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
    WorkflowStep,
    WorkflowStepStatus,
)


def _public_session_operations() -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(Session, predicate=callable)
        if not name.startswith("_")
    }


def test_catalog_covers_every_public_session_operation() -> None:
    assert set(OPERATION_CATALOG) == _public_session_operations()


def test_catalog_entries_are_substantive_and_link_known_concepts() -> None:
    narrative_fields = (
        "mechanism",
        "inputs",
        "outputs",
        "usual_ordering",
        "alternatives",
        "selection_rationale",
        "assumptions",
        "failure_modes",
        "leakage_risks",
        "anti_patterns",
        "state_changes",
        "result_reading",
        "next_considerations",
        "concept_links",
    )
    for name, operation in OPERATION_CATALOG.items():
        assert operation.name == name
        assert len(operation.definition) >= 30
        assert len(operation.purpose) >= 30
        assert len(operation.pipeline_role) >= 10
        for field_name in narrative_fields:
            assert getattr(operation, field_name), f"{name}.{field_name} is empty"
        assert set(operation.concept_links) <= set(CONCEPT_NOTES)
        json.dumps(operation.to_dict())


def test_explanation_schemas_serialize_nested_enums_and_mappings() -> None:
    evidence = Evidence(
        key="missing-age",
        kind=EvidenceKind.METRIC,
        summary="Training age values contain missing entries.",
        value={"missing_count": 3, "missing_rate": 0.125},
        source="train.quality",
        limitations=("Does not establish why values are missing.",),
    )
    finding = Finding(
        key="age-missingness",
        title="Age needs a missing-value policy",
        detail="Three of 24 training values are missing.",
        severity=FindingSeverity.MEDIUM,
        evidence=(evidence,),
        affected_columns=("age",),
        confidence=0.9,
    )
    recommendation = Recommendation(
        key="try-age-imputation",
        title="Compare a median baseline",
        rationale="Median replacement is resistant to the observed skew.",
        priority=ActionPriority.BEFORE_MODELING,
        action=Action(
            key="impute-age",
            label="Impute age",
            operation="impute",
            parameters={"columns": ["age"], "strategy": "median"},
        ),
        based_on=(finding.key,),
    )
    step = WorkflowStep(
        operation="impute",
        status=WorkflowStepStatus.READY,
        origin=DecisionOrigin.RECOMMENDED,
        summary="A train-only imputation plan can be fitted.",
        evidence=(evidence,),
    )
    transition = BeforeAfterExplanation(
        operation="impute",
        before={"missing_age": 3, "columns": ["age", "target"]},
        after={"missing_age": 0, "columns": ["age", "target"]},
        changes=("Missing age values were replaced.",),
        unchanged=("Rows and split membership were preserved.",),
        origin=DecisionOrigin.EXPLICIT,
    )

    payload = {
        "finding": finding.to_dict(),
        "recommendation": recommendation.to_dict(),
        "step": step.to_dict(),
        "transition": transition.to_dict(),
    }
    serialized = json.dumps(payload)
    assert '"recommended"' in serialized
    assert payload["finding"]["evidence"][0]["kind"] == "metric"  # type: ignore[index]
    assert payload["recommendation"]["action"]["parameters"]["columns"] == ["age"]  # type: ignore[index]


def test_finding_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        Finding(
            key="bad-confidence",
            title="Invalid confidence",
            detail="Confidence values are probabilities.",
            confidence=1.1,
        )

