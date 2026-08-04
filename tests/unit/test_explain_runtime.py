import json
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.explain import (
    HISTORY_SCHEMA_VERSION,
    AfterOperationExplanation,
    BeforeOperationExplanation,
    DecisionOrigin,
    WorkflowStepStatus,
    explain,
    normalize_history,
)
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.session.walkthrough import _unusual_order


def _session() -> Session:
    return Session.ingest(
        pd.DataFrame(
            {
                "age": [20.0, 30.0, None, 50.0, 60.0, 70.0, 80.0, 90.0],
                "city": ["a", "b", "a", "b", "a", "b", "a", "b"],
                "target": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
    )


def test_workflow_resolves_all_operations_with_exact_blocker_chains() -> None:
    session = Session()
    workflow = {step.operation: step for step in session.workflow()}

    assert set(workflow) == set(OPERATION_CATALOG)
    assert workflow["ingest"].status == WorkflowStepStatus.AVAILABLE
    assert workflow["fit"].status == WorkflowStepStatus.BLOCKED
    dataset_blocker = (
        "No materialized dataset is attached. "
        "Run ingest or checkpoint_load or reattach first."
    )
    assert dataset_blocker in workflow["fit"].blockers
    assert (
        "fit requires split via split or inject_split or group_split or time_split"
        in workflow["fit"].prerequisite_chain
    )


def test_workflow_reports_done_available_skipped_and_repeatable() -> None:
    session = _session().set_roles(
        {"age": "feature", "city": "feature", "target": "target"}
    ).split(test_size=0.25, random_state=4)
    workflow = {step.operation: step for step in session.workflow()}

    assert workflow["ingest"].status == WorkflowStepStatus.DONE
    assert workflow["split"].status == WorkflowStepStatus.DONE
    assert workflow["inject_split"].status == WorkflowStepStatus.SKIPPED
    assert workflow["fit"].status == WorkflowStepStatus.AVAILABLE
    assert workflow["eda"].repeatable is True


def test_split_resolver_matches_conditional_runtime_role_requirement() -> None:
    session = _session()
    before = session.explain("split", moment="before")

    assert before.status == WorkflowStepStatus.AVAILABLE
    assert before.prerequisite_status == {"dataset": True}
    assert any("random split needs no roles" in item.lower() for item in before.appropriateness)

    session.split(test_size=0.25, random_state=3)
    assert session.split_plan is not None
    assert session.split_plan.kind == "random"

    stratified = _session()
    with pytest.raises(ValidationError, match="target"):
        stratified.split(test_size=0.25, stratify=True, random_state=3)
    stratified.set_roles({"target": "target"}).split(
        test_size=0.25,
        stratify=True,
        random_state=3,
    )
    assert stratified.split_plan is not None
    assert stratified.split_plan.kind == "stratified"


def test_before_and_after_explanations_use_state_and_observed_record() -> None:
    session = _session()
    before = session.explain("fit", moment="before")
    after = session.explain("ingest", moment="after")

    assert isinstance(before, BeforeOperationExplanation)
    assert before.status == WorkflowStepStatus.BLOCKED
    assert before.prerequisite_status == {"dataset": True, "roles": False, "split": False}
    assert before.risks
    assert before.likely_state_changes

    assert isinstance(after, AfterOperationExplanation)
    assert after.sequence == 1
    assert after.parameters["source_type"] == "dataframe"
    assert after.decision_origin == DecisionOrigin.AUTOMATIC
    assert after.result_summary["format"] == "pandas.DataFrame"
    assert after.state_changes
    assert after.interpretation
    assert after.limitations
    assert after.next_valid_choices


def test_explain_accepts_facade_operation_names() -> None:
    session = _session()
    before_flat = session.explain("fit", moment="before")
    before_facade = session.explain("classical.fit", moment="before")
    before_prefixed = session.explain("session.classical.fit", moment="before")
    assert before_facade.operation == before_prefixed.operation == "fit"
    assert before_facade.status == before_flat.status
    after = session.explain("session.data.ingest", moment="after")
    assert after.operation == "ingest"
    assert after.sequence == 1


def test_after_explanation_for_unrun_repeatable_operation_is_explicit() -> None:
    explanation = explain(_session(), "eda", moment="after")
    assert isinstance(explanation, AfterOperationExplanation)
    assert explanation.sequence is None
    assert explanation.interpretation == ("There is no result to interpret.",)


def test_history_v2_is_sequenced_serializable_and_backward_compatible() -> None:
    session = _session().set_roles(
        {"age": "feature", "city": "feature", "target": "target"}
    )
    assert [record["sequence"] for record in session.history] == [1, 2]
    for record in session.history:
        assert record["schema_version"] == HISTORY_SCHEMA_VERSION
        assert record["timestamp"]
        assert record["operation_id"] == record["action"]
        assert record["parameters"] == record["details"]
        assert record["decision_origin"] in {"automatic", "recommended", "explicit"}
        assert set(record["state_transition"]) == {"before", "after", "changes"}
        assert isinstance(record["warnings"], list)
        assert isinstance(record["result_summary"], dict)
    json.dumps(session.history)

    old = normalize_history([{"action": "ingest", "details": {"format": "csv"}}])
    assert old[0]["schema_version"] == 2
    assert old[0]["operation_id"] == "ingest"
    assert old[0]["parameters"] == {"format": "csv"}


def test_checkpoint_preserves_v2_and_loads_old_history(tmp_path: Path) -> None:
    session = _session()
    path = session.checkpoint_save(tmp_path / "checkpoint")

    saved = json.loads((path / "history.json").read_text(encoding="utf-8"))
    assert saved[0]["schema_version"] == 2
    restored = Session.checkpoint_load(path, trusted=True)
    assert all(record["schema_version"] == 2 for record in restored.history)
    assert [record["sequence"] for record in restored.history] == list(
        range(1, len(restored.history) + 1)
    )

    from buildml.core.serialization import sha256_file

    (path / "history.json").write_text(
        json.dumps([{"action": "ingest", "details": {"format": "parquet"}}]),
        encoding="utf-8",
    )
    # Keep MANIFEST hashes aligned so this exercises legacy history normalization,
    # not tamper detection (integrity still refuses mismatched digests).
    manifest_path = path / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest.get("hashes")
        if isinstance(hashes, dict) and "history.json" in hashes:
            hashes["history.json"] = sha256_file(path / "history.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    old_restored = Session.checkpoint_load(path, trusted=True)
    assert old_restored.history[0]["schema_version"] == 2
    assert old_restored.history[0]["parameters"]["format"] == "parquet"


def test_public_data_access_and_exports_are_recorded_without_walkthrough_recursion(
    tmp_path: Path,
) -> None:
    session = _session().split(test_size=0.25, random_state=2)
    session.head(2)
    session.partition("train")
    session.to_pandas()
    session.to_engine("pandas")
    destination = session.to_parquet(tmp_path / "data.parquet")
    before_walkthrough = len(session.history)
    report = session.walkthrough()

    operations = [record["operation_id"] for record in session.history]
    assert destination.exists()
    assert {"head", "partition", "to_pandas", "to_engine", "to_parquet"} <= set(operations)
    assert len(session.history) == before_walkthrough
    assert "walkthrough" not in operations
    assert len(report.timeline) == before_walkthrough


def test_unusual_order_applies_split_target_role_only_when_stratifying() -> None:
    before = {"has_dataset": True, "roles": {}, "has_split": False}
    random_record = {
        "sequence": 1,
        "operation_id": "split",
        "parameters": {"stratify": False},
        "state_transition": {"before": before},
    }
    stratified_record = {
        **random_record,
        "parameters": {"stratify": True},
    }

    assert _unusual_order([random_record]) == []
    unusual = _unusual_order([stratified_record])
    assert len(unusual) == 1
    assert "target role required by stratify=True" in unusual[0]["reason"]
