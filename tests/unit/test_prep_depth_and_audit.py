"""Leakage-safe prep depth, custom transforms, dry-run, and history audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.preprocess import list_transforms, unregister_transform


def _text_frame(n: int = 40) -> pd.DataFrame:
    texts = [
        "good product fast delivery",
        "bad quality slow shipping",
        "average item okay value",
        "excellent service great price",
    ]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "review": [texts[i % len(texts)] for i in range(n)],
            "x": rng.normal(size=n),
            "seg": (["a", "b"] * ((n // 2) + 1))[:n],
            "y": ([0, 1] * ((n // 2) + 1))[:n],
        }
    )


def test_text_features_requires_split_and_is_train_fitted() -> None:
    frame = _text_frame()
    session = Session.ingest(frame).set_roles(
        {"review": "feature", "x": "feature", "seg": "feature", "y": "target"}
    )
    with pytest.raises(LeakageError):
        session.text_features(columns=["review"], method="tfidf", max_features=16)

    session.split(test_size=0.25, stratify=True, random_state=0)
    session.text_features(columns=["review"], method="tfidf", max_features=16)
    assert session.text_plan is not None
    assert session.text_plan.method == "tfidf"
    assert "review" not in session.dataset.columns
    assert any(name.startswith("review__") for name in session.dataset.columns)
    assert session.last_preprocess is not None
    assert session.last_preprocess.operation == "text_features"


def test_reduce_dimensions_pca_reports_explained_variance() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
            "c": rng.normal(size=50),
            "y": ([0, 1] * 25),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.2, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(n_components=2, prefix="pc")
    )
    assert session.reduce_plan is not None
    assert session.reduce_plan.n_components == 2
    assert len(session.reduce_plan.explained_variance_ratio_) == 2
    assert "pc_1" in session.dataset.columns
    assert "a" not in session.dataset.columns
    total = session.reduce_plan.to_dict()["total_explained_variance"]
    assert 0.0 < total <= 1.0 + 1e-9


def test_custom_transform_train_fit_and_registry() -> None:
    unregister_transform("unit_clip")

    def fit(train: pd.DataFrame, params: dict) -> dict:
        col = train.columns[0]
        lo = float(train[col].quantile(0.1))
        hi = float(train[col].quantile(0.9))
        return {"column": col, "lo": lo, "hi": hi}

    def transform(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
        col = artifact["column"]
        out = frame.copy()
        out[col] = out[col].clip(artifact["lo"], artifact["hi"])
        return out

    Session.register_transform(
        "unit_clip",
        fit=fit,
        transform=transform,
        description="Clip to train deciles",
        overwrite=True,
    )
    assert any(spec.name == "unit_clip" for spec in Session.list_transforms())
    assert any(spec.name == "unit_clip" for spec in list_transforms())

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    train_before = session.partition("train")["x"].copy()
    session.apply_custom_transform("unit_clip", columns=["x"])
    assert session.custom_plan is not None
    assert session.custom_plan.name == "unit_clip"
    train_after = session.partition("train")["x"]
    assert float(train_after.max()) <= float(train_before.quantile(0.9)) + 1e-9
    assert session.last_preprocess is not None
    assert session.last_preprocess.operation == "apply_custom_transform"
    unregister_transform("unit_clip")


def test_dry_run_and_summarize_history_do_not_mutate() -> None:
    frame = _text_frame(24)
    session = Session.ingest(frame).set_roles(
        {"review": "feature", "x": "feature", "seg": "feature", "y": "target"}
    )
    before_history = len(session.history)
    preview = session.dry_run("split")
    assert preview.would_mutate is False
    assert preview.steps
    assert preview.steps[0].operation == "split"
    assert preview.steps[0].available is True
    assert isinstance(preview.ranked_risks, list)
    assert isinstance(preview.suggested_next_ops, list)
    assert preview.prerequisite_graph.to_dict()["n_nodes"] >= 1
    assert len(session.history) == before_history

    blocked = session.dry_run("fit")
    assert blocked.steps[0].available is False
    assert blocked.steps[0].blocked_reasons
    assert blocked.prerequisite_graph.missing_required or blocked.steps[0].blocked_reasons

    session.split(test_size=0.25, stratify=True, random_state=1)
    summary = session.summarize_history()
    assert summary.n_operations >= 2
    assert "split" in summary.operation_counts
    assert session.last_history_summary is not None
    assert summary.suggested_next_ops
    assert all("operation" in item for item in summary.suggested_next_ops)
    assert summary.prerequisite_graph.to_dict()["n_nodes"] >= 1
    # summarize_history / dry_run do not append history records
    assert all(
        record.get("operation_id") not in {"dry_run", "summarize_history"}
        for record in session.history
    )


def test_walkthrough_includes_audit_summary() -> None:
    frame = _text_frame(32)
    session = (
        Session.ingest(frame)
        .set_roles({"review": "ignore", "x": "feature", "seg": "ignore", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=4)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    report = session.walkthrough()
    assert report.audit_summary
    assert report.audit_summary["has_fit"] is True
    assert "ranked_risks" in report.audit_summary
    assert "suggested_next_ops" in report.audit_summary
    assert report.next_actions


def test_text_and_pca_roundtrip_through_pipeline(tmp_path) -> None:
    frame = _text_frame(48)
    session = (
        Session.ingest(frame)
        .set_roles({"review": "feature", "x": "feature", "seg": "ignore", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
        .text_features(columns=["review"], method="count", max_features=12)
        .scale(columns=["x"], method="standard")
        .reduce_dimensions(columns=["x"], n_components=1, prefix="pc")
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    path = tmp_path / "pipe"
    session.save_pipeline(path)
    restored = Session().load_pipeline(path, trusted=True)
    assert restored.text_plan is not None
    assert restored.reduce_plan is not None
    assert restored.fit_result is not None


def test_error_slices_by_segment() -> None:
    frame = _text_frame(40)
    session = (
        Session.ingest(frame)
        .set_roles({"review": "ignore", "x": "feature", "seg": "ignore", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=3)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    report = session.error_slices(by="seg", partition="test", min_segment_n=2)
    assert report.kind == "segment_errors"
    assert report.payload["by"] == "seg"
    assert report.payload["by_columns"] == ["seg"]
    assert report.payload["segments"]
    assert "precision" in report.payload["segments"][0]
    assert "small_segments" in report.payload
    assert any(record.get("operation_id") == "error_slices" for record in session.history)

    multi = session.error_slices(
        by=["seg", "y"],
        partition="test",
        min_segment_n=1,
        max_segments=10,
    )
    assert multi.payload["by_columns"] == ["seg", "y"]
    assert multi.payload["segments"] or multi.payload["small_segments"]
    rows = multi.payload["segments"] or multi.payload["small_segments"]
    assert all("seg=" in row["segment"] and "y=" in row["segment"] for row in rows)

    tiny = session.error_slices(by="seg", partition="test", min_segment_n=10_000)
    assert tiny.payload["segments"] == []
    assert tiny.payload["small_segments"]
    assert any("min_segment_n" in tip for tip in tiny.interpretation)


def test_unknown_custom_transform_raises() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(ValidationError, match="Unknown custom transform"):
        session.apply_custom_transform("does_not_exist", columns=["x"])
