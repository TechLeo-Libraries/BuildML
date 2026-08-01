"""Teaching Studio / walkthrough surface for fold-local vs Session-global preprocess."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.dashboard.teaching import build_teaching_studios
from buildml.preprocess.fold import SESSION_GLOBAL_ONLY_STEPS, PreprocessRecipe
from buildml.session.walkthrough import preprocess_scope_status


def _text_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    texts = [f"good product {i}" if i % 2 == 0 else f"bad item {i}" for i in range(n)]
    x = rng.normal(size=n)
    y = (x + (np.arange(n) % 2) > 0).astype(int)
    return pd.DataFrame({"review": texts, "x": x, "y": y})


def test_preprocess_scope_empty_without_history() -> None:
    status = preprocess_scope_status([])
    assert status["enabled"] is False
    assert status["disclosures"] == []
    assert status["session_global_only"] == list(SESSION_GLOBAL_ONLY_STEPS)


def test_preprocess_scope_from_fold_recipe_history() -> None:
    history = [
        {
            "operation_id": "cv_score",
            "sequence": 3,
            "parameters": {
                "fold_preprocess": {
                    "text": "tfidf",
                    "reduce": "pca",
                    "session_global_only": list(SESSION_GLOBAL_ONLY_STEPS),
                }
            },
            "result_summary": {"mean": 0.7},
        }
    ]
    status = preprocess_scope_status(history)
    assert status["enabled"] is True
    assert status["fold_local"]["text"] == "tfidf"
    assert status["fold_local"]["reduce"] == "pca"
    assert 3 in status["fold_local"]["history_sequences"]
    assert any("fold-local" in note.lower() and "tfidf" in note for note in status["disclosures"])
    assert any("pca" in note.lower() for note in status["disclosures"])
    assert any("session-global-only" in note.lower() for note in status["disclosures"])


def test_preprocess_scope_session_global_custom_and_resample() -> None:
    history = [
        {
            "operation_id": "apply_custom_transform",
            "sequence": 2,
            "parameters": {"name": "clip"},
            "result_summary": {},
        },
        {
            "operation_id": "resample",
            "sequence": 3,
            "parameters": {"strategy": "random_over"},
            "result_summary": {},
        },
    ]
    status = preprocess_scope_status(history)
    assert status["enabled"] is True
    assert status["session_global"]["apply_custom_transform"] is True
    assert status["session_global"]["resample"] is True
    assert status["fold_local"]["text"] is None
    assert any("custom" in note.lower() for note in status["disclosures"])
    assert any("resample" in note.lower() for note in status["disclosures"])


def test_walkthrough_and_teaching_surface_preprocess_scope(tmp_path: Path) -> None:
    session = (
        Session.ingest(_text_frame(100))
        .set_roles({"review": "feature", "x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    session.cv_score(
        LogisticRegression(max_iter=200),
        cv=2,
        preprocess=PreprocessRecipe(
            text="tfidf",
            reduce="pca",
            text_max_features=20,
            reduce_n_components=2,
        ),
    )

    walk = session.walkthrough()
    assert walk.preprocess_scope_status["enabled"] is True
    assert walk.preprocess_scope_status["fold_local"]["text"] == "tfidf"
    assert walk.preprocess_scope_status["fold_local"]["reduce"] == "pca"
    doc = walk.export_html(tmp_path / "walk_scope.html").read_text(encoding="utf-8")
    assert "Preprocess scope" in doc
    assert "tfidf" in doc
    assert "pca" in doc.lower()
    assert "Session-global" in doc or "session-global" in doc.lower()

    eda = session.eda(include_plots=False)
    assert eda.overview["preprocess_scope_status"]["fold_local"]["text"] == "tfidf"
    studios = build_teaching_studios(eda.to_dict())
    cockpit = studios["cockpit"]
    assert cockpit["worked_example"]["values"]["fold_local_text"] == "tfidf"
    assert cockpit["worked_example"]["values"]["fold_local_reduce"] == "pca"
    assert any("fold-local" in line.lower() for line in cockpit["interpretation"])
    assert "text-features" in cockpit["concepts"]
    assert "principal-components" in cockpit["concepts"]
    assert "custom-transforms" in cockpit["concepts"]

    multi = studios["multivariate"]
    assert multi["worked_example"]["values"]["fold_local_reduce"] == "pca"
    assert any("eda pca" in line.lower() for line in multi["interpretation"])
