from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session


def test_extract_dates_compare_models_and_persist(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "x": list(range(20)),
            "y": [i % 2 for i in range(20)],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"event_date": "time", "x": "feature", "y": "target"})
        .extract_dates(columns=["event_date"], include_time=False, drop_original=True)
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(columns=["x"], method="standard")
    )
    assert any(c.startswith("event_date_") for c in session.dataset.columns)

    comparison = session.compare_models(
        {
            "logreg": LogisticRegression(max_iter=500),
            "rf": RandomForestClassifier(n_estimators=30, random_state=0),
        },
        task="classification",
    )
    assert comparison.rows
    assert session.fit_result is not None

    model_dir = tmp_path / "model"
    session.save_model(model_dir)
    restored = Session.ingest(frame).load_model(model_dir)
    assert restored.fit_result is not None
    assert restored.fit_result.feature_columns == session.fit_result.feature_columns
