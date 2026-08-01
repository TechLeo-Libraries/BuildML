import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session


def test_encode_scale_fit_evaluate_flow() -> None:
    frame = pd.DataFrame(
        {
            "age": [21, 25, 30, 35, 40, 45, 50, 55],
            "city": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "city": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .encode(columns=["city"], method="onehot")
        .scale(columns=["age"], method="standard")
        .fit(LogisticRegression(max_iter=500), task="classification")
    )

    report = session.eda()
    assert report.overview["n_rows"] == 8
    assert report.recommendations
    assert report.adaptive_plan
    assert "city" not in session.dataset.columns

    preds = session.predict(partition="test")
    assert len(preds) == len(session.partition("test"))

    metrics = session.evaluate(partition="test")
    assert metrics.task == "classification"
    assert "accuracy" in metrics.metrics
    assert "confusion_matrix" in metrics.diagnostics
    assert session.fit_result is not None
    assert session.fit_result.n_train_rows > 0
