"""Real optional MAPIE backend acceptance, including persisted coverage metadata."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

pytest.importorskip("mapie")


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("method", ["split", "cv_plus", "jackknife_plus"])
def test_real_mapie_coverage_and_bundle(tmp_path, task, method):
    rng = np.random.default_rng(192)
    x = rng.normal(size=(220, 2))
    y = 2 * x[:, 0] + rng.normal(size=220)
    if task == "classification":
        y = (y > 0).astype(int)
    session = Session.ingest(pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y}))
    session.set_roles({"a": "feature", "b": "feature", "y": "target"})
    session.split(test_size=.2, validation_size=.2, random_state=1)
    session.probabilistic.fit(backend="mapie", estimator=method, task=task, alpha=.1)
    before = session.probabilistic.predict_interval(partition="test")
    evaluation = session.probabilistic.evaluate(partition="test")
    metric = "interval_coverage" if task == "regression" else "set_coverage"
    assert metric in evaluation.metrics, evaluation.warnings
    assert before.alpha == .1
    modern = session.probabilistic.plan.estimator_.api == "modern"
    if modern:
        for operation in (session.probabilistic.predict_interval, session.probabilistic.evaluate):
            with pytest.raises(ValidationError, match="differs from calibrated alpha"):
                operation(alpha=.2)
    bundle = session.probabilistic.save_bundle(tmp_path / "bundle")
    session.probabilistic.load_bundle(bundle, trusted=True)
    after = session.probabilistic.predict_interval(partition="test")
    assert after.alpha == before.alpha
    assert after.lower == before.lower
    assert after.upper == before.upper
    assert after.prediction_sets == before.prediction_sets
