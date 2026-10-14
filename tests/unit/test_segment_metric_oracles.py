"""Segment diagnostics must match independent residuals on the named partition."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError


def _session():
    rng = np.random.default_rng(710)
    x = rng.normal(size=100)
    frame = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(size=100),
                          "site": ["east", "west", None, "small"] * 25,
                          "cohort": np.arange(100) % 2})
    return (Session.ingest(frame)
            .set_roles({"x": "feature", "y": "target", "site": "group", "cohort": "ignore"})
            .split(test_size=0.2, validation_size=0.2, random_state=3)
            .fit(Ridge(), task="regression"))


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
@pytest.mark.parametrize("minimum", [1, 100])
def test_segment_residual_metrics_match_each_actual_partition(partition, minimum, tmp_path):
    session = _session()
    part = session.partition(partition)
    prediction = session.fit_result.estimator.predict(part[["x"]])
    residual = part["y"].to_numpy() - prediction
    report = session.error_slices(by=["site", "cohort"], partition=partition,
                                  min_segment_n=minimum, max_segments=20,
                                  export_html=tmp_path / "segments.html")
    rows = report.payload["segments"] + report.payload["small_segments"]
    labels = "site=" + part.site.astype("string").fillna("<NA>") + " | cohort=" + part.cohort.astype("string")
    assert sum(row["n"] for row in rows) == len(part)
    for row in rows:
        errors = residual[(labels == row["segment"]).to_numpy()]
        assert row["mae"] == pytest.approx(np.abs(errors).mean())
        assert row["rmse"] == pytest.approx(np.sqrt(np.square(errors).mean()))
        assert row["median_ae"] == pytest.approx(np.median(np.abs(errors)))
        assert row["mean_residual"] == pytest.approx(errors.mean())
        assert row["max_abs_error"] == pytest.approx(np.abs(errors).max())
    assert bool(report.payload["segments"]) == (minimum == 1)
    assert (tmp_path / "segments.html").is_file()


@pytest.mark.parametrize("partition", ["vaildation", "TEST", "", None])
def test_unknown_partition_cannot_silently_analyze_the_test_set(partition):
    session = _session()
    with pytest.raises(ValidationError, match="partition"):
        session.error_slices(by="site", partition=partition)


@pytest.mark.parametrize("options", [
    {"by": "missing"}, {"by": []}, {"by": "site", "min_segment_n": 0},
    {"by": "site", "max_segments": 0},
])
def test_invalid_segment_specification_is_refused(options):
    with pytest.raises(ValidationError):
        _session().error_slices(**options)
