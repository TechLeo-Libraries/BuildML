"""Observed segment populations, scalar metrics and exported report consumers."""

import copy
import json
from collections import Counter
from html.parser import HTMLParser

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from buildml import Session
from buildml.core.errors import ValidationError


def _session(kind, storage="object", *, collision=None, problem=None,
             partition="validation", fit=True, unused_collision=False):
    rng = np.random.default_rng(1937)
    x, z, noise = rng.normal(size=(3, 180))
    y = (np.where(x - .6*z + .7*noise > 0, 10., 2.) if kind != "ridge"
         else 3*x - .4*z + .7*noise)
    groups = (["alpha"]*10 + ["beta"]*8 + [None]*4 + ["<NA>"]*5 + ["rare"]*3)*6
    if collision is not None:
        groups = list(collision)*90
    if unused_collision:
        groups = ["1"]*180
    frame = pd.DataFrame({"x": x, "z": z, "y": y,
                          "g": pd.Series(groups, dtype=object),
                          "region": ["north|west", 'say "east"', r"back\slash"]*60})
    if storage != "object":
        categories = ([1, "1", "unused"] if unused_collision else
                      list(collision) + ["unused"] if collision else
                      ["alpha", "beta", "<NA>", "rare", "unused", "never"])
        frame["g"] = pd.Categorical(frame.g, categories=categories,
                                    ordered=storage == "ordered")
    if problem is not None:
        row = 120 if partition == "validation" else 150
        if isinstance(problem, str):
            frame["y"] = frame.y.astype(object)
        frame.loc[row, "y"] = problem
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature",
        "y": "target", "g": "group", "region": "ignore"})
    session.inject_split(train_indices=list(range(120)),
                         validation_indices=list(range(120, 150)),
                         test_indices=list(range(150, 180)))
    if fit:
        estimator = (Ridge(alpha=.7) if kind == "ridge" else
                     make_pipeline(StandardScaler(), LogisticRegression(C=.6))
                     if kind == "pipeline" else LogisticRegression(C=.6))
        session.fit(estimator, task="regression" if kind == "ridge" else "classification")
    return session


@pytest.fixture(params=["logistic", "pipeline", "ridge"])
def kind(request):
    return request.param


@pytest.fixture(params=["object", "categorical", "ordered",
                        "categorical_nonmissing", "ordered_nonmissing"])
def fitted(kind, request):
    storage = request.param.split("_")[0]
    collision = ("alpha", "beta") if request.param.endswith("_nonmissing") else None
    return _session(kind, storage, collision=collision)


def _label(row, by):
    parts = []
    for column in by:
        value = row[column]
        if pd.isna(value):
            text = "<NA>"
        else:
            text = str(value)
            if text == "<NA>" or any(c in text for c in ('|', '"', "\\")):
                text = json.dumps(text, ensure_ascii=False)
        parts.append(f"{column}={text}")
    return " | ".join(parts)


def _scalar_predictions(session, part):
    estimator = session.fit_result.estimator
    values = part[["x", "z"]].to_numpy()
    if hasattr(estimator, "steps"):
        scaler, estimator = [step for _, step in estimator.steps]
        values = (values - scaler.mean_) / scaler.scale_
    if session.fit_result.task == "regression":
        return np.array([sum(float(a)*float(b) for a, b in zip(row, estimator.coef_))
                         + float(estimator.intercept_) for row in values])
    return np.array([estimator.classes_[int(sum(float(a)*float(b) for a, b in
                     zip(row, estimator.coef_[0])) + float(estimator.intercept_[0]) > 0)]
                     for row in values])


def _metric_oracle(truth, prediction, regression):
    n = len(truth)
    if regression:
        errors = [float(a)-float(b) for a, b in zip(truth, prediction)]
        absolute = sorted(abs(error) for error in errors)
        return {"n": n, "mae": sum(absolute)/n,
                "rmse": (sum(error*error for error in errors)/n)**.5,
                "median_ae": (absolute[(n-1)//2] + absolute[n//2])/2,
                "mean_residual": sum(errors)/n, "max_abs_error": max(absolute)}
    correct = sum(a == b for a, b in zip(truth, prediction))
    tp = sum(a == b == 10 for a, b in zip(truth, prediction))
    actual = sum(a == 10 for a in truth)
    predicted = sum(b == 10 for b in prediction)
    return {"n": n, "n_correct": correct, "n_incorrect": n-correct,
            "accuracy": correct/n, "error_rate": 1-correct/n,
            "precision": tp/predicted if predicted else 0.,
            "recall": tp/actual if actual else 0.,
            "f1": 2*tp/(actual+predicted) if actual+predicted else 0.,
            "support_positive": actual, "predicted_positive_rate": predicted/n}


class _Tables(HTMLParser):
    """Read actual HTML tables as an independent downstream consumer."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tables = {}
        self.caption = ""
        self.headers = []
        self.rows = []
        self.cells = []
        self.tag = None
        self.buffer = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.caption, self.headers, self.rows = "", [], []
        if tag == "tr":
            self.cells = []
        if tag in {"caption", "th", "td"}:
            self.tag, self.buffer = tag, []

    def handle_data(self, data):
        if self.tag:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == self.tag:
            value = "".join(self.buffer)
            if tag == "caption":
                self.caption = value
            elif tag == "th":
                self.headers.append(value)
            else:
                self.cells.append(value)
            self.tag = None
        if tag == "tr" and self.cells:
            self.rows.append(dict(zip(self.headers, self.cells)))
        if tag == "table":
            self.tables[self.caption] = self.rows


def _check(session, partition, by, maximum, minimum, destination):
    part = session.partition(partition)
    before = session.to_pandas()
    estimator_hash = joblib.hash(session.fit_result.estimator)
    labels = [_label(row, by) for row in part.to_dict("records")]
    counts = Counter(labels)
    prediction = _scalar_predictions(session, part)
    np.testing.assert_allclose(session.fit_result.estimator.predict(part[["x", "z"]]),
                               prediction, atol=1e-12)
    report = session.error_slices(by=by, partition=partition, max_segments=maximum,
                                  min_segment_n=minimum, export_html=destination)
    payload = report.payload
    assert payload["n_rows"] == len(part)
    assert payload["n_unique_segments"] == len(counts)
    assert payload["n_omitted_segments"] == max(0, len(counts)-maximum)
    rows = payload["segments"] + payload["small_segments"]
    assert len(rows) == min(maximum, len(counts))
    selected = {row["segment"] for row in rows}
    assert len(selected) == len(rows)
    if selected != set(counts):
        assert max(counts[key] for key in set(counts)-selected) <= min(counts[key] for key in selected)
    regression = session.fit_result.task == "regression"
    for key in ("segments", "small_segments"):
        score = "mae" if regression else "error_rate"
        assert [row[score] for row in payload[key]] == sorted(
            [row[score] for row in payload[key]], reverse=True)
        for row in payload[key]:
            positions = [i for i, label in enumerate(labels) if label == row["segment"]]
            expected = _metric_oracle(part.y.to_numpy()[positions], prediction[positions], regression)
            assert row["n"] == counts[row["segment"]]
            assert (row["n"] >= minimum) == (key == "segments")
            for metric, value in expected.items():
                assert row[metric] == pytest.approx(value, abs=1e-12)
    assert payload["n_segments_reported"] == len(payload["segments"])
    assert payload["n_small_segments"] == len(payload["small_segments"])
    json.dumps(report.to_dict(), allow_nan=False)
    tables = _Tables(destination.read_text(encoding="utf-8")).tables
    for key, caption in [("segments", "Segments"), ("small_segments", "Small Segments")]:
        if not payload[key]:
            continue
        exported = {row["segment"]: row for row in tables[caption]}
        assert set(exported) == {row["segment"] for row in payload[key]}
        for row in payload[key]:
            for metric, value in row.items():
                if metric != "segment":
                    assert float(exported[row["segment"]][metric]) == pytest.approx(value, rel=1e-5, abs=1e-6)
    pd.testing.assert_frame_equal(before, session.to_pandas())
    assert joblib.hash(session.fit_result.estimator) == estimator_hash
    return report


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("by", [["g"], ["g", "region"]])
@pytest.mark.parametrize("maximum,minimum", [(2, 1), (20, 5)])
def test_observed_populations_metrics_and_real_html(fitted, partition, by, maximum, minimum, tmp_path):
    _check(fitted, partition, by, maximum, minimum, tmp_path / "segments.html")


def _refuses_atomically(session, options, match, destination, *, seed_partition="validation"):
    prior = session.error_slices(by="region", partition=seed_partition, min_segment_n=1,
                                 export_html=destination)
    contents = destination.read_bytes()
    history = copy.deepcopy(session._history)
    model_hash = joblib.hash(session.fit_result.estimator)
    with pytest.raises(ValidationError, match=match):
        session.error_slices(export_html=destination, **options)
    assert session._last_diagnostic is prior
    assert session._history == history
    assert destination.read_bytes() == contents
    assert joblib.hash(session.fit_result.estimator) == model_hash


@pytest.mark.parametrize("collision", [(1, "1"), (True, "True"), (1.25, "1.25")])
@pytest.mark.parametrize("storage", ["object", "categorical"])
@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("by", [["g"], ["g", "region"]])
def test_display_key_collision_refuses_before_replacing_reports(kind, collision, storage, partition, by, tmp_path):
    session = _session(kind, storage, collision=collision)
    _refuses_atomically(session, {"by": by, "partition": partition, "min_segment_n": 1},
                        "collision|ambiguous", tmp_path / "segments.html")


@pytest.mark.parametrize("field", ["max_segments", "min_segment_n"])
@pytest.mark.parametrize("value", [True, 2.5, "2", np.nan, 0])
def test_invalid_cardinality_is_an_atomic_validation_error(kind, field, value, tmp_path):
    _refuses_atomically(_session(kind), {"by": "g", field: value}, field,
                        tmp_path / "segments.html")


@pytest.mark.parametrize("kind,problem", [("logistic", np.nan), ("pipeline", np.nan),
    ("ridge", np.nan), ("ridge", np.inf), ("ridge", -np.inf), ("ridge", "bad")])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_invalid_scoring_truth_cannot_publish_nan_or_false_error_rates(kind, problem, partition, tmp_path):
    session = _session(kind, problem=problem, partition=partition)
    _refuses_atomically(session, {"by": "g", "partition": partition, "min_segment_n": 1},
                        "target", tmp_path / "segments.html", seed_partition="train")


def test_numpy_integral_limits_produce_portable_json(kind, tmp_path):
    report = _check(_session(kind), "validation", ["g"], np.int64(20), np.int64(1),
                    tmp_path / "segments.html")
    assert type(report.payload["max_segments"]) is int
    assert type(report.payload["min_segment_n"]) is int


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_unused_typed_category_is_not_an_observed_collision(kind, partition, tmp_path):
    session = _session(kind, "categorical", unused_collision=True)
    report = _check(session, partition, ["g"], 20, 1, tmp_path / "segments.html")
    assert report.payload["n_unique_segments"] == 1
    assert report.payload["segments"][0]["n"] == 30


@pytest.mark.parametrize("by", [["g"], ["g", "region"]])
def test_owned_model_bundle_retains_categorical_report_semantics(kind, by, tmp_path):
    source = _session(kind, "categorical")
    bundle = source.save_model(tmp_path / "owned-model")
    restored = _session(kind, "categorical", fit=False).load_model(bundle, trusted=True)
    expected = _check(source, "test", by, 20, 1, tmp_path / "source.html")
    observed = _check(restored, "test", by, 20, 1, tmp_path / "restored.html")
    assert expected.payload == observed.payload


@pytest.mark.parametrize("kind", ["logistic", "pipeline"])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_unseen_numeric_truth_is_counted_as_an_error(kind, partition, tmp_path):
    # Accuracy can score an unseen true class; probability calibration cannot.
    session = _session(kind, problem=99., partition=partition)
    _check(session, partition, ["g"], 20, 1, tmp_path / "segments.html")
