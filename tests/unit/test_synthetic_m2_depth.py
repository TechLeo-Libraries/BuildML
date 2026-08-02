"""Depth tests for synthetic generators (copula joints, SMOTE wrap, privacy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.synthetic.models import GaussianCopulaGenerator, build_column_specs


def _session_with_cats() -> Session:
    rng = np.random.default_rng(0)
    n = 180
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n) * 2 + 1,
            "x3": rng.integers(0, 5, size=n),
            "cat": rng.choice(["u", "v", "w"], size=n, p=[0.5, 0.3, 0.2]),
            "y": rng.integers(0, 2, size=n),
        }
    )
    # Induce correlation
    frame["x2"] = frame["x1"] * 0.7 + frame["x2"] * 0.3
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "x3": "feature",
                "cat": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )


def test_gaussian_copula_preserves_rough_correlation() -> None:
    session = _session_with_cats()
    session.fit_synthesizer(method="gaussian_copula", random_state=0)
    sample = session.sample_synthetic(n=400, random_state=1)
    assert sample.frame is not None
    real_corr = (
        session.dataset.frame.iloc[list(session._split_plan.train_indices)][["x1", "x2"]]
        .corr()
        .iloc[0, 1]
    )
    syn_corr = sample.frame[["x1", "x2"]].corr().iloc[0, 1]
    assert np.isfinite(syn_corr)
    # Same sign and roughly similar magnitude
    assert real_corr * syn_corr > 0
    assert abs(real_corr - syn_corr) < 0.45


def test_copula_condition_rejection() -> None:
    session = _session_with_cats()
    session.fit_synthesizer(method="gaussian_copula", random_state=0)
    sample = session.sample_synthetic(
        n=30, condition={"cat": "u"}, random_state=2
    )
    assert (sample.frame["cat"].astype(str) == "u").all()


def test_bootstrap_plain_is_subset_of_train_values() -> None:
    session = _session_with_cats()
    session.fit_synthesizer(method="bootstrap", smooth_sigma=0.0, random_state=0)
    train = session.dataset.frame.iloc[list(session._split_plan.train_indices)]
    sample = session.sample_synthetic(n=25, random_state=3)
    # Every sampled x1 must appear in train (plain bootstrap)
    train_vals = set(np.round(train["x1"].to_numpy(), 10))
    for val in np.round(sample.frame["x1"].to_numpy(), 10):
        assert val in train_vals


def test_smote_method_optional_extra() -> None:
    x, y = make_classification(
        n_samples=200,
        n_features=4,
        weights=[0.8, 0.2],
        random_state=0,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    try:
        import imblearn  # noqa: F401
    except ImportError:
        with pytest.raises(MissingExtraError):
            session.fit_synthesizer(method="smote", random_state=0)
        return

    fit = session.fit_synthesizer(method="smote", random_state=0)
    assert fit.method == "smote"
    sample = session.sample_synthetic(n=20, random_state=1)
    assert sample.n_rows == 20
    assert "y" in sample.frame.columns


def test_resample_still_works_alongside_synthetic() -> None:
    """Cross-link: resample path must remain intact."""
    x, y = make_classification(
        n_samples=200,
        n_features=4,
        weights=[0.85, 0.15],
        random_state=1,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=1)
    )
    session.fit_synthesizer(method="bootstrap", random_state=0)
    assert session.synthesizer_plan is not None
    try:
        import imblearn  # noqa: F401
    except ImportError:
        pytest.skip("imbalanced-learn not installed")
    session.resample(sampler="random_oversample")
    assert session.resample_plan is not None
    # Synthesizer plan still attached (resample does not clear it)
    assert session.synthesizer_plan is not None


def test_unknown_method_raises() -> None:
    session = _session_with_cats()
    with pytest.raises(ValidationError, match="Unknown synthesizer"):
        session.fit_synthesizer(method="ctgan")  # type: ignore[arg-type]


def test_column_specs_mixed() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"], "c": [1, 2, 1]}
    )
    specs = build_column_specs(frame)
    kinds = {s.name: s.kind for s in specs}
    assert kinds["a"] == "continuous"
    assert kinds["b"] == "categorical"
    gen = GaussianCopulaGenerator.fit(frame, specs, random_state=0)
    out = gen.sample(10, random_state=1)
    assert len(out) == 10
