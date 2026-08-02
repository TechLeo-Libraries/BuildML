"""Integration smoke: synthetic Session loop + bundle + walkthrough."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from buildml import Session


def test_synthetic_session_smoke(tmp_path: Path) -> None:
    x, y = make_classification(
        n_samples=280,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=0,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["bucket"] = pd.cut(frame["f0"], bins=3, labels=["lo", "mid", "hi"]).astype(str)

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in frame.columns if c.startswith("f")},
                "bucket": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
    )

    session.fit_synthesizer(method="gaussian_copula", random_state=0)
    session.sample_synthetic(n=60, random_state=1)
    fid = session.evaluate_synthetic(mode="fidelity", partition="test")
    assert fid.n_synthetic >= 60
    tstr = session.evaluate_synthetic(mode="tstr", partition="test")
    assert "score" in tstr.metrics

    session.fit_synthesizer(method="bootstrap", smooth_sigma=0.0, random_state=2)
    session.sample_synthetic(n=10, merge_mode="extend_train")

    bundle = tmp_path / "syn"
    session.save_synthetic_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "synthetic_plan.joblib").is_file()

    walk = session.walkthrough()
    assert walk.synthetic_status.get("has_synthesizer_plan") is True
    assert any("privacy" in d.lower() for d in walk.synthetic_status.get("disclosures", []))
