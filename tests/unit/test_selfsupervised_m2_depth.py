"""Deeper self-supervised coverage: encoder unit, scarce-label head, explain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.selfsupervised.encoder import MaskedTabularEncoder
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.fit import fit_ssl_pretext


def _frame(n: int = 140, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(-1.0, 0.8, size=(n // 2, 3))
    x1 = rng.normal(1.5, 0.8, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b", "c"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def test_masked_encoder_roundtrip() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    enc = MaskedTabularEncoder(latent_dim=5, hidden=(16,), max_iter=60, random_state=0)
    enc.fit(x)
    z = enc.transform(x)
    assert z.shape == (80, 5)
    recon = enc.reconstruct(x)
    assert recon.shape == x.shape


def test_low_level_pretext_head_eval(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_ssl_pretext(
        session.dataset,
        session.split_plan,
        method="masked_tabular",
        latent_dim=5,
        max_iter=70,
        prefer_reduce_components=False,
    )
    assert fit.n_train_rows > 0
    head_plan, head_fit = finetune_ssl_head(
        session.dataset, session.split_plan, plan, estimator="logistic_regression"
    )
    assert head_fit.n_labeled_train == fit.n_train_rows
    ev = evaluate_ssl(
        session.dataset, plan, head_plan, session.split_plan, partition="test"
    )
    assert "f1_macro" in ev.metrics

    from buildml.selfsupervised.checkpoint import save_ssl_bundle

    out = save_ssl_bundle(
        tmp_path / "direct",
        plan,
        fit_result=fit,
        head_plan=head_plan,
        head_fit_result=head_fit,
        eval_result=ev,
    )
    assert (out / "ssl_plan.joblib").is_file()


def test_head_skips_unlabeled_train() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_ssl_pretext(method="masked_tabular", latent_dim=4, max_iter=50, random_state=0)
    # Blank half of train labels
    rng = np.random.default_rng(7)
    full = session.to_pandas().copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=len(idx) // 2, replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    head = session.finetune_ssl_head()
    assert head.n_unlabeled_skipped > 0
    assert head.n_labeled_train + head.n_unlabeled_skipped == len(idx)


def test_invalid_mask_ratio() -> None:
    with pytest.raises(ValidationError):
        MaskedTabularEncoder(mask_ratio=0.0).fit(np.ones((10, 2)))


def test_explain_before() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    before = session.explain("fit_ssl_pretext", moment="before")
    assert before.operation == "fit_ssl_pretext"
    assert before.prerequisite_status.get("split") is True
