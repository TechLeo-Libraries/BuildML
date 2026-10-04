"""Text embeddings may define similarity without numeric feature columns."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.cbr.fit import fit_cbr
from buildml.core.errors import ValidationError


def test_embedding_fit_accepts_text_only_features(monkeypatch):
    session = Session.ingest(pd.DataFrame({
        "text": ["billing question", "delivery question"] * 20,
        "target": [0, 1] * 20,
    })).set_roles({"text": "feature", "target": "target"})
    session.split(test_size=0.2, random_state=0, stratify=True)
    monkeypatch.setattr("buildml.cbr.extras.text_embedding_available", lambda: True)
    monkeypatch.setattr("buildml.cbr.retrieval_build.cbr_industry_available", lambda: True)
    monkeypatch.setattr("buildml.cbr.retrieval_build.windows_industry_ann_refused", lambda: True)
    seen = []

    def embed(frame, columns, *, model_name, numeric_matrix):
        seen.append(numeric_matrix)
        values = np.array([[1.0, 0.0] if "billing" in value else [0.0, 1.0] for value in frame["text"]])
        return values, "test-embedding"

    monkeypatch.setattr("buildml.cbr.retrieval_build.embed_text_cases", embed)
    plan, _ = fit_cbr(
        session.dataset, session.split_plan, backend="embedding", metric="cosine",
        task="classification", text_columns=["text"],
    )
    assert plan.columns == ()
    assert seen == [None]
    assert plan.case_base.ann_index_ is None
    assert plan.case_base.search_matrix_.shape == (len(session.split_plan.train_indices), 2)
    with pytest.raises(ValidationError, match="No numeric columns"):
        fit_cbr(session.dataset, session.split_plan, backend="sklearn", task="classification")
