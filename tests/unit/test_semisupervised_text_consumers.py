"""Offline real text semi-supervision agrees with independent sklearn fitting."""
import numpy as np
import pandas as pd
import pytest
import test_offline_local_encoder_contracts as offline_encoders
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

from buildml import Session
from buildml.data.splits import frame_for_partition
from buildml.semisupervised.checkpoint import load_semisupervised_bundle, save_semisupervised_bundle
from buildml.semisupervised.evaluate import evaluate_semisupervised
from buildml.semisupervised.fit import fit_semisupervised
from buildml.semisupervised.predict import predict_semisupervised

local_sentence_encoder = offline_encoders.local_sentence_encoder
tiny_bert = offline_encoders.tiny_bert


@pytest.mark.parametrize("dtype,explicit", [(object, False), ("string", False), ("string", True)])
@pytest.mark.parametrize("marker", [None, "unknown"])
def test_real_text_fit_matches_train_only_oracle(local_sentence_encoder, tmp_path, dtype, explicit, marker):
    st = pytest.importorskip("sentence_transformers")
    texts = ["red apple", "blue berry fruit", "red fruit", "engine valve"] * 12
    labels = ["bad", "good", "bad", "good"] * 12
    for i in range(0, len(labels), 5):
        labels[i] = marker
    frame = pd.DataFrame({"text": pd.Series(texts, dtype=dtype), "y": labels})
    session = Session.ingest(frame).set_roles({"text": "feature", "y": "target"}).split(test_size=0.25, random_state=9)
    train = frame_for_partition(session.dataset, session.split_plan, "train")
    holdout = frame_for_partition(session.dataset, session.split_plan, "test")
    kwargs = dict(backend="hf", method="text_pseudo_label", text_model_name=str(local_sentence_encoder),
                  threshold=0.65, max_self_train_iter=4, random_state=3, unlabeled_marker=marker)
    if explicit:
        kwargs["text_column"] = "text"
    plan, result = fit_semisupervised(session.dataset, session.split_plan, **kwargs)
    encoder = st.SentenceTransformer(str(local_sentence_encoder), device="cpu")
    codes = np.array([{"bad": 0, "good": 1}.get(value, -1) for value in train["y"]])
    reference = SelfTrainingClassifier(LogisticRegression(max_iter=500, random_state=3),
                                       threshold=0.65, max_iter=4)
    reference.fit(np.asarray(encoder.encode(train["text"].tolist()), dtype=float), codes)
    vectors = np.asarray(encoder.encode(holdout["text"].tolist()), dtype=float)
    np.testing.assert_allclose(plan.estimator_.predict_proba(holdout["text"].tolist()),
                               reference.predict_proba(vectors), atol=1e-9)
    np.testing.assert_array_equal(plan.estimator_.estimator_.transduction_, reference.transduction_)
    assert plan.classes_ == ("bad", "good")
    assert result.columns == ("text",)
    assert result.n_unlabeled_train == int((codes == -1).sum())
    _, predictions = predict_semisupervised(session.dataset, plan, session.split_plan, partition="test")
    expected = np.array(["bad", "good"])[reference.predict(vectors)]
    assert predictions.predictions == tuple(expected)
    evaluation = evaluate_semisupervised(session.dataset, plan, session.split_plan, partition="test")
    labeled = holdout["y"].notna().to_numpy() & (holdout["y"].to_numpy() != "unknown")
    assert evaluation.metrics["accuracy"] == pytest.approx(np.mean(expected[labeled] == holdout["y"].to_numpy()[labeled]))
    path = save_semisupervised_bundle(tmp_path / "text-bundle", plan, fit_result=result, eval_result=evaluation)
    restored = load_semisupervised_bundle(path, trusted=True)  # Created above from a benign fixture.
    _, reloaded = predict_semisupervised(session.dataset, restored, session.split_plan, partition="test")
    assert reloaded.predictions == predictions.predictions
    pd.testing.assert_frame_equal(session.to_pandas(), frame)
    # Poison only held-out rows: training and pseudo-label acceptance must not move.
    poisoned = frame.copy(deep=True)
    poisoned.loc[list(session.split_plan.test_indices), "text"] = "heldout unseen tokens"
    poisoned.loc[list(session.split_plan.test_indices), "y"] = "unseen holdout class"
    other = Session.ingest(poisoned).set_roles({"text": "feature", "y": "target"}).split(test_size=0.25, random_state=9)
    second, _ = fit_semisupervised(other.dataset, other.split_plan, **kwargs)
    np.testing.assert_allclose(second.estimator_.estimator_.estimator_.coef_, reference.estimator_.coef_, atol=1e-9)
    np.testing.assert_array_equal(second.estimator_.estimator_.transduction_, reference.transduction_)
