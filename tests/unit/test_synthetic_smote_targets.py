import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

pytest.importorskip("imblearn")


def _session(populations):
    train = pd.concat(
        [
            pd.DataFrame({"x": values, "y": 2 * values + 1, "site": "train", "label": label})
            for label, values in populations.items()
        ],
        ignore_index=True,
    )
    holdout = pd.DataFrame(
        {"x": [900, 901], "y": [1801, 1803], "site": "holdout", "label": "eligible"}
    )
    return (
        Session.ingest(pd.concat([train, holdout], ignore_index=True))
        .set_roles({"x": "feature", "y": "feature", "site": "feature", "label": "target"})
        .inject_split(
            train_indices=list(range(len(train))), test_indices=[len(train), len(train) + 1]
        )
    )


def test_smote_resamples_only_the_requested_train_class():
    session = _session(
        {"majority": np.arange(12), "eligible": np.arange(20, 26), "tiny": np.arange(60, 62)}
    )
    result = session.synthetic.fit(
        method="smote", k_neighbors=3, sampling_strategy={"eligible": 10}, random_state=17
    )
    assert result.partition == "train"
    assert result.n_rows == 20
    sample = session.synthetic.sample(n=125, random_state=21).frame
    assert list(sample.columns) == ["x", "y", "site", "label"]
    assert sample["label"].tolist() == ["eligible"] * 125
    assert sample["site"].tolist() == ["train"] * 125
    assert sample["x"].between(20, 25).all()
    assert (~sample["x"].isin(np.arange(20, 26))).any()
    np.testing.assert_allclose(sample["y"], 2 * sample["x"] + 1, rtol=0, atol=1e-12)
    original_rows = set(sample.iloc[:120][["x", "y"]].itertuples(index=False, name=None))
    assert set(sample.iloc[120:][["x", "y"]].itertuples(index=False, name=None)) <= original_rows
    pd.testing.assert_frame_equal(sample, session.synthetic.sample(n=125, random_state=21).frame)


@pytest.mark.parametrize("strategy", ["auto", {"eligible": 6}])
def test_noop_smote_fit_keeps_the_previous_session_plan(strategy):
    session = _session({"majority": np.arange(6), "eligible": np.arange(20, 26)})
    previous_fit = session.synthetic.fit(method="bootstrap", random_state=17)
    previous_plan = session.synthesizer_plan
    previous_sample = session.synthetic.sample(n=4, random_state=21)
    previous_history = list(session.history)

    with pytest.raises(ValidationError, match="no synthetic rows"):
        session.synthetic.fit(method="smote", k_neighbors=3, sampling_strategy=strategy)

    assert session.synthesizer_plan is previous_plan
    assert session.synthetic_fit_result is previous_fit
    assert session.synthetic_sample_result is previous_sample
    assert session.history == previous_history
    pd.testing.assert_frame_equal(
        previous_sample.frame, session.synthetic.sample(n=4, random_state=21).frame
    )


def test_smote_still_rejects_too_few_neighbors_in_a_requested_class():
    session = _session(
        {"majority": np.arange(12), "eligible": np.arange(20, 26), "tiny": np.arange(60, 62)}
    )
    with pytest.raises(ValidationError, match="k_neighbors"):
        session.synthetic.fit(method="smote", k_neighbors=3, sampling_strategy={"tiny": 5})
    assert session.synthesizer_plan is None
