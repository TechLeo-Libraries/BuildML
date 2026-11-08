"""Episode sampling keeps support/query rows disjoint and preserves feature names."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from buildml.metalearning.features import sample_support_query


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("scarce", [False, True])
@pytest.mark.parametrize("n_way", [None, 2])
def test_sampling_preserves_columns_and_positional_row_identity(
    collision, duplicates, scarce, n_way
):
    per_class = 4 if scarce else 10
    n = per_class * 3
    feature = "__y_code__" if collision else "feature"
    frame = pd.DataFrame(
        {
            feature: np.arange(n, dtype=float),
            "label": np.repeat(["a", "b", "c"], per_class),
            "row_id": np.arange(n),
        }
    )
    frame.index = np.arange(n) // 2 if duplicates else np.arange(6000, 6000 + n)
    original = frame.copy()
    encoder = LabelEncoder().fit(frame.label)
    kwargs = dict(
        target_column="label",
        columns=[feature],
        label_encoder=encoder,
        k_shot=2,
        n_query=30 if scarce else 6,
        n_way=n_way,
    )
    support, query, chosen = sample_support_query(frame, rng=np.random.default_rng(21), **kwargs)
    again = sample_support_query(frame, rng=np.random.default_rng(21), **kwargs)
    assert list(support.columns) == list(query.columns) == list(frame.columns)
    assert len(chosen) == (3 if n_way is None else n_way)
    assert set(support.label) == set(query.label) == set(chosen)
    assert set(support.row_id).isdisjoint(query.row_id)
    assert support.row_id.is_unique and query.row_id.is_unique
    assert support.groupby("label").size().tolist() == [2] * len(chosen)
    expected_query_per_class = 2 if scarce else max(1, 6 // len(chosen))
    assert query.groupby("label").size().tolist() == [expected_query_per_class] * len(chosen)
    pd.testing.assert_frame_equal(support, original.iloc[support.row_id.to_numpy()])
    pd.testing.assert_frame_equal(query, original.iloc[query.row_id.to_numpy()])
    pd.testing.assert_frame_equal(support, again[0])
    pd.testing.assert_frame_equal(query, again[1])
    pd.testing.assert_frame_equal(frame, original)
