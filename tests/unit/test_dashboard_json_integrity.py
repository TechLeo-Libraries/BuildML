"""Degenerate analytical measurements must remain valid strict JSON."""

import json

import numpy as np
import pytest

from buildml.dashboard.serialize import json_safe


@pytest.mark.parametrize("scalar", [float, np.float32, np.float64])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_measurements_are_missing_values_in_strict_json(scalar, value):
    payload = json_safe({"association": scalar(value), "nested": [{"vif": scalar(value)}]})
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == {"association": None, "nested": [{"vif": None}]}


def test_finite_measurements_and_numpy_flags_retain_numeric_types():
    payload = json_safe({"score": np.float32(0.5), "count": np.int64(12),
                         "flagged": np.bool_(False), "bounds": (float("-inf"), 2.0)})
    assert json.loads(json.dumps(payload, allow_nan=False)) == {
        "score": 0.5, "count": 12, "flagged": False, "bounds": [None, 2.0]
    }
