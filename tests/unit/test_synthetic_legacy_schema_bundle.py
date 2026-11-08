"""An actual older v1 bundle remains a usable public consumer."""

import json
from pathlib import Path

import joblib
import pandas as pd

from buildml.synthetic import load_synthetic_bundle, sample_synthetic, save_synthetic_bundle
from buildml.synthetic.validation import validate_synthetic


def test_actual_older_slotted_schema_supports_comparison_sampling_and_resaving(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures/synthetic-legacy-62a90ad"
    plan = load_synthetic_bundle(fixture, trusted=True)
    spec = plan.column_specs[0]
    assert spec == spec
    before = joblib.hash(plan.generator_)
    sample = sample_synthetic(plan, n=80, random_state=11)
    assert set(sample.frame.category) == {"left", "right"}
    assert joblib.hash(plan.generator_) == before
    destination = save_synthetic_bundle(tmp_path / "resaved", plan)
    assert json.loads((destination / "meta.json").read_text(encoding="utf-8"))["format"] == "buildml.synthetic_bundle.v1"
    reloaded = load_synthetic_bundle(destination, trusted=True)
    pd.testing.assert_frame_equal(sample_synthetic(reloaded, n=80, random_state=11).frame,
                                  sample.frame)
    result = validate_synthetic(reloaded, sample.frame, run_great_expectations=False)
    assert result.passed
