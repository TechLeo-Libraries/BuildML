"""Exercise SDMetrics metadata conversion and CTGAN parameter validation."""

import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.synthetic.adapters.sdmetrics_eval import sdmetrics_quality_scores
from buildml.synthetic.adapters.sdv import _build_synthesizer


def test_ctgan_rejects_batch_size_incompatible_with_packing():
    with pytest.raises(ValidationError, match="positive multiple of 10"):
        _build_synthesizer("ctgan", None, epochs=1, batch_size=256)


@pytest.mark.parametrize("metadata_kind", ["inferred", "object", "dictionary"])
def test_real_sdmetrics_accepts_supported_metadata_forms(metadata_kind):
    pytest.importorskip("sdmetrics")
    metadata_module = pytest.importorskip("sdv.metadata")
    frame = pd.DataFrame({"amount": [float(i % 7) for i in range(40)], "category": ["a", "b"] * 20})
    metadata = None
    if metadata_kind != "inferred":
        metadata = metadata_module.SingleTableMetadata()
        metadata.detect_from_dataframe(frame)
        if metadata_kind == "dictionary":
            metadata = metadata.to_dict()
    scores, warnings = sdmetrics_quality_scores(frame, frame.copy(), metadata)
    assert scores["sdmetrics_overall"] == pytest.approx(1.0)
    assert bool(warnings) == (metadata_kind == "inferred")
