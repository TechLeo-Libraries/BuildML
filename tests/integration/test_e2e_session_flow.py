from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_ingest_roles_split_impute_checkpoint_reattach(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "noise": [0, 0, 0, 0, 0, 0, 0, 0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"feature": "feature", "target": "target"})
        .drop_columns(["noise"])
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(columns=["feature"], strategy="median")
    )
    session.assert_can_fit()

    path = tmp_path / "flow_ckpt"
    session.checkpoint_save(path)
    restored = Session.checkpoint_load(path, trusted=True)

    assert restored.reattach_result is not None
    assert restored.reattach_result.status == "resume"
    assert "noise" not in restored.dataset.columns
    assert restored.split_plan is not None
    assert restored.to_pandas()["feature"].isna().sum() == 0
