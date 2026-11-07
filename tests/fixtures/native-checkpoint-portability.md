`native-checkpoint-portability.zip` contains four native snapshots written with
BuildML's public checkpoint API. It lets the same reader tests run when DuckDB,
Polars, both, or neither are installed. It contains only Parquet and JSON, with
no serialized Python plans. The original writer manifests are intact.

The writer was BuildML source `108e752ce5efbe5dac3f46287c204d875b3ad9bc` on
Windows Python 3.12.8, with DuckDB 1.5.6, Polars 2.0.0, and PyArrow 25.0.0.
The source data and split can be reproduced with this recipe:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session

root = Path("native-checkpoint-fixtures")
root.mkdir()
x = np.arange(32, dtype=float) / 7
raw = root / "raw.parquet"
pd.DataFrame({"x": x, "z": np.sin(x), "y": 3 * x + np.cos(x)}).to_parquet(raw, index=False)
for engine in ("duckdb", "polars"):
    for layout in ("single", "partitioned"):
        session = Session.ingest(raw, engine=engine, mode="lazy")
        try:
            session.set_roles({"x": "feature", "z": "feature", "y": "target"})
            session.split(test_size=0.25, validation_size=0.25, random_state=13)
            session.checkpoint_save(
                root / f"{engine}-{layout}", sidecar_layout=layout, sidecar_partition_rows=7
            )
        finally:
            session.close_native()
```

Archive the contents of `root`, with writer versions and source revision in
`provenance.json`. History timestamps and Parquet bytes can vary with the writer;
the tests check the rows, splits, and predictions that a reader consumes.
