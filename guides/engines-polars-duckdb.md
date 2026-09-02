# Engines: Pandas, Polars, and DuckDB

```bash
pip install buildml
pip install "buildml[engines]"
```

Pandas is what sklearn sees. Polars and DuckDB are for ingest, filter,
project, and aggregate before that materialization. Engine choice does not
make sklearn train out of core. Lazy Polars collects at the sklearn
boundary.

`with session:` closes an owned DuckDB connection. After Session
preprocess, native handles rebuild so later `project` /
`prepare_design_matrix` can still use the engine where that path exists.

Related: [classical end-to-end](classical-end-to-end.md),
[workflow-guide](../docs/workflow-guide.rst), [features](../docs/features.rst).

---

## What engines are for

Real tables are often larger than a notebook demo. Ingest with DuckDB or
Polars for typed IO, narrow rows and columns with engine-native ops, then
materialize a design matrix for sklearn only when needed.

BuildML records engine and mode in ingest metadata. After Session
preprocess it rebuilds native handles so `Dataset.project` /
`prepare_design_matrix` can prefer engine ops where that path exists.

---

## Use case: DuckDB ingest, filter, then classical fit

```python
from pathlib import Path

import pandas as pd

from buildml import Session
from buildml.data import portable_filter_expr

# Write a small CSV for the demo
path = Path("artifacts/txns.csv")
path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "amount": [10, 120, 30, 200, 15, 90, 110, 40],
        "velocity": [1, 4, 1, 5, 2, 3, 4, 2],
        "fraud": [0, 1, 0, 1, 0, 0, 1, 0],
    }
).to_csv(path, index=False)

with Session.ingest(str(path), engine="duckdb") as session:
    # Context manager calls close_native(): release owned DuckDB connections.
    pred = portable_filter_expr("amount", ">", 20)
    narrowed = session.dataset.filter_expr(pred)
    # Continue on the Session after syncing / working with the frame:
    session.set_roles(
        {"amount": "feature", "velocity": "feature", "fraud": "target"}
    )
    session.split(test_size=0.25, stratify=True, random_state=0)
    session.scale(method="standard")
    from sklearn.linear_model import LogisticRegression

    session.fit(LogisticRegression(max_iter=500), task="classification")
    print(session.evaluate(partition="test").metrics)
```

`portable_filter_expr` builds simple quoted comparisons for Polars and
DuckDB. Complex SQL stays engine-specific.

---

## Use case: Polars lazy ingest and projection

```python
# pip install "buildml[polars]"
from buildml import Session

session = Session.ingest("artifacts/txns.csv", engine="polars", mode="lazy")
session = session.with_engine("polars")
native = session.to_engine("polars")
print(type(native))

# Escape hatch when you need a Pandas copy explicitly:
pdf = session.to_pandas()
session.sync_native()  # rebuild native from current Pandas frame after edits
```

Lazy Polars frames collect at `to_pandas()` / sklearn materialization
boundaries. That is not zero-copy Torch loading and not out-of-core
`fit`.

---

## Use case: prepare_design_matrix before sklearn

```python
from buildml import Session
from sklearn.linear_model import Ridge

session = (
    Session.ingest(pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "y": [1, 2, 3, 4]}))
    .set_roles({"a": "feature", "b": "feature", "y": "target"})
    .split(test_size=0.25, random_state=0)
)

prep = session.prepare_design_matrix(partition="train")
print(prep)  # engine-aware projection metadata / handle
session.fit(Ridge(), task="regression")
```

---

## Switching engines mid-session

```python
session.with_engine("pandas")
session.with_mode("memory")  # or "lazy" where supported
print(session.metadata())
session.head(3)
session.to_parquet("artifacts/snapshot.parquet")
```

Missing extras raise `MissingExtraError` naming `polars` or `duckdb`.

---

## Large-path ingest honesty

For large file paths, BuildML may refuse blind full Pandas loads. Use:

- `dry_run=True` on ingest to inspect recommendations
- `read_nrows=...` for samples
- Engine extras for native IO

```python
report_session = Session.ingest("huge.parquet", dry_run=True)
print(report_session.ingest_report)
```

---

## Failure modes / limits

| Limit | Honest statement |
| --- | --- |
| Out-of-core sklearn | Not supported: engines help prep, not lazy `fit` |
| Torch loaders | Materialize via Pandas/NumPy bridge: no Polars zero-copy into DataLoaders |
| DuckDB leaks | Always `close_native()` or `with session:` |
| Complex SQL | Not portable: keep engine-specific logic outside `portable_filter_expr` |
| After preprocess | Native handles rebuild; verify with `sync_native` if you edited frames externally |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [EDA / Teaching Studio](eda-teaching-studio.md)
- [Torch deep](torch-deep.md) (materialization limits)
