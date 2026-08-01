# Ingest, engine, and checkpoint design

Shared spine for BuildML v2. Designed from day one so classical ML, deep learning, and RAG domains can share one data contract.

Related: [reconstruction-roadmap.md](./reconstruction-roadmap.md)

---

## 1. Goals

1. Ingest user data into a **Dataset handle**, not “always a Pandas DataFrame.”
2. Support **memory / lazy / out-of-core** modes with the same session method names.
3. Make **mid-loop take-out and bring-back** first-class via checkpoints + validation.
4. Preserve **leakage safety** across engines, materialization, and reattach.
5. Be honest about limits: large-data prep is in scope for v2; “train any model on a full TB in RAM” is not.

---

## 2. Canonical data contract

| Layer | Choice | Role |
| --- | --- | --- |
| Interchange | Apache Arrow / Parquet | Source of truth on disk and between tools |
| Large tabular compute | Polars **and** DuckDB (both supported) | Scan, filter, aggregate, many transforms |
| ML materialization bridge | Pandas / NumPy (and later framework tensors) | When estimators require dense in-memory `X, y` |
| Optional legacy | Existing CSV paths; Excel/Datatable as extras | Compatibility, not core identity |

**Rule:** Pandas is a **view/materialization target**, not the forever architecture.

---

## 3. Dataset handle

Conceptual object (names illustrative):

```text
Dataset
  - uri / frames / partitions
  - schema (names, dtypes, nullability)
  - roles (feature, target, group, time, id, weight, ignore)
  - engine (pandas | polars | duckdb | ...)
  - mode (memory | lazy | out_of_core)
  - split_plan / split_membership
  - stats (row_estimate, nbytes_estimate, sample_fingerprint)
  - history (operations applied)
```

### Modes

| Mode | When | Behavior |
| --- | --- | --- |
| `memory` | Small/medium fits RAM | Materialized table(s); fastest iteration |
| `lazy` | Larger than comfortable RAM | Deferred plans; collect/sample on demand |
| `out_of_core` | Partitioned multi-GB / TB-oriented | Partition scans; batched ops; sample-first EDA |

Engine selection can be explicit or suggested from estimates at ingest.

---

## 4. Ingest pipeline (automated by default)

```text
Source (path | DataFrame | Arrow | cloud URI later)
   → detect format
   → infer / validate schema
   → estimate scale (rows, bytes, partitions, load characteristics)
   → recommend engine + mode (auto)
   → allow user override (explicit engine/mode)
   → register Dataset in Session
   → optional sample profile
```

### Automation goals

Ingest should do the mechanical work for the user:

- Detect source type and format
- Infer schema / dtypes / nullability
- Estimate size and likely memory pressure
- Recommend `memory` vs `lazy` vs `out_of_core`
- Recommend an engine when large-data extras are available
- Surface a short ingest report the user can accept or override

Users should not have to manually wire loaders unless they want control.

### Multi-option engines (locked)

Do **not** pick only Polars or only DuckDB. Support **both** behind the same Dataset API:

| Engine | Strength | Role |
| --- | --- | --- |
| Pandas | Ecosystem familiarity; sklearn bridge | Default small/medium materialization |
| Polars | Fast lazy/eager tabular transforms | Primary large-frame engine extra |
| DuckDB | SQL scan/aggregate over files/partitions | Query/scan ally extra |

Auto-ingest picks a sensible default from scale + installed extras; users may override per session or per operation. If a recommended engine extra is missing, BuildML explains how to install it and continues on the best available path when safe.

### Ingest requirements

- Support at least: Pandas object, CSV, Parquet, Arrow IPC
- Return clear errors for corrupt/mismatched schemas
- Never silently read a multi-GB file into Pandas without a mode decision
- Provide `dry_run=True` style estimates where practical
- Emit install hints when a better engine exists but is not installed

### Large-data honesty gates

Before an operation runs, BuildML should know whether it:

1. Can run lazy/out-of-core
2. Requires a sample
3. Requires full materialization
4. Should refuse by default (with override)

Example: full Sweetviz-style report on a TB source → sample or refuse with guidance.

---

## 5. Session + leakage interaction

```text
Session
  owns Dataset(s)
  owns roles / split plan
  owns fitted artifacts (preprocessors, models)
  exposes methods that delegate to domain packages
```

Method pattern:

```text
session.drop_columns([...])
session.eda()
session.split(...)
session.impute(...)
session.fit(...)
session.predict(...)
session.checkpoint.save(...)
session.checkpoint.reattach(...)
```

Internals never reimplement transforms in the session body.

---

## 6. Leakage rules across engines

1. **Roles first.** Target/group/time/id columns are declared before modeling transforms.
2. **Split before fit-capable prep** on the modeling path (or use pipeline that fits on train fold only).
3. **Fit on train partition only**; apply to valid/test/infer.
4. **Resampling is train-only.**
5. **Search/CV refits per fold**; no fitting on full data then “evaluating.”
6. **Materialization does not reset fit scope.** Collecting a lazy plan for train does not authorize fitting on test.
7. **Reattach can invalidate fitted artifacts** when schema/roles/split integrity break.

---

## 7. Mid-loop export / reattach (checkpoint)

### Why

Professionals will leave the loop: export to another tool, custom code, or teammate workflow, then continue. v1 had no safe contract for this.

### Checkpoint bundle (proposed)

```text
checkpoint_name/
  data/
    frame.parquet              # canonical interchange (always)
    native_sidecar.parquet     # optional single-file Polars/DuckDB snapshot
    native_sidecar/            # optional partitioned layout (large frames)
      part-********-********.parquet
  meta.json             # schema, roles, mode, engine, native_sidecar, versions
  splits.json           # split membership or split recipe + seed
  artifacts/            # optional fitted pipeline pieces
  history.json          # operation log
  MANIFEST.json         # hashes, created_at, buildml_version
```

Alternative single-file archive (zip/tar) wrapping the same layout is fine for UX; contents stay structured.

**Native sidecars:** When a Polars/DuckDB handle is attached at save time, BuildML
writes an optional Parquet sidecar and records `lazy_intent`, `layout`, and
`compression` (default zstd) in `meta.json`. Small/medium frames use
`data/native_sidecar.parquet`; large frames (≥50k rows by default) use a
partitioned `data/native_sidecar/` directory. Public knobs on
`save_checkpoint` / `Session.checkpoint_save`:

| Kwarg | Default | Role |
| --- | --- | --- |
| `sidecar_layout` | `'auto'` | `'auto'` / `'single'` / `'partitioned'` |
| `sidecar_partition_rows` | `25_000` | Rows per part when partitioned |
| `sidecar_compression` | `'zstd'` | Parquet codec for the sidecar |

Restore prefers the sidecar (`scan_parquet` / `read_parquet`) so reattach does
not always rebuild eagerly from the Pandas-exported frame alone. Older
checkpoints with only `frame.parquet` or a legacy single-file sidecar remain
valid. Engine-native *query plans* are not serialized — lazy restore is a new
scan over sidecar bytes (collect-on-promote still applies; sklearn still needs
RAM).

**DuckDB connection ownership:** `get_engine('duckdb')` returns a cached adapter
that does not open a connection per call. Connections live on `DuckDBTable` /
root `Dataset` (`_owns_native_connection=True`). Derived project/filter handles
share the connection without ownership. Call `Dataset.close_native()` on the
owner when finished, or use `with dataset:` / `with session:` so owned
connections close on exit.

### Export APIs (conceptual)

| API | Purpose |
| --- | --- |
| `to_pandas()` / `to_polars()` / `to_parquet()` | Raw escape hatch |
| `checkpoint.save(path)` | Full resumable bundle |
| `export_predictions()` / `export_frame(role=...)` | Narrow exports |

Raw escape is always allowed. **Continuing modeling after external edits requires reattach validation.**

### Reattach validation matrix

| Case | Result |
| --- | --- |
| Schema + roles + split membership unchanged | Resume; artifacts valid if present |
| New columns added | Allow; require role assignment before modeling ops |
| Columns removed that artifacts depend on | Block predict; require refit or artifact drop |
| Dtype changes on modeled columns | Warn/block depending on severity |
| Row identity/order changed after split | Invalidate split integrity; require re-split or explicit partition inject |
| User supplies only train or only test externally | Allowed via explicit partition APIs |
| External transforms after fit | Mark artifacts stale; refuse silent predict |
| Metadata missing (data-only import) | Open as new ingest; guided mode restarts split/fit lifecycle |

### Flexibility modes

| Mode | Default behavior |
| --- | --- |
| **Guided** | BuildML owns split/fit scope; reattach strict |
| **Professional** | User may inject partitions and override more freely; still no silent leakage |
| **Inference** | Load fitted artifacts + new data only; no refit unless asked |

---

## 8. Materialization and deep learning / RAG readiness

Classical sklearn path:

```text
Dataset partition → (transform) → materialize X, y → estimator
```

Deep learning later:

```text
Dataset partition → batches/dataloaders → framework tensors → trainer
```

RAG later:

```text
Document sources → chunk table → embedding table → index artifact → retrieve
```

Shared pieces: ingest, schema, checkpoint, history, result objects, docs standard.  
Domain-specific pieces: trainers, indexes, metrics.

Designing Dataset/checkpoint now prevents classical ML from painting us into a Pandas-only corner.

---

## 9. Packaging implications (extras)

Full table lives in [reconstruction-roadmap.md §H](./reconstruction-roadmap.md). Summary:

| Extra | Contents |
| --- | --- |
| core | Session, Dataset, pandas/pyarrow bridge, sklearn baseline, automated ingest |
| `polars` | Polars engine adapter |
| `duckdb` | DuckDB engine adapter |
| `engines` | Both large engines together |
| `eda` / `reports` / `viz` | Plotting and profiling stacks |
| `excel` | Excel IO |
| `all-classical` | Common professional classical workstation set |
| `dl` / `rag` / `ai` | Later domain extras |

`import buildml` must succeed on core alone. Extras expand flexibility — lighter default install, not a lesser product.

---

## 10. Phase-1 implementation entry (recommended)

1. Define core types: schema, roles, result/error types  
2. Dataset handle with `memory` mode + Parquet/CSV/DataFrame ingest  
3. Session skeleton delegating to empty/light domain modules  
4. Split container + leakage guards  
5. Checkpoint save/load for memory-mode round-trip  
6. CI smoke: install, import, ingest, checkpoint reattach  
7. Add Polars/DuckDB adapter behind the same Dataset API  
8. Only then rebuild classical transform/model depth on top

---

## 11. Remaining implementation choices (non-blocking)

1. **Split membership storage:** row IDs vs position indices vs recipe+seed only  
2. **Artifact format:** joblib vs skops vs custom directory  
3. **Multi-table datasets:** defer to later; single-table v2 foundation  
4. **Auto-engine heuristic weights:** tune from benchmarks once both adapters exist  

**Locked:** support **both** Polars and DuckDB; automate ingest detection/recommendation; user override always available.
