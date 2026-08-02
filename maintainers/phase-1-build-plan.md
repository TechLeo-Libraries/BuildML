# Phase 1 build plan — Foundation

Implementation plan for BuildML 2.0 foundation.  
Grounded in: [reconstruction-roadmap.md](./reconstruction-roadmap.md) · [ingest-engine-checkpoint-design.md](./ingest-engine-checkpoint-design.md) · [classical-ml-capability-map.md](./classical-ml-capability-map.md)

**Status:** First implementation slice landed (`2.0.0a1`). Continue milestones D→G.  
**Python:** 3.10–3.13  
**North star:** flexibility · depth · functionality

---

## 1. Phase-1 goal

Ship an installable, tested **foundation** that:

1. Installs from modern packaging (`pyproject.toml`) with lean **core** deps
2. Imports cleanly: `import buildml` without plotting/profiling/engine extras
3. Ingests data with **automated detection** (format, schema, scale) and mode recommendation
4. Exposes a thin **Session** API that owns a **Dataset** handle
5. Supports **roles**, **splits**, and leakage guards (even before full transform library)
6. Saves/loads **checkpoints** (memory-mode round-trip)
7. Has CI: lint + tests + import smoke on Python 3.10–3.12
8. Leaves clear extension points for Polars/DuckDB adapters and classical depth (Phase 2+)

Phase 1 is **not** full classical parity. It is the spine everything else hangs on.

---

## 2. Non-goals (Phase 1)

- Full reimplementation of all v1 preprocess/model/EDA methods
- Deep learning / RAG / LLM operator
- Production server/CLI/SaaS
- Perfect auto-engine heuristics under every edge case (ship sensible v1 heuristics + overrides)
- Byte-compatible `SupervisedLearning` 1.0.9 API

---

## 3. Migration strategy (2.0 clean break)

| Approach | Decision |
| --- | --- |
| Version | Jump to `2.0.0a1` (alpha) then `2.0.0` when foundation+parity gates pass |
| Legacy code | Move current packages under `buildml/_legacy/` for reference during port; **not** exported from `buildml.__init__` |
| Public API | New `Session` (+ helpers) only at root |
| Docs | Mark 1.x as legacy; new quickstart targets Session |
| Compatibility shim | None in Phase 1 |

During Phase 1 implementation order:

1. Add new packages alongside legacy (briefly)
2. Point root exports at new API
3. Relocate old modules to `_legacy/`
4. Delete `_legacy/` once Phase 2 parity no longer needs it as a reference

---

## 4. Target package layout

```text
buildml/
  __init__.py                 # exports Session, __version__, maybe ingest helpers
  py.typed                    # typing marker
  core/
    __init__.py
    types.py                  # ColumnRole, DType info, enums
    results.py                # Result base / IngestReport / SplitResult stubs
    errors.py                 # BuildMLError hierarchy + missing-extra errors
    validation.py             # shared validators
  ingest/
    __init__.py
    detect.py                 # format / schema / scale detection
    report.py                 # IngestReport
    loaders.py                # DataFrame, CSV, Parquet, Arrow loaders (core)
  data/
    __init__.py
    dataset.py                # Dataset handle
    roles.py                  # role assignment
    splits.py                 # split plan + membership
    modes.py                  # memory | lazy  (legacy out_of_core coerces → lazy; no OOC fit)
    engines/
      __init__.py
      base.py                 # Engine protocol
      pandas_engine.py        # core engine
      polars_engine.py        # optional extra (stub/raise if missing)
      duckdb_engine.py        # optional extra (stub/raise if missing)
  checkpoint/
    __init__.py
    bundle.py                 # save/load layout
    validate.py               # reattach validation matrix (subset in P1)
  session/
    __init__.py
    session.py                # OOP facade (delegates only)
  preprocess/                 # Phase 1: package scaffold + maybe 1–2 ops
  model/                      # Phase 1: package scaffold only
  eda/                        # Phase 1: package scaffold + minimal summary optional
  pipeline/                   # Phase 1: package scaffold only
  _legacy/                    # relocated 1.x modules (temporary)

tests/
  unit/
  integration/
  fixtures/

pyproject.toml
README.md                     # updated 2.0 alpha story (can be staged)
.github/workflows/ci.yml
```

Root export sketch:

```python
from buildml.session import Session

__version__ = "2.0.0a1"
__all__ = ["Session", "__version__"]
```

---

## 5. Packaging (`pyproject.toml`) sketch

```toml
[project]
name = "buildml"
version = "2.0.0a1"
requires-python = ">=3.10,<3.13"
dependencies = [
  "numpy",
  "pandas",
  "pyarrow",
  "scikit-learn",
]

[project.optional-dependencies]
polars = ["polars"]
duckdb = ["duckdb"]
engines = ["buildml[polars,duckdb]"]
imbalanced = ["imbalanced-learn"]
viz = ["matplotlib", "seaborn"]
reports = ["sweetviz", "ydata-profiling"]
eda = ["buildml[viz,reports]"]
excel = ["openpyxl"]
all-classical = ["buildml[engines,imbalanced,eda,excel]"]
dev = ["pytest", "ruff", "mypy", "build", "pytest-cov"]
```

Exact lower/upper bounds set during Phase 1 Milestone A using current stable compatible ranges.

---

## 6. Public API surface (Phase 1 minimum)

### `Session`

| Method / property | Behavior |
| --- | --- |
| `Session()` / `Session.ingest(...)` | Create session from path/frame |
| `session.dataset` | Current Dataset handle |
| `session.ingest_report` | Last automated ingest report |
| `session.set_roles(...)` | Assign feature/target/group/time/id/weight |
| `session.split(...)` | Create train/test (and optional valid) membership |
| `session.to_pandas()` | Escape hatch |
| `session.to_parquet(path)` | Escape hatch |
| `session.checkpoint_save(path)` | Write bundle |
| `Session.checkpoint_load(path)` / `session.reattach(...)` | Restore + validate |
| `session.with_engine(...)` / `session.with_mode(...)` | Overrides |

### Automated ingest report (minimum fields)

- source type / format  
- schema summary  
- row/byte estimates  
- recommended mode  
- recommended engine  
- installed engines available  
- warnings (e.g. large file → avoid blind Pandas load)

### Dataset

- schema, roles, mode, engine name  
- `sample(n)` / `head(n)`  
- partition accessors after split  
- materialize guards

---

## 7. Milestones

### Milestone A — Packaging & import graph

**Build**

- Add `pyproject.toml`; stop relying on broken `requirements.txt` for install metadata
- Single-source version
- New package scaffolds + lean `__init__.py`
- Relocate legacy under `_legacy/` (or isolate so it is not imported at root)
- Dev extras: pytest, ruff, mypy

**Acceptance tests**

- [ ] `pip install -e .` in clean venv succeeds on 3.10/3.11/3.12
- [ ] `python -c "import buildml; print(buildml.__version__)"` works **without** seaborn/sweetviz/polars/duckdb
- [ ] Root does not import `_legacy` or optional viz stacks
- [ ] `python -m build` produces wheel + sdist

### Milestone B — Core types, errors, results

**Build**

- `ColumnRole`, schema structures, mode/engine enums
- `BuildMLError`, `MissingExtraError` (message includes install hint)
- `IngestReport`, basic result bases

**Acceptance tests**

- [ ] Unit tests for role validation (invalid role rejected)
- [ ] Missing-extra error message names the extra (e.g. `polars`)
- [ ] mypy/ruff clean on new packages (legacy can be excluded)

### Milestone C — Automated ingest + Dataset (memory mode)

**Build**

- Loaders: DataFrame, CSV, Parquet, Arrow
- Detection: format, schema, scale estimates
- Mode recommendation heuristics (thresholds documented)
- Engine recommendation (pandas default; suggest polars/duckdb when large + installed/missing)
- `Dataset` memory-mode handle with sample/head

**Acceptance tests**

- [ ] Ingest DataFrame fixture → report + dataset
- [ ] Ingest small CSV/Parquet fixtures → schema matches
- [ ] Large-file *simulated* estimate recommends non-memory mode (fixture with mocked size)
- [ ] Never auto-loads a “huge” path into Pandas without warning/mode decision (tested via hooks/mocks)
- [ ] Docstrings present on public ingest APIs

### Milestone D — Roles, splits, leakage guards

**Build**

- Role assignment API
- Random + stratified split (train/test; optional valid)
- Split membership stored with dataset/session
- Guard: modeling-oriented fit hooks refuse full-data fit when split exists (scaffold hook even if models come later)
- Explicit external partition injection API stub/minimal

**Acceptance tests**

- [ ] Stratified split preserves class proportions within tolerance
- [ ] Target role required before stratified split on classification target
- [ ] Accessors return disjoint train/test index sets
- [ ] Attempting a guarded “fit on full data” path raises a clear leakage error (can be a dedicated guard function in P1)

### Milestone E — Checkpoint save / reattach

**Build**

- Bundle layout: `data/`, `meta.json`, `splits.json`, `history.json`, `MANIFEST.json`
- Save/load memory-mode round-trip
- Reattach validation subset:
  - unchanged schema/roles/splits → resume
  - missing metadata → treat as fresh ingest
  - removed required columns → block with clear error
  - row/split integrity change → invalidate splits

**Acceptance tests**

- [ ] Save → load restores schema, roles, split membership
- [ ] Data-only reattach (no meta) opens as new ingest path
- [ ] Column removal after export fails validation as specified
- [ ] Manifest includes buildml version + content hashes

### Milestone F — Session API + engine extension points

**Build**

- `Session` methods wiring ingest/roles/split/checkpoint
- Engine protocol + pandas implementation
- Polars/DuckDB modules: real adapter if deps present, else `MissingExtraError` with install hint
- Override APIs: `with_engine`, `with_mode`

**Acceptance tests**

- [ ] End-to-end integration: ingest → roles → split → checkpoint → reattach
- [ ] `with_engine("polars")` without extra installed → clear MissingExtraError
- [ ] With `polars`/`duckdb` installed (optional CI job), basic dataset construction works
- [ ] Session methods have extensive docstrings (purpose, args, returns, example, leakage/scale notes)

### Milestone G — CI skeleton

**Build**

- GitHub Actions: matrix Python 3.10–3.12 on Ubuntu (+ Windows if feasible)
- Jobs: ruff, mypy (new code), pytest, import smoke, build wheel
- Optional job: `extras-engines` with polars+duckdb

**Acceptance tests**

- [ ] CI green on main/PR
- [ ] Failure of import smoke fails the pipeline
- [ ] Coverage report produced for new packages (floor can start modest, e.g. ≥80% on `core`/`ingest`/`data`/`checkpoint`/`session`)

---

## 8. Definition of Done (Phase 1 exit)

Phase 1 is complete when all of the following are true:

1. Milestones A–G acceptance boxes checked  
2. `import buildml` works on core-only install  
3. Documented Session quickstart (short) exists in docs or README alpha section  
4. No root dependency on `_legacy`  
5. Roadmap Phase 1 exit criteria satisfied: *packaging, import graph, core types, ingest handle, CI*  
6. Ready to start Phase 2 (classical parity) without redesigning Dataset/Session/Checkpoint

---

## 9. Suggested execution order (engineering days, indicative)

| Order | Work | Depends on |
| --- | --- | --- |
| 1 | Milestone A packaging | — |
| 2 | Milestone B types/errors | A |
| 3 | Milestone C ingest/dataset | B |
| 4 | Milestone D roles/splits | C |
| 5 | Milestone E checkpoint | D |
| 6 | Milestone F session + engine stubs | E |
| 7 | Milestone G CI harden | A–F in parallel late |

Parallelism: docs/docstrings continuous; CI workflow can land early in A and tighten through G.

---

## 10. First implementation slice (when coding starts)

Do this as the first PR-sized slice:

1. `pyproject.toml` + version `2.0.0a1`  
2. `buildml/core/{errors,types,results}.py`  
3. `buildml/ingest` minimal DataFrame ingest + report  
4. `buildml/data/dataset.py` memory handle  
5. `buildml/session/session.py` with `ingest` + `to_pandas`  
6. `tests/integration/test_import_and_ingest.py`  
7. Root `__init__.py` export `Session`  
8. Park legacy so it cannot break import  

That slice alone should prove the new direction is real.

---

## 11. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Scope creep into full parity during foundation | Enforce Phase-1 non-goals; park features on capability map |
| Legacy import side effects | Isolate `_legacy`; never import from root |
| Engine extras complexity | Protocol + MissingExtraError first; adapters thin |
| Thresholds for auto mode wrong | Document thresholds; always allow override; tune later |
| Docs lag code | Docstring standard required in Milestone F acceptance |

---

## 12. Go / no-go

| Question | Answer in this plan |
| --- | --- |
| Start greenfield Session API? | Yes |
| Support Polars and DuckDB? | Yes — stubs/adapters in P1, both as extras |
| Automate ingest? | Yes — core Milestone C |
| Keep 1.x API live? | No — `_legacy` only |
| Begin coding after this doc approved? | Yes |

**Owner action:** Approve this Phase-1 plan (or note edits). After approval, implementation starts at §10 first slice.
