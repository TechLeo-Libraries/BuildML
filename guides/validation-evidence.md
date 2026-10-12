# Validation evidence and supported scope

Support belongs to a version, platform, dependency set and exercised method.
Use the reproducible runners below to generate evidence for your environment.
Missing dependencies, crashes, timeouts and skipped tests remain visible; they
are not counted as successful backend verification.

The optimization extra constrains HiGHS below 1.15. HiGHS 1.15.1 conflicts with
OR-Tools 9.15 native symbols after CVXPY import. Linux checks with HiGHS 1.14.0,
CVXPY 1.9.3 and OR-Tools 9.15.6755 passed both import orders and native solver
construction. CI exercises those import orders before the solve contracts.
See the [upstream OR-Tools report](https://github.com/google/or-tools/issues/5246).

## Real public-data evaluation

```bash
python -m benchmarks.timeseries.m4_real --download --series 24
python -m benchmarks.rag.beir_scifact --download --queries 300
```

M4 uses real hourly series from a pinned official repository revision: two
historical fixed origins and the official competition test horizon. Hyperparameters
are fixed before evaluation. Reports contain series IDs, source URLs/hashes,
per-origin sMAPE/seasonal MASE, time, and independent seasonal-naive parity.
The default is a reproducible 24-series subset; `--series 414` evaluates all
hourly series. It is not the full M4 competition across every frequency.

SciFact uses BEIR's full corpus and official test relevance judgments, with
document-level nDCG/recall/MRR at 10, per-query latency, actual retrieval mode,
and an independent TF-IDF baseline. Only document titles/abstracts are indexed;
claims and relevance judgments remain outside the index. Hashing embeddings
are lexical rather than trained semantic embeddings. Lower scores are reported
without changing the test set or choosing hyperparameters against it.

Downloads require `--download`; failures raise rather than fall back to synthetic
data. Cached file SHA256 values are retained in every result. TLS verification
remains enabled; Windows can use the optional `truststore` package for OS roots.
These real-data evaluations are separate from small offline smoke benchmarks.

Sources: [M4 official data](https://github.com/Mcompetitions/M4-methods),
[BEIR dataset registry](https://github.com/beir-cellar/beir/wiki/Datasets-available).

## Backend/platform evidence table

Follow-up assessment, 2026-10-12, review branch `codex/coverage-cuda-ci-20261012`:

| Configuration | Outcome | Qualification |
|---|---|---|
| Remote core Linux, Python 3.10–3.13 | All configurations passed; Python 3.12: 1,886 passed, 301 skipped, 70.19% line coverage (43,123 / 61,437 statements) | [CI run 38314965748](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38314965748), commit `3b4df47`; enforced floor remains 69%. Other Python configurations are not represented by the quoted Python 3.12 measurement |
| Remote backend Windows/Linux matrix | Linux 156 passed / 10 skipped; Windows 157 passed / nine skipped | [Backend run 38314965754](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38314965754), commit `3b4df47`; nine required groups, configuration-specific skips remain in artifacts |
| Remote release acceptance | Passed | [Acceptance run 38314965734](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38314965734), commit `3b4df47`; this is verification, not a published release |
| Local Linux CUDA | BuildML regression fit/evaluation passed | Quadro RTX 3000, Torch 2.7.1+cu126, CUDA 12.6; five epochs, finite metrics, R² 0.98486; peak Torch allocation 29.38 MiB, excluding driver/context allocations |
| Concurrent ASGI soak | 358,036 requests in 30 minutes, zero errors | Four workers; sampled p95 29.21 ms; peak RSS 205.49 to 210.80 MiB; bounded latency sampling; in-process evidence, not a deployment SLA |
| Linux targeted repairs | Five DDP tests passed; PDF, TorchScript and DEC/IDEC checks passed | Distributed CPU/Gloo checks use the CUDA-capable Torch installation; multi-GPU throughput is unmeasured |
| Expanded local Linux suite with CUDA/extras | 81.74% line coverage; 2,058 passed, 45 skipped, six failed | Commit `505cc55`; diagnostic evidence, not a passing gate. All six failures were the HiGHS/OR-Tools native symbol conflict described above. The compatible dependency repair passed targeted checks; a fresh whole-package measurement on `3b4df47` is running. The 90% target remains outstanding |

Exact local measurements are preserved in `reports/coverage90-gpu-linux.json`,
`reports/coverage90-soak-30min.json` and `reports/coverage90-linux-full.json`.
The broader local environment and the remote core environment have different
dependencies; their percentages are not interchangeable.

Earlier snapshot:

Local assessment, 2026-10-11, modified 2.6.4 source, Windows 11 / Python 3.12.8:

| Configuration | Outcome | Qualification |
|---|---|---|
| Polars/DuckDB | 9 passed, 1 skipped | Real engine and connection checks; dependency-absence case skipped |
| statsmodels forecasting | 5 passed, 1 skipped | Real ETS fitting/generation/evaluation; dependency-absence case skipped |
| imbalanced-learn | 5 passed, 1 skipped | Real training-only resampling and holdout preservation |
| Gymnasium | 24 passed, 1 skipped | Real tabular environment/policy checks |
| ripser/persim | 12 passed, 0 skipped | Real TDA persistence/vectorization contracts |
| MAPIE | 16 passed, 0 skipped | Real method, alpha and persistence contracts |
| hnswlib industry ANN | 15 passed, 2 skipped | Worker and Session fitting/prediction/evaluation; text-embedding and Windows torch-CBR paths remain unverified |
| FastAPI/httpx serving | 18 passed, 0 skipped | Authentication and scoring-schema contracts |
| CUDA | Unverified | Windows Torch is CPU-only; Linux CUDA installation failed when host disk space ran out |

The machine-readable local reports preserve package versions and skip reasons.
These eight groups passed 104 tests with six disclosed skips in a consistent
optional configuration (`pip check` passed). Dependencies were installed in
disposable directories; the existing environment was preserved.
Update this dated snapshot after changing backend behavior; CI artifacts from
the workflow below are the ongoing platform evidence. Adding a workflow alone
does not establish that its configurations pass.

The local full Windows suite passed 1,418 tests with 167 skips and measured
72.87% line coverage. That Windows measurement did not justify raising the Linux
floor; the follow-up above records subsequent Linux evidence. Additional targeted checks
cover final worker and persisted-index compatibility refinements.

```bash
python scripts/backend_support_report.py
python scripts/backend_support_report.py --groups engines forecasting mapie ann
```

| Group | Required real libraries | Exercised contracts |
|---|---|---|
| Engines | Polars, DuckDB | Ingestion and connection lifecycle |
| Forecasting | statsmodels | Real classical forecasting and fit/evaluation behavior |
| Resampling | imbalanced-learn | Training-only resampling and lineage |
| RL | Gymnasium | Tabular state/action and environment behavior |
| TDA | ripser, persim | Persistence/feature and domain behavior |
| Graph | NetworkX | Graph structure, classical node fitting and graph bundle contracts; optional GCN skips remain visible |
| MAPIE | MAPIE | Real split/CV+/jackknife+ methods, alpha, persistence |
| ANN | hnswlib | Persistent worker, neighbour parity, retention, restore, fault containment |
| Serving | FastAPI, httpx, Uvicorn | Authentication configuration, scoring schema, real loopback HTTP predictions and shutdown |

The runner writes `support.json`, `support.md`, logs and JUnit XML with exact
versions, tested counts and every skip reason. Requesting a missing or failed
group returns nonzero. `.github/workflows/backend-evidence.yml` runs these
contracts on Windows/Linux Python 3.12, independently of the broader core
matrix. A successful job does not certify unexecuted optional methods.

## Scale, memory and bounded operational reliability

```bash
python scripts/verify_operational_limits.py --rows 100000 1000000 --soak-seconds 120
# Bounded child: Linux address space or Windows Job Object allocation limit:
python scripts/verify_operational_limits.py --memory-budget-mib 2048
python scripts/verify_operational_limits.py --gpu
```

Classical fitting is in memory. The scale runner compares predictions with an
independent sklearn pipeline, records peak process RSS and elapsed time, and
isolates memory failures. The serving probe sends authenticated concurrent ASGI
requests, checks prediction parity and unauthorized rejection, and reports errors,
latency and peak RSS before/after. A two-minute soak is regression evidence;
it does not establish an internet deployment SLA or months of stable operation.

CUDA absence is explicitly unavailable. A successful CUDA probe measures a
Torch runtime baseline; BuildML device-path training/evaluation must also be
exercised for a claim about its GPU workflow. Device memory and throughput are
specific to the recorded hardware/configuration.

## Statistical interpretation

Conformal fitting refuses Session-global fitted preprocessing because it has
already used future calibration rows. The fitting/calibration carve does not
consult labels. External feature engineering, dependent units, temporal shifts
and exchangeability remain the caller's study-design responsibility. See the
[probabilistic guide](probabilistic-deep.md) for the safe workflow.

EDA reports carry exploratory interpretation metadata and warnings. Target and
numeric drift tests include Benjamini–Hochberg-adjusted q-values computed over
the complete tested family before displayed lists are truncated. Numeric drift
flags combine an adjusted q-value threshold with effect size. Multiple-comparison
correction does not turn exploration into causal evidence or repair dependent
sampling. Categorical/boolean mutual information already uses discrete-feature
estimation. Target-type inference is a heuristic, and univariate drift cannot
exclude joint-distribution changes.
