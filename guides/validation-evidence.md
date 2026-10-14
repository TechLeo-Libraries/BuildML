# Validation evidence and supported scope

Support belongs to a version, platform, dependency set and exercised method.
Use the reproducible runners below to generate evidence for your environment.
Missing dependencies, crashes, timeouts and skipped tests remain visible; they
are not counted as successful backend verification.

The expanded job at `b9b9e48` terminated with an illegal instruction while
importing hnswlib during test collection. It produced no new coverage result.
ANN availability probes now run in a subprocess on every platform. The expanded
CI profile rebuilds hnswlib without `-march=native`, bypasses cached wheels, and
requires a real ANN fit/predict/evaluate contract before the coverage suite.
The portable import and real ANN contract passed remotely at `f7700ce`, whose
complete CI workflow passed. The newer `e3f7b93` expanded gate also passed;
its core jobs stopped at a missing parameter docstring. That documentation
regression was repaired without increasing the docstring budget.

For bounded real HTTP and process recovery evidence, install `buildml[serve]`
and `psutil`, then run:

```bash
python scripts/verify_http_reliability.py --seconds 3600 --out artifacts/http-reliability.json
python scripts/verify_http_reliability.py --seconds 3600 --continuous --out artifacts/http-continuous.json
```

The runner checks predictions against a known linear function, authentication,
malformed requests, RSS samples and sampled latency at three client/batch loads.
It restarts after graceful shutdown and a forced process termination. Traffic
pauses during these deliberate failures; startup and shutdown times are recorded.
Each recovery-run process handles one third of the requested load duration.
The continuous mode keeps one process through all three loads. Results record
source hashes and dependency versions and reject source changes during the run.
They describe this loopback configuration and duration, not an availability SLA
or long-term production reliability.

The isolated RL profile pins imitation 1.0.1, SB3 2.2.1, Gymnasium 0.29.1,
Torch 2.7.1 CPU and Arrow 21 with NumPy 1.26. The earlier declared SB3 minimum
contradicted imitation's upper bound. BC now uses the installed trainer's
constructor/optimizer API, and GAIL uses a vectorized environment. GAIL refuses
variable horizons by default; accepting them requires an explicit boolean and
discloses possible reward leakage before agent execution and in the fit result.
Seeded CPU policies repeat, and trusted GAIL bundles use SB3's archive format
instead of pickling live environment internals. These contracts establish
interoperation on bounded workloads, not control performance on difficult tasks.

The serving follow-up also reproduced cross-app model contamination: two
servers in one process returned the most recently loaded model's predictions.
Apps now own independent bundle state, and each handle clears only its own app
on shutdown. Two concurrent real HTTP servers are tested against different
known linear functions, including both shutdown orders. The module-global
state accessor remains a compatibility view; clearing it without an app only
resets that view. Boolean and fractional port configuration is also rejected
instead of silently selecting another endpoint.

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

Follow-up assessment, 2026-10-14, review branch `codex/coverage-cuda-ci-20261012`:

| Configuration | Outcome | Qualification |
|---|---|---|
| Remote core Linux, Python 3.10–3.13 | All configurations passed; Python 3.12: 2,039 passed, 329 skipped, 70.64% line coverage | [CI run 38542047517](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38542047517), commit `c3bd24a`; enforced floor remains 69%. The later `e3f7b93` core jobs stopped at the docstring audit before testing |
| Remote backend Windows/Linux matrix | Both jobs passed, including real HTTP load and process recovery | [Backend run 38543420923](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38543420923), commit `e3f7b93`; required backend groups, bounded allocation checks and separate 60-second real HTTP probes. Configuration-specific skips remain visible |
| Remote release acceptance | Passed | [Acceptance run 38314965734](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38314965734), commit `3b4df47`; this is verification, not a published release |
| Remote expanded CPU whole-package gate | Passed: 83.02% (51,045 / 61,488 statements); 2,386 passed, 50 skipped | [CI run 38543420905](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38543420905), PR merge `c28427c`, whose Git tree equals `e3f7b93`; the expanded floor is raised to 82% from this passing measurement. Core docstring failure is separate from the passing expanded suite |
| Remote EconML and compatible causal profiles, Windows/Linux | All four jobs passed | [CI run 38353328266](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38353328266), `e7d0a5b`; repaired bounded policy behavior, adjusted effect reporting, graph-name integrity and real backend imports are required |
| Local Linux CUDA | BuildML regression fit/evaluation passed | Quadro RTX 3000, Torch 2.7.1+cu126, CUDA 12.6; five epochs, finite metrics, R² 0.98486; peak Torch allocation 29.38 MiB, excluding driver/context allocations |
| Local Linux EconML 0.17.0 policy learner | Sixteen known-benefit and reporting contracts passed | Continuous/binary outcomes, numeric/string treatment labels, two simulation seeds and requested bootstrap counts; adjusted population summaries and omitted intervals are explicit. Repaired behavior also passed remote Windows/Linux jobs |
| Isolated Linux DoWhy 0.14 | Estimation/refutation passed; four name and bundle contracts passed | NumPy 2.4.1 / SciPy 1.15.3 / scikit-learn 1.9.1 / pandas 3.0.6; graph node names retain spaces, punctuation, quotes and Unicode. This compatible profile is separate from expanded NumPy 2.5 / SciPy 1.18; remote Windows/Linux jobs also passed |
| Isolated Linux giotto 0.6.2 | Fifteen real topology tests passed | NumPy 1.26.4 / SciPy 1.11.4 / scikit-learn 1.3.2 / Arrow 20; real Arrow execution, diagram-axis integrity, bounded Mapper sampling and true igraph node/edge counts. This is a separate compatible profile, not a full-suite measurement |
| Concurrent ASGI soak | 358,036 requests in 30 minutes, zero errors | Four workers; sampled p95 29.21 ms; peak RSS 205.49 to 210.80 MiB; bounded latency sampling; in-process evidence, not a deployment SLA |
| Real Windows HTTP recovery assessment | 1,022,102 requests in one hour, no unexpected failures | Immutable `e3f7b93` source hash unchanged; three 20-minute process lifetimes at 1/4/8 clients and 1/16/64 rows. Sampled p95 4.89/18.58/35.79 ms; graceful/forced restart startup 6.50/17.80 seconds. Deliberate downtime is separate from scored requests |
| Isolated Linux RL/imitation profile | Seventeen tests passed, one intentional absence-test skip | Real PPO, BC holdout behavior at two seeds, GAIL horizon guard, seeded repeatability, agent parameters and trusted bundle parity. Separate CPU profile; remote Windows/Linux results require their CI jobs |
| Linux targeted repairs | Five DDP tests passed; PDF, TorchScript and DEC/IDEC checks passed | Distributed CPU/Gloo checks use the CUDA-capable Torch installation; multi-GPU throughput is unmeasured |
| Expanded local Linux suite with CUDA/extras | Passed: 83.00% line coverage (51,022 / 61,472 statements); 2,327 passed, 46 skipped | Commit `c3bd24a`, measured 2026-10-14; immutable whole-package monolith completed in 30 minutes. The CUDA probe also passed. The 90% target remains outstanding |

Exact local measurements are preserved in `reports/coverage90-gpu-linux.json`,
`reports/coverage90-soak-30min.json` and `reports/coverage90-linux-full.json`.
The broader local environment and the remote core environment have different
dependencies; their percentages are not interchangeable.

The `expanded-coverage` CI job reproduces the assessed dependency profile using
`requirements-expanded-coverage.txt` and its dated constraints. It runs all
tests against all BuildML modules, with an 82% floor, and uploads source,
installed package, coverage and test-result artifacts. CPU Torch replaces local
CUDA Torch in that job; its outcome must be verified independently. The core
floor remains 69%, and the 90% target remains outstanding.

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
