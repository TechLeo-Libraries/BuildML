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

The `3647366` comparison added `--methods naive seasonal_naive lag_ridge ets arima`
on the same 24 hourly series and three origins (72 evaluations per method).
Lower values are better:

| Method | Mean sMAPE | Mean seasonal MASE |
|---|---:|---:|
| Independent seasonal naive / BuildML seasonal naive | 21.56 | 1.307 |
| Lag ridge | 17.61 | 1.026 |
| ETS with fixed defaults | 31.87 | 2.002 |
| ARIMA with fixed defaults | 65.56 | 15.419 |

The default industry configurations did not beat seasonal naive on this subset.
Tune configurations using historical validation before judging them on a fresh
holdout; these results do not establish general superiority for any method.

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

Maintained assessment, updated 2026-10-19, review branch `codex/coverage-cuda-ci-20261012`:

| Configuration | Outcome | Qualification |
|---|---|---|
| Remote core Linux, Python 3.10-3.13 | All core jobs passed: 71.28-71.29%; 2,172 passed, 398 skipped per job | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; corrected thread/process collection permits core floor 70% under the one-point headroom rule |
| Remote backend Windows/Linux matrix | Both jobs passed, including real HTTP recovery and continuous probes | [Backend run 38589015595](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015595), `2837fb3`; real backend groups and bounded allocation checks; skips remain visible |
| Remote release acceptance | Passed | [Acceptance run 38589015461](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015461), `2837fb3`; verification does not publish a release |
| Remote expanded CPU whole-package gate | Passed: 85.1201% (52,640 / 61,842); 3,185 passed, 91 skipped | [CI run 39354834562](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39354834562); tested merge `8eec7730` tree equals `02a9337`; all three workflows passed; scope/exclusions unchanged; floor84%; target90 needs3,018 further lines |
| Remote EconML and compatible causal profiles, Windows/Linux | All four jobs passed | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; real estimation/refutation, graph-name, policy and persistence contracts |
| Local Linux CUDA | BuildML regression fit/evaluation passed | Quadro RTX 3000, Torch 2.7.1+cu126, CUDA 12.6; five epochs, finite metrics, R² 0.98486; peak Torch allocation 29.38 MiB, excluding driver/context allocations |
| Local Linux EconML 0.17.0 policy learner | Sixteen known-benefit and reporting contracts passed | Continuous/binary outcomes, numeric/string treatment labels, two simulation seeds and requested bootstrap counts; adjusted population summaries and omitted intervals are explicit. Repaired behavior also passed remote Windows/Linux jobs |
| Isolated Linux DoWhy 0.14 | Estimation/refutation passed; four name and bundle contracts passed | NumPy 2.4.1 / SciPy 1.15.3 / scikit-learn 1.9.1 / pandas 3.0.6; graph node names retain spaces, punctuation, quotes and Unicode. This compatible profile is separate from expanded NumPy 2.5 / SciPy 1.18; remote Windows/Linux jobs also passed |
| Isolated Linux giotto 0.6.2 | Fifteen real topology tests passed | NumPy 1.26.4 / SciPy 1.11.4 / scikit-learn 1.3.2 / Arrow 20; real Arrow execution, diagram-axis integrity, bounded Mapper sampling and true igraph node/edge counts. This is a separate compatible profile, not a full-suite measurement |
| Concurrent ASGI soak | 358,036 requests in 30 minutes, zero errors | Four workers; sampled p95 29.21 ms; peak RSS 205.49 to 210.80 MiB; bounded latency sampling; in-process evidence, not a deployment SLA |
| Real Windows HTTP recovery assessment | 1,022,102 requests in one hour, no unexpected failures | Immutable `e3f7b93` source hash unchanged; three 20-minute process lifetimes at 1/4/8 clients and 1/16/64 rows. Sampled p95 4.89/18.58/35.79 ms; graceful/forced restart startup 6.50/17.80 seconds. Deliberate downtime is separate from scored requests |
| Isolated RL/imitation profile | Seventeen tests passed locally; both remote Windows/Linux jobs passed | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; real PPO/BC/GAIL and trusted persistence on a separate compatible CPU stack |
| Linux targeted repairs | Five DDP tests passed; PDF, TorchScript and DEC/IDEC checks passed | Distributed CPU/Gloo checks use the CUDA-capable Torch installation; multi-GPU throughput is unmeasured |
| Expanded local Linux suite with CUDA/extras | Tests passed, coverage qualified: 84.88% (52,448 / 61,792 lines); 3,096 passed, 64 skipped | Immutable `d9af9b6` completed in 45:43. Installed FLAML integration skipped following an availability probe; an unreadable worker coverage-file warning also occurred. Not used to adjust floors. Strict installation-based FLAML gating and a bounded 90-second heavy-import probe are new repairs; the original probe failure's cause is unestablished. Exclusions remain 566 |

Exact local measurements are preserved in `reports/coverage90-gpu-linux.json`,
`reports/coverage90-soak-30min.json` and `reports/coverage90-linux-full.json`.
The broader local environment and the remote core environment have different
dependencies; their percentages are not interchangeable.

The `expanded-coverage` CI job reproduces the assessed dependency profile using
`requirements-expanded-coverage.txt` and its dated constraints. It runs all
tests against all BuildML modules, with an 84% floor, and uploads source,
installed package, coverage and test-result artifacts. CPU Torch replaces local
CUDA Torch in that job; its outcome must be verified independently. The core
floor remains 70%, and the 90% target remains outstanding.

## Correctness and coverage reach

Use coverage gaps to select workflows, then verify their outputs with independent
numerical or consumer checks. A subset percentage is not a whole-suite result.
The artifact hand-off batch ran 25 real-consumer cases on Linux and exercised
54 unchanged production lines missing from the passing `18f0199` remote baseline.
Changed/new executed lines are counted separately, with source hashes and line
mapping in the report. This does not project the next full-suite percentage.

```bash
python scripts/measure_coverage_reach.py baseline.json observed.json \
  --baseline-ref BASELINE_COMMIT --current-root EXACT_OBSERVED_SOURCE \
  --output coverage-reach.json
```

`EXACT_OBSERVED_SOURCE` must match the measured run, preferably an immutable
archive. The script reads the baseline from Git once and maps unchanged source
lines across edits. It neither combines coverage databases nor changes floors.
Profile differences and unmapped files must be reviewed before interpreting reach.

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

Import preflights isolate native crashes, log each module's start/completion,
and limit numerical-library threads. A 30-second import timeout permits one
bounded retry with a 90-second limit, with both attempts recorded. Missing or
crashed imports still fail, and numerical contract failures are never retried
into passing support. This addresses the observed Windows TDA import timeout;
the next remote run must verify the adjusted preflight independently.

## Scale, memory and bounded operational reliability

Statsmodels rolling evaluation updates filtered/smoothed states from prior
observations while preserving the originally fitted parameters. Historical
exogenous values stay aligned across validation and scoring windows. Tests
compare random-walk ARIMA/SARIMAX with independent origin formulas and ETS with
fixed-coefficient Holt/seasonal recurrences; future holdout changes cannot alter
earlier predictions. The M4 runner also accepts `--methods` to compare installed
industry methods on the same series/origins, with strict forecast alignment.

The local Windows continuous HTTP assessment at `ab968bc` completed one hour
with one server process, 1,057,585 requests and no unexpected errors. The three
loads used 1/4/8 clients with batch sizes 1/16/64; sampled p95 latencies were
4.72/17.47/33.54 ms. Source hashes matched before and after. This is evidence
for the tested local service and linear model.

The Linux continuous assessment at `9e6e039` also passed one hour with one
server process and unchanged source hashes: 1,006,128 requests, no unexpected
errors, and process RSS 204.77 to 208.65 MiB. Sampled p95 latencies were
4.78/21.49/39.94 ms. Concurrent coverage and M4 jobs ran on that host, so these
latencies are not a controlled Windows/Linux comparison. Longer operation,
other models and deployment populations remain separate evidence requirements.

The requested 24-hour paced HTTP checks at immutable `bc4bb17` completed
successfully on Windows and Linux (Ubuntu under WSL2). Each retained one server
process through three eight-hour loads: 1/4/8 clients with 1/16/64 rows per batch,
targeting ten requests per second per platform. Windows completed 861,372
requests; Linux completed 862,328. Both recorded zero errors, matching source
hashes before/after and graceful final shutdowns. Sampled RSS was 204.78 to
212.77 MiB on Windows (213.02 maximum), and 202.30 to 206.80 MiB on Linux.
These are local linear-model service results for the recorded dependency stacks,
not a months-long SLA or evidence for every deployment and model.

Corrected thread/process collection at `58e4309` passed remotely at 84.06%
(51,817 of 61,645 executable lines), with 2,723 tests passed and 82 skipped.
The merge tree equals the source tree. Its entire CI, backend matrix and release
acceptance passed. Expanded floor 83% retains the required headroom; the core
floor remains 70%, based on its separately measured profile.

The same immutable source passed locally on Linux at 84.12%
(51,854 of 61,645 lines), with 2,745 tests passed and 60 skipped in 27:30.
CPU/CUDA contracts passed. Coverage exclusions remain 564. Reaching 90% still
requires 3,627 additional covered lines at this measured denominator. Subset
coverage gains must not be added to these complete measurements.

Serving configuration supplies omitted arguments. Explicit arguments override
it, including `trusted=False`, a loopback host and CPU placement. Permission
flags accept booleans only. Agent training options are checked at the dispatch
boundary, with real NGBoost and Torch fitting checks for numerical/device paths.

Agent integer fields refuse boolean, fractional and non-finite arguments before
Session mutation. Checkpoint partition sizes follow the same rule. Fault-injected
sidecar writes verify canonical data, role and split recovery; missing agent
bundle paths are checked across every registered save/load bundle tool.
DuckDB fallback writes release unique temporary registrations after SQL failure
and preserve caller registrations. Both Parquet fallback paths materialize Arrow
data; they do not establish bounded-memory streaming.

One older Torch CI job aborted in native code during buffered two-rank validation.
Ten local repetitions passed on Torch 2.7.1, whereas the failed CI stack used
2.14.1. This mismatch does not establish a repair. Torch CI now repeats the real
check on each installed Python/Torch stack, retains every attempt and records
native diagnostics. A failed or skipped attempt fails the check without retries.
The reporter defaults to `INFO` diagnostics: `DETAIL` reproduced a native
Torch 2.14.1 segmentation fault while constructing DDP, including in a minimal
Torch-only program. `INFO` passed that independent probe. `--distributed-debug
DETAIL` remains available to reproduce the diagnostic-mode failure. This does
not establish the cause of the older intermittent abort without that setting.
The corrected `INFO` check passed ten fresh attempts locally on Torch
2.14.1 CPU and ten each remotely on Python 3.11/3.12 with Torch 2.14.1 CUDA-build
wheels (CPU Gloo execution). These are bounded reproductions, not a general
distributed-training reliability claim.

A subsequent Python 3.11 CI repetition again aborted natively on attempt three
(`39067808325`). The intermittent failure remained unresolved at that source. The repeat checker
now enables `BUILDML_DDP_TRACE=1`, recording each worker's initialization,
training, result persistence and process-group teardown in its attempt log.
These traces help locate failures; they do not suppress them or establish a fix.

The traces at `95566fb` show both ranks returning after successful training and
process-group destruction before a native abort. Torch CI now captures bounded
native cores and extracts debugger backtraces to identify the failing library.
The expanded whole-package job at that source passed independently at 84.09%
(51,850 / 61,661 lines; 2,749 passed, 82 skipped), but its complete workflow failed
in the Python 3.11 Torch job. Its tested merge `edc10a96` has the same source tree.

Local cold startup exposed another boundary: importing capability workers loaded
the full Session API. Package-level Session loading is now lazy, preserving
`from buildml import Session`. All 14 previously failing/erroring local cases
passed in the subsequent 22-test immutable Linux repair suite at `0d51d53`,
including ANN fault containment, NGBoost, TDA and the anomaly example. This
targeted result does not replace a whole-package coverage measurement.

Native cores from subsequent Python 3.11 failures locate the abort in
`ProcessGroupGloo::AsyncBroadcastWork` destruction: a Torch tensor decrement
tries to acquire the GIL after CPython finalization begins. A Torch-only probe
confirms that Torch 2.14 can retain Gloo threads even after model deletion,
garbage collection and `destroy_process_group()`.

Single-node Linux Gloo now uses PyTorch's supported forkserver launcher. Its
normal worker exit path avoids CPython finalization; the parent joins every
rank before reading the serialized result. Deliberate rank-factory failures
must still fail, without retries. Six local contracts pass on both Torch 2.7.1
and Python 3.11.17/Torch 2.14.1 CPU, including numerical model use, validation
with buffers, caller RNG preservation and worker failure propagation. Repeated
verification passed 50 attempts locally on Python 3.11.17/Torch 2.14.1 CPU
and 50 on each remote Python 3.11/3.12 stack in CI run `39079847470`,
without retries or skipped attempts. The complete workflow passed. This is a
launcher workaround for the observed native lifetime issue, not an upstream
Torch repair or a general distributed-training reliability guarantee.

Follow-up runtime repairs at `f8de1ea` passed 224 Linux contracts with coverage
instrumentation (one Giotto-profile skip). They include all previous failed cases,
weighted blending and error propagation, public exports, TDA row identity and
holdout isolation, probe recovery and bounded graph queries. Session no longer
probes optional clustering backends on import; one local import measured 5.28
seconds versus the earlier 16.68-second profile during concurrent coverage work.
These timings describe those local runs, not a general throughput benchmark.

Categorical EDA checks use independent SciPy association values. Cramér's V
is uncorrected for finite-sample bias and remains an exploratory effect size,
not a significance test or causal claim. Degenerate tables yield no effect
estimate; the documented eight-column screening cap is tested explicitly.

The isolated AutoGluon 1.6.0/PyArrow 24 CPU profile passed eleven real contracts:
binary and multiclass classification, regression, train-only model fitting,
weighted F1, signed-loss normalization and internal validation score reporting.
GBM seed hints reach the backend. This adapter limits base models to LightGBM;
it does not exercise the full AutoGluon model zoo. The compatible CI job uses
`requirements-autogluon-verification.txt` and refuses skipped contracts.

AutoGluon bundle checks relocate the bundle and remove the original model
directory before comparing probabilities. Bundles include the backend's model
files and cached training artifacts. Untrusted deserialization, escaped asset
paths and recursive destinations inside the predictor directory are refused.

Tabular SSL tools expose native Torch methods and explicit training budgets.
With a fixed random_state, Torch initialization and augmentation noise are
seeded as well as NumPy shuffling. Invalid budgets and non-finite training
matrices are refused before model construction. Real tests compare encoders
after changing only held-out rows, on CPU and CUDA. They do not establish
that every method improves prediction accuracy on every population.

```bash
python scripts/verify_operational_limits.py --rows 100000 1000000 --soak-seconds 120
# Bounded child: Linux address space or Windows Job Object allocation limit:
python scripts/verify_operational_limits.py --memory-budget-mib 2048
python scripts/verify_operational_limits.py --gpu
# Longer local-service check with limited traffic and memory sampling:
python scripts/verify_http_reliability.py --continuous --seconds 86400 --requests-per-second 10 --rss-sample-seconds 60
```

The paced rate is shared across clients, with small scheduling bursts possible.
Latency samples exclude pacing sleeps. A 24-hour run keeps the same process
through three eight-hour loads; its progress is saved each minute. Use an
immutable source snapshot so ongoing development cannot alter the assessment.

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

Cost-sensitive threshold tuning minimizes empirical policy cost across all
observed probability breakpoints and all-positive/all-negative policies. It
retains the estimator's positive class identity; FP/FN counts and supplied
benefits are checked against independent policy enumeration, including tied
scores. This is a validation-partition optimum, not a guarantee of future cost.
Confirm the selected fixed policy on untouched test data and monitor prevalence.

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

Fold-local selectors now use the training target type rather than a cardinality
cutoff. Explicit F-test scorers are honored, including many-class classification.
The default scorer is `auto`. CV and search carry their resolved task into
automatic selection, including regression with integer outcomes. Direct
preprocessing treats integers as discrete; choose `f_regression` and a regression
estimator when those integers represent a continuous quantity. Explicit choices
are preserved, including values stored in older recipes. A selector retaining
one feature when thresholding would discard all features remains serializable.
Non-finite numeric targets are refused before supervised selection.

Fold-local date extraction parses valid mixed formats consistently per value,
so a row's calendar features do not depend on other rows in the prediction
batch. Mixed timezone offsets require explicit normalization. Ambiguous numeric
date formats still require the caller to choose a date convention.

Native columnar previews pass their row budgets into Polars Parquet/IPC readers,
rather than reading the full table and trimming it afterward. Design preparation
converts only requested columns and materializes repeated names once, preserving
first-occurrence order. Sampling still changes the fitted population, and classical
training still requires an in-memory Pandas/NumPy matrix. A configured hard memory
gate refuses the materialized frame; it is not a guarantee against allocation-time OOM.

Torch search and cross-validation seed model construction before each fold's
training. CPU tests compare initial weights and scores after changing the ambient
Torch RNG, including dropout. Custom factories' external randomness and GPU
bitwise determinism are separate concerns. Device-string validation precedes
hardware fallback, and internal seeded-sampler failures propagate. The combined
Linux repair check passed 115 tests with three expected dependency-state skips,
including actual imodels RuleFit execution under installation-based test gating;
this targeted check does not replace a complete-suite coverage measurement.

Current expanded coverage evidence (2026-10-19, source `3d35017`): remote Linux
passed at **85.0410%** (52,580 / 61,829 lines), with 3,125 passed and 91 skipped.
[CI run 39326055971](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39326055971)
passed, as did backend evidence and release acceptance. Tested merge `a75c6d26`
and source trees are equal. This supports expanded floor **84%** with the required
one-point headroom. Scope and 566 excluded lines are unchanged; core floor stays 70%.
Local Linux at the same source passed at **85.1397%** (52,641 / 61,829 lines),
3,153 passed and 63 skipped in 24:29, without coverage collection warnings.
Installed FLAML integration passed. Six new ONNX consumer cases skip remotely
because that profile lacks their dependencies; they pass locally. These records
supersede earlier latest-measurement statements above; historical results retain
their original floors and qualifications. Reaching 90% requires 3,067 further
covered lines at the current remote denominator; subset percentages are not added.

Latest remote whole-package result at `02a9337`: **85.1201%** (52,640 / 61,842),
3,185 passed and 91 skipped. [CI run 39354834562](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39354834562)
and both companion workflows passed. Tested merge `8eec7730` and source trees
are equal. Core/expanded floors remain 70%/84%, exclusions remain 566; 90%
requires 3,018 additional covered lines at this denominator. The latest local
whole-package measurement remains the separately recorded `3d35017` result.

RAG bundle restoration now requires a reconstructable hashing or named
sentence-transformer embedding space. Callable, unknown or absent backend
identities are refused rather than replaced with hashing. Rebuild older bundles
marked callable with the original embedder. Explicit built-in HashingEmbedder
objects now record the hashing backend. Loading also checks stored vector width
against metadata and refuses non-finite values before normalizing vectors.
