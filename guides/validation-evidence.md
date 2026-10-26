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

Maintained assessment, updated 2026-10-26, review branch `codex/coverage-cuda-ci-20261012`:

| Configuration | Outcome | Qualification |
|---|---|---|
| Remote core Linux, Python 3.10-3.13 | All core jobs passed: 71.28-71.29%; 2,172 passed, 398 skipped per job | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; corrected thread/process collection permits core floor 70% under the one-point headroom rule |
| Remote backend Windows/Linux matrix | Both jobs passed, including real HTTP recovery and continuous probes | [Backend run 38589015595](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015595), `2837fb3`; real backend groups and bounded allocation checks; skips remain visible |
| Remote release acceptance | Passed | [Acceptance run 38589015461](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015461), `2837fb3`; verification does not publish a release |
| Current accepted remote expanded CPU whole-package gate | Passed:87.7196% (54,930/62,620);6,838 passed,77 skipped | [CI40265925278](https://github.com/TechLeo-Libraries/BuildML/actions/runs/40265925278), source `2c8b7c4`, tested merge `38229adc`; trees match and all three workflows passed.566 exclusions unchanged;floor86 retained from passing87.0013% source a05ec41;90% gap1,428 lines. Ranking cutoff/averaging repair accepted with coverage7.16.2;2781 pytest warnings, no coverage collection warnings |
| Remote expanded CPU whole-package gate, before profile expansion | Passed: 85.2993% (52,773 / 61,868); 3,305 passed, 91 skipped | [CI run 39376275463](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39376275463); merge `a373d1fe` tree equals `4fcceb2`; all three workflows passed. NumPy 2.5.4/SciPy 1.18.1 profile; broader profile subsequently passed at 85.4319% (`0b31402`); scope/exclusions and floor 84% unchanged |
| Remote EconML and compatible causal profiles, Windows/Linux | All four jobs passed | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; real estimation/refutation, graph-name, policy and persistence contracts |
| Local Linux CUDA | BuildML regression fit/evaluation passed | Quadro RTX 3000, Torch 2.7.1+cu126, CUDA 12.6; five epochs, finite metrics, R² 0.98486; peak Torch allocation 29.38 MiB, excluding driver/context allocations |
| Local Linux EconML 0.17.0 policy learner | Sixteen known-benefit and reporting contracts passed | Continuous/binary outcomes, numeric/string treatment labels, two simulation seeds and requested bootstrap counts; adjusted population summaries and omitted intervals are explicit. Repaired behavior also passed remote Windows/Linux jobs |
| Isolated Linux DoWhy 0.14 | Estimation/refutation passed; four name and bundle contracts passed | NumPy 2.4.1 / SciPy 1.15.3 / scikit-learn 1.9.1 / pandas 3.0.6; graph node names retain spaces, punctuation, quotes and Unicode. This smaller causal-only profile passed remote Windows/Linux jobs; DoWhy is now also included in the broadened full expanded profile, whose remote acceptance passed at `0b31402` |
| Isolated Linux giotto 0.6.2 | Fifteen real topology tests passed | NumPy 1.26.4 / SciPy 1.11.4 / scikit-learn 1.3.2 / Arrow 20; real Arrow execution, diagram-axis integrity, bounded Mapper sampling and true igraph node/edge counts. This is a separate compatible profile, not a full-suite measurement |
| Concurrent ASGI soak | 358,036 requests in 30 minutes, zero errors | Four workers; sampled p95 29.21 ms; peak RSS 205.49 to 210.80 MiB; bounded latency sampling; in-process evidence, not a deployment SLA |
| Real Windows HTTP recovery assessment | 1,022,102 requests in one hour, no unexpected failures | Immutable `e3f7b93` source hash unchanged; three 20-minute process lifetimes at 1/4/8 clients and 1/16/64 rows. Sampled p95 4.89/18.58/35.79 ms; graceful/forced restart startup 6.50/17.80 seconds. Deliberate downtime is separate from scored requests |
| Isolated RL/imitation profile | Seventeen tests passed locally; both remote Windows/Linux jobs passed | [CI run 38589015544](https://github.com/TechLeo-Libraries/BuildML/actions/runs/38589015544), `2837fb3`; real PPO/BC/GAIL and trusted persistence on a separate compatible CPU stack |
| Linux targeted repairs | Five DDP tests passed; PDF, TorchScript and DEC/IDEC checks passed | Distributed CPU/Gloo checks use the CUDA-capable Torch installation; multi-GPU throughput is unmeasured |
| Historical qualified local Linux run | Tests passed, coverage qualified: 84.88% (52,448 / 61,792 lines); 3,096 passed, 64 skipped | Immutable `d9af9b6` completed in 45:43. Installed FLAML integration skipped following an availability probe; an unreadable worker coverage-file warning also occurred. Not used to adjust floors. Strict installation-based FLAML gating and a bounded 90-second heavy-import probe are new repairs; the original probe failure's cause is unestablished. Exclusions remain 566 |
| Local Linux original-profile whole suite | Passed: 85.3947% (52,832 / 61,868); 3,333 passed, 63 skipped | Immutable `4fcceb2`, 25:31; no coverage collection warnings; same source and package scope as the broadened-profile comparison |
| Local Linux broadened-profile whole suite | Passed: 85.5192% (52,909 / 61,868); 3,337 passed, 59 skipped | Immutable `4fcceb2`, 24:50; no coverage collection warnings. NumPy 2.4.1/SciPy 1.15.3/mlxtend 0.23.4 with DoWhy 0.14; 80 lines gained, 3 lost, net 77. Five positive DoWhy cases execute; its missing-package case skips |

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



RAG bundle restoration now requires a reconstructable hashing or named
sentence-transformer embedding space. Callable, unknown or absent backend
identities are refused rather than replaced with hashing. Rebuild older bundles
marked callable with the original embedder. Explicit built-in HashingEmbedder
objects now record the hashing backend. Loading also checks stored vector width
against metadata and refuses non-finite values before normalizing vectors.


The broadened expanded manifest matches the recorded passing local environment:
124 selected requirement/constraint checks pass. It includes ONNX Runtime 1.31.0,
which was already used by the passing local ONNX consumers, so those checks can
execute remotely. This does not assert that the new remote job has passed.

The paired full runs used the same immutable `4fcceb2` archive and unchanged
statement/exclusion sets (61,868 statements; 566 exclusions). Coverage databases
were not combined. Only the dependency profile changed. The original runtime
inventory stayed unchanged; changed packages ran in a separate overlay.
The latest remote result is85.4319% on the broadened profile (52,855/61,868; 3,316 passed,80 skipped), CI39396524174. The remote90% gap is2,827 lines. The
broader local result is 85.5192%; reaching 90% still needs 2,773 covered lines at
that local denominator. Core 70% and expanded 84% floors stay unchanged: this
result does not provide the one-point headroom required for an 85% floor.


### Seeded Torch semi-supervised refits

The FixMatch/MixMatch-style tabular adapter validates finite feature matrices,
aligned integer action codes, and training controls before fitting. Seeded runs
isolate Torch initialization and stochastic training from ambient RNG state;
NumPy shuffling/augmentation use the same requested seed. Fitted module, class
mapping, feature width and pseudo-label count publish together after success.
This promises repeatability on the tested CPU stack, not bitwise equality across
Torch releases, devices or CUDA kernels. Poisoned-holdout checks compare the
train-only Session path with a separately fitted training population.

Latest accepted remote whole-package measurement: `38e6d5a`, 85.4298%
(52,858 /61,873), 3,328 passed and80 skipped, CI39403400421. All three
workflows passed; merge/source trees match. The90% gap is2,828 lines at this
source; floors70/84 and566 exclusions remain unchanged. The semi-supervised
changes above require their own subsequent full CI measurement.


### Diagnostic populations and segment identity

Calibration and threshold reports refuse scored target labels that are missing
or absent from the classifier's fitted classes. They retain numeric class
equality across equivalent integer/float dtypes; formatting a class for display
does not define positive membership. Segment accuracy and positive counts use
the same class identity. Segment labels quote reserved values and delimiters,
keeping missing values separate from the literal `<NA>` and preventing compound
slice values from collapsing into one group. These are observational diagnostics;
they retain the existing small-sample and multiple-slicing disclosures.

Latest accepted full-suite result is `a862a5b`:85.4312% (52,893/61,913),
3,350 passed,80 skipped. [CI39410559363](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39410559363), backend evidence and release acceptance all passed.
Tested merge `e524f780` and source trees match. The exact90% gap is2,829 lines
(`ceil(0.9*61913)-52893`). Earlier handwritten remote gaps for `0b31402` and
`38e6d5a` were corrected to2,827 and2,828 respectively; percentages were unchanged.
Diagnostic changes require a fresh whole-suite measurement.


### Resampling provenance

Random over/undersampling gathers complete donor rows using imbalanced-learn's
positional sample indices. Repeated or non-range pandas index labels do not
alter row selection. Auxiliary identifiers, groups, timestamps, weights,
ignored columns and their dtypes travel with the selected donor; validation
and test remain unchanged. Weighted-classifier consumers compare the resulting
fit with a separate sklearn fit on the actual donor rows and weights.

SMOTE, ADASYN and Borderline-SMOTE now require a feature/target-only dataset.
They refuse auxiliary metadata because interpolated feature vectors do not
supply justified identifiers, group membership, timestamps or weights. Choose
random resampling to retain complete donor records, or deliberately prepare a
feature/target-only dataset for synthesis. No-feature datasets are refused.

Latest accepted remote full-suite source `e507442`:85.4582% (52,926/61,932),
3,365 passed,80 skipped. [CI39417610204](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39417610204), backend evidence and release acceptance passed;
tested merge `b9e53433` and source trees match. Exact90% gap:2,813 lines.
The resampling changes need their own subsequent whole-package measurement;
core70/expanded84 floors and566 exclusions remain unchanged.


### Outlier numeric input contracts

Outlier fitting requires positive finite numeric fence controls, excluding
booleans. Fitting, flagging, capping and score-time replay refuse infinite
values and numeric conversions that would discard non-null values. Genuine
missing values remain missing and valid numeric strings can be scored. A
score-only dropping plan retains all rows and uses its frozen fences as caps,
with the existing warning; fitted plans are not modified during replay.

Numerical consumers independently compute IQR and population-standard-deviation
fences from train positions, inspect duplicate-index partition remapping,
and compare score-only clipping with NumPy. The outlier changes have targeted
evidence; they need a new full-suite measurement beyond accepted `8fbfe07`.


### Feature-selection estimator ownership and controls

Model-based selection fits a clone of a caller-supplied sklearn estimator.
Pre-existing fitted state remains unchanged, and an unfitted caller remains
unfitted. Variance cutoffs must be finite nonnegative numbers, univariate
feature counts must be positive integers, and score-function names must be
recognized. Protected-only candidate lists are refused before sklearn fitting.

Train-only consumers compare population variances and correlation-derived F
statistics with independent NumPy calculations, and exercise multiclass and
regression model importances, coefficient handling, zero-coefficient fallback,
mutual-information task dispatch and protected-column replay. This does not
establish that selected features are causal or that task-type heuristics fit
every population. Tune selection within the appropriate training CV folds.


### Frozen binning output contracts

Binning refuses generated output names that already exist in the scoring
frame, including identifiers, targets, weights and other features. It checks
all outputs before mutation. Both ordinal and one-hot output preserve genuine
missingness; a missing source leaves every one-hot indicator missing instead
of implying a valid all-zero encoding. Numeric coercion retains the existing
lenient policy for non-numeric input, which is disclosed as missing.

Only finite training values determine uniform or quantile edges. Scoring
extremes are absorbed by the frozen open-ended bins; they do not change edges.
A column with no finite training support is refused. Bin counts require integers
of at least two. Independent searchsorted/indicator checks cover ties at bin
boundaries, missing rows, duplicate indices and constant train columns.


### Real active-learning host integration

Windows Python3.12 with scikit-activeml1.0.1 passed16 required host consumers
and15 existing industry checks. The adapter supplies numeric missing-label
arrays, wraps fitted sklearn committee members without refitting, and selects
utilities for the pool positions rather than including labeled rows. It asks
for one scoring round because only initial pool priorities are returned; this
does not implement iterative greedy batch selection.

Host CoreSet scores are distances to the nearest labeled row; with no labels,
the host uses negative sums of distances to the pool. Host QBC KL is the mean
KL divergence from each member to the mean probability distribution. These
cold-start/KL definitions differ from the documented native fallback (normalized
norms and mean pairwise KL). Host success and native fallback remain explicitly
disclosed; do not compare their raw scores as interchangeable measures.

Independent NumPy distance, vote-count and probability-disagreement oracles
passed for binary/multiclass committees, labeled/empty-labeled populations and
pool order. Tree values remain unchanged. Distance tolerances allow the observed
4.2e-8 self-distance roundoff in the host's pairwise kernel. No query scores are
claimed to establish improved downstream model accuracy or label efficiency.

The candidate expanded profile now pins scikit-activeml1.0.1 and makefun1.16.0
and requires real host imports before full testing. Six direct metadata checks
match the recorded Linux baseline dependencies; Windows pip check passed.
Neither check is a Linux full-suite result. This profile and its fixes subsequently passed the complete remote Linux gate
at `a87e8e6` (CI39462402101); subset reach remains recorded separately.


### Multi-task target identity and population disclosures

Classification encoders retain original class values, including numeric-looking
strings such as `001` and `1.00`; decoding converts NumPy scalar containers
without parsing label text as numbers. Incompatible mixed class-label types
are refused instead of collapsing their string representations. New encoder
metadata uses serializable Python scalar classes. Older encoders preserve the
class values actually stored in their bundle; refit if original types were lost.

Uniform and mixed regression targets must be numeric and finite. Mixed target
blocks reject nulls before encoding and require valid head-kind metadata for
every requested target. These checks supplement the existing train-population
validation. Evaluation still applies its documented missing-target-row policy.
Train/all evaluations explicitly identify in-sample metrics; they are not
presented as unseen holdout estimates. Real sklearn predictions and Torch
mixed-head residual/accuracy checks verify the reported metrics independently.


### EconML refutation population and estimand

Policy-tree refutation uses DRPolicyTree with the same nuisance model family
and tree controls as policy fitting. It refits on the perturbed training
population and reports mean doubly robust treatment effects. A raw difference
in observed treated/control outcomes is not substituted for the adjusted ATE.
Placebo treatment permutes training treatment codes; random-confounder checks
append seeded noise to the declared train confounders.

Real DML, causal-forest and policy-tree consumers test both refutations on a
confounded simulation with known constant effect. Poisoned holdouts and duplicate
indices check population boundaries; original fitted predictions and source
frames remain unchanged. The observed raw arm contrast is over one unit from
the true effect, so recovering the adjusted effect is consequential. Placebo
and sensitivity checks remain diagnostics under declared assumptions; they do
not establish identification or valid effects for arbitrary deployments.


### LightFM sparse side-feature consumers

LightFM side-feature preparation is checked using real SciPy CSR matrices and
independent per-entity means, population standard deviations and matrix-vector
products. Missing/non-finite or complex numeric inputs are refused before
conversion; numeric side features require explicit preprocessing when missing.
These checks verify matrix preparation, not LightFM training, recommendation
quality or availability of the optional compiled model runtime on every host.

### SSL head class identity consumers

New frozen SSL heads retain raw numeric or text class identity and JSON-safe
class summaries. Independent accuracy counts include integer/float-equivalent
numeric labels and distinct numeric-looking strings. Mixed incompatible class
types are refused, as are unseen evaluation labels; unlabeled-only partitions
return empty metrics. Legacy string-label head fixtures remain evaluable after
serialization with a warning to refit for raw class types. Legacy labels cannot
recover type information discarded when those heads were originally fitted.

### Decision cost accounting consumers

Cost-matrix decision consumers calculate each action's expected loss from
predicted class probabilities and a named asymmetric cost table. They check
class-axis permutations, selected actions, held-out realized cost and accuracy.
For this method, fit `expected_cost` is the total probability-weighted loss;
observed realized totals and means remain separate metrics. Earlier saved
plans retain their serialized fit summary; refit the policy to refresh it.
These calculations do not establish calibration of the underlying probabilities.

### Real serving CLI consumers

Real serving CLI consumers launch a fitted linear-model bundle on temporary
loopback ports and compare HTTP predictions to the known function `2*x+3`.
Unauthenticated, API-key and Basic-auth configurations exercise the actual
blocking CLI, HTTP authorization and graceful shutdown. Tests observe the real
Uvicorn server solely to stop their owned instance; transport and prediction
are not mocked. Startup logs must omit credentials and the port must close.
This bounded integration check does not establish a long-term service SLA.

### Automatic ARIMA selection consumers

Automatic ARIMA selection is checked against an independently enumerated
restricted statsmodels AIC grid, with forecasts compared to a direct fit of
the winning order. Explicit orders are also compared to direct fits and are
disclosed as supplied, without claiming a search ran. This synthetic numerical
regression check does not establish real-workload forecasting superiority.

### Causal partition bootstrap accounting

Causal partition `bootstrap_samples` reports performed resamples. Native
fixed-nuisance bootstrap quantiles and standard deviation are checked against
independent NumPy resampling at zero, one and twenty draws. DoWhy and EconML
partition bootstraps are unavailable: positive requests produce an explicit
warning, zero performed samples and no interval. This does not add uncertainty
support or establish nominal coverage for those backends.

### DoWhy partition effect consumers

DoWhy holdout consumers assign different treatment effects to train, validation
and test rows. Both partition estimation and evaluation must match an independent
NumPy least-squares treatment coefficient on the requested rows. Changing only
training outcomes after fitting must not change that holdout re-estimate.
DoWhy fits a fresh backdoor model on that partition; this differs from native
and EconML scores using train-fitted nuisances. Declared identification
assumptions remain necessary, and these checks do not validate confidence
interval coverage or identify effects in arbitrary deployment populations.

### Dashboard zero observations

Dashboard gate context distinguishes zero observations from unavailable values.
An observed zero complete-case count, regression median, analysis-row count or
anomaly measurement remains zero; absent values may use documented legacy
aliases. Real EDA frames and independent Pandas counts check this translation.
Readiness findings remain analytical aids and do not establish causal effects.

### LangChain passage-budget consumers

LangChain adapter consumers record prompts from a real offline QA chain and
verify that the positive integer `k` cap limits actual input documents,
recorded context and citations together. The framework's FakeListLLM supplies
fixed responses: this verifies integration and context accounting, not LLM
answer quality or grounding. Faithfulness remains unset and token usage empty.
The accepted expanded profile pins classic1.0.8/community0.4.2/core1.6.12 and
requires real imports; Linux CI39629685433 passed86.0131%. The installed
community package emits a maintenance deprecation warning; this result does
not remove that upstream lifecycle limitation.

### Session SHAP attribution consumers

Session-level SHAP consumers independently select train, validation and test
rows, reproduce capped sampling, and verify feature order and importances.
Linear-model attributions match known coefficients times centered feature
values; tree attributions also reconstruct predictions from base values plus
feature contributions. Duplicate input index labels are included. These tests
verify model attribution, not causal interpretation or feature-policy effects.

### DuckDB native-reader ownership and previews

DuckDB native-reader consumers exercise CSV, TSV, Parquet, Arrow file and
Arrow stream inputs at full, zero and three-row previews, including nulls,
quoted paths and embedded delimiters. Relation handles remain lazy; Arrow
bridges first materialize an Arrow table, disclosed separately from scanning.
Failed readers close their owned connection. This bounded check does not
establish out-of-core classical training or memory-pressure reliability.

### Synthetic sample quality validation

Synthetic sample quality checks explicitly detect invalid numeric values and
fractional values in modeled integer columns. Missing values remain subject
to the separate null-rate check; coercion failures cannot silently pass as
missing observations. Means use finite observations. Rate tolerances are
bounded to [0,1], range slack is finite and non-negative, and duplicate columns
are refused. Independent frequency counts verify OOV, range and null metrics.
These are schema and sample-quality screens, not privacy or leakage audits.

### Online cumulative refit

Disclosed online refit fallback is limited to reconstructable Session train
rows. It fits a fresh batch estimator and preserves the prior plan's estimator.
External-frame refits are refused because prior external batches are not
retained for full retraining; use native `partial_fit` or retain an explicit
complete batch dataset. Independent Ridge and logistic-regression fits verify
cumulative train updates, overlap handling and holdout exclusion. This does
not establish transactional behavior for every native incremental backend.

### Recommender bundle restoration

Direct legacy recommender plans and separate-state payloads both rebuild absent
user/item indexes from stored catalog order. Real fitted item/user KNN, SVD,
NMF and content models are checked after restoration, including cold users and
train-item exclusion. Factorization scores also match independent NumPy dot
products. Malformed or non-object metadata raises a validation error before
deserialization. This does not grant trust to pickle artifacts: loading still
requires explicit `trusted=True`.

### SB3 interaction accounting and partial runtime readiness

SB32.7.0 with Gymnasium1.2.3 supports the accepted expanded NumPy2/Torch2.7
requirements according to six direct metadata checks. It is a separate runtime
from the older imitation1.0.1 profile. Explicit `industry/gym_sb3` readiness
requires working Gymnasium and SB3; BC/GAIL readiness still requires imitation.
The complete `rl-industry` availability flag continues to describe that full
stack. Modern SB3 is included directly in the accepted assessment profile without
asserting that the incompatible imitation profile can be merged with it.

Training reports actual executed environment interactions from the real model
counter and preserves the requested count separately. PPO, DQN and A2C can
finish rollout blocks past that request; this is disclosed and warned rather
than presented as a hard resource cap. Session training-row counts for online
SB3 reflect executed interactions, not Session tabular rows. Evaluation requires
positive integer episode/step counts and returns observed episode statistics.
The action-score vector remains a one-hot action indicator, not probabilities.

Real-model consumers replay deterministic seeded Gym episodes independently,
compare saved/reloaded model actions and verify actual rollout counters. They
are bounded functional checks, not evidence of policy superiority, convergence,
robotics support or long-running environment reliability. The SB3 profile passed full Linux CI39490465704 at source `b833554`:85.7786%,
3,560 passed and80 skipped. It remains separate from the imitation profile.


For an SB3-only setup, the assessed dependency selection is
`pip install "buildml[rl]" stable-baselines3==2.7.0`. The expanded CI constraints
pin the rest of that profile. The separate imitation1.0.1 verification file
retains its own compatible dependency stack. A measured Windows import under
coverage took19.34s, beyond the former12s generic probe; SB3 now shares the90s
bounded scientific-import policy, with import failures and transient negative
caching still preserved. Direct uninstrumented import measured10.55s; these are
observations from this machine rather than universal startup expectations.


### Maintained SkopeRules consumer verification

The adapter prefers imodels 3.0.4 SkopeRulesClassifier, already present in the
expanded profile; legacy skrules remains a fallback whose runtime was not verified
in this batch. The expanded job now requires the maintained class import. Four
real consumer cases use two seeds and feature names containing spaces, `and`,
and reversed positional aliases. Every exported positive-class mask matches an
independent Pandas query of the original miner rule; confidence matches its
reported precision. Unsupported or unknown conjuncts raise ValidationError.
RuleFit positional aliases remain covered by existing real-consumer tests.
Local targeted result: 23 passed, one expected missing-extra guard skip, 786
warnings (784 upstream sklearn RuleFit warnings and two flat API deprecations).
Subset reach against source 1c3a427: 39 previously missing unchanged production
lines and 16 changed/new executed lines. This is not aggregate coverage or
predictive superiority evidence. Remote whole-suite acceptance subsequently passed
at c4adda9 (see current accepted measurement above).


### DuckDB chained positional masks and fallback

A second SQL mask previously reused the source view name `s`, producing an
infinite-recursion BinderException. Distinct source-view identifiers preserve
earlier relations on the shared connection. Masks now require Python or NumPy
booleans; strings, numbers, None and NaN are refused instead of using truthiness.
Eleven new cases plus existing identifier, mask and lifecycle consumers passed:
33 passed, no skips or warnings. Independent Pandas iloc selections agree with
real DuckDB results for single and chained masks, including user `_bml_rn`
columns, tuple masks and NumPy booleans; the source relation remains unchanged.
An import fault is injected only at the adapter's direct Arrow imports to verify
its real Pandas fallback. DuckDB and Arrow remain installed; this is not a
verification of an environment without Arrow or the separate Arrow-filter fallback.
Subset reach against verified c4adda9 is 14 previously missing unchanged lines
and three changed/new executed lines. Scoped mypy and Ruff pass. Full remote
whole-package coverage for this DuckDB batch passed at a3fd9e7.


### Autonomous failure and recovery consumers

Eight new real Session cases exercise missing estimator arguments, invalid split
values and fitting before splitting with stop_on_error both enabled and disabled.
Each failure has one audit error; stopped runs leave split/model state untouched.
Continued runs split and fit a real linear model whose holdout predictions match
an independent sklearn LinearRegression fit on the same training partition.
Skipped steps consume the iteration budget, preventing a later fit, and an
allowlisted destructive step is blocked before changing the dataset.
No product defect was found. An initial test incorrectly treated argument
validation as pre-confirmation schema enforcement; the registry resolves tool
names, while these argument errors are reported by execution.
Combined new and existing autonomy tests: 42 passed, no skips, one upstream
LangChain community maintenance warning. Ruff passed. Subset reach against
verified a3fd9e7 is 15 previously missing unchanged production lines, zero
changed/new lines. Production source is unchanged; this does not establish
planner quality, a complete sandbox or long-term service reliability.
Whole-package remote acceptance of these autonomy tests passed at 8e8d38a.


### Offline semi-supervised text consumers

Automatic HF text selection now accepts Pandas object and string dtypes; explicit
text_column selection remains supported. Six new cases cover both missing labels
and an explicit unknown marker. A locally initialized tiny BERT/SentenceTransformer
encoder is saved and reloaded; no model weights or datasets are downloaded.
Independent float64 embeddings plus sklearn SelfTrainingClassifier and logistic
regression reproduce probabilities, accepted pseudo-labels and holdout accuracy.
Replacing only held-out text and labels leaves coefficients and transduction
unchanged. Trusted bundles created by the tests preserve holdout predictions.
The Session target remains unchanged throughout. This verifies plumbing and
partition isolation, not useful embedding quality or superiority on real text.
The six new and eight existing semi-supervised tests passed locally:14 passed,
no skips or warnings. Ruff, scoped mypy, dependency and YAML checks passed.
Nine text-runtime dependencies were installed locally without replacing existing
packages; the expanded Linux profile already pins sentence-transformers6.1.0
and transformers5.19.0. CI now requires a usable SentenceTransformer import.
Subset reach against verified 8e8d38a:47 previously missing unchanged production
lines and one changed/new executed line. Remote whole-suite acceptance of this
text batch passed at be8fc01.


### Native first-order MAML outer-update correctness

The native fallback previously backpropagated query loss on a deep-copied
adapted network, then stepped an optimizer owning the disconnected base model.
Consequently the initialization remained unchanged even though support adaptation
and episodic query scoring ran. It now transfers detached query gradients to
base parameters before the outer Adam update. Using autograd.grad avoids adding
the final support gradients still held by the adapted network; the inner-update
Jacobian is deliberately omitted for first-order MAML.
Four real Torch cases (two seeds, one or three inner steps) reproduce the outer
update with independent explicit support gradient descent and Adam's analytical
first-update formula. All parameters and query accuracy agree. A real
insufficient-data episode reports no training score and an explicit warning.
New consumers plus existing Session meta-learning checks:15 passed, one expected
installed-Torch missing-extra guard skip, no emitted warnings. Ruff and scoped
mypy passed. Subset reach against verified be8fc01:four previously missing
unchanged lines and three changed/new executed lines. The complete MAML suite
passed at aca522a after correcting the native-import test hook.
This is a correctness repair, not evidence of
few-shot superiority, second-order MAML or an independently verified learn2learn
execution path. Previously saved native MAML plans require refitting to obtain
the corrected initialization; loading alone does not update learned weights.


### Core partial-autocorrelation numerical checks

The NumPy fallback left one intermediate Levinson-Durbin coefficient unset at
each order, changing PACF from lag three onward. Its recurrence now updates
all k-1 prior coefficients. Four cases (two AR-series seeds, lag caps three
and eight) match independent dense Yule-Walker linear solves at absolute
tolerance 1e-12. Direct autocovariance sums independently verify ACF.
Translation and positive rescaling preserve the results; zero-lag PACF is one.
The tests force the core routing decision while statsmodels remains installed;
this verifies real fallback arithmetic, not an absent-statsmodels environment.
The statsmodels implementation is unchanged. Recompute previously generated
core-fallback diagnostics to obtain corrected values.
Five new and twelve existing time-series cases passed:17 passed, no skips,
14 upstream warnings (statsmodels tuple-return/decomposition deprecations and
KPSS lookup-table p-value bounds). Ruff and scoped mypy passed.
Subset reach against verified aca522a:57 unchanged previously missing lines
plus one changed/new executed line. Of these,30 unchanged lines are in
diagnostics.py;27 occur elsewhere under this host's optional-backend routing
and existing cases. They are not attributed to new CI coverage. Package-wide
remote acceptance of the PACF fix passed at b9267e6; the current table records the measured aggregate.


### Penalized changepoint and numeric-boundary consumers

Binary segmentation previously ignored the documented penalty and forced a
length-based count of breaks. It now calls ruptures Binseg.predict(pen=penalty).
Six real-library comparisons cover PELT/BinSeg at penalties .2,5,1000; high
penalty produces no breaks. Independent sums verify segment means and counts.
Invalid penalties (nonfinite, negative, bool, string, None, overflowing real)
and nonfinite, complex, nonnumeric or multidimensional series raise ValidationError
before detection. CUSUM constant-series/high-threshold cases return no breaks.
Recompute older BinSeg reports to obtain the documented penalty semantics.
This does not establish changepoint power or guarantee a detected break is causal.
Twenty-one new and nine existing time-series checks passed locally:30 passed,
no skips,12 upstream statsmodels future/p-value-lookup warnings. Ruff, scoped
mypy, dependency and YAML checks passed. Local ruptures1.1.10 installation added
only that package; the expanded profile already pins it and now requires its
import. Subset reach against verified b9267e6:23 previously missing unchanged
lines and19 changed/new executed lines. Full remote acceptance and aggregate
coverage for this changepoint batch passed at e33a194 (see the current accepted gate).


### EDA measured-zero ledger consumers

List-format mutual information previously used truthiness when choosing mi,
score or value: zero could become unavailable or be replaced by another field.
Selection now uses the first non-None value. Six cases verify shared ledger,
rendered static HTML and cockpit output, alias priority, dictionary/list parity
and unchanged source reports. A balanced Cartesian category/target sample has
empirical mutual information zero, independently checked with sklearn. These
checks verify display fidelity, not population independence or feature quality.
The new consumers and existing full EDA/cockpit endpoint checks passed:
9 passed, no skips, one upstream LangChain community maintenance warning.
The initial restricted Windows run blocked in asyncio loopback socket setup
before the endpoint handler; only its verified owned worker was stopped. The
same scope passed with loopback access; interrupted coverage was discarded.
Ruff passed. Scoped mypy retains one verified pre-existing Any|None-to-float
error at sheet_coverage.py454; no clean typing claim is made.
Subset reach against verified e33a194:11 previously missing unchanged lines
and one changed/new executed line. Remote whole-suite acceptance of this ledger
fix passed at 51eeff2; the current table records its measured aggregate.


### Persisted scoring uses coerced estimator inputs

Scoring initialized its working input before schema coercion; without replayable
preprocess plans, estimators still received the original values. Numeric estimators
can hide this by coercing inputs themselves. The working input is now selected
after contract coercion. A real saved sklearn OneHotEncoder/LogisticRegression
pipeline requires trained string categories, making ignored integer-to-string
coercion observable. Eight cases cross DataFrame/Dataset, loaded/path bundle
and apply_plans flags. Independent one-hot/logistic calculations match labels
and probabilities; duplicate row indices and unchanged caller data are preserved.
Existing valid/invalid numeric, boolean and datetime contract checks also pass.
Final:35 passed, no skips, two existing Pandas malformed-date parsing warnings.
Ruff and scoped mypy passed. Subset reach against verified 51eeff2 is two
previously missing unchanged lines and one changed/new executed line. This is
a correctness-led repair; it does not imply a large coverage gain. Bundle
weights remain usable without refitting; use corrected scoring to apply their
persisted contract. Remote whole-suite acceptance passed at 5ac9580.


### Semi-supervised raw class identities

New fits encode raw labels and preserve their types in classes, predictions and
JSON-safe scalar summaries. Previously all classes became strings and prediction
guessed numeric types, so code "001" became integer1. Incompatible mixed labels
(such as1,"1",2) now raise ValidationError instead of collapsing, and raw-class
evaluation compares encoded identities rather than normalized strings.
Twenty new checks cover three real sklearn methods and string codes, decimal
strings, integers, floats and booleans. Independent explicitly coded sklearn
fits agree with predictions and labeled-holdout accuracy. Trusted bundle
roundtrips preserve types; plan/fit summaries serialize to JSON. Numerical
holdout labels cannot impersonate trained string classes. Existing real Torch
and offline tiny-encoder text consumers also pass:51 passed, no skips or
warnings. The Torch class-summary assertion now expects the source numeric
domain, matching the intentional metadata correction.
Encoders in older plans lack the raw-identity marker. Their original decoding
and string-evaluation behavior is retained, with refit disclosures/warnings;
three real legacy bundle cases cover integer, decimal and text labels. Loading
cannot recover previously lost identities. Refit to obtain new raw-label behavior.
Ruff passed. Scoped mypy retains two verified pre-existing PartitionOrAll alias
errors under --follow-imports=skip in evaluate.py29/predict.py23. Subset reach
against verified5ac9580:five previously missing unchanged production lines and
21 changed/new executed lines. Whole-package remote acceptance passed at603a4f9.

### Evolutionary nested selection numerical audit

Four new real-consumer cases check group and time nested evaluation, with and
without fold-local scaling. Every actually sampled Ridge candidate is independently
scored with sklearn pipelines on the corresponding inner folds. The checks compare
candidate means/spreads, selected parameters, evaluation counts and outer scores,
including aggregate mean/sample spread. Recorded fitting targets must exactly
match permitted inner/outer training populations; poisoning only Session holdout
targets leaves selection and scores unchanged. These checks found no new production
defect and do not establish that evolutionary search explores every configuration.

Final local verification:23 passed,4 skipped, no warnings in68.11s. The four skips
are existing real Optuna consumers because Optuna is absent from this Windows
runtime; the expanded remote profile verifies that backend separately. Ruff and
format checks pass. Against verified603a4f9, the broader subset reaches three
previously missing unchanged lines, all in missing-Optuna handling; the evolutionary
numerical checks themselves add zero previously missing lines. This strengthens
correctness evidence without claiming an evolutionary coverage gain. Subsequent
whole-suite acceptance passed at4c06bd3:86.4734%,3801 passed,77 skipped.

### Fairness policy numerical and input contracts

Threshold suggestions and application now require one-dimensional inputs and
finite real scores. Application validates all supplied thresholds before producing
labels: NaN/Inf thresholds previously silently selected the negative label, and
complex or malformed values escaped as raw conversion errors. Grid resolution
must be an integer of at least five; fractional values are no longer truncated.
NumPy integer resolutions remain supported, and unknown groups retain the
explicit 0.5 default. Threshold search still uses the documented [0,1] grid;
other ranking scales require caller mapping. Reweighing also checks vector shape.

Four independent scalar exhaustive-grid calculations cover demographic parity
and equal opportunity, including a group without positives, tied policy values,
applied decisions, selection rates and TPRs. Joint/marginal count calculations
verify mean-one row weights and missing-positive-cell warnings. The reweighing
documentation now accurately distinguishes normalized row weights from raw table
weights and describes omitted unobserved cells. Valid policy mathematics is unchanged.

Initial28-case reproduction had20 failures, including one incorrect fixture
expectation about missing-negative-cell warnings. Corrected baseline:19 failed,
9 passed. Final broader verification:47 passed, no skips or warnings in46.21s.
Ruff passes. Scoped mypy retains the same pre-existing Any-return error in
selection_rates_from_scores (baseline372/current409 under --follow-imports=skip).
Reach against verified4c06bd3:23 previously missing unchanged production lines
and39 changed/new executed lines, all in fairness mitigation. This is subset
evidence; a new whole-suite gate must pass before accepting an aggregate gain.

### Exact Z3 bounds and coverage measurement reproducibility

The optional Z3 verifier now constructs numeric bounds without converting through
float. Real solver checks show why this matters: neighboring integers above2^53
and narrow Decimal/Fraction intervals previously collapsed, reversing SAT/unsat
results. Thirty scalar comparison cases cover all six operators, including
Python/NumPy boolean bounds. Two exact rational midpoint checks prove nonempty
intervals independently of Z3. Unsupported/category/non-finite predicates are
skipped rather than raising conversion errors. Partial numeric SAT checks now
warn about unchecked predicates; a real public Session fit carries the warning
through without changing rule predictions. This remains a numeric antecedent
check, not a complete rule-set consistency proof.

Initial real-solver reproduction:16 failed,12 passed. Final broader run:61 passed,
one installed-backend missing-extra guard skipped,786 existing warnings in89.27s
(784 sklearn RuleFit deprecations and two flat-API notices). Ruff and scoped mypy
pass. Against verified4c06bd3, subset reach is17 unchanged missing lines:13 in
the verifier, three in rule consumers and one optional Torch guard. Of14 changed
or added executed lines,11 belong to this verifier repair; three are imports from
the preceding fairness repair. No combined percentage is claimed.

The fairness source8594200 passed all three workflows (3834 passed,77 skipped,
1900 warnings,12:47; no collection warnings). Its mergeff016ce9 matched source
treefbacba31e290aee2038d5d45c3216171a93ca80b. However, automatic coverage.py
upgrade7.16.2 to7.17.0 changed exclusion accounting:41 additional TYPE_CHECKING
lines across40 existing files were excluded. Raw86.5151% (53738/62114,607 excluded)
is recorded as noncomparable and is not the accepted baseline or a ratchet input.
Development coverage is now pinned7.16.2. Both static package analysis and the
final targeted report confirm566 exclusions again. Latest accepted whole-suite
coverage remains86.4734% at4c06bd3; a new complete gate must verify the pin and
both correctness repairs. Scope, exclusion rules and core70/expanded85 floors
are unchanged. Z3-solver5.1.0.0 was already pinned in the expanded profile;
local verification installed only that backend and changed only coverage tooling.

The fresh91b34d6 gate subsequently passed with coverage.py7.16.2 and566
exclusions:53762/62122=86.5426%,3875 passed,77 skipped. All three workflows
passed; tested mergeabb82ccd matched source tree64feecb0527523dd3f145345dd78e369dc7a4246.
Full14:20,1900 warnings, no collection warnings. This is the accepted comparable
measurement for both fairness and exact-Z3 repairs. Exact90% gap:2148 lines.

### Frozen task semantics for synthetic utility

TSTR previously inferred classification/regression from holdout target cardinality.
A small regression holdout could therefore report accuracy, while new labels in
a classification holdout could switch evaluation to regression. Task inference
now uses the fitted plan's frozen target column specification. The existing
heuristic remains explicit: numeric targets with more than20 distinct training
values imply regression; review numeric class codes and target semantics. Valid
existing plans already contain these specifications and need no generator refit.
Malformed plans missing target metadata request refitting instead of falling back
to holdout inference. Explicit n_synthetic budgets must be positive integers;
zero, negative, fractional, boolean and non-finite budgets no longer default,
truncate or leak conversion errors. NumPy integers and automatic None remain valid.

Independent bootstrap row draws and Ridge normal equations verify TSTR and TRTR
R², median imputation, training-only scaling and utility gaps across two seeds
and test/validation. Reducing holdout targets to two values keeps the same task;
generator frames and caller data remain unchanged. Independent logistic/ridge
classification probes also agree when holdout labels introduce new classes.
Initial18 checks failed (including missing task disclosures); final43 passed,
one SDV-not-installed skip and two existing flat-API warnings in51.49s. Ruff
passes; scoped mypy retains the verified pre-existing Any-return alias error
in _resolve_eval_backend (baseline300/current312 under follow-imports=skip).
Subset reach against verified91b34d6:15 unchanged missing lines, comprising9
evaluation lines and6 local absent-backend catalog/import paths;11 changed/new
evaluation lines executed. Those6 host absence paths are not a projected remote
gain. Whole-suite acceptance of this batch remains pending. Coverage scope,
566 exclusions, pinned7.16.2 and core70/expanded85 floors remain unchanged.

The subsequent d90a0e5 gate accepted the frozen utility repair:53777/62128=
86.5584%,3894 passed,77 skipped; all three workflows passed. Tested merge
b5caaa5f matched source tree76fb48249b4d1a56abc94ea00ee4b0c15d3c8993.
Full17:36,1900 warnings, no collection warnings. Coverage7.16.2/566 exclusions
and core70/expanded85 floors were unchanged; exact90% gap2139 lines.

### Anomaly threshold selection and honest partition metadata

All four threshold methods are checked against independent scalar confusion
counts on real IsolationForest scores, including a manually interpolated
contamination quantile. Validation tuning ignores poisoned test labels. Applied
thresholds produce the expected real test-set scores and flags; update_plan=False
leaves the detector threshold unchanged. Zero beta remains valid (precision),
and NumPy fractional beta values retain the scalar optimum.

Unknown metric names previously silently selected F-beta. Invalid beta values
either reached sklearn errors or were accepted without a clear numeric contract;
they now raise BuildML ValidationError before scoring. F-beta requires a finite
nonnegative real number excluding booleans. Exploratory allow_test_tuning=True
previously emitted 'test was not used'; it now explicitly discloses optimistic
test-selected metrics. Stored policies also previously said validation_tuned
for train/test selections; they now record train_tuned or test_tuned. These
post-fit names are catalogued but cannot be requested directly during detector
fitting. Validation policy names remain unchanged. Older non-validation plans
with incorrect metadata should be retuned to record their actual population;
loading alone does not reconstruct past threshold selection.

Initial reproduction:10 failed,8 passed; all eight numerical oracles passed
before the repair. Final broader verification:43 passed, three PyOD-absent
skips, no warnings in83.93s. Ruff passes. Scoped mypy retains two verified
pre-existing follow-imports=skip errors (TuningPartition alias27->28 and
label-column Any-return219->237). Subset reach against verifiedd90a0e5 is23
unchanged missing lines:16 threshold, one post-fit guard and six optional
catalog/import availability paths;14 changed/new lines executed. Host availability
paths are not projected remote gains. Coverage7.16.2/566 exclusions and
core70/expanded85 floors are unchanged; full-suite acceptance remains pending.

The subsequent b95cb80 gate accepted the anomaly repair:53806/62140=86.5883%,
3921 passed,77 skipped. All three workflows passed; tested merge875457e3
matched source tree e7c8ae6494233e2c3308dae81f26c1a5f96fc34b. Full11:22,
1900 warnings, no collection warnings; coverage7.16.2/566 exclusions unchanged.
Exact90% gap2120 lines; core70/expanded85 retained.

### Native committee acquisition invariance

Native qbc_kl previously averaged only KL(P_i||P_j) for i<j. Since KL is
directional, merely reordering an unchanged fitted committee changed its scores
and could change labeling priorities. It now averages both directions for each
pair, equivalent to the mean over all ordered distinct member pairs. Three real
multiclass logistic BaggingClassifiers match independent scalar logarithmic
calculations and retain scores/rankings when member order is reversed. Native
variation ratios match scalar vote counts; core-set scores match nearest labeled
Euclidean distances. Empty pools, the empty-label normalized-norm seed heuristic,
single-member zero disagreement and invalid committee protocols are also checked.
The native core-set batch scores are static priorities, not an independently
verified iterative k-center optimization guarantee.

This repair affects the native industry scoring path, including fallback when
the optional host backend cannot run. The separate installed host backend can
use different acquisition mathematics; these checks do not claim backend
equivalence or better learning efficiency. Existing fitted committees need no
retraining. Future native KL queries may change after the correction; historical
query choices and labels are not rewritten.

Initial16-case reproduction:7 failed,9 passed, including one test fixture
accessing estimator_ on an unfitted bagger before reaching the adapter. Corrected
baseline:6 failed,10 passed in16.50s, all remaining failures in KL scalar or
permutation checks. Final broader verification:88 passed, no skips or warnings
in223.51s, including existing public query/annotation flows, Torch consumers and
installed host-backend checks. The direct native numerical checks do not fake
package absence. Ruff and scoped mypy pass. Reach against verifiedb95cb80:
43 unchanged missing production lines (42 native scorer, one optional Torch
availability guard), and one changed/new executed scorer line. Scope and
566 exclusions remain unchanged under coverage7.16.2; full-suite aggregate
acceptance remains pending.

The subsequent958b390 gate accepted native KL invariance at53849/62140=
86.6575%,3937 passed,77 skipped. All three workflows passed; tested merge
a54169fc matched source tree5e189439de15194100d1490bf357e04fc172fe71.
Full14:26,1900 warnings, no collection warnings. Coverage7.16.2/566 exclusions
and core70/expanded85 unchanged; exact90% gap2077 lines.

### Stateful online class identities and evaluation populations

New online encoders retain raw class labels and mark that representation in
persisted encoder state. Previously class stringification plus numeric guessing
turned code "001" into integer1 and booleans into text; numeric updates could
impersonate trained string classes. Discovery and explicit classes now retain
their raw values; incompatible mixed numeric/string labels are refused. New
evaluation compares encoded identities, and unseen typed classes are refused
before updating weights. NumPy class scalars are normalized for JSON summaries.

Twenty real classifier cases cross SGD/BernoulliNB, discovered/explicit vocabularies,
and string codes, decimal strings, integer, float and boolean domains. Independent
raw LabelEncoders and fresh coded estimators match successive partial_fit chunks,
typed predictions, holdout accuracy and continuation after trusted reload. Model
frames remain unchanged. Metadata retains raw classes and serializes as JSON.
Four actual unmarked legacy bundle fixtures retain their original decode behavior
and disclose refitting. Old weights/string ordering are preserved; loading does
not recover lost original types. Refit old classifiers for raw identity behavior.

Evaluation of train/all previously described those rows as never used in training;
it now identifies training-row inclusion and avoids claiming independent holdout
evidence. Partially/all-null holdout targets are excluded or return empty metrics,
with coefficients and update counters unchanged. Initial18 failures included one
Pandas string-dtype assignment fixture failure;17 reached the intended product
contracts. The corrected typed-holdout fixture passes, and the archived legacy
helper independently confirmed accepting numeric1.0 as string-class"1.0" code0.
Final broader run:94 passed, one River-not-installed skip, no warnings in71.51s.
Ruff passes. Scoped mypy retains the same three baseline follow-imports=skip
errors: feature DataFrame Any-return444->453 and predict/evaluate alias21/31.
Subset reach against verified958b390:18 unchanged missing lines (9 evaluation,
4 sklearn compatibility and5 optional availability paths) plus28 changed/new
executed lines. Compatibility/availability paths are not projected remote gains.
Scope, coverage7.16.2/566 exclusions and core70/expanded85 remain unchanged;
new full-suite aggregate acceptance remains pending.

### Synthetic sampling budget boundary (2026-10-22)

Public synthetic sampling now refuses boolean, fractional, string and non-finite
row budgets before drawing or merging; None still uses the fitted train count
and NumPy integer counts remain supported. Thirty-two invalid-budget cases
exercise native bootstrap/copula with and without train appends; six independent
NumPy bootstrap consumers verify exact sampled rows, train appends and unchanged
 validation/test frames. Rejected calls preserve dataset/split/plan.
Baseline24 failed/14 passed11.16s; final106 passed,1 absent-SDV skip,2 existing
flat-alias deprecation warnings53.58s. Ruff and scoped mypy pass. Reach against
verified3f9acc9:0 unchanged sampling lines,5 changed/new sampling lines;6 unchanged
missing availability/catalog lines are local-profile differences, not projected
remote gains. Coverage7.16.2/566 exclusions. No aggregate percentage projected.
Internal generator direct-call coercion is unchanged; public Session sampling
is the repaired boundary. New whole-suite acceptance pending.

### Symbolic raw class identity (2026-10-23)

Symbolic classification now preserves raw class identities in induced tree/list
rules, all three sklearn neural modes, metadata and trusted bundles. Numeric-looking
strings, floats, ints and booleans match independent DecisionTree/LogisticRegression
consumers (the decision-list fixture has a known perfectly separating boundary).
Incompatible mixed labels and numeric impersonation of explicit string vocabularies
are refused. New-fit accuracy, agreement and hard-rule repair counts distinguish
string consequents from numeric targets. Existing plan comparison behavior remains;
three actual legacy string-encoder bundles retain their original decoding with
refit notices. Older plans need refitting to obtain raw identities.
Corrected baseline31 failed/5 passed22.94s. Final98 passed,1 installed-backend
missing-extra-guard skip,786 existing warnings91.60s; real Torch/imodels/Z3
checks pass. Ruff passes; scoped mypy has the same8 baseline diagnostics.
Reach against verified4f9eb4d:4 unchanged missing lines (3 symbolic,1 optional Torch availability) and17 changed/new;
no projected aggregate gain. Coverage7.16.2/566 exclusions unchanged. Expanded
job passed but overall CI failed the docstring audit; repaired in the next batch.

### Active-learning raw label continuation and CI audit repair (2026-10-23)

The symbolic expanded job passed4042 tests with77 skips (raw86.6873%),
but overall CI39872285177 failed its four core docstring-audit jobs because the
new legacy_labels parameter was undocumented. Backend39872285036 and
acceptance39872284910 passed. That raw percentage is NOT accepted or used for
floors; accepted4f9eb4d remains86.6835%. Parameter documentation is now repaired
and local audit passes without changing its budgets.

New active learners preserve raw class labels through user annotation rounds,
full vocabulary-expanding refits, metadata and trusted bundle continuation.
Five label domains match independently fitted LogisticRegression models before
and after two human-label rounds, including a reload. Three new-class rounds
reorder codes and reproduce fresh fits; original holdouts remain unchanged.
Mixed incompatible labels are refused. Raw factorization gives metrics a shared
vocabulary without conflating strings and numbers, including unseen holdout
classes. Three actual legacy string-encoder bundles retain codes, numeric decoding
and string-based refits; evaluation asks for a fresh fit for raw identity.
Missing holdout labels are counted and excluded without model/budget mutation.
Corrected initial9 cases all failed14.17s; repaired9 passed13.71s. Expanded18
new cases plus real host/Torch/native acquisition/annotation/integration checks:
107 passed,zero skips/warnings83.50s. Ruff and docstring audit pass; scoped mypy
retains its one baseline PartitionOrAll alias error. Subset reach against accepted
4f9eb4d:4 unchanged missing lines (3 active-learning,1 optional Torch availability)
and28 changed/new lines (26 active-learning,2 earlier symbolic import/function
definitions). Coverage7.16.2/566 exclusions; no projected whole-suite gain.
Also fixes the prior symbolic docstring CI failure by documenting legacy_labels;
audit passes586 existing findings within unchanged budgets. Fresh full CI pending.

### Agent text Torch counts and numerical replay (2026-10-23)

Text Torch vocabulary/loader controls now refuse boolean, non-integral,
string, non-finite and out-of-range counts before attaching Session loader state.
Vocabulary caps require two reserved tokens; frequency, sequence length and
batch size require positive integers. NumPy integer controls remain supported.
Actual confirmed agent text-loader→CPU training→evaluation consumers independently
reconstruct train-only token vocabulary, reserved/unknown IDs, truncation/padding,
all partition tensors and batch sizes at three sequence lengths. A NumPy embedding
mean-pool/MLP replay matches actual Torch logits and reported accuracy after two
epochs; refused confirmation leaves state empty and caller frames stay unchanged.
Initial39 cases:26 failed,13 passed15.69s (24 invalid vocab controls,2 agent
zero/negative max_len failures); valid numerical workflows already matched.
Expanded72 new checks pass16.73s. Broader five-file suite93 passed,1 absent-ONNX
skip,47 deprecation warnings76.63s. Ruff, scoped mypy and docstring audit pass.
Reach against verifiedffd295f:9 unchanged missing lines (8 dispatcher,1 optional
Torch availability) and6 changed/new lines. Coverage7.16.2/566 exclusions unchanged.
No projected aggregate gain, CUDA/ONNX throughput or model-quality claim.
Fresh whole-suite acceptance pending.

### Offline audio backbone/head and core dependency checks (2026-10-23)

CI39884934089 at44e6ade failed in its core profiles:37 new tests expected
Torch loader validation while Torch was absent. Expanded job, backend39884934033
and acceptance39884934043 passed; this overall result was not promoted. The core
test assumptions are corrected here and verified in a subprocess with Torch
imports deliberately blocked. This simulates dependency absence, not every minimal
profile/package version; fresh remote core CI remains required.

Backbone heads refuse non-integral, boolean, string and non-finite class
counts, and invalid freeze overrides, before changing parameter flags. Session
heads now retain the latest freeze metadata when replaced without an override.
Actual offline Wav2Vec2/HuBERT encoders loaded from tiny local configurations
exercise confirmed agent load/head dispatch. NumPy mean-pooling and affine-head
math matches logits; real SGD steps verify frozen parameters stay unchanged and
unfrozen parameters receive gradients and change. These are random architectures
(weights=none), not evidence of pretrained quality or speech accuracy.
Baseline20 cases10 failed/10 passed20.74s; repaired20 passed20.98s. Broader102
passed,1 missing-torchvision skip,77 deprecation warnings77.11s. Ruff/docstring
audit pass; scoped mypy retains the same2 baseline diagnostics. Reach against
acceptedffd295f:33 unchanged missing lines (26 dispatcher including8 earlier text
lines,6 zoo,1 optional Torch availability),13 changed/new lines (7 this repair,
6 earlier text validation); no projected whole-suite percentage.
Also fixes prior CI's37 Torch-absent core test failures: dependency-free vocabulary
and agent schema checks still run; real-loader tests explicitly require Torch.
Installed text checks72 passed; combined simulated Torch-absent checks32 passed,
60 explicit dependency skips. No production validation was weakened or floor lowered.
Fresh full CI acceptance pending.

### Symbolic evaluation population and regression boundaries (2026-10-23)

Symbolic and neuro-symbolic evaluation now describe the requested population
without a universal holdout claim. Train/all results explicitly warn that fitting
rows are included; validation/test retain the no-refit behavior. Regression scoring
refuses non-finite, complex and unscorable targets or rule predictions with an
actionable ValidationError rather than raw numerical-library failures. Valid
classification and regression metric formulas remain unchanged.
Independent LogisticRegression fits, Ridge normal equations and declared-rule
row formulas check all four partitions across symbolic and three neural modes.
Missing/null targets are refused without silently dropping rows; frames and
fitted coefficients remain unchanged. Corrected baseline45 cases27 failed/18
passed20.31s; repaired49 passed18.01s. Broader147 passed,1 installed-backend
guard skip,786 existing warnings82.40s; actual Torch/imodels/Z3 consumers pass.
Ruff/docstring audit pass; scoped mypy retains its4 baseline diagnostics.
Reach against accepted4b822d4:26 unchanged missing lines (25 symbolic,1 optional
Torch availability),15 changed/new lines; coverage7.16.2/566 exclusions unchanged.
No projected whole-suite gain. Fresh full CI acceptance pending.

### EDA gate populations, costs and executable teaching (2026-10-23)

EDA gate class shares now divide by observed target counts, including the
displayed majority-share and class-balance teaching calculations. Target medians
are computed by the real regression analyzer; legacy reports lacking a median
show it as unavailable. These EDA summaries are descriptive and do not claim a
held-out baseline score. Threshold teaching asks for a positive label and costs,
uses calibrated posterior probabilities, and removes the incorrect balanced-prior
requirement and display-order positive-class guess. Weighting/resampling advice
now calls for task costs and held-out checks.
Generated evaluation examples safely quote column names, restrict modeling to
observed labels with a labeling-bias notice, exclude targets from feature sampling,
prepare numeric/categorical columns separately, and use real estimator/evaluate
APIs. Actual numeric/categorical classification/regression examples execute with
ordinary and quoted names. Three missing-label class distributions and two actual
regression reports match independent Pandas/NumPy counts and medians.
Original16 cases all failed89.28s; repaired16 passed41.99s. Strengthened8 examples
exposed target-in-feature selection and categorical-path failures, now repaired.
Final20 new checks within broader77 passed,zero skips,1 existing LangChain warning
114.23s. Ruff/docstring audit pass; scoped mypy retains4 baseline diagnostics.
Reach against acceptedd10877f:6 unchanged missing lines (4 workflow,2 optional
availability) and17 changed/new lines; coverage7.16.2/566 exclusions unchanged.
No projected aggregate gain. Fresh full CI acceptance pending.


Time-series diagnostics now refuse non-positive/non-integer lag requests,
multidimensional, complex, non-finite or constant series and unsupported
deterministic term codes before backend work. KPSS disclosures distinguish
level stationarity (`c`) from trend stationarity (`ct`); ADF disclosures record
the selected deterministic terms. Valid short-window numerical failures remain
visible as unavailable statistics while preserving correlations.
New tests compare ACF/PACF against dense covariance/linear-solve oracles and
all eight ADF/KPSS term combinations against real statsmodels calls. A real
Session consumer perturbs held-out observations and checks training-only results
and input preservation. Final five-file verification:57 passed,zero skips,
81 statsmodels future/interpolation warnings,82.59s;no collection warnings.
Ruff,mypy and docstring audit pass;586 audit findings within unchanged budgets.
Subset reach against verified6c048ff:9 unchanged previously missing lines
(4 workflow/result lines,5 availability lines) plus16 changed/new executed lines.
This is subset reach,not an aggregate coverage gain. Fresh full CI is pending.


Nested CV now refuses unknown inner_search names and mixed grid/randomized
families before fitting. Previously unknown names fell through to auto and
cross-family estimator/recipe combinations could omit a supplied space.
Both estimator and recipe spaces must use the same family. Existing explicit
Optuna/evolutionary configuration refusals remain intact.
New29 tests include12 before-fix failures,11 existing invalid-configuration
contracts and6 independent weighted Ridge consumers (grid/randomized by
MAE/MSE/R2). Dense weighted normal equations independently verify every inner
winner,inner mean/spread,outer metric and aggregate mean/spread. Invalid requests
never fit or publish Session state; valid searches preserve the input.
Final five-file batch:90 passed,zero skips,2 existing sklearn warnings,65.67s.
Ruff and docstring audit passed;586 findings within unchanged budgets.
Scoped mypy has the same4 pre-existing errors (three Any-return DataFrame
methods and optional SplitPlan in search refit);no new errors or ignores.
Subset reach vs verified7aa4f0c:8 unchanged previously missing production lines
and4 changed/new executed lines,all in model/selection.py. No projected
whole-suite gain. Fresh full CI is pending.


Tabular RL training/evaluation now require positive integer episode and step
budgets before backend allocation. Epsilon-greedy distributions refuse complex
or non-finite action values and invalid/non-finite exploration rates; finite
out-of-range exploration rates retain documented clipping.
New57 cases include36 budget requests across training/evaluation and both
controls,10 action-value/rate refusals,5 independent finite probability vectors
and6 real registered Gym chain consumers. Q-learning,SARSA and Expected SARSA
match independent TD recurrences for true termination and time truncation;
reported rewards,visit counts,TD errors,interaction budgets and environment
closure are checked. NumPy integer counts work. Evaluation returns are known
exactly and do not mutate learned tables or training visits.
Final two-file verification:81 passed,1 installed-Gym missing-backend guard
skipped,zero warnings,81.13s. No collection warnings. Ruff,mypy and docstring
audit pass;586 findings within unchanged budgets. Baseline57 cases had39 genuine
failures and18 passes in2.86s,including passing numerical rollout consumers.
Subset reach vs verified01430f9:2 unchanged previously missing production lines
plus22 changed/new executed lines,all in rl/tabular.py. No projected aggregate
coverage gain. Fresh full CI pending.


Real offline Transformers ASR consumers now execute the production pipeline
with local tiny Wav2Vec2 CTC models,not a substituted pipeline or stub.
Zero head weights and fixed output biases yield known frame logits; CTC collapse
must produce A or blank. Explicit,auto and default backend selection,short/padded/
truncated arrays,all four Dataset partitions and unchanged source audio are
checked. Actual convolution failures on one-sample inputs produce per-clip
error sentinels/warnings; a subsequent valid inference succeeds.
Initial11 new tests passed46.41s. Final broader speech batch:20 passed,
1 installed-Transformers missing-backend guard skipped,zero warnings,78.24s.
No collection warnings. Ruff and docstring audit passed;586 audit findings
within unchanged budgets. No production changes were needed. No speech-quality,
pretrained-quality or generalization claim is made for these tiny fixtures.
Subset reach vs verified4197d8e:23 unchanged previously missing production lines
(22 speech pipeline/Dataset lines,1 Torch availability line),zero changed/new
executed lines. No combined or projected whole-suite gain. Fresh full CI pending.


Fold-local PCA now honors the documented full-variance1.0 endpoint and NumPy
integer/real scalar component controls. Booleans,strings,complex and unsupported
controls are refused. Both fit and replay refuse non-finite/complex feature
values before lossy conversion or backend work. Component names cannot collide
with retained input columns; removed source names can be reused safely.
New36 numerical/boundary consumers reproduced18 failures and18 passes in11.25s.
Independent centered NumPy SVD subspaces verify fourteen count/fraction/drop
variants,including a held-out shift that cannot change the fitted center.
Other cases cover invalid controls,fit/replay values,fit/replay name collisions
and legitimate reuse of a removed source name. Input values and indices persist.
Final five-file verification:84 passed,zero skips/warnings,52.31s;no collection
warnings. Ruff and docstring audit pass;586 findings within unchanged budgets.
Scoped mypy has the same20 pre-existing normalized diagnostics as baseline;
no new errors,ignores or budget changes.
Subset reach vs verifiedfa5b2f6:1 unchanged previously missing production line
plus32 changed/new executed lines,all in preprocess/fold.py. No projected
aggregate gain. Fresh full CI pending.


External sklearn transformer handoffs now preserve scipy sparse output as a
Pandas sparse frame and validate row index/order,unique feature names and shape
compatibility. This fixes valid sparse one-hot/ColumnTransformer designs failing
ordinary DataFrame construction and prevents silently ambiguous/reordered input
from reaching a consumer. Dense and unnamed-array handoffs remain supported.
Initial15 cases reproduced8 failures and7 passes in21.64s. Two additional
array-row/name mismatch consumers were added before repair verification.
Real sparse/dense encoders and ColumnTransformers feed Ridge; independent
one-hot/scaling designs and normal equations verify predictions for unknown
categories,normal indices and duplicate row labels. DataFrame row changes and
real encoder name collisions are refused. No source values are changed.
Final five-file verification:86 passed,zero skips,2 existing sklearn selector
warnings,57.06s;no collection warnings. Ruff/docstring audit pass;586 findings
within unchanged budgets. Scoped mypy has19 existing diagnostics versus20 before,
with no new normalized messages or ignores;one previous Any-return resolved.
Subset reach vs verified5853b6b:0 unchanged previously missing lines,16 changed/new
executed lines in preprocess/fold.py. This is a correctness repair with no
projected aggregate coverage gain. Fresh full CI pending.


Native AutoML nested selection now preserves built-in grouped/temporal inner
populations and custom inner splitters. Evaluation sample weights apply to both
inner candidate ranking and outer scoring,matching weighted fitting. Inner
splits are validated once per outer fold before candidate-error handling;
outer/inner indices are checked before indexing,including overlap refusals.
Fifteen independent weighted Ridge procedure oracles reproduced15 failures
in29.66s; after repair all15 passed18.38s. NumPy normal equations and explicit
fold populations verify MAE/MSE/R2 outer means/spreads across ordinary,group-role,
explicit-group,temporal and custom-inner modes. The full input remains unchanged.
Seven further splitter-constraint cases include stratified groups and sklearn
class-name aliases; eight malformed-membership cases and missing-group refusal
exercise validation. Final four-file verification:45 passed,zero skips/warnings,
81.83s;no collection warnings. Ruff and docstring audit pass;586 findings within
unchanged budgets. Scoped mypy retains the same2 pre-existing tournament-trial
type diagnostics,no new errors or ignores.
Subset reach vs verified85774fb:22 unchanged previously missing lines
(16 AutoML search workflow lines,6 catalog/availability lines) plus37 changed/new
executed lines. No projected whole-suite gain. Fresh full CI pending.


Nested AutoML now repeats the configured search method,trial budget,recipe and
ensemble choices inside each outer-training subset using canonical selection=cv.
This replaces the randomized proxy for adaptive methods,removes the hidden
4..12 trial substitution and includes requested ensembles. Outer studies receive
fresh budgets without persistent Optuna storage. Original row positions,group
labels,temporal order and evaluation weights remain aligned.
Weighted sklearn voting now routes sample weights through its known fit_params
API after validating every active member;unsupported members are refused before
fitting and dropped members are ignored. Voting-family selection follows metric
direction instead of preferring the largest regression losses.
Verified immutable946098b reproduction:8 failures,2 absent-Optuna skips13.81s.
Three additional weighted-voting consumers failed11.38s; loss-direction case
failed while its R2 control passed16.08s. New tests wrap and call the actual
search,never substitute its output. Independent weighted normal equations verify
Ridge trial scores and predictions; real voting candidates and weighted output
metrics verify ensemble propagation. Direct voting priors/means use hand-derived
weighted expectations. Public population oracles use canonical CV folds now.
Final five-file verification:58 passed,2 absent-Optuna skips,zero warnings,
105.20s;no collection warnings. Ruff/docstring audit pass;586 findings within
unchanged budgets. Scoped mypy has the same9 pre-existing diagnostics across
search/supervised,no new errors,ignores or budgets.
Subset reach vs verified946098b:9 unchanged missing lines (3 search/space,
6 catalog/availability) plus20 changed/new executed lines. Source refactoring
changes statement counts;scope/exclusions are preserved and no projected
aggregate gain is claimed. Fresh full CI pending.


The configured nested AutoML repair at `400b8dd` passed all three workflows:
4,457 passed,77 skipped,2044 warnings in17:00,without coverage collection warnings.
Both locally absent Optuna cases ran in the expanded profile. Full coverage is
86.8912%,down0.0345 percentage points from86.9257%: covered lines/statements
changed by-47/-26 in search,+7/+7 in supervised and+2/0 in spaces. The canonical
search now replaces the manual nested path; its legacy private helpers remain
for compatibility. This correctness repair is not a claimed coverage gain.
Scope,566 exclusions and core70/expanded85 floors remain unchanged. The exact
90% gap is1,937 lines. Same-source whole-suite reach separately found3 previously
missing unchanged lines and20 changed/new executed lines; it is not an aggregate
projection. Tested merge7cfa5072 and source400b8dd have the same tree.


Native AutoML custom CV replay and configuration repair:
one-shot fold iterators are normalized once using sklearn's reusable check_cv
wrapper, so all candidate and ensemble scorers receive the supplied folds.
Previously the first candidate exhausted a generator and later candidates were
silently discarded. Custom cv/outer_cv metadata is now null plus the supplied
type in extras; integer requests are unchanged and cv_strategy is disclosed.
This metadata is not an observed count or complete custom-fold replay definition.
The indices are materialized in memory; custom splitter objects retain their
split method, and nested indices must fit each outer-training population.

Initial18-case reproduction:14 failures,4 absent-Optuna skips in16.73s.
Final seven-file source-frozen verification:83 passed,6 absent-Optuna skips,
zero warnings in138.87s. New21-case consumer file contributes17 passed/4 skipped;
real grid/randomized/evolutionary searches use splitter/list/tuple/generator CV.
Independent Ridge normal equations verify all trial means and sample deviations.
Real voting/stacking list-vs-generator controls and bundle prediction/config
roundtrip pass. No fabricated estimators, scores or search outputs are used.
Ruff and docstring audit pass;586 findings within unchanged budgets.
Scoped mypy --follow-imports=silent has the same2 existing tournament type
diagnostics,no new ignores/budgets. An accidentally unscoped type check followed
the entire package and was stopped using only its verified own process IDs;
its incomplete broad log is not a new baseline or a clean type-check claim.

Subset reach against accepted400b8dd:11 unchanged previously missing lines
(5 search candidate/stacking,6 optional catalog/availability),5 changed/new
executed lines. These counts are not a projected whole-suite percentage.
Latest accepted whole-package source400b8dd remains54115/62279=86.89124745%,
4457PASS77SKIP,coverage7.16.2/566 exclusions,core70/expanded85,exact90gap1937.
Fresh full CI required for this source;no90 or long-term SLA claim.


AI executor bundle restoration repair:25 explicit load handlers now forward
trusted without coercion;26 load schemas including the existing NLP path expose
boolean trust and retain confirmation. Defaultfalse/omitted requests still
refuse executable deserialization. Proposal warnings disclose explicit trust.
The existing schema boolean validation refuses strings,numbers and null before
path access; trust alone never bypasses confirmation.

Initial real round-trip reproduction on ff93cc8 production source:
34FAIL1PASS(NLP)1SKIP(absentTDA) in67.56s,176 warnings. The strengthened existing
consumer now restores through execute_tool instead of switching to directAPI.
Final four-file source-frozen verification:373 passed, 5 skipped, 502 warnings in 289.45s.
No coverage collection warnings. New boundary contracts cover all26 tool schemas,
malformed trust4values each and confirmation withholding (156cases).
Two LinUCB consumers use independent per-arm normal equations and exploration
bonuses to verify action scores,labels,direct-method/action-match offline metrics
and restored policy output. AutoML retains predictions,metrics and selected
parameters after real agent-driven save/load. No fake models or load outputs.
Five local skips are absent-ripser TDA consumers; no TDA runtime validation here.
SSL,KG,recommender and decision load schemas/boundaries are checked; their
positive runtime round-trips are outside this local batch.

Ruff/docstring audit PASS,586 findings within unchanged budgets. Scoped mypy
reports7 existing diagnostics; independently archived ff93cc8 source produced
the same normalized diagnostics,verified exactly. No new ignores or budgets.
Subset reach against verified ff93cc8 expanded artifact:
78 previously missing unchanged lines (59 executor workflow lines and19 optional
catalog/availability lines) and
22 changed/new executed lines. This is not a projected whole-suite percentage.
Fresh full CI required for the repair. Scope,566 exclusions,core70/expanded85
ratchet preserved;90percent and general long-term SLA remain unestablished.


Passing source `a05ec41` crosses87% whole-package line coverage:
54,187/62,283=87.00126840389834%,4,637 passed,77 skipped,zero failures/errors.
CI39990338088,backend39990338984 and release acceptance39990338099 passed.
Tested merge4b8b634c/source trees equale687668f89d67f80c138aaa4380648b996b694b8.
XML911.552s/log911.61s15:11,2059 pytest warnings,no collection warnings;
coverage7.16.2/566 exclusions unchanged. Covered lines increased64 while
statements increased2 versusff93cc8. The expanded floor is raised85 to86
under the existing one-point-headroom rule (actual headroom1.0012684pp).
Core floor remains70. Exact90% gap1,868 lines; no90% or long-term SLA claim.
Same-source whole-suite reach separately reports61 previously missing unchanged
lines and27 changed/new executed lines. This differs from Windows subset reach
78/22 because profile/platform and scope differ; percentages are never combined.


Frozen agent model consumer verification (production source unchanged):
native KG TransE L1/L2 and DistMult scores match independent embedding algebra
before and after actual agent save/load. Item/user kNN recommendations match
train-only pivoted rating matrices,cosine scores,eligibility and top-k score
thresholds,with and without train-history exclusion. Restored legacy SSL MLP
bottleneck activations match explicit matrix/ReLU forward calculations at two
latent widths,and the supervised head's held-out accuracy is preserved.
Both multiclass decision cost consumers now restore through the executor and
recheck their independently derived Bayes actions,not just direct API loading.
These tests close prior local positive-runtime gaps for SSL,KG,recommender and
decision load tools; native/sklearn paths do not validate every optional backend.

Eight additional real temporal consumers exercise the agent's previously missing
time_split handler on shuffled UTC dates,with/without validation. Assert disjoint
chronological prefix/middle/tail populations and unchanged dataset,then derive
naive/mean/drift/seasonal-naive forecasts independently from training history.
Initial bundle/decision15PASS74warnings22.91s; temporal8PASS16warnings9.13s.
Final five-file source-frozen verification:188 passed, 128 warnings in 116.38s.
Ruff/docstring audit PASS,586 findings within unchanged budgets. No production
code changes or new failures found; no redundant production type check run.
Warnings include known flat-alias deprecations and legacy sklearn SSL notices;
no warnings suppressed and no coverage collection warnings.

Subset reach vs verified7cdbeb7:
32 previously missing unchanged production lines,
0 changed/new executed production lines.
This is not a projected whole-suite percentage. Fresh full CI required to assess
aggregate coverage. Package scope,coverage7.16.2/566 exclusions and core70/
expanded86 ratchet unchanged. The 90 percent target and long-term SLA remain open.

Reach breakdown:18 workflow lines (16 AI dispatch,2 temporal split) and14 optional catalog/availability lines. Zero changed/new production lines. Subset counts are separate from full-suite percentages.

Session/agent Torch CV outer-population repair:
the Session wrapper previously supplied its entire Dataset to the intentionally
whole-input low-level CV API,so reserved validation/test rows entered folds.
It now validates an attached split and supplies a copied train-only Dataset.
Unsplit Session CV remains whole-input; low-level Dataset CV is unchanged.
Result config records population,n_rows,held_out_partitions and disclosures
distinguish split/unsplit behavior. Existing global preprocessing limitations
remain: reserving holdouts does not undo transformations learned outside folds.

Initial6-case reproduction:6FAIL2warnings17.76s. Two unsplit numerical controls
reached the new missing-metadata assertion; two split controls exposed whole-
Dataset fold populations,and two real agent consumers failed on poisoned outer
holdouts. After repair initial6PASS2warnings16.96s. Added three malformed outer
split controls (emptytrain/testoverlap/validationoverlap),refused before factory.
Independent linear arithmetic computes each real trained fold model's MSE from
its weights and fold-only normalization,with split/unsplit and normalizeon/off.
Non-RangeIndex rows and poisoned holdout feature/target values remain unchanged.
Both regression and classification run through the actual AI execution path.
Final five-file source-frozen verification32PASS,zero skips/failures,
24warnings141.16s;no collection warnings. Existing CPU Torch CV/search seed,
sampling and real runtime training/retrieval consumers pass too.
Ruff/scopedmypy/docstring audit PASS (586 findings within unchanged budgets).
Subset reach vs verifieddc1b162:20 unchanged previously missing lines,
14 changed/new executed lines. Not an aggregate projection.
Fresh full CI required. Scope,coverage7.16.2/566 exclusions and core70/expanded86
ratchet unchanged;90percent and general long-term SLA remain unestablished.

Reach breakdown:18 workflow lines (13 low-level CV,3 AI dispatch,2 split checks) and2 optional-availability lines;14 changed/new Session-wrapper lines. Subset reach is separate from aggregate coverage.

Agent Torch selection numerical consumers (production unchanged):8 new cases
cover grid/randomized search and nested CV for regression/classification.
Actual default factories and actual training/scoring run; the wrapper only
records returned real models and params,never substitutes models or outputs.
Grid/distribution spaces select linear tabular architectures and two learning
rates. NumPy affine predictions from fitted weights independently verify MSE
or argmax accuracy. Independent seeded fold populations verify inner means/std,
metric-direction winners and outer selected-model refits/aggregate metrics.
All8 cases run56 real model fits. Poisoned reserved holdout features/targets stay
outside both loops,and the source Dataset remains unchanged. Tiny epoch counts
are numerical workflow checks,not learned-model quality or search superiority.
Initial regression4PASS2warnings18.84s; then classification cases added.
Final four-file source-frozen verification29PASS,zero skips/failures,
6warnings66.89s;no collection warnings. Prior SessionCV holdout,search seed and
sampling contracts included. Ruff/docstring audit PASS,586 findings within
unchanged budgets. No production type check repeated for this tests-only batch.
Subset reach vs verifiedb9280a9:20 previously missing unchanged lines,
0 changed/new executed lines. Not an aggregate projection.
Fresh full CI required;scope,coverage7.16.2/566 exclusions and core70/expanded86
unchanged. The 90 percent target and general long-term SLA remain open.

ASR corpus counting and transcript input repair:
empty utterance references previously added artificial denominator units,
under-reporting corpus WER/CER. Totals now pool actual reference word and
non-whitespace character counts. Only an entirely zero corpus denominator is
floored at one;reported reference totals remain zero and the convention is
disclosed. Per-utterance zero-reference convention remains denominatorone.
The pure metric API requires list/tuple containers of strings. Session and AI
wrappers no longer coerce scalar strings/dicts into character/key utterances.
Malformed transcripts fail before result/history state changes. Valid empty
strings remain supported. No claim of general ASR model quality.

Initial public consumer matrix35FAIL16PASS17warnings12.54s;afterrepair51PASS/
17warnings9.75s. Known substitution/insertion/deletion/casing counts and empty
reference populations are checked across direct,Session and actualAI paths.
18 additional required-input refusal cases verify unchanged dataset/roles/history,
no model/split attachment and no files written. These use real tool dispatch,
not fake session implementations. Real local Wav2Vec2 CTC consumers also pass.
Final explicit four-file/node source-frozen verification81PASS,zero skips/failures,
17warnings87.96s;no collection warnings. HF offline flags enabled,no downloads.
Ruff/docstring audit PASS (586 findings within unchanged budgets). Scoped mypy
has11 existing diagnostics; an archived4ad7a3c baseline,verified against git show,
produced the identical normalized diagnostic multiset. No new ignores/budgets.
Subset reach vs verified4ad7a3c:31 unchanged previously missing lines,
7 changed/new executed lines. Not an aggregate projection.
Fresh full CI required. Package scope,coverage7.16.2/566 exclusions and core70/
expanded86 preserved;90percent and general long-term SLA remain outstanding.

## Agent anomaly support-vector consumers - 2026-10-23

Eight real AI-dispatch consumers check One-Class SVM across unsupervised/normal-only novelty,
quantile/decision-zero threshold policies and validation/test populations. Independent RBF
support-vector algebra checks scores; separate sklearn fits on exact train/normal-only rows
check fit populations. Scalar confusion counts and top-seven rankings check labeled metrics.
Override thresholds leave the fitted plan unchanged. All-row attachments preserve source rows
and split indices; holdout-only and duplicate attachments refuse without data/history mutation.
Initial8PASS/48 deprecation warnings43.61s. No production source changed or optional backends
simulated. Ruff/doc audit PASS (586 existing findings, unchanged budgets).

Final four-file source-frozen subset51PASS,zero skips/failures,48 expected flat-alias
deprecation warnings235.68s;no collection warnings. Reach vs verified31ae0db:
19 unchanged previously missing lines (17 workflow,2 optional availability),zero
changed/new lines. No production changes. This is not a whole-package projection.
Evidence reports/agent-anomaly-{initial,final,docstrings}.log and
reports/agent-anomaly-{coverage,reach,measurement}.json. No runtime installs or
local full-suite overlap. Fresh remote CI must pass independently.

## Agent DBSCAN density geometry - 2026-10-24

Twelve real agent consumers verify DBSCAN using an independent scalar neighborhood graph
and Euclidean nearest-core assignment. Cases cover populated/all-noise train fits,
min_samples2/3/99,eps.25/.5,validation/test,exact epsilon and nextafter-outside points.
Frozen plans,train populations,source rows and split indices remain unchanged during
assignment. All-row attachments match the independent labels; holdout-only and duplicate
attachments refuse without data/history changes. Initial12PASS60 expected flat-alias
deprecation warnings10.46s. No production changes or simulated optional backends.
Ruff/docstrings PASS586 findings within unchanged budgets. Final measurement below.

Final four-file source-frozen verification32PASS0SKIP/FAIL60 expected flat-alias
warnings72.10s;no collection warnings. Reach vs verified74ecbf4:12 unchanged
previously missing lines (9 workflow,3 availability),zero changed/new lines.
Subset reach is separate from accepted full87.1750%;no aggregate projection.
Evidence reports/agent-dbscan-{initial,final,docstrings}.log and
reports/agent-dbscan-{coverage,reach,measurement}.json. No runtime installs,
production edits or local full-suite overlap. Fresh remote CI required.

## Meta-learning raw label identity - 2026-10-24

New meta-learning encoders preserve raw string,numeric and boolean labels rather than
stringifying them and guessing their types on output. Numeric-looking strings such as
"01" and "1" remain distinct in decoded episodes and adaptation prototypes. Invalid
mixed types and unseen support/query labels refuse; raw-vocabulary validation precedes
sklearn transform because boolean encoders can coerce unknown strings. A persisted
marker selects the new behavior. Unmarked legacy encoders retain historical string
transforms and decode/prototype conventions; bundle format is unchanged.

Seventeen new independent label/centroid consumers and actual agent adaptation/bundle
roundtrips cover prototypical and warm-start fits with string,int,float,bool classes.
Before repair15FAIL2PASS7warnings14.16s;firstrepair16PASS1FAIL40warnings14.71s exposed
boolean unknown-label coercion;final focused17PASS40warnings13.53s. Support means are
computed independently from selected rows;the frozen meta-plan and dataset remain
unchanged. Ruff/doc audit PASS586 existing findings;scoped mypy clean2files,existing
unused-config-section note. No new type ignores or audit budgets.

Final source-frozen five-file verification55PASS1SKIP40 expected flat-alias warnings
68.11s;no collection warnings. The skip is the torch-missing negative control when
Torch is present;real Torch prototypical/MAML consumers ran. Reach vs verified
d2df2ee:5 unchanged previously missing lines (4 workflow,1 availability),14 changed/new
executed production lines. Separate from accepted87.1878% whole-package coverage;
no aggregate projection or floor raise. Scope,coverage7.16.2/566 exclusions and
core70/expanded86 preserved. Fresh remote CI required;local Torch differs from CI.
Evidence reports/meta-labels-{before,after,repaired,final,docstrings,mypy}.log and
reports/meta-labels-{coverage,reach,measurement}.json. No runtime installs/full overlap.

## Agent future exogenous forecasting - 2026-10-24

The agent could fit an exogenous ForecastPlan but its generation schema/dispatch
omitted future_exog despite describing it as required. The tool now accepts a matrix
of finite JSON numbers in fitted column order and forwards it to Session generation.
Malformed rows,booleans and nonfinite values refuse before result/history mutation;
existing shape checks remain shared. Caller-supplied covariates must be available at
the forecast origin or explicitly assumed scenarios;offline holdout covariates do
not establish deployment availability.

Twenty-four real agent consumers independently solve Ridge normal equations with an
unpenalized intercept,then calculate recursive forecasts and expanding one-step,
fixed-origin and windowed-origin holdout paths. Two lag sets,one/two exogenous columns
including reversed order,validation/test and real bundle reloads are checked. Nine
malformed-input cases check frozen plans,data,history and result state. Focusedfinal
33PASS157 expected flat-alias warnings10.73s. Initial fixture split-size assumption
was corrected before reproducing the missing-forward defect. After forwarding,
a bool-vector acceptance failure prompted strict finite numeric-row validation.
Ruff/doc audit PASS586 unchanged findings;scopedmypy has the same7 existing diagnostics
as an archived,gitshow-verified5c6e158 baseline. No ignores/budgets changed.

Final source-frozen five-file verification66PASS0SKIP/FAIL177 expected flat-alias
warnings73.52s;no collection warnings. Reach vs verified5c6e158:4 unchanged previously
missing workflow lines,6 changed/new executed lines (including the isfinite import).
No availability padding. Subset reach is separate from accepted full87.1880%;no
aggregate projection/floor raise. Scope,coverage7.16.2/566 exclusions/core70/expanded86
preserved. Evidence reports/agent-exog-{before,before-forwarding,after,repaired,final,
docstrings,mypy,baseline-mypy}.log and reports/agent-exog-{coverage,reach,measurement}.json.
Baseline archive reports/exog-baseline-source. No runtime installs/full-suite overlap.
Fresh remote CI required independently before promoting any aggregate change.

## Torch resume patience and scalar optimizer consumers - 2026-10-24

Resuming early-stopped training previously reset elapsed patience: a constant-loss
run that stopped at epoch3 uninterrupted stopped at epoch4 when resumed after epoch2.
The trainer now derives consumed patience from cumulative epoch offset minus best
epoch,independent of sparse logged history. Changing monitor,mode or minimum delta
starts a fresh window with a warning. Existing serialized epoch metadata is reused;
no new checkpoint fields or format version are required.

Twenty real Torch consumers independently calculate scalar SGD momentum,gradient
clipping,Step/Cosine/Plateau learning rates and losses across full8epoch vs3+5resume
runs. Constant and moving losses,parameter state,contract mismatch refusals before
weight restore,patience in min/max modes with sparse logs,and changed-monitor windows
are checked. Initial15PASS14.51s;patience regression2FAIL15PASS14.66s;repaired20PASS14.64s.
Ruff/docstrings PASS586 unchanged findings. Scopedmypy has the same3 pre-existing
diagnostics as an archived,gitshow-verified4dbd273 baseline. No ignores/budgets changed.

Final source-frozen three-file verification34PASS1expected Torch-missing-control SKIP,
zero warnings/failures59.36s;no collection warnings. Real Torch consumers ran;newfixture
uses importorskip for core profiles without optionalTorch. Earlier broader log34PASS/
1SKIP65.60s preserved;guard added only after completion,then final rerun on frozen tests.
Reach vs verified4dbd273:19 unchanged previously missing lines (18training workflow,
1availability),10 changed/new executed lines. Separate from accepted87.1973% fullsuite;
no aggregate projection/floor raise. Scope,coverage7.16.2/566 exclusions/core70/expanded86
preserved. Evidence reports/torch-sgd-{initial,before-patience,repaired,first-final,final,
docstrings,mypy,baseline-mypy}.log and reports/torch-sgd-{coverage,reach,measurement}.json.
Archived control reports/torch-sgd-baseline-source. No runtime installs/full-suite overlap.
Fresh remote CI required independently before accepting aggregate changes.

Agent evolutionary search can receive an unfitted `preprocess` object with
`impute`, `scale`, `select`, `select_k` and `select_threshold`. For example,
`{"impute":"median","scale":"standard","select":"univariate"}` supplies
the base recipe for `recipe_space={"select_k":[1,2]}`. Preprocessing is fitted
inside each training fold and refitted on train for the selected model.
This does not undo preprocessing previously applied globally to the Session;
the existing leakage refusal remains in force.

## Agent evolutionary fold-local base recipes - 2026-10-24

The agent exposed recipe_space but omitted the required base preprocess argument.
It now accepts a validated,unfitted base recipe with impute,scale,select,select_k and
select_threshold settings,constructing PreprocessRecipe before the native search.
Unsupported settings/invalid enums/nonmapping objects,nonpositive/fractional/bool
selection counts and negative/nonfinite/bool thresholds refuse before search state
changes. No executable estimator payloads are accepted. Existing Session-global
preprocessing leakage refusals and fold-local fitting/refit rules remain in force.

Twenty-four real-agent consumers independently compute per-fold medians,standard/
minmax scaling,correlation-ranked feature selection and Ridge normal equations with
an unpenalized intercept. Every explored candidate's MSE/sampleSD,winner and refit
predictions are checked;poisoned reserved holdouts do not alter train-foldscores.
GA caching/budget accounting is checked for observed configurations;no claim that
a finite genetic search finds the global optimum. Twelve malformed recipes check
unchanged data/history and no fitted/search result. Focused24PASS10.70s.
Initialmissing-base1FAIL8.89s;first numericmatrix12FAIL12.17s from the oracle using
population SD while classicalCV reports sample SD,corrected after means matched.
Ruff/doc audit PASS586 unchanged findings;scopedmypy same7 existing diagnostics as
an archived,gitshow-verified1bf7160 source control. No ignores/budgets changed.

Final source-frozen four-file verification78PASS0SKIP/WARN/FAIL53.70s;no collection
warnings. Reach vs verified1bf7160:1 unchanged previously missing AI workflow line,
16 changed/new executed production lines. Separate from accepted87.2256% fullsuite;
no aggregate projection/floor raise. Scope,coverage7.16.2/566 exclusions/core70/expanded86
preserved. Evidence reports/agent-evo-{before,after,repaired,final,docstrings,mypy,
baseline-mypy}.log and reports/agent-evo-{coverage,reach,measurement}.json. Archived
control reports/evo-baseline-source. No runtime installs/local full-suite overlap.
Fresh remote CI required independently before accepting aggregate changes.

## Causal assumption booleans and native refutation - 2026-10-24

Python Session/mapping declarations previously used bool() on acknowledgement flags,
so truthy strings/numbers could acknowledge assumptions;the agent already required
JSON booleans. Central CausalAssumptions.validate now requires actual booleans for
unconfoundedness,positivity and empty-confounder waivers. Session and mapping parsers
preserve values until validation. Refusals preserve prior assumptions,data/history.
Literal True empty-confounder waivers remain explicit;declarations do not prove
identification or positivity in a deployment population.

Twenty-four native real-agent refutation consumers independently solve arm-specific
OLS/Ridge nuisance fits and calculate original/placebo/random-confounder ATEs. Three
treatment label types,poisoned reserved holdouts,plan immutability and real trusted
bundle replay are checked. Initial24PASS168 flat-alias warnings10.96s. Seventy-three
flag boundary/control cases across instance,mapping,Session,agent exposed30FAIL/
43PASS9.60s before repair. Combinedfocused97PASS168warnings11.53s. Ruff/docstrings
PASS586 unchanged findings;scopedmypy clean2files,existing unused-config-section note.

Final source-frozen five-file117PASS0SKIP/FAIL168 expected flat-alias warnings72.36s;
no collection warnings. Reach vs verified4e047bc:16 unchanged previously missing
lines (15workflow,1availability),3 changed/new executed lines. Separate from accepted
87.2321% fullsuite;no aggregate projection/floor raise. Scope,coverage7.16.2/566 exclusions
and core70/expanded86 preserved. Evidence reports/agent-causal-{initial,repaired,final,
docstrings,mypy}.log,reports/causal-flags-before.log and reports/agent-causal-{coverage,
reach,measurement}.json. No runtime installs/local full-suite overlap. Fresh remote
CI required independently before accepting aggregate changes. Next candidate evidence:
reports/next-bundle-boundary-gaps.json ranks persisted-bundle failure paths from4e047bc;
not yet tested defects or projected gains.

KG bundle loads validate declared sizes,vocabulary/index alignment,finite embedding
matrices,integer train triples and graph-cache consistency before replacing Session
state. Valid direct-plan and redundant-array layouts remain supported. Complex
representations retain their dtype during restoration;verification of frozen complex
representations exercises serialization/scoring and does not claim a PyKEEN training run.

## KG persisted-state contracts and reconstruction - 2026-10-24

Real fitted-bundle tests exposed malformed contracts accepted at load,missing inbound
adjacency not rebuilt when outbound adjacency existed,and redundant complex embeddings
cast to float with imaginary components discarded. Loading now preserves embedding/
index-array dtype,validates metadata sizes,vocabulary/index alignment,finite embedding
shapes,aligned integer train-id bounds and true-triple/adjacency consistency. Missing
derived caches rebuild independently. Session attachment occurs only after validation.
Valid direct-plan,redundant-array and derived-cache legacy layouts remain supported.

New72-case matrix uses actual native TransE L1/L2 and DistMult fits,independent scores/
adjacency checks and real serialized corruption. Frozen analytic ComplEx/RotatE arrays
verify dtype preservation and independent NumPy scoring;they do NOT invoke or simulate
PyKEEN training. Before repair51FAIL12PASS9ComplexWarnings14.66s;initialrepair63PASS12.01s;
expandedfocused72PASS12.92s. Corrupt load refusals preserve an already-fitted Session's
state by identity and plan fingerprint. Ruff/doc audit PASS586 unchanged findings;
scopedmypy clean1file with existing unused-config-section note.

Final source-frozen four-file108PASS2SKIP0WARN/FAIL48.22s;no collection warnings. Both
skips are real PyKEEN positive tests absent locally;native query/scoring consumers ran.
Reach vs verified45bd975:14 unchanged previously missing lines (13workflow,1availability),
49 changed/new executed lines including2imports. Separate from accepted87.2552% fullsuite;
no aggregate projection/floor raise. Scope,coverage7.16.2/566 exclusions/core70/expanded86
preserved. Evidence reports/kg-bundle-{before,after,repaired,final,docstrings,mypy}.log
and reports/kg-bundle-{coverage,reach,measurement}.json. No runtime installs/full overlap.
Fresh remote CI must independently verify native/optional profile compatibility.

AutoML persistence checks that its selected plan and paired classical `FitResult`
share the same model and ordered feature,target,task and training-row contracts.
Invalid saves refuse before creating the destination;invalid loads leave existing
Session state intact. Legacy direct-plan/missing-fit layouts still reconstruct the
classical fit contract from the plan. Existing AutoGluon asset path guards remain.

AutoML persisted model contract repair (2026-10-24): real fitted regression and
classification consumers reproduce mismatched AutoMLPlan/FitResult metadata or
estimators. Save now refuses before creating a destination; load refuses before
replacing Session state. Legacy missing-fit reconstruction remains supported.
Independent affine predictions and malformed artifact controls: focused48PASS;
four-file frozen subset72PASS,zero skips/warnings/failures78.15s. Ruff/doc audit
PASS586 unchanged findings;scopedmypy clean1file (existing unused config note).
Reach vs verified82556fb:9 previously missing unchanged lines,25 changed/new
executed lines;subset reach is not a whole-package percentage. Local AutoGluon
filesystem controls do not establish installed-backend numerical execution.

Torch SSL sidecar contract repair (2026-10-24): five real tabular methods,
current/legacy direct/dictionary bundles and NumPy affine/ReLU (VAE eval mean)
checks validate restored embeddings. Loader refuses method/input/latent contract
disagreements, duplicate representation names and malformed state before Session
replacement. Tabular weight restore errors become ValidationError.
Focused65PASS0WARN23.94s;four-file frozen subset88PASS6CUDA/visionSKIP,
25 existing facade deprecation warnings,zero failures124.82s;no collection warnings.
Ruff/doc auditPASS586;scopedmypy clean1file,existing unused config note.
Reach vs verifieda2ea071:6 previously missing unchanged lines (3 checkpoint,
3 availability controls),20 changed/new executed including1import. Subset reach
is not a whole-package percentage;no current CUDA/vision/HF training claim.

Ensemble paired-model persistence repair (2026-10-24): actual Ridge and logistic
voting fits with both nonuniform weight orders, regression/classification and
current/direct/missing-fit/legacy-fit layouts retain predictions checked by
NumPy affine, sigmoid and weighted-mean arithmetic. Save refuses inconsistent
plan/FitResult before creating files;load refuses before replacing Session state.
Before repair80FAIL16PASS;focused repaired96PASS;four-file frozen suite114PASS,
zero skips/failures,160 existing facade deprecations48.86s,no collection warnings.
Ruff/doc auditPASS586;scopedmypy clean1file,existing unused config note.
Reach vs verified0e76c4e:4 previously missing unchanged lines,24 changed/new
executed lines including1import. Subset reach is not an aggregate percentage.

Model-driven allocation consumer evidence (2026-10-24):72 new cases check real
logistic-score top-k,exact integer knapsack and continuous LP allocations across
string/integer/bool labels,two budgets,both holdout partitions and empty-eligible
sets. Independent NumPy sigmoid scores,exhaustive integer subset search and
fractional value/cost arithmetic verify selected ids,values,costs,utilization,
selected-row positive rates,in-sample disclosures and frozen bundle replay.
No production defect reproduced;production source unchanged. Focused72PASS0WARN;
four-file frozen subset92PASS0SKIP/FAIL26existing facade deprecations71.33s,
no collection warnings. Ruff/doc auditPASS586. Reach vs verifiedb71f97b:
28 previously missing unchanged lines (including backend catalog/availability
controls),zero changed/new lines. Subset reach is not a whole-package percentage.

Frozen multimodal preprocessing repair (2026-10-24):actual saved object/JSON
contracts with raw/normalized fitting,replaced replay training rows and conflicting
runtime settings revealed geometry/normalization drift. Frozen preprocessing now
owns numeric/image/audio normalization,image geometry,audio rates/lengths and
source rate;runtime batching remains configurable without mutating caller config.
Missing fitted text vocab is refused instead of silently learning again.
Independent train-only NumPy transform/token calculations and original actual
Torch tensors agree after replay. Reliable pre-fix18FAIL;focused repaired18PASS;
four-file frozen suite42PASS2absentONNXSKIP4existing facade deprecations0FAIL69.14s,
no collection warnings. Ruff/doc auditPASS586. Scopedmypy retains the same6errors
as extracted1b71c34 baseline (4fusiontyping,2Pandasstuberrors);not a clean type check.
Reach vs verified1b71c34:4 previously missing unchanged lines,5changed/new executed
lines. Subset reach is not a whole-package percentage or ONNX execution claim.

Fairness positive-class consumer repair (2026-10-24):public fairness AUC and
threshold suggestions previously chose the last probability column when the
caller declared the first class positive;actual AUC0.076923 instead of independent
0.923077 reproduced. LinearSVC hard predictions were also mistaken for scores.
The helper now matches Session proba_LABEL columns and checks predict_proba
availability before the classical label fallback. Existing raw-label layouts
remain supported. Actual logistic,sklearn Pipeline and LinearSVC consumers across
string/integer/bool labels,both positive choices and both holdouts verified by
NumPy sigmoid,pairwise AUC and exhaustive per-group threshold arithmetic.
Pre-fix24FAIL12PASS;repaired36PASS;four-file frozen suite91PASS0SKIP/WARN/FAIL43.59s.
Ruff/doc auditPASS586;scopedmypy clean1file,existing unused config note.
Reach vs verified2ded027:1 previously missing unchanged line,6changed/new executed
lines;this correctness repair does not establish an aggregate coverage gain.
Fairness metrics remain observational;threshold suggestions are opt-in.

Ordered symbolic overlay consumer evidence (2026-10-24):64 new cases compare
actual fitted Ridge/Logistic hybrid outputs against independent affine model
and scalar rule calculations. Overlay/repair modes,regression and raw string/
integer/bool classification,soft threshold nextafter(.5,0)/.5 boundaries,
priority/constraint order,repair counts and complete per-row traces are checked
on validation/test. Trace-off and actual bundle replay retain results without
changing the fitted plan/data. No production defect reproduced;test-only batch.
Focused64PASS;four-file frozen suite113PASS0SKIP/WARN/FAIL225.52s.
Ruff/doc auditPASS586. Reach vs verified09fb559:17 previously missing unchanged
lines,zerochanged/new. Subset reach is not a whole-package percentage.
Accepted fairness repair full measurement87.3433% is slightly below prior87.3485%;
all gates passed and no coverage gain was claimed for that repair.

LTR consumer evidence (2026-10-24):112 new cases use actual Ridge/RankSVM fits,
string/integer query ids,three cutoffs,both holdouts,reverse query order and
current/directlegacy/redundant bundle layouts. Independent train means/std,
Ridge normal equations,RankSVM affine scores and scalar nDCG/MAP/MRR check scores,
ids,metrics and zero-relevance-query skipping. Ten invalid column contracts must
preserve the existing fitted ranker. Legacy payloads derive from real fitted
models;no fake optional backend is introduced. No production defect reproduced.
Focused112PASS;four-file frozen suite134PASS1absentXGBoostSKIP0WARN/FAIL82.24s.
Existing real LightGBM and Torch ranking checks ran;this does not establish
XGBoost execution locally. Ruff/doc auditPASS586. Reach vs verifieda2a802e:
16 previously missing unchanged lines,zerochanged/new;not an aggregate percentage.

Real two-rank DDP numerical evidence (2026-10-24):eight new CPU-only Gloo cases
use BuildML's single-node launcher and a real two-process torchrun-compatible
rendezvous on localhost. Independent NumPy Adam globally averaged gradients for
one/three epochs and shuffle on/off match returned parameters,predictions and
validation loss. Rank-zero result-by-value,rank-one empty result,process-group/
owned-worker cleanup and original loader/RNG preservation checked. The oracle
also distinguishes unaveraged rank-local three-step updates beyond tolerance.
This is same-host functional evidence,not GPU throughput or real multi-host
reliability. CUDA-enabled runtimes explicitly skip these CPU-only checks.
Focused8PASS160.21s;three-file frozen suite16PASS0SKIP/WARN/FAIL389.71s.
Ruff/doc auditPASS586;actual spawned-worker coverage collected by existing
multiprocessing/subprocess instrumentation. Reach vs verifiede9ade35:72 previously
missing unchanged production lines,zerochanged/new. Production source matches
that baseline exactly;subset reach is not an aggregate coverage percentage.


### Probabilistic numerical consumer checks (2026-10-24)

`tests/unit/test_probabilistic_numerical_consumers.py` checks actual native
BayesianRidge and GaussianProcessRegressor output against independent posterior
matrix and kernel solves. Gaussian CRPS is checked by numerical integration of
squared CDF error, separately from the scoring implementation's closed form.
GaussianNB probability columns and NLL, Brier, ECE and accuracy are checked from
class-conditional Gaussian densities for string, integer and boolean labels.

The consumers cover validation/test populations, partially missing targets,
posterior interval alpha changes and trusted bundle replay. Evaluation must retain
the fitted plan and dataset. These checks establish numerical agreement for these
workflows; they do not establish exchangeability, population-wide nominal coverage,
or the behavior of unavailable optional backends. A subset reach count must not be
added to the accepted whole-package percentage.

Local four-file verification passed **61 tests**, with zero skips, warnings or
failures, in 57.25 seconds. The 45 new consumers passed in the focused run.
Ruff and the docstring budget passed (586 existing findings unchanged).
Source-mapped reach against verified `72a39e9` measured **18 previously missing
unchanged production lines**, with zero changed/new production lines. This is
subset evidence; the accepted full-suite result remains 87.5082% pending fresh CI.


The initial new scoring tests exposed a test-oracle compatibility issue in
Python 3.10 CI: scikit-learn through 1.8 computes BayesianRidge predictive variance
on raw features, while 1.9 centers those features. Both versions expose an offset
attribute, so attribute presence cannot select the convention. The test oracle
uses the verified version boundary while retaining an independent posterior
matrix calculation. A passing local subset did not justify accepting the failed
CI run or changing the recorded whole-package coverage.

Corrected verification passed 61 tests with scikit-learn 1.9 (54.95 seconds) and
all 45 new consumers with an actual isolated scikit-learn 1.8 installation
(16.39 seconds), with zero skips, warnings or failures. Ruff and the unchanged
586-finding docstring budget passed. Recomputed subset reach remains 18 unchanged
previously missing production lines, zero changed/new lines. Remote acceptance
is still required; the recorded full-suite percentage remains unchanged.


### Pipeline inference replay and empty-batch boundary (2026-10-25)

`tests/unit/test_pipeline_numerical_replay.py` exercises real saved Ridge and
LogisticRegression pipelines with fitted imputation and scaling. Predictions
are checked with independent fill/standardization arithmetic and affine or
sigmoid calculations. Raw and already-transformed requests cover DataFrame and
Dataset inputs, filesystem and loaded bundles, legacy missing schema contracts,
duplicate row indices, probability class labels and unchanged fitted state.

Malformed numeric values, missing features, non-frame inputs and empty batches
must be refused without changing fitted state. Before the repair, four empty
batch cases leaked StandardScaler ValueError from preprocessing. Scoring now
rejects empty input with ValidationError before coercion or replay, consistently
with replay enabled or disabled. These checks concern local inference correctness,
not long-term reliability or a broader deployment guarantee.

Final three-file verification passed **99 tests**, with no failures or skips,
and two existing datetime parsing warnings, in 60.11 seconds. Ruff and the
unchanged 586-finding docstring budget passed. Source-mapped reach against
verified `05e1c37` measured **2 previously missing unchanged lines and 2 new
executed guard lines**. This is a correctness repair with a small measured reach;
it is not an aggregate percentage increase. Remote full-suite acceptance remains
required before updating the recorded 87.5370% whole-package measurement.


### Multi-target numerical and attachment consumers (2026-10-25)

`tests/unit/test_multitask_numerical_consumers.py` fits real sklearn multi-output
regression and classification plans. Independent Ridge normal equations and
logistic affine decision calculations check per-target predictions over train,
validation, test and all rows, including trusted bundle replay. Classification
fixtures use string, integer and boolean labels.

All-row attachment retains duplicate row indices and the original targets while
adding feature-role prediction columns. Conflicts on either the first or second
output column, missing features or splits, and attempted holdout attachment must
leave the dataset and fitted plan unchanged. These are correctness checks for
local native workflows; they do not validate unavailable optional backends or
long-term deployment reliability. Subset reach is separate from full coverage.

Three-file verification passed **102 tests**, with zero failures, skips or
warnings, in 62.64 seconds. All 68 new consumers passed in the focused run.
Ruff and the unchanged 586-finding docstring budget passed. Source-mapped reach
against verified `e9a7259` measured **18 previously missing unchanged production
lines**, with zero changed/new production lines. No production defect was
reproduced in these workflows. The accepted whole-package measurement remains
87.5390% until a passing full remote measurement establishes the next result.


### Agent temporal workflow numerical consumers (2026-10-25)

`tests/unit/test_agent_temporal_numerical_consumers.py` executes actual registered
tools for role assignment, chronological splitting, imputation, scaling, Ridge
fitting and evaluation. Independent timestamp ordering and row-count calculations
check train/validation/test positions for fractional and absolute sizes, implicit
and explicit time columns, and forward/reversed/shuffled source rows.

Independent train medians and standard deviations check preprocessing on every
row. Ridge normal equations check held-out predictions and RMSE. Duplicate row
indices and original targets must survive unchanged. No provider service or
optional backend is substituted. These finite local consumers do not establish
future stationarity, exchangeability or long-term service reliability.

Three-file verification passed **49 tests**, with no failures or skips, and 16
existing forecast-alias deprecation warnings, in 62.35 seconds. The 36 new cases
passed in 8.29 seconds. Ruff and the unchanged 586-finding docstring budget passed.
Source-mapped reach against verified `01b4c80` was **1 previously missing unchanged
production line**, zero changed/new lines. No defect was reproduced. This is
additional numerical correctness evidence with minimal coverage reach; it must
not be presented as a material aggregate coverage improvement. The accepted
whole-package measurement remains 87.5614% pending fresh remote acceptance.


### Evolutionary numerical candidate consumers (2026-10-25)

`tests/unit/test_evolutionary_numerical_candidates.py` runs actual evolutionary
search with integer, categorical, linear-float and log-float domains. Three
seeds exercise Ridge and decision-tree configurations with full mutation,
crossover, three generations and a fourteen-candidate evaluation budget.

Every evaluated candidate's fold RMSE is checked against independent Ridge
normal equations or a separately fitted reference tree traversed through its
arrays. Fold sample standard deviation and minimum-score ranking are checked.
Parameter bounds must hold after mutation. Repeating searches with the same
seed and extreme replacement of validation/test rows must retain the candidate
sequence and scores; refit=False must leave the Session without a fitted model.
These checks validate search mechanics and population isolation for these finite
workloads, not optimizer superiority or a generalization guarantee.

Final three-file verification passed **32 tests**, with zero failures, skips or
warnings, in 47.31 seconds. Ruff and the unchanged 586-finding docstring budget
passed. Source-mapped reach against verified `01b4c80` was **13 previously missing
unchanged production lines**, zero changed/new lines. No production defect was
reproduced. The initial test oracle used population instead of sample fold
standard deviation; that test-only assumption was corrected before accepted
verification. Whole-package coverage remains 87.5614% pending fresh remote CI.


### EDA readiness evidence consumers (2026-10-25)

`tests/unit/test_dashboard_gate_evidence_consumers.py` builds real classification
and regression EDA reports from clean and missing/duplicate frames. Missing-cell,
duplicate-row, numeric range and target summary evidence is checked against
independent frame calculations. Reports are replayed from memory, strict JSON
and supported legacy VIF/correlation/cardinality shapes.

Readiness status totals must match the actual rows. Framing remains a human
judgment, and causal interpretation remains open for observational data. Gate
judgments are not persisted, and rendering must not mutate report evidence.
These are analytical aids; numerical consistency does not settle substantive
methodological questions or establish causality.

Final two-file verification passed **22 tests**, with no failures or skips, and
one upstream langchain-community deprecation warning, in 234.62 seconds. The 16
new consumers passed in 130.60 seconds. Ruff and the unchanged 586-finding
docstring budget passed. Source-mapped reach against verified `cd0bbe6` was
**8 previously missing unchanged production lines**, zero changed/new lines.
No production defect was reproduced. Whole-package coverage remains 87.5855%
pending fresh remote verification. Session capability probes and package-wide
coverage reporting contribute to local runtime; no backend availability was
substituted to bypass them.


### Offline transformer sentiment label semantics (2026-10-25)

`tests/unit/test_transformer_sentiment_offline_consumers.py` uses actual local
BertForSequenceClassification checkpoints and a saved tokenizer, with model
network access disabled. Zero weights and known classifier biases provide an
independent softmax oracle for signed sentiment scores, neutral output and
threshold behavior across validation/test/all populations.

These fixtures validate inference plumbing and semantics, not sentiment quality:
they are deliberately untrained constant-output heads. Before the repair, 18
unknown-label refusal cases failed because LABEL_0/custom labels were silently
interpreted as negative. Transformer sentiment now requires recognizable
positive, negative or neutral output labels; otherwise it raises ValidationError
with id2label guidance. A generic classifier index does not establish polarity.
The 18 recognized-label consumers already passed before the repair.

Final two-file verification passed **56 tests**, with zero failures, skips or
warnings, in 103.56 seconds. Ruff and the unchanged 586-finding docstring budget
passed. Source-mapped reach against verified `6645945` was **20 previously missing
unchanged lines and 7 changed/new executed lines**. No external model was fetched.
Whole-package coverage remains 87.5951% pending fresh full remote verification;
subset reach must not be added to the aggregate percentage.


### Multi-target persisted contract and Torch replay (2026-10-25)

`tests/unit/test_multitask_bundle_contracts.py` fits real native regression and
string/integer/boolean classification plans. Malformed feature/target contracts,
unfitted or absent models, inconsistent dimensions and stored metadata must be
refused before creating output directories or replacing a Session's existing
learner. Label encoders must agree with the class vocabulary. Valid current and
legacy direct-plan or summary-free bundles retain numerical predictions.

The native pre-fix run had 92 failed refusal cases and 16 passing valid cases.
A separate actual mixed-target Torch consumer reproduced a PicklingError from
the function-local neural module. Torch learners now persist CPU weight arrays
and reconstruct the module with strict weight checks on load, without refitting
or restoring optimizer state. Reload defaults to CPU. Independent NumPy dense/
ReLU/head calculations verify predictions; damaged tensor shapes must fail before
Session attachment. Saving leaves the original model weights unchanged.

Available summaries provide consistency checks, not proof of training provenance.
Legacy bundles without summaries remain readable; semantic history cannot be
reconstructed from estimator dimensions alone. Trusted deserialization is still
required. These local checks do not establish GPU throughput or long-term service
reliability.

Final four-file verification passed **217 tests**, with zero failures, skips or
warnings, in 66.48 seconds. Ruff, mypy (both changed source files) and the unchanged
586-finding docstring budget passed. Source-mapped reach against verified
`6645945` was **4 previously missing unchanged lines and 85 changed/new executed
lines**. This includes contract validation and real Torch persistence, and is not
an aggregate percentage. The accepted full-suite measurement remains 87.5951%
until the changed production source passes fresh remote verification.

Recomputed multi-task bundle subset reach against the newer verified sentiment
source `a81f4c8` remains 4 unchanged previously missing lines and 85 changed/new
executed lines. The newer accepted full measurement is 87.6231% at `a81f4c8`;
the multi-task production repair still needs its own remote acceptance.


### Semi-supervised persisted model and missingness contracts (2026-10-25)

`tests/unit/test_semisupervised_bundle_contracts.py` fits actual LabelPropagation
and SelfTrainingClassifier plans with string, integer and boolean labels.
Independent RBF-weighted label-distribution calculations and logistic score
arithmetic check valid current, direct-plan and summary-free bundle replay.

Malformed fitted shapes, absent/unfitted estimators, label encoders or class-code
mismatches, invalid labeled/unlabeled counts and conflicting stored summaries
must be refused before creating output directories or replacing the receiver's
usable plan. Boolean vocabularies cannot silently become integer vocabularies.
Before the repair, 198 refusal cases failed while 24 valid reloads passed.

Available metadata establishes consistency with the persisted plan, not training
provenance or deployment-population validity. Summary-free legacy artifacts
remain supported; trusted deserialization is still required. These native checks
do not establish behavior for every optional semi-supervised backend.

Final three-file verification passed **251 tests**, with zero failures, skips or
warnings, in 87.06 seconds. Ruff, mypy (the changed source file) and the unchanged
586-finding docstring budget passed. Source-mapped reach against verified
`0d9f52e` was **4 previously missing unchanged lines and 52 changed/new executed
lines**. Whole-package coverage remains 87.6323% pending fresh full remote
verification; subset reach is not an aggregate percentage increase.


### Recommender persisted catalog identity (2026-10-25)

Actual item-kNN, user-kNN, SVD, NMF and content bundles accepted stale nonempty
catalog indexes, duplicate IDs and interaction-matrix dimensions inconsistent
with the catalog. A stale index can silently select another user's ratings or
attribute scores to the wrong item. Before repair, 70 refusal cases failed while
13 valid restoration/metadata cases passed. Save now validates before directory
creation; load validates after legacy array/catalog restoration and before the
plan reaches consumers. Unhashable IDs are refused as validation errors.

The final four-file Windows subset passed **160 tests**, no skips/failures, in
75.15 seconds, with 77 existing deprecated Session-alias warnings from actual
agent consumers. Independent cosine, weighted-rating, content-profile and factor
arithmetic checks cover all five native methods, with train-item exclusion and
valid direct/separate legacy payloads. This establishes catalog consistency,
not artifact provenance or behavior of absent industry backends; trusted loading
is still required.

Ruff and the unchanged 586-finding docstring budget passed. Targeted mypy with
`--follow-imports=skip` passed the changed source file. A broader import-following
check reported **533 errors in 92 files**, with none in the changed checkpoint
file; package-wide type cleanliness is not claimed. Source-mapped subset reach
against verified `90088a5` was **1 previously missing unchanged line and 17
changed/new executed lines**. The accepted aggregate remains **87.6404%** pending
fresh full remote verification; no coverage floor, exclusion or package scope
was changed. An earlier successful 129-test subset inherited the whole-suite
coverage threshold; the final report-only subset explicitly disables that
threshold without changing the full-suite gate.


### Search-domain validation and existing-study preservation (2026-10-25)

The real CI-pinned Optuna 5.0.0 backend accepted some malformed declared domains
through coercion, including boolean bounds, fractional integer bounds and a
string logarithm flag. Other invalid domains leaked backend ValueError, KeyError
or TypeError instead of BuildML ValidationError. Before repair, 14 of 38 public
Optuna/evolutionary refusal consumers failed. Optuna now shares the existing
validated domain parser with evolutionary search, preserving valid linear,
logarithmic, integer and categorical domains.

Domain normalization alone still allowed invalid requests to add failed trials
to a caller's existing study: 19 standalone public API consumers demonstrated
that mutation. Declared domains now validate before study creation or advancement.
Caller-provided callable spaces remain runtime suggestions; this check does not
promise rollback of failures inside arbitrary user callables.

Final six-file Windows verification with real isolated Optuna 5.0.0 passed
**91 tests**, no failures or warnings, in 59.58 seconds. Two existing tests for
missing Optuna intentionally skipped because it was installed. Native GA and
Optuna candidate domains, two-fold RMSE and sample standard deviation were
checked using independent Ridge normal equations. Existing fitted model
predictions remain unchanged after refused searches and searches without refit;
invalid declared domains cannot consume an existing study trial.

Ruff, copy lint and the unchanged 586-finding docstring budget passed. Targeted
mypy reports three no-any-return findings at lines 268, 509 and 757, reproduced
exactly on the verified baseline source; a clean type check is not claimed.
Source-mapped subset reach against verified `c1317b5` was **6 previously missing
unchanged lines and 20 changed/new executed lines**. This is separate from
whole-package coverage, whose accepted result remains **87.6453%** pending a
passing full remote measurement. Package scope, exclusions and floors are
unchanged. The optional backend was installed only under this task's reports
directory; the main environment was preserved.


### CV group shape and configuration refusal (2026-10-26)

Explicit groups must be a one-dimensional sequence with one value per training
row. Short, long and matrix-shaped groups previously leaked raw sklearn or
pandas ValueError through cv_score, grid_search, randomized_search,
evolutionary_search and nested_cv_score. Fifteen refusal cases failed before
the repair while 59 other configuration and valid-group cases passed.
The shared splitter resolver now checks this boundary before backend use.

Seventy-four added consumers cover those five entry points, nested search-family
conflicts, invalid fold counts/strategies/metrics, missing group/time roles and
insufficient groups. Refused requests preserve the existing fitted model,
search result, history, holdout frame and independently calculated predictions.
Valid GroupKFold and nested grouped scores match independent Ridge normal
equations, including duplicated and reversed external Series index labels:
group values retain their positional meaning rather than aligning by index.

The final eight-file Windows subset passed **209 tests**, no failures or warnings,
in 66.17 seconds. One existing missing-Optuna test intentionally skipped because
the real isolated Optuna 5.0.0 backend was installed. Ruff, copy lint and the
unchanged 586-finding docstring budget passed. Targeted mypy retains the same
three no-any-return findings at lines 268, 509 and 757, reproduced exactly on
archived verified `8934aed` source; no clean package-wide typecheck is claimed.
Source-mapped reach against that verified baseline was **8 previously missing
unchanged lines and 5 changed/new executed lines**. Accepted whole-package
coverage remains **87.6701%** pending fresh remote full verification. Package
scope, exclusions and core/expanded coverage floors are unchanged.


### Frozen KG ranking counts and filters (2026-10-26)

Knowledge-graph link prediction accepted fractional counts: k=1.5 could return
two candidates while reporting k=1. Non-boolean filtering controls silently
changed the candidate population. Other malformed counts leaked Python errors,
and NumPy boolean filter flags produced result metadata that JSON could not
serialize. Before repair, 78 of 198 ranking consumers failed; a separate six-case
NumPy-flag serialization check also failed completely.

Link prediction now requires a positive integral count, refuses boolean counts,
requires a boolean filter and normalizes supported NumPy flags to Python bool.
Valid NumPy integer counts remain supported. Invalid requests cannot replace a
previous ranking or append a successful history entry. The embedding state
remains frozen.

Real two-epoch native TransE (L1/L2) and DistMult fixtures cover all three
prediction modes, filtered/unfiltered populations, multiple count limits,
missing/unequal query fields and aligned OOV outputs. Expected rankings use
independent translation-distance or trilinear-product equations and the raw
training partition's known triples. These checks establish count, score and
candidate-population consistency; this batch does not measure ranking quality
on large deployment graphs.

Final four-file Windows verification passed **236 tests**, no failures, skips or
warnings, in 39.76 seconds. Ruff, targeted mypy for the changed source file,
copy lint and the unchanged 586-finding docstring budget passed. Source-mapped
reach against verified `bba9d20` was **17 previously missing unchanged lines and
6 changed/new executed lines**. Accepted whole-package coverage remains
**87.6823%** pending fresh full remote verification. Package scope, exclusions
and core/expanded floors remain unchanged.


### Recommendation and retrieval cutoff contracts (2026-10-26)

Recommendation and RAG retrieval/evaluation paths accepted some malformed
cutoffs by coercion and leaked Python errors for others. Recommender exclusion
flags also accepted non-booleans. Of 230 initial real-consumer checks, 164 failed
before repair. Cutoff validation now consistently requires a positive integral
count, while preserving NumPy integer counts. NumPy exclusion flags normalize
to ordinary boolean metadata. Retrieval k=None remains an override sentinel
that selects the configured cutoff; an actual configured/evaluation cutoff
cannot be None.

Recommender evaluation had described its metrics as micro averages even though
it computes an equal-weight mean over scored warm users. Documentation and
point-of-use disclosures now correctly identify macro averaging. Independent
per-user hit, recall, discount and average-precision counts verify the published
averages with unequal relevance support; the numerical averaging is unchanged.

The final five-file Windows subset passed **405 tests**, no failures or skips,
in 86.77 seconds. Its 77 warnings are existing deprecated Session-alias calls
from agent consumers. The new consumers exercise all five native recommender
methods and dense/BM25/hybrid local retrieval, direct/configured/evaluation
cutoffs, refusal-state preservation, valid defaults and JSON result metadata.
Independent dense cosine scores and rank-metric counts supplement the existing
numerical recommender consumers. These are contract/regression checks; the
maintained comparative retrieval benchmark remains separate.

Ruff, copy lint and the unchanged 586-finding docstring budget passed. Targeted
mypy on six source files retained 10 findings in two files, all matched to
unchanged source lines and messages from verified `cff4e9d`; no new finding was
observed and no clean package-wide typecheck is claimed. Source-mapped reach
against that accepted baseline was **3 previously missing unchanged lines and
17 changed/new executed lines**. Whole-package coverage remains **87.7102%**
pending fresh full remote verification. Scope, exclusions and coverage floors
remain unchanged.

### Provider plan structure and argument retention (2026-10-26)

Provider planning previously accepted malformed JSON structures or raised raw
AttributeError/TypeError, and silently discarded each operation's parameters.
The response is now validated as a whole before steps are constructed. Invalid
structures return the existing empty failure result with a schema reason, usage,
raw response and egress manifest. Valid arguments survive into confirmed and
explicit autonomous execution; optional text-list defaults remain supported.

The seven-file regression set passed207 tests with zero failures/skips in108.84s.
There was one upstream langchain-community deprecation warning. Eighty-four new
consumers cover malformed roots/steps/fields, accounting and audit records,
confirmation pauses, unmapped operations, missing estimator refusal and advisor
read-only tool rounds. Actual split populations match an independently generated
permutation, and real fitted predictions match NumPy least-squares arithmetic in
both execution modes. The builtin MockProvider scripts the protocol offline;
these checks do not measure an external model's planning quality.

Source-mapped reach against verified2c8b7c4 is42 previously missing unchanged
lines and30 changed/new executed lines. These are subset counts, not a combined
whole-package percentage. Ruff, targeted mypy, copy lint and the unchanged586
finding docstring budget passed. The accepted remote full result remains87.7196%
(54,930/62,620;6,838 passed,77 skipped), with the90% target still open. Core70,
expanded86 and566 exclusions are unchanged; fresh full CI must pass separately.
