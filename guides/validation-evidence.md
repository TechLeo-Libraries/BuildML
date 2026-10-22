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
| Current accepted remote expanded CPU whole-package gate | Passed:86.4734% (53,687/62,085);3,801 passed,77 skipped | [CI39804246676](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39804246676), source `4c06bd3`, tested merge `414d33ee`; trees match and all three workflows passed.566 exclusions unchanged;floor85 set from this passing86.4734% gate;90% gap2,190 lines. Evolutionary nested numerical audit accepted;1900 pytest warnings, no coverage collection warnings |
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
