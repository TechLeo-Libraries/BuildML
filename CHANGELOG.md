# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with pre-release tags for alpha (`aN`) builds.

Each entry describes its release at the time it was prepared. Historical
installation instructions, dependency limits, and validation counts may differ
from the current release; use the installation guide for current requirements.

## [Unreleased]

### Validation

- EDA list-format mutual-information ledgers preserve measured zero values and
  primary-field priority. Shared, static HTML and cockpit consumers agree with
  dictionary-format reports instead of showing zero as unavailable.

- Binary segmentation now uses the documented penalty instead of forcing a
  length-based break count. Changepoint inputs require finite real vectors
  and finite non-negative numeric penalties; real ruptures consumers verify
  break locations and independent segment means. Recompute earlier BinSeg output.

- Core NumPy PACF updates all Levinson-Durbin coefficients, correcting values
  from lag three onward. Independent dense Yule-Walker solves verify the
  recurrence and affine invariance. Recompute earlier core-fallback diagnostics.

- Native first-order MAML transfers adapted query gradients to its base model.
  Previously the outer optimizer silently left the initialization unchanged.
  Independent SGD/Adam calculations verify updates; existing native MAML plans
  need refitting to benefit. The learn2learn path is unchanged.

- Semi-supervised text inference accepts Pandas string feature columns. Offline
  tiny-encoder consumers verify self-training probabilities, pseudo-labels,
  holdout isolation and trusted bundle reloads against independent calculations.

- Real autonomous Session consumers verify failure audits, stop/continue policy,
  skipped-step iteration limits and destructive-step refusal. Continued fits
  match independent sklearn models; no new product defect was found.

- DuckDB positional filters use distinct source views so chained masks do not
  create recursive views. Non-boolean masks are refused before row selection;
  real SQL and injected Arrow-import-failure consumers match positional oracles.

- SkopeRules uses the maintained imodels implementation when available. Exported
  predicates preserve positional feature aliases and reject unsupported conjuncts
  instead of silently broadening rules; independent real-consumer masks verify this.

- LightFM side-feature preparation refuses missing, non-finite and complex
  inputs before sparse conversion. Real SciPy consumers verify entity means,
  catalog ordering, constant-feature scaling and matrix-vector products.

- New SSL heads preserve raw class labels and JSON-safe summaries; evaluation
  compares encoded identities and refuses unseen labels instead of conflating
  numeric and textual classes. Older stored string-label heads retain their
  legacy evaluation contract with an explicit refit warning.

- Cost-matrix policy fit summaries report probability-weighted expected total
  loss as expected cost. Observed-label realized costs remain separate metrics;
  independent multiclass oracles verify reordered cost axes and held-out loss.

- Real blocking serving CLI consumers verify linear-model HTTP predictions,
  API-key/Basic authentication, credential-free startup logs and graceful
  shutdown with released loopback ports.

- Automatic ARIMA disclosures distinguish an explicitly supplied order from a
  completed AIC grid search. Real statsmodels consumers verify the restricted
  AIC winner and resulting forecasts against independent direct fits.

- Causal partition bootstrap counts now describe performed resamples. DoWhy
  and EconML disclose unsupported requests and report zero with no interval.
  Invalid negative, boolean, fractional and non-finite budgets are refused.
  Native bootstrap quantiles and spread match independent NumPy resampling.

- DoWhy partition estimation now returns a fresh effect for the requested rows.
  Evaluation no longer overwrites that holdout effect with the stored training
  ATE. Native/EconML evaluation still reuses fitted nuisances; the differing
  DoWhy procedure is disclosed explicitly.

- Expanded Linux coverage passed86.0131%; its floor rises to85% under the
  one-point headroom policy, preserving package scope and exclusions.
- Dashboard readiness context preserves measured zero complete cases, target
  median, anomaly counts and scored rows instead of substituting defaults.

- LangChain generation applies the positive integer passage budget to actual
  chain input, recorded context and citations consistently. Offline framework
  consumers check prompts without contacting a model service. A pinned
  LangChain profile is submitted for independent expanded Linux validation.

- Real Session SHAP consumers verify train/validation/test selection, capped
  sampling and feature order against known linear effects and tree prediction
  additivity, including duplicate input index labels.

- DuckDB native readers explicitly disclose their lazy relation handle and
  close the owned connection when reading fails. Real multi-format previews
  check row limits, quoted paths, values and connection ownership.

- Synthetic sample validation explicitly fails unparseable, non-finite and
  complex numeric values and fractional integer values. Summary metrics use
  finite observations. Tolerances require valid finite bounds, and duplicate
  sample columns are refused before validation.

- Disclosed online batch-refit fallback fits a fresh estimator on cumulative
  Session train rows. It refuses external-frame refits whose complete history
  cannot be reconstructed, rather than silently dropping earlier stream data.
  Native incremental estimators continue to support external updates.

- Recommender bundles rebuild missing catalog indexes for both direct legacy
  plans and separate-state payloads. Invalid JSON metadata is refused with a
  validation error before pickle loading. Real fitted model restoration checks
  preserve recommendation scores and train-item exclusions.

- SB3 online training reports actual executed interactions and requested counts
  separately, disclosing rollout overruns. Invalid training/evaluation counts
  are refused. SB3 readiness no longer requires the separate imitation runtime.
  The candidate expanded profile includes SB32.7.0 for full CI verification.

- EconML policy-tree refutation now uses the doubly robust policy learner
  and mean adjusted training effects, matching policy fitting rather than
  calling the incompatible raw PolicyTree API and reporting an arm contrast.

- Multi-task decoding preserves numeric-looking string classes rather than
  parsing them as numbers. Mixed targets reject nulls/invalid head metadata,
  regression targets require finite values, and train/all evaluation identifies
  in-sample metrics. Incompatible class-label types are refused explicitly.

- Repair real scikit-activeml host execution: compatible missing labels,
  fitted committee wrappers and pool-only utility mapping. Pin host1.0.1 in
  the candidate expanded CI profile and require real imports. Numerical host
  consumers passed on Windows; the expanded Linux profile needs full CI acceptance.

- Binning refuses generated-column overwrites, preserves missing values in
  every one-hot indicator, and learns edges from finite training values only.
  Bin counts must be integers of at least two; scoring keeps frozen edge bounds.

- Random over/undersampling preserves complete donor rows, including IDs,
  groups, timestamps, weights and ignored columns, using positional sampler
  provenance. Synthetic resampling refuses auxiliary metadata instead of
  inventing mode values. Resampling also refuses datasets with no feature
  columns rather than falling back to ID/ignored columns.

- Expand the full Linux assessment to include DoWhy 0.14 using a compatible
  NumPy 2.4.1/SciPy 1.15.3/mlxtend 0.23.4 stack. Same-source complete local runs
  passed at 85.5192% (broader profile) and85.3947% (original):80 gained lines,
  three lost, net 77. Five DoWhy positive cases execute; its missing-package
  case skips. Scope, exclusions and coverage floors remain unchanged. Remote
  acceptance of the broadened profile is verified separately.
- Require ONNX Runtime 1.31.0 in the expanded assessment, matching the passing
  local full profile so its real ONNX consumer checks also execute remotely.

- Scoring-contract coercion refuses numeric/datetime conversions that would
  discard non-null values and rejects boolean inputs outside0/1/null. Invalid
  columns remain intact, while valid conversions preserve existing nulls and
  row indexes. Persisted pipeline scoring tests cover null-capable estimators.

- Recommendation exclusions are enforced as catalog eligibility before ranking
  and fallback. Seen training items cannot reappear with zero backend scores,
  including oversized kNN requests and degenerate content profiles. Real agent
  checks cover five native methods, independent scores/metrics and bundle trust.

- Real knowledge-graph agent consumer checks cover native TransE L1/L2 and
  DistMult fitting, frozen triple scores, candidate ranks, train-only queries,
  independently calculated filtered metrics and bundle trust boundaries.
- Same-source whole-suite coverage at `6e072fd`: remote85.2237% and local
  Linux85.3207%; no test failures or local coverage collection warnings.

- Dashboard CSV exports choose the highest-count top category and preserve
  target summary statistics plus exploratory interpretation and limitations.
  Consumer checks independently verify correlations, missingness, PCA curves
  and categorical association charts across supported report layouts/themes.

- RAG bundle restore refuses unreconstructable embedding backends, incompatible
  vector dimensions and non-finite vectors. It no longer silently substitutes
  hashing for arbitrary embedding spaces. Explicit HashingEmbedder instances
  now record the reconstructable hashing backend rather than callable. Old
  bundles recorded as callable require rebuilding with their original embedder.

- RAG directory ingest ignores matching folders. Retrieval evaluation rejects
  empty mapping labels and scalar relevance collections with domain errors,
  instead of splitting strings into characters or leaking TypeError. Real
  local index tests verify all document/chunk label forms and rank metrics.

- Autonomous runs now validate boolean permission/policy flags and positive
  integer step caps before any planning or mutation. Truthy strings and numeric
  permissions cannot grant autonomy; NumPy integer caps remain supported.

- Raise expanded Linux coverage floor to 84% from passing whole-suite CI at
  85.0410% (3,125 passed, 91 skipped), source `3d35017`. Local Linux passed
  at 85.1397% (3,153 passed, 63 skipped), without collection warnings.
  Package scope and exclusions remain unchanged; the 90% target stays open.

- Export/serving hand-off checks consume real relocated TorchScript/ONNX models.
  Generated handlers preserve request groups, packaging references remain relative,
  and recipes quote filenames. Serving model selectors are isolated; Indexed DDP
  pods have the matching DNS subdomain and quoted script paths. Invalid scripting
  methods are refused and multi-output dynamic-axis metadata covers every output.
  A reusable reach report maps unchanged source lines against a full-suite baseline;
  the 25-case Linux workflow exercised 54 previously missing production lines.

- Required FLAML integration skips only when the package is absent. Its bounded
  import probe now uses the 90-second scientific-import policy. Actual fitting
  and recovery checks passed; the prior skipped probe's cause remains unestablished.

- Torch continual initialization honors its seed without consuming ambient CPU
  Torch RNG state. EWC consolidates learned first-buffer weights and estimates
  empirical Fisher diagonals from per-example squared gradients. Invalid stream
  updates are rejected before buffer/model contract changes; replay copies and
  row/label alignment have numerical checks. The batch adds 29 cases; broader
  Linux checks passed all 56 tests, while Windows passed 55 with one River-absence
  skip. Full-suite coverage is measured separately.

- Remote whole-package Linux coverage at `18f0199` passed at 84.96% (3,104
  passed, 85 skipped), with the complete CI, backend evidence and release
  acceptance workflows passing. The newer local `d9af9b6` run passed its tests
  at 84.88% but skipped installed FLAML and reported an unreadable worker coverage
  file. That qualified local result does not set floors. The 90% goal remains open.

- Meta-learning adaptation reports classes actually present in support and
  rejects missing support targets before per-class sampling. Explicit task
  columns must identify one consistent task. Episode sampling preserves user
  feature columns, including `__y_code__`, without temporary-column collisions.
  Independent support centroid/refit and row-integrity checks passed with the
  existing meta-learning backend group: 54 passed and one expected dependency
  skip on each of Windows and Linux; subset results are not coverage figures.

- Active-learning annotations use dataset row positions rather than pandas
  index labels, reject duplicate positions and missing/marker labels, and
  publish budget/history changes only after successful refit. Failed model or
  committee refits leave the Session unchanged; new classes remain supported.
  Acquisition scores and holdout isolation match independent probability/vote
  calculations. The broader Linux active-learning group passed all 56 tests.

- Federated evaluation identifies its requested partition without claiming
  verified exclusion from original training data. Training/all-data scores
  warn about possible training overlap. Distinct client ids with colliding
  string report keys are refused instead of silently overwriting metrics.
  New frozen-population and binary causal numerical contracts passed alongside
  existing Linux federation/Flower/causal checks: 75 passed, one expected skip.

- Probabilistic fitting and calibration select SplitPlan row positions rather
  than pandas index labels. Offset and duplicate labels no longer cause missing-row
  errors; permuted integer labels cannot substitute other rows. Real native,
  MAPIE split and NGBoost regression contracts check index and holdout isolation.
  Independent grouped/time-ordered nested-CV oracles and frozen probabilistic
  evaluation checks cover scaling, missing labels, empty populations and schema
  failures. The combined Linux regression run passed 101 tests with one expected
  dependency-state skip; this subset is not a coverage measurement.

- Polars Parquet/IPC previews pass row limits to the reader before table
  materialization. Design-frame preparation materializes repeated requested
  columns once, preserving first-occurrence order across engines.

- Torch search and cross-validation seed model construction for every fold.
  Malformed CUDA indices are refused even without CUDA. A seeded sampler's
  internal failure propagates instead of triggering an unseeded retry.

- The required imodels integration skips only when the package is absent,
  making installed-backend failures visible. Its scientific import probe uses
  the existing bounded 90-second heavy-import policy.

- AutoGluon probability outputs follow the NumPy estimator contract. AutoML
  bundles clone backend model assets into a relative directory and can load
  after relocation without the original training directory; trust and path
  containment checks remain mandatory.

- Single-node Linux Gloo workers use forkserver to avoid native cleanup racing
  Python finalization. All workers are joined before loading the serialized
  result; injected worker failures must propagate. Fifty repeated checks passed
  locally on Python 3.11/Torch 2.14 CPU and on each remote Python 3.11/3.12 stack
  at `7eec279`. This is a tested launcher workaround, not an upstream Torch repair.

- AutoGluon classification distinguishes binary and multiclass targets, preserves
  weighted F1, normalizes signed loss scores before ranking, and reports internal
  validation scores instead of training scores. GBM seed hints and class labels
  are retained. A separate compatible profile exercises actual CPU fits.

- Categorical EDA effect sizes are checked against independent SciPy reference
  values, including degenerate tables and the documented pair cap. Guidance
  explicitly identifies uncorrected Cramér's V as an exploratory effect size.

- Session import no longer probes optional HDBSCAN/UMAP runtimes. Clustering
  defaults resolve on use, including the existing constant and wildcard imports.
- Knowledge-graph queries reject fractional, boolean and string hop limits;
  graph oracles cover bounded shortest paths, cycles, direction and vocabulary
  failures.

- TDA subsampling maps global split positions to train-frame positions instead
  of using them as Pandas labels or indexing a reduced Series twice. Custom
  indices, stratification, holdout membership and cap validation have regression
  contracts. Transient import-probe failures can recover after a bounded cooldown;
  scientific probes retain bounded cold-start allowances.

- Blending no longer retries internal weighted-fit `TypeError` failures without
  weights. The unweighted fallback applies only when the fit signature rejects
  `sample_weight`. Public export checks now include seven additional domains
  and verify star-import identities.

- Weighted holdout blending is checked against independent sklearn fits across
  refitting, passthrough and pandas/NumPy inputs. Unfitted prediction now raises
  the documented `NotFittedError` for both blend classes.

- Capability-worker imports no longer construct the complete Session API.
  Session loads on first access while preserving the public class and import
  syntax; import-boundary contracts protect lightweight dependency probes.

- Repeated DDP checks retain opt-in rank lifecycle traces, including successful
  process-group teardown. Native worker aborts remain failures without retries;
  the intermittent CI abort is still under investigation.

- The immutable `58e4309` suite passed at 84.12% locally and 84.06% remotely;
  its complete CI, backend matrix and release acceptance passed. Windows/Linux
  paced 24-hour HTTP assessments completed with zero recorded errors and
  unchanged source hashes, on the documented local workload.
- Expanded Linux coverage floor raised to 83% after its full remote suite passed
  at 84.02%; local CPU/CUDA coverage passed at 84.08%. The overall matrix still
  failed separate Torch DETAIL-mode diagnostics; this is not an all-green claim.
- Torch CI repeats buffered two-rank validation on each Python stack, recording
  the installed Torch version, every attempt and native failure diagnostics.
  A failed or skipped attempt fails the check; repetitions are not retries.
- Core Linux coverage floor raised from 69% to 70% after all four Python
  configurations passed at 71.28-71.29% with corrected thread/process tracing.
- Coverage records both worker threads and multiprocessing. A matching six-test
  HTTP comparison confirmed omitted threaded execution; fresh whole-suite results
  must qualify this measurement change separately from earlier percentages.
- Expanded whole-package CI coverage floor raised from 81% to 82%, following
  a passing 83.02% remote CPU measurement; the 90% target remains outstanding.
- Real HTTP load/recovery runner records source identity, dependencies, sampled
  latency and process-tree RSS; it distinguishes deliberate downtime from load.
- Isolated giotto profile verifies the compatible numerical stack and real graph
  contracts separately from the newer expanded stack.

### Fixed

- Fold-local supervised selection uses target type rather than a cardinality
  cutoff, honors explicit F-test scorers, refuses infinite numeric targets,
  and retains a serializable fallback when thresholding would discard all features.
  The default univariate scorer is now `auto`; explicit choices remain supported.
- CV and search propagate the resolved task into automatic feature selection,
  including integer-valued regression. Fold-local date parsing handles valid
  mixed formats consistently across batches and rejects mixed timezones clearly.
- Tabular SSL agent tools expose native methods and forward training budgets.
  SSL random_state now seeds Torch initialization and stochastic augmentations,
  in addition to NumPy shuffling, for repeatable fits on the same backend.
  Invalid budgets and non-finite feature matrices fail before model construction.
- Agent integer arguments and checkpoint partition sizes reject malformed values
  before mutation. DuckDB Parquet fallbacks use unique temporary registrations
  and release them after SQL failures as well as successful writes.
- Evolutionary search rejects boolean, fractional and non-finite budgets before
  fitting, instead of truncating evaluation limits or raising low-level errors.
  NumPy integer budgets remain supported.
- TorchScript serving places inputs on the model's parameter/buffer device and
  floating-point dtype. Empty, ragged, unbatched and non-finite inputs return
  client errors; non-finite model outputs return a sanitized server error.
- Cost-sensitive threshold tuning considers every observed probability breakpoint
  and all-positive/all-negative policies, including optima below 0.05 or above
  0.95. Sorted prefix counts avoid quadratic rescoring of the partition.
- Statsmodels rolling forecasts condition on prior observed history with fixed
  parameters, instead of repeating the forecast from the training endpoint.
  Exogenous history remains aligned across validation and rolling windows.
- DuckDB preserves embedded quotes in projected, sampled and aggregated column
  names, and reads Parquet directories containing apostrophes.
- SHAP attribution rejects invalid sample budgets and malformed/non-finite
  output dimensions rather than producing empty or misleading summaries.
- Explicit serving arguments take precedence over configuration, including
  `trusted=False`, loopback binding and CPU placement. Omitted arguments inherit
  configuration; permission flags still reject non-boolean values.
- Agent training requests preserve registered AutoML, probabilistic, symbolic,
  neuro-symbolic, recommender and ranker options instead of silently dropping them.
- Native ANN capability probes contain interpreter crashes on every platform;
  expanded CI requires a portable source-built hnswlib and a real ANN contract.
- Concurrent serving apps retain independent models through either shutdown
  order. Unicode Basic credentials no longer fail during secret comparison;
  boolean and fractional ports are rejected instead of coerced.
- Custom CV splitters reject repeated, negative, fractional and out-of-range
  positions before fitting. Conformal carving preserves observation IDs and
  rejects duplicates and invalid minimum sizes. Null-target evaluation retains
  calibrated MAPIE intervals/sets on the same labeled population.
- Giotto Mapper summaries count the real igraph vertices and edges.
- The RL extra's SB3 minimum matches imitation's supported series. Real BC
  uses the trainer's constructor/optimizer API; GAIL uses a vectorized environment,
  seeds its reward network, and archives models without live environment state.
  Variable horizons require an explicit choice and retain risk disclosures.
  Agent imitation requests preserve the selected method and training budget.

- DoWhy causal graphs preserve dataframe names containing spaces, quotes,
  punctuation and Unicode by passing native node IDs rather than interpolated DOT.
- Doubly robust policy trees bound depth and leaf size to reduce unstable rules
  from small noisy leaves; their numerical profile is shared across platform CI.
- Evolutionary search rejects missing, nonfinite, boolean, and fractional integer
  bounds before fitting; logarithmic domains require positive bounds and a
  boolean flag. Invalid declarations raise actionable ValidationError messages.
- EconML policy-tree fitting uses the observed-outcome API of DRPolicyTree.
  ATE and neighborhood effect dispersion come from cross-fitted reward estimates;
  reports disclose omitted bootstrap intervals and the actual refit count.
- Fold-local model selection clones caller-supplied estimators and rejects null
  string labels before factorization. Outlier fences and bin edges learn only
  from finite training values, refusing columns with no finite observations.
- Calibration ECE weights quantile bins by their actual sample counts when
  probability ties produce unequal bin sizes, and reports bin counts and method.
- EDA CSV exports preserve fields from later heterogeneous rows, including
  drift statistics and multivariate anomaly measurements. Dashboard JSON maps
  undefined nonfinite measurements to null instead of failing serialization.
- Segment error diagnostics reject unknown partition names instead of silently
  analyzing the test set.
- Managed serving refuses startup timeouts, stops the unready worker and uses
  a monotonic readiness deadline.

- AutoML leaderboards rank loss metrics in ascending order and unavailable
  scores last, with score gaps measured in the correct direction.
- Text SSL installs training dependencies, respects the requested device and
  leaves unavailable final training losses unset instead of reporting zero.

- Federated classification retains encoded training classes and decodes arbitrary
  original labels correctly. Client evaluation restores classifier metadata.
- Ensemble contribution reports decode internally encoded base labels; binary
  average precision uses the fitted positive class instead of assuming label 1.
- Zero-budget knapsack allocation keeps eligible positive-value zero-cost items.
- The optimization extra uses HiGHS below 1.15 to avoid its native symbol conflict
  with OR-Tools after CVXPY import; CI checks both import orders.

- DuckDB lazy row masks retain independent sidecars, so later filters cannot
  replace the row selection of earlier unmaterialized relations.
- Deserialization requires literal `trusted=True`; serving APIs reject non-boolean
  permissions, and provider error tracebacks suppress secret-bearing exceptions.
- RuleFit indexed features map to the fitted input columns and zero coefficients
  remain zero; unknown online, probabilistic, symbolic and RL backends fail clearly.
- DDP preserves requested shuffle behavior. Torch consistency inference disables
  dropout, and concept-bottleneck predictions follow the fitted model device.
- OR-Tools feasible-only knapsack solutions disclose that optimality is unproven.
  The exact PuLP/CBC path explicitly requests zero relative and absolute gaps.

- Retrieval-grounded advisors now use the actual retrieval entry point and
  retain document IDs in source evidence instead of silently omitting context.
- Single-node DDP no longer serializes an unused training-loader random
  generator; rank samplers still use the configured seed and caller state is preserved.

- Coverage enforcement rises to 69% from a passing Linux CI monolith at
  70.04–70.05% across Python 3.10–3.13 (1,645 passed, 261 skipped). The broader local CUDA environment
  remains a separate measurement; 90% is still a target.

- Agent tool execution preserves explicit backend and online drift settings.
  Non-boolean permission values cannot enable bundle trust or test-set tuning.
- Invalid neuro-symbolic backend names fail explicitly rather than selecting
  a different model implementation.

- The breast-cancer example now encodes `malignant=1`, so positive-class
  probabilities and threshold costs refer to malignancy.
- Session conformal fitting rejects previously fitted Session preprocessing,
  which may have used future calibration rows. Calibration carving no longer
  stratifies on labels; unsupported fit/calibration class combinations fail clearly.
- Explicit Windows industry ANN runs in a persistent isolated worker with
  bounded waits, fault containment, retention and lazy restoration.
- BM25 retrieval reuses its sparse index until the corpus or parameters change.

### Added

- Real locally generated transformer checkpoints verify pooling, embedding
  widths, persistence and weight updates without model downloads.
- Independent graph-ranking and adjacency oracles and evolutionary search
  budget checks cover frozen prediction and held-out data boundaries.
- Coverage measures spawned Python and multiprocessing workers using the
  supported coverage.py instrumentation, with existing exclusions retained.

- Agent/direct workflow parity, holdout-preservation and bundle-trust regression
  checks across native fitting domains; bounded latency sampling for longer soaks.

- EDA interpretation metadata and full-family Benjamini–Hochberg q-values for
  target tests and numeric drift screens.
- Real M4 hourly and BEIR SciFact comparative benchmarks, conformal simulation,
  backend evidence reports, and bounded scale, serving and CUDA verification tools.
- Windows/Linux backend evidence CI and a maintained validation evidence guide.

## [2.6.4] - 2026-10-04

### Fixed

- Documentation examples include their imports, demonstration data, and setup.
  Dashboard lessons preserve complete examples and distinguish synthetic
  demonstrations from findings about the uploaded data.
- Dashboard examples demonstrate missing values, duplicate observations,
  feature interactions, and partial dependence with the appropriate data and
  calculations. Explanations and API guidance have been corrected.
- EDA handles missing nullable boolean and integer features, and regression
  targets with only one observed value. Mutual information treats categorical
  and boolean features as discrete variables.
- Giotto retains requested homology dimensions when their diagrams are empty,
  keeping feature values and names aligned between training and prediction.
- PuLP optimization accepts both legacy status codes and PuLP 4 solver results,
  while continuing to require a proven optimal solution.
- Dashboard tables identify missing-value percentages, role charts account for
  undeclared columns, and navigation safely handles charts still loading.
  Readiness gates distinguish sampling from unrelated report warnings.
- Text-only CBR embeddings no longer require an unrelated numeric feature.
  Embedding and Torch retrieval use exact search when ANN is disabled on Windows.
- SDMetrics accepts SDV metadata objects, and CTGAN rejects invalid batch sizes
  before training. Synthetic categorical samples preserve their category types.
- PyG uses the Torch tensor API, Giotto vectorization preserves one row per
  sample, and Stable-Baselines3 handles array-valued discrete actions.
- Causal bundles serialize NumPy treatment values, and causal-forest bootstrap
  models use a valid number of estimators.
- Distributed Torch training returns its fitted model after worker shutdown
  and runs validation on every rank. Unsupported distributed early-stopping
  and plateau-scheduler combinations fail before launching workers.

### Changed

- The LangChain adapter uses the separately declared `langchain-classic`
  dependency and its invocation interface.
- LightFM's extra excludes Python 3.12 and newer; the recommender guide
  documents a tested Linux/Python 3.11 source-build procedure.
- Optional-backend probes allow additional startup time for scientific stacks
  that otherwise could be incorrectly reported as unavailable.
- CI checks documentation examples for syntax and missing imports, alongside
  regression checks for generated dashboard examples and user-facing wording.

## [2.6.3] - 2026-09-23

### Fixed

- MAPIE jackknife-plus regression uses leave-one-out cross-conformal fitting.
- Fold-local preprocessing implements the scikit-learn estimator interface.
- Torch adapters safely handle read-only input arrays, and label validation
  rejects unknown labels before categorical conversion.
- Online passive-aggressive learning supports current scikit-learn releases.
- PuLP optimization supports current solver interfaces and requires an optimal
  solution before reporting an exact result.
- Reinforcement-learning examples use supported Gymnasium environments.
- Documentation references, source links, and workflow explanations have been
  corrected and checked against the implemented behavior.

### Changed

- Documentation builds enforce strict reference checks. CI exercises real
  optional backends and installed wheels across Windows, Linux, and macOS on
  Python 3.10–3.13.
- Tests use supported facades and explicitly check expected compatibility
  notices, making unexpected warnings easier to identify.
- Releases use PyPI Trusted Publishing with a protected GitHub environment
  and verification of the tested distribution hashes and source commit.

## [2.6.2] - 2026-09-17

### Changed

- External docs state the Session as a unified, stateful ML lifecycle
  (leakage safeguards, fold-local preprocessing, teaching, workflow
  guidance, checkpointing, auditable export) without first-or-only
  library claims.
- RAG Session docs and teaching overlays no longer say core ingest,
  chunk, evaluate, save, or load require ``buildml[rag]``. That extra
  is only for sentence-transformer embeddings and rerank. Hashing,
  BM25, and the NumPy index run on core.
- ``buildml[metalearning-industry]`` installs ``learn2learn`` only when
  ``python_version < '3.12'``. Windows 3.12 fails the Cython build
  (``longintrepr.h``); 3.13 still hits qpth/numpy pins. Native
  first-order MAML/Reptile stays on the industry path.
- Releases ``2.4.0a3``, ``2.4.0``, ``2.5.0``, and ``2.6.0`` were yanked
  from PyPI because their wheels omit ``operation_index.json`` and fail
  to import. Install ``2.6.1`` or later. The legacy 1.x line remains available.

### Added

- ``Session.ingest`` reads ``.xlsx`` / ``.xlsm`` when ``buildml[excel]``
  (openpyxl) is installed. Workbooks stay on the pandas path; Polars and
  DuckDB engines are refused with an explicit error.

### Fixed

- At this release, `release.yml` used the `PYPI_API_TOKEN` secret and
  `skip-existing` to handle previously uploaded distributions. Version 2.6.3
  replaced this arrangement with protected Trusted Publishing.
- `examples/multitask_multioutput_loop.py` and
  `examples/online_partial_fit_loop.py` print ASCII ``->`` instead of
  ``→``, so a Windows cp1252 console no longer raises
  ``UnicodeEncodeError`` after a successful run.
- UMAP availability no longer uses the 12s Torch-oriented subprocess
  timeout. First Windows import of ``umap`` / numba can exceed that and
  made ``buildml[unsupervised]`` report missing while the package
  imported in-process.
- Graph fit disclosures use ASCII ``<->`` so Windows cp1252 consoles can
  print ``fit.to_dict()``.
- PyKEEN 1.11 ``TriplesFactory`` requires ``entity_to_id`` /
  ``relation_to_id``. The KG industry adapter now supplies those maps so
  ``session.kg.fit(backend='pykeen')`` runs against current PyKEEN.
- ``buildml[reports]`` pins ``setuptools>=69,<82`` because
  ``ydata_profiling`` still imports ``pkg_resources``, which setuptools
  82+ no longer provides.
- AutoGluon 1.6 removed ``TabularPredictor.get_model_best()``. The
  AutoML adapter now reads ``model_best`` (with method and leaderboard
  fallbacks) so ``session.automl.run(backend='autogluon')`` completes.
- giotto-tda persistence batches pad **per homology dimension** with
  trivial birth=death bars. A global zero-pad made H0/H1 counts disagree
  across samples and raised ``ValueError`` in giotto vectorizers.

## [2.6.1] - 2026-09-10

### Fixed

- **PyPI wheels can import BuildML again.** `2.4.0`–`2.6.0` omitted
  `buildml/explain/generated/operation_index.json` from the wheel.
  `import buildml` loads the Session catalog at import time, so a clean
  `pip install buildml` raised `FileNotFoundError`. The index is now
  declared in `[tool.setuptools.package-data]` and `MANIFEST.in`. CI
  inspects `dist/` and clean-installs the wheel before a release can
  ship. Install `buildml>=2.6.1` (source checkouts were never broken).

## [2.6.0] - 2026-09-04

### Changed

- **Proofs and examples are a user contract.** Paste scripts live in
  `examples/` and match the guides. Proofs are checkout-only evidence
  for one job (`proofs._lib` plus repo root). Composition examples combine
  library capabilities; they are not separately shipped applications.
  Core examples run in CI; bundles land in
  `examples/.artifacts/`; the classical file uses 120 rows so printed
  metrics are not three-row noise.
- **`session.calibration` and `session.tune_threshold` default to
  validation.** Test is for a frozen policy. A split without validation
  raises instead of silently scoring test.
- **Classical / AutoML proofs prefer public tables.**
  `loan-approval-classical` and `mortgage-default-classical` load OpenML
  German Credit (`credit-g`) and fall back to the in-repo credit draw
  with `loader_selected` recorded. `churn-automl-search` uses sklearn
  Wisconsin breast cancer. Slugs stay; READMEs say what the table
  actually is. The shorter examples use small demonstration datasets.
- **CBR `backend=None` stays sklearn on Windows.** In-process hnswlib
  and faiss can hard-crash the interpreter even when a subprocess probe
  succeeds. Explicit `backend="industry"` raises unless
  `BUILDML_ALLOW_CBR_INDUSTRY_WINDOWS=1`. Explicit `backend="torch"`
  raises unless `BUILDML_ALLOW_CBR_TORCH_WINDOWS=1`.
- **Package summary.** PyPI and the README opener describe one object
  that holds your data, split, preprocessing, model, and run history,
  and also runs the domain catalog. That object is named a Session.
- **Coverage floor is 66 on the Linux CI monolith.** The previous 70
  figure was a Windows isolated combine, not the gate that actually
  runs `pytest --cov=buildml`. This cut measured 66.67%.

### Added

- **`examples/breast_cancer_classical_loop.py`.** Wisconsin breast
  cancer from sklearn, no `proofs._lib`, so a public table can be pasted
  without a checkout harness.
- **REAL_PUBLIC_DATASET Tier C twins.** Same-split sklearn industry
  comparisons for `breast-cancer-classical`, `diabetes-progression-regression`,
  `wine-cluster-segments`, and `adult-fairness-observational`
  (`baseline_industry.py` → `results/comparison.json`).
- **`cbr-faiss` extra.** Optional faiss-cpu ANN peer, marker-skipped on
  Windows and Python 3.13, same pattern as `recommenders-lightfm`.
  `buildml[cbr-industry]` stays hnswlib. `production` now also pulls
  `imbalanced` and `engines`.
- **Session extra probes.** `scripts/probe_industry_extras.py` now covers
  `shap`, `imbalanced`, `polars`, `duckdb`, `optuna`, `speech`, `serve`, and
  `ai`, and no longer treats `faiss` or `optuna` as part of extras that do
  not install them. Runtime stability gained `imbalanced_resample_smote` and
  `engines_polars_ingest` use-case probes.

### Fixed

- **PuLP knapsack selection.** Read solved binaries with ``pulp.value``
  instead of treating ``LpVariable.value`` as a number. Current PuLP
  exposes ``value`` as a method, which crashed ``backend='pulp'``
  knapsack on Windows after solve.
- **Sphinx API inventory.** Enum members and schema fields in
  `buildml.core.types` are documented once (member/field docstrings), so the
  package reference no longer emits duplicate-object warnings.
  Contributor notes `pypi-2x-publish` and `session-facade-migration` sit on a
  hidden toctree so Read the Docs links resolve without a sidebar entry.

## [2.5.0] - 2026-08-06

### Changed

- **Industry EDA surfaces.** Static research HTML is BUILDML STATIC EDA (Industry
  readiness sheet, `html_format="research"`). The live dashboard is the Industry
  EDA App: Cockpit spine 01-08, Readiness Gates (UI-only session marks), Concept
  Academy (~204 `CONCEPT_NOTES` lessons plus readiness-path slugs). Shared
  analytic coverage and adapt helpers keep App and Static aligned. Design-source
  folder `redesigning_eda/` removed from the tree; launch/preview/gauntlet
  helpers documented under `scripts/`.
- **Static Offline HTML primary.** BUILDML STATIC EDA header export is Offline
  HTML only (re-download of the offline snapshot). CSV and PDF briefing header
  buttons removed from Static; App keeps Offline HTML as the primary header
  export with CSV/PDF on API routes for automation.

### Added

- **EDA adaptability proof.** Tier A `proofs/eda-industry-adaptability/` evidences
  Static research HTML and Dashboard App sheet/API payloads across 12 datasets
  (sklearn real-world tables plus synthetic stress cases). Regenerates via
  `python proofs/eda-industry-adaptability/script.py` or
  `scripts/eda_adaptability_gauntlet.py`. Suite count: Tier A **63/63**.

## [2.4.0] - 2026-08-04

First **stable** Session 2.x release on PyPI. At release time, an unpinned
installation selected 2.4.x instead of legacy 1.0.9. The 2.4.0 and 2.4.0a3
distributions were subsequently yanked because their wheels omitted a required
catalog file; version 2.6.1 corrected the packaging issue.

### Changed

- **Graduated from alpha.** Package version `2.4.0` (non-pre-release); PyPI
  classifier `Development Status :: 5 - Production/Stable`. Install docs teach
  bare `pip install buildml` for Session 2.x.

### Added

- **KG / causal / federated / RL evaluation depth.** Native evaluation metrics
  documented in capability matrices (KG filtered MRR/Hits/mean_rank protocol;
  causal nuisance + ATE metrics including `outcome_mae`; federated
  `balanced_accuracy` / binary `roc_auc`); Flower adapter no longer eagerly
  imports `flwr` (local-sim disclosed); federated/RL catalogs mirror KG
  `industry_extra_present` vs `industry_runtime_present` separation. Bundle
  save→load→re-score unit tests + Tier A proofs
  (`logistics-kg-linkpred`, `causal-treatment-effect`, `federated-hospital-sim`,
  `tabular-q-frozenlake`, `imitation-cartpole-control`).
- **Ensemble / AutoML / Torch speech depth.** Ensemble evaluate attaches
  `base_contributions` / `diversity` / `ensemble_report`
  (`build_ensemble_eval_report`); AutoML `leaderboard()` with nested-CV
  disclosure + catalog `default_selection='cv'` / prominent `nested`; ASR
  default prefers transformers when installed (stub disclosed for CI);
  RAG callable-embedder restore warnings. Guides + proofs updated.
- **Serve / deploy production surface.** Declarative `ServeConfig` (YAML / env /
  CLI) in `buildml.serving.config`; HTTP Basic auth alongside API-key/Bearer
  (either may authorize); OpenAPI/docs default **closed** when auth is enabled;
  first-party `deploy/serve/Dockerfile` + `docker-compose.example.yml` (non-root,
  healthcheck, no `--allow-insecure-public-bind`); hardened K8s serve
  template/renderer with Secret `secretKeyRef`, readiness/liveness probes, and
  `buildml-serve:local` image from `deploy/serve`. Tests:
  `tests/unit/test_serving_production.py`.
- **Fairness observational depth.** Intersectional sensitive columns, bootstrap /
  stratified-subsample stability bands, per-group classical metrics bridge,
  richer `FairnessReport` (`to_markdown` / scope / warnings), and opt-in
  mitigation helpers (`suggest_group_thresholds`, `suggest_reweighing_weights`
  + Session facades `attach_to_last_eval` / `suggest_thresholds` /
  `suggest_reweighing`). Catalog discloses observational scope and refuses
  silent fairness washing / legal certification. Guide:
  `guides/fairness-deep.md`.
- **REAL_PUBLIC_DATASET Tier A proofs.** Four leakage-safe proofs on real public
  tables (not synthetic blobs): `breast-cancer-classical` (sklearn breast_cancer),
  `diabetes-progression-regression` (sklearn diabetes), `wine-cluster-segments`
  (sklearn wine + external ARI), `adult-fairness-observational` (OpenML Adult /
  credit-g with disclosed offline proxy fallback). Loaders and provenance meta in
  `proofs/_lib/datasets.py`; harness `normalize_dataset_meta` /
  `refuse_perfect_scores`; wired into `CI_SMOKE_TIER_A` and `proofs/README.md`.

## [2.4.0a3] - 2026-08-04

### Added

- **Namespaced Session facades.** Every domain exposes `session.<attr>.*`
  (e.g. `session.fairness.evaluate`, `session.anomaly.fit`, `session.rag.retrieve`)
  delegating to existing flat methods. Catalog via `Session.list_facades()` /
  `session.<domain>.describe()`. Migration guide:
  `docs/session-facade-migration.md`.
- **Facade-aware discovery.** `list_capabilities` / `describe_method` report
  `preferred_path`, `stability_tier` (`core` | `domain` | `experimental`), and
  `flat_deprecated`. EDA facade is `session.explore`; workflow/teaching facade
  is `session.audit` (avoids shadowing flat `eda` / `workflow`).
- **Dual-form operation resolution.** `resolve_operation_name` accepts flat and
  facade spellings (`fairness.evaluate` / `session.fairness.evaluate`); wired
  through explain catalog, resolver, AI tool sync, and generic executor
  dispatch. Catalog keys remain canonical flat.
- **Domain-variable shadow warning.** Accessing a facade when the Session is
  bound to a domain-named local (`rag = Session()`) emits a one-shot
  `UserWarning` (prefer `session = Session()`).
- **PyPI 2.x publish checklist.** Added `docs/pypi-2x-publish.md` covering
  publisher configuration, project ownership, and distribution validation.

### Changed

- **Domain flat actions deprecated (warnings only).** Calling flat domain
  methods (e.g. `session.evaluate_fairness`) emits `DeprecationWarning` pointing
  at the preferred facade. Classical core (`ingest` / `fit` / `evaluate` / ...)
  remains dual first-class without warnings. Flat domain alias removal is
  deferred to BuildML 3.0.
- **Facades are the supported domain API for 2.4.x.** Flat aliases remain
  supported-but-deprecated until 3.0. Documented in README, `docs/stability.md`,
  `docs/session-facade-migration.md`, `docs/pypi-2x-publish.md`, and CONTRIBUTING.
- **Teaching content prefers facades.** Examples, guides, proof READMEs, docs
  sources, and explain concepts/beginner/overlays teach `session.<domain>.*` as
  the primary domain path; classical chains may stay flat. See
  `docs/session-facade-migration.md`.
- **Coverage floor raised to 70.** `fail_under` 60 → **70** after full-suite
  measure ~70.71% (`scripts/coverage_ratchet.json`).
- **CI mypy widened.** Scoped typing expands beyond session/core/explain to
  packages verified clean with `--follow-imports=silent` (fairness, serving,
  pipeline, and other clean domains; see `.github/workflows/ci.yml`).
- **Industry capability honesty.** KG / federated / RL / AutoML / anomaly /
  probabilistic industry `available` flags prefer runtime import probes;
  find_spec-only signals disclosed separately.
- **Synthetic proof score ceilings.** Harder generators + metric ceilings on
  payment-rail, cluster/SKU, policy/compliance symbolic, credit/mortgage
  Bernoulli labels, IoT overlap; proofs README clarifies what smoke proves.

### Added

- **CI proof smoke gate.** `python -m proofs._lib.run_all --smoke` runs a fixed
  Tier A subset without `--skip-existing`; GitHub Actions job `proofs-smoke`.
- **Coverage ratchet process.** Full-suite measure via
  `scripts/run_full_coverage.py` (per-module isolation on Windows + combine):
  **70.7%** (39539/55914). `fail_under` raised through 60 to **70**
  (`scripts/coverage_ratchet.json`).
- **Lazy/cached walkthrough capability probes.** Default
  `walkthrough(capability_probe="lazy")` skips inactive domain industry imports;
  process-wide matrix + subprocess import caches; `eager` / `skip` modes.
- **Fairness disparity reporting.** `Session.evaluate_fairness` /
  `fairness_capability_matrix` (observational DP / DI / equalized odds).
- **Optional SHAP attribution.** `Session.explain_shap` behind `buildml[shap]`.
- **Release workflow.** `.github/workflows/release.yml` (manual/tag build +
  optional PyPI publish).
- **Surface stability policy.** `docs/stability.md` for alpha churn control.
- **Harder RAG proof corpora.** Hard distractors / paraphrases in
  `support-kb-rag` and `policy-handbook-rag` so hashing cannot trivially score
  perfect retrieval metrics.
- **Proof suite expansion (+30 Tier A, +30 Tier B, +30 Tier C twins).** Inventory
  is now **57/57** Tier A, **36/36** Tier B, **57/57** Tier C. New Tier A covers
  ensembles (`fit_voting` / `fit_stacking` / `fit_blending`), Torch tabular/text,
  and additional industry scenarios across anomaly, forecast, RAG, recommenders,
  LTR, KG, TDA, semi/AL/SSL, online, multitask, meta, symbolic, CBR, decisions,
  synthetic, unsupervised, causal, federated, and graph. New Tier B products
  compose ≥3 Session surfaces each (Meridian, Helix, Prism, Orbit, Quasar,
  Forge, Canyon, Vector, Citadel, Nova, Sentinel, Ballast, Parchment, Lattice,
  Beacon, Rivulet, Cornerstone, Apex, Relay, Mosaic, Kiln, Aurora, Compass,
  Folio, Dynamo, Scaffold, Terrace, Volt, Keystone, Zenith). Shared synthetic
  loaders extended in ``proofs/_lib/datasets.py``. Harness:
  ``python -m proofs._lib.run_all --tier all``. Docs: ``proofs/README.md``,
  root README, ``guides/README.md``, Sphinx ``features.rst``.

### Security

- **Trusted deserialize gate.** All joblib/pickle/torch bundle and checkpoint
  plan loads require keyword-only ``trusted=True`` (default ``False``) via
  ``buildml.core.serialization``. Session facades, domain loaders,
  ``predict_from_pipeline``, and managed serving (``create_serving_app`` /
  ``serve_bundle`` / ``buildml-serve --trusted``) thread the flag;
  ``data_only=True`` checkpoint loads skip plans without needing ``trusted``.
  Residual pickle risk remains inherent: the gate makes it opt-in.
- **Path allowlist + integrity hashes.** Loaders refuse URI-shaped paths
  (``https://``, ``s3://``, ``file://``, …). Optional ``sha256`` of joblib
  payloads is recorded in anomaly / pipeline ``meta.json`` and Session
  ``MANIFEST.json`` hashes are verified on load when present (tamper detection;
  not authenticity). JSON sidecars load without executing code.
- **TorchScript trust gate.** ``load_torchscript(..., trusted=True)`` required
  (aligned with serving TorchScript loads).
- **Windows Torch import probing.** ``torch_available`` / RAG semantic stack
  probes import Torch in a subprocess on Windows so a broken DLL load cannot
  hard-crash the parent process during capability matrices or EDA.
- **AI injection heuristics.** Expanded patterns (DAN, prompt exfil, role-play,
  base64 / atob / hex smuggle, multi-line instruction overrides), NFKC +
  zero-width / bidi strip, Latin-homoglyph fold, structured ``InjectionFinding``
  reason codes and ``refuse_injection``. ``ToolRegistry.register`` always
  refuses: allowlist is closed at construction; confirm-on-write unchanged.

### Changed

- **Session typing + facade docs.** Removed ``warn_return_any=false`` for
  ``buildml.session.*``; CI mypy covers the full ``buildml/session`` package
  cleanly. Mixin methods are short facades pointing at canonical ops docs;
  ``scripts/audit_docstrings.py`` allows that facade shape. Mixin LOC cut
  substantially (~18.4k → ~11.2k).
- **Domain maturity floor.** ``scripts/domain_maturity_index.py`` ratchets
  claimed-complete domains to artifact score ≥ 6 plus ``explain_hooks`` /
  ``checkpoint`` (unless ``analysis_only``). Ensemble gained a dedicated
  ``ensemble_capability_matrix``; timeseries documents analysis-only floor.
  CONTRIBUTING defines the domain floor explicitly.
- **Industry probe honesty.** ``probe_industry_extras.py`` aligns probes with
  extras, emits platform tags + markdown artifacts, and surfaces
  ``skipped_by_marker``. On Windows every module is probed in a subprocess so
  native AV/DLL crashes cannot kill the CI parent. Capability matrices expose
  ``platform_markers`` for LightFM / giotto-tda / learn2learn / skope-rules /
  neuralforecast.
- **Session monolith split.** Public
  ``buildml.Session`` API unchanged (474 public attributes preserved). Domain method
  signatures/docstrings moved into ``buildml/session/mixins/`` (34 domain
  mixins + ``_shared`` annotation bag); ``session.py`` is now the thin
  assembler (~840 LOC) owning ``__init__``, context manager, and state glue.
  Orchestration remains in ``*_ops.py``. Eliminated all
  ``from buildml.session._imports import *`` star-imports in ops (explicit
  imports; ``Literal``/``Any`` from ``typing``). Coverage
  ``fail_under`` raised 20 → 25 (classical+checkpoint smoke ~26.5%).

- **Quality / hygiene.** Deleted stale root ``audit_session2.txt``. Added
  ``[tool.coverage.report] fail_under`` (later raised to 25). Widened scoped
  CI mypy to ``buildml/_version.py`` plus ``explain``
  capability/glossary/prerequisites (and fixed ``worked_example_pattern``
  typing in anomaly concept notes). Added GitHub Actions ``windows-classical``
  job (core+dev, import smoke, ruff, classical alpha smoke; Torch/PyG remain
  Linux-only). Replaced silent ``except: pass`` blocks under ``buildml/`` with
  ``logger.debug(..., exc_info=True)`` + intent comments. Replaced all 259
  Session/ops ``Controls …; see the function signature`` template Parameter
  blurbs with pedagogical text (docstring audit still 0). Aligned
  ``requirements.txt`` / ``requirements-dev.txt`` to ``pyproject.toml`` ranges
  with install-honesty notes; CONTRIBUTING documents coverage/mypy/Windows CI.

- **Graph / federated / Tier C teaching polish.** Graph beginner examples
  aligned to real ``fit_graph`` kwargs; ``Session.export_round_history`` wired
  to ``buildml.federated.results.export_round_history``; proof industry
  comparison stubs filled with Tier C notes; ``nlp_capability_matrix`` exposes
  ``backends_available``; anomaly / RL teaching notes expanded.

- **Docstring coverage (mid-size domains).** Enforced at 0 findings for:
  ``online`` (114), ``federated`` (107), ``automl`` (99), ``multitask`` (96),
  ``activelearning`` (95), ``anomaly`` (95), ``synthetic`` (93),
  ``semisupervised`` (90), ``unsupervised`` (83).
- **Docstring audit at 0.** Every audited package (including Session and
  ``*_ops.py``) passes ``scripts/audit_docstrings.py --check``.
  ``buildml/session/`` is in ``ENFORCED_PREFIXES``.

### Added

- **Forecasting / ranking / causal / KG capability gaps.** Forecasting refuses
  ``exog_columns`` on univariate-only methods via ``method_supports_exog``; AI
  operator gets ``time_split``; ranking accepts ``lambdarank`` →
  ``lambdarank_lgbm``; DoWhy holdout evaluate re-estimates ATE; KG evaluate adds
  relation-prediction ranks; ``dl_capability_matrix`` + Session/AI introspection;
  unsupervised/synthetic explain status embed live matrices.

- **Capability-matrix walkthrough / audit hooks.**
  ``buildml/explain/capability_status.py`` centralizes matrix loading,
  domain-status attachment, walkthrough routing, and audit suggestions.
  Domain ``explain_hooks`` status payloads that were missing
  ``capability_matrix`` (forecasting, timeseries, RAG, CBR, unsupervised,
  SSL, online, meta-learning, federated, graph, NLP, ranking, recommender,
  synthetic) now embed the live matrix plus ``capability_introspection`` API
  hints. Walkthrough reports add ``capability_introspection_status`` and HTML
  orientation tables; audit ``suggest_next_operations`` prioritizes unchecked
  matrices before open fit paths; the workflow resolver keeps every
  ``*_capability_matrix`` operation permanently ``available``.
- **Capability-matrix wiring audit (foundational introspection).** Domains that
  already published honest backend matrices in catalog code but hid them from
  Session, explain overlays, and the AI operator now expose them consistently:
  `rl_capability_matrix`, `causal_capability_matrix`, `federated_capability_matrix`,
  `graph_capability_matrix`, `kg_capability_matrix`, `metalearning_capability_matrix`,
  `multitask_capability_matrix`, `online_capability_matrix`,
  `probabilistic_capability_matrix`, `recommender_capability_matrix`,
  `semisupervised_capability_matrix`, `activelearning_capability_matrix`, and
  `automl_capability_matrix`, plus AI tools for every matrix peer domains already
  had (SSL, unsupervised, forecast, timeseries, RAG, TDA, CBR, symbolic).
- **Time-series analysis teaching layer.** New concept notes and beginner layers
  for decomposition, stationarity diagnostics, changepoint detection, and the
  analysis-before-forecast workflow (`buildml/explain/concepts/timeseries.py`,
  `buildml/explain/beginner/timeseries.py`).
- **Tier A proof: tabular Q-learning** (`proofs/tabular-q-frozenlake/`) :
  end-to-end `fit_rl(mode='tabular_q')` → `evaluate_rl` → `act_rl` → bundle,
  complementing the existing imitation-cartpole proof.
- **AI executor generic read-only dispatch.** Registered read-only tools with a
  `session_method` now fall back to Session dispatch when no bespoke branch
  exists, fixing unwired capability-matrix tools (anomaly, ranking, decision,
  synthetic, and the new matrices above).

### Changed

- **The explain system now teaches beginners.** Explanations were written for
  people who already knew the material: terse, jargon-first, and silent on what
  a term meant or why a step existed: which defeats the point of an explain
  surface. Every explanation is now layered, and the beginner layer is the
  default.
  - **Reading levels.** `Session.explain(...)` and the new `Session.learn(...)`
    accept `level="beginner"` (default), `"intermediate"`, or `"advanced"`. The
    level controls how much scaffolding is rendered, never which facts are true:
    assumptions, leakage risks, and failure modes appear at every level, while
    `advanced` drops the analogy and glossary and widens the parameter and
    pitfall lists.
  - **An operation primer on every explanation.** All 288 catalog operations
    carry a beginner briefing: plain-language summary, analogy, why it exists,
    the steps in order, prerequisites in ordinary words with the calls that
    satisfy them, what each key parameter means in practice and how to move it,
    what changes on the session, how to read the result, the common pitfalls, a
    glossary of the jargon the answer itself used, a worked example, and the
    neighbouring tools. The primer is *derived* in `buildml/explain/pedagogy.py`
    from the catalog entry and its linked concept notes rather than hand-copied,
    to keep the shared content consistent. Operations can override individual
    sections with authored prose.
  - **A beginner layer on all 188 concept notes**, across every domain:
    supervised, unsupervised, forecasting, anomaly, NLP, RAG, RL, online,
    federated, causal, graph, knowledge graphs, symbolic, CBR, TDA,
    meta-learning, probabilistic, recommenders, ranking, synthetic, AutoML, and
    the AI operator. Each adds a plain summary, an analogy, beginner steps, when
    to use and when not to, misconceptions paired with corrections, a worked
    example, self-check questions, the BuildML tools that apply the idea, and
    prerequisite/follow-on links that turn the note set into a reading order
    instead of an index.
  - **A machine-learning glossary** (`buildml/explain/glossary.py`): 234 terms
    with aliases, in plain language, detected automatically in explanation prose
    so jargon is defined where it is used. Every term now resolves to the concept
    note or operation that teaches it, so a definition is never a dead end.
- **Prerequisite handling consolidated.** `buildml/explain/prerequisites.py` now
  owns how each precondition is checked, which operations satisfy it, and how it
  is phrased for a beginner. The resolver previously carried a long `if`/`elif`
  chain that had fallen behind the catalog and could not evaluate a third of its
  prerequisite keys; the three answers now live in one table and are covered by a
  test that fails if a catalog prerequisite has no probe.
- **AI operator teaching tools.** `explain_operation` takes a `level`, and the
  new read-only `learn_concept` tool lets an operator answer "what is this?"
  before proposing any write. Both are in the default registry and the autonomy
  allowlist.

### Added

- **`Session.learn(topic, level=...)`.** Answers the question that comes before
  `explain`: what *is* this, and what should I understand first. The topic may be
  a concept key (`"leakage-boundary"`), an operation name (`"split"`), or the
  word that tripped you up (`"stratified"`), with spacing and hyphenation
  forgiven and close matches suggested when nothing resolves. Called with no
  topic it returns the foundation concepts in reading order. Returns a
  `LearningBrief` carrying the subject plus `read_first` and `read_next` notes.

### Fixed

- **Single-sentence explain content no longer renders one bullet per letter.**
  497 fields across the operation catalog and the concept notes were authored as
  a bare string where a tuple was expected: a missing trailing comma, which no
  type checker catches at runtime. Anything iterating them, including the new
  beginner primer, walked the characters. `OperationSpec` and `ConceptNote` now
  normalize their prose fields on construction, and a test fails if a prose
  field is ever a string again.
- **The beginner primer no longer repeats itself.** "When not to use" and
  "common pitfalls" both drew on the same anti-patterns and leakage risks, so a
  reader met the same warning twice in one briefing; pitfalls now exclude what
  the avoidance list already said. "When to use" no longer reprints the ordering
  notes already shown in the expert appropriateness section, and the related
  tools list drops any call the authored alternatives already recommend.

### Documentation

- **Docstring standard, enforced in CI.** BuildML's promise is to make the
  complex easy, and the API docstrings were not holding up their end: most were
  a single line, parameters were listed by type without saying what they do, and
  almost nothing explained *when* to reach for one option over another. There is
  now a written standard in [`CONTRIBUTING.md`](CONTRIBUTING.md): NumPy style,
  with a beginner-readable summary, a description of the concept and its role in
  the pipeline, parameters explained by effect rather than type, returns
  explained by meaning, raises, notes covering leakage and alternatives, and
  examples for anything non-trivial.
  - `scripts/audit_docstrings.py` checks the standard mechanically. Run it with
    `--report` for a per-package coverage table, `--path` to see every finding in
    one area, and `--check` for the CI gate. Wired into
    [`ci.yml`](.github/workflows/ci.yml).
  - The gate is a two-part ratchet. Packages listed in `ENFORCED_PREFIXES` must
    stay at zero findings; every other package must stay at or below its recorded
    count in `scripts/docstring_budget.json`. New shallow docstrings fail CI while
    the existing backlog does not block unrelated work, and the recorded counts
    can only fall: `--write-budget` refuses to raise one unless `--rebaseline`
    is passed, which prints exactly what it ratified.
  - Variadic parameters documented in NumPy's `*args` / `**kwargs` spelling are
    now recognised. The parser skipped every line beginning with `*` to avoid
    reading bullets as parameter names, which meant a correctly documented
    `**kwargs` was reported as undocumented. Bullets and block quotes always
    carry a space after their marker, so the filter now requires one.
- **`buildml.preprocess` rewritten to the standard and locked at zero findings.**
  Every public function, class, and method across all 15 modules now explains the
  technique, not just the call. Each fit function states why a split plan is
  mandatory and what leaks without it; each method choice (`'iqr'` against
  `'zscore'`, `'onehot'` against `'target'`, `'quantile'` against `'uniform'`)
  explains the trade-off rather than naming the option; each plan class explains
  why the learned state is stored rather than recomputed at inference.
  Target encoding, the easiest step in the package to misuse, now documents the
  out-of-fold mechanism that makes it safe and why `transform_encoder` demands a
  split plan when the other methods do not.
- **`buildml.nlp` rewritten to the standard and locked at zero findings.** All 26
  modules: the supervised path, topics, keyphrases, sentiment, entities,
  summaries, language detection, corpus profiling, normalisation, vectorisation,
  the result dataclasses, the three optional-backend adapters, and the history
  hooks. Text carries failure modes tabular data does not, so the docstrings name
  them: every fit function states that the vocabulary is learned on train alone
  and what a holdout score means once it is not; the out-of-vocabulary rate is
  explained as the signal that a metric is measuring words the model never saw;
  extractive summarisation documents that it selects sentences rather than
  writing them; sentiment documents that a lexicon which recognises none of a
  corpus's vocabulary will report it as uniformly neutral; and
  `interpret_text_prediction` explains why it refuses hashing and embedding
  representations instead of returning attributions that cannot be traced to a
  word. Backend choices (`sklearn` against `embedding` against `transformer`,
  NMF against LDA, TF-IDF against RAKE against TextRank) explain the trade-off
  rather than listing the option.
- **`buildml.rl` rewritten to the standard and locked at zero findings.** All 18
  modules across imitation learning and reinforcement learning. Decision-making
  is the domain where a metric is most easily read as more than it is, so the
  docstrings draw the lines explicitly: behavioural cloning documents that it
  reproduces a demonstrator rather than succeeding, and that agreement with a
  poor demonstrator is still a poor policy; offline bandit evaluation explains
  why the direct method and inverse propensity scoring fail in opposite
  directions, and that disagreement between them means neither should be
  trusted; `action_match_rate` is documented as the diagnostic to read before
  either estimate; the `offline` flag is carried through every result, summary,
  and bundle so a counterfactual estimate is never later mistaken for a measured
  one; and tabular control documents `unseen_state_rate` as the point at which
  a return becomes a measure of luck. Algorithm choices (LinUCB against
  epsilon-greedy against softmax, Q-learning against SARSA, PPO against DQN
  against A2C) explain the trade-off rather than naming the option, and
  `act_sb3_observation` documents that its one-hot scores are not probabilities
  and why Stable-Baselines3 cannot supply real ones.
- **`buildml.dl` rewritten to the standard and locked at zero findings.** All 25
  modules: the tabular and text Torch path, loaders, training, evaluation,
  curves, cross-validation, nested search, DDP, export, packaging, the
  Kubernetes renderers, the multimodal and modality helpers, the pretrained
  backbone hooks, and the speech path. Deep learning fails in ways classical
  models do not, so the docstrings name them: `make_loaders` documents the group
  and time checks it runs and why it raises rather than warns; `fit_standardize`
  and the image and audio statistics helpers each state that they learn from
  train rows only and what a holdout distribution should therefore look like;
  `nested_cv_torch` explains what its estimate covers and why `search_torch`
  alone reports an optimistic number; the export helpers document that tracing
  records one path through the model and that data-dependent control flow is
  silently lost; `TorchBundle` documents what a bundle does and does not contain,
  and why the module must be supplied on load. The pretrained hooks carry
  `weight_mode` through every result so `'mock'` weights can never be mistaken
  for real ones, and the speech module refuses foundation-model pretraining by
  name rather than approximating it, while labelling its stub transcription
  backend as test scaffolding wherever the text appears.
- **`buildml.ai` rewritten to the standard and locked at zero findings.** All 13
  modules: the tool registry, egress controls, provider layer, advisor,
  executor, planner, autonomy mode, transcript, and security hardening. This is
  the domain where a docstring that overstates a guarantee is itself a hazard,
  so each control now says what it actually does: `detect_pii_columns` documents
  that it matches column *names* and will miss a `notes` column full of
  addresses; `sanitize_tool_result` and `detect_injection_attempt` state that a
  finite phrase list raises the cost of an attack rather than preventing one,
  and that the real bound is the closed registry; `build_stats_payload` explains
  that a minimum and maximum are literal values from your data and that an
  aggregate over few rows can still identify someone; `EgressManifest` documents
  that it accounts for the payload and not the prompt. The confirmation model is
  documented as structural rather than procedural: `ToolRegistry` explains that
  an unregistered name is refused rather than matched to the nearest tool, and
  `requires_confirmation` explains why it answers `True` for a tool it does not
  recognise. `run_autonomous` names its residual risks and recommends reviewing
  a plan before executing it unattended.
- **`buildml.data` rewritten to the standard and locked at zero findings.** All
  11 modules: the `Dataset` handle, split planning, the engine protocol, the
  pandas, Polars, and DuckDB adapters, the shared aggregation vocabulary, the
  portable filter helpers, and the design-matrix prep path. Two things decide
  whether a result is trustworthy here, and both are now stated wherever they
  apply. The first is leakage: `create_split` explains why a random split is
  wrong for grouped or time-ordered data and what the resulting score would
  overstate, `create_time_split` documents why forecast evaluation must respect time
  ordering, and `guard_fit_partition` explains its training-partition
  restriction. The second
  is memory: every method now says whether it defers work or forces it, so
  `select_columns` and `filter_expr` are documented as the operations that keep
  unnecessary rows and columns out of memory, while `sample_rows`, `filter_rows`, and
  `to_pandas` are marked as the points where a lazy plan collects. Native
  handles are documented as narrowing what must be materialised and explicitly
  *not* as out-of-core fitting. `DuckDBTable` explains connection ownership :
  who closes, who shares, and what breaks when the owner closes first: and
  `prepare_design_frame` records that a sampled fit describes the sample rather
  than the population.
- **`buildml.rag` rewritten to the standard and locked at zero findings.** All 18
  modules: corpus ingest, chunking, the three embedding backends, the vector
  store, index build and incremental update, dense, BM25 and hybrid retrieval,
  fusion, cross-encoder reranking, grounded generation, retrieval and generation
  evaluation, bundle persistence, the capability matrix, the dependency gates,
  the history hooks, and the LangChain adapter. The `retrieve` documentation explains
  that ranking does not establish relevance: a question the corpus cannot answer
  may still return ranked passages. The `rrf_fuse` documentation
  explains why fusing by rank avoids comparing a BM25 score against a cosine
  similarity, and `weighted_fuse` explains why its per-query normalisation makes
  scores unstable across queries. Grounding claims are bounded rather than
  implied: `score_faithfulness` states that it measures citation coverage and
  lexical overlap, not truth, and that a fluent, well-cited, entirely wrong
  answer scores well; `generate_from_retrieve` documents that empty retrieval and
  provider errors are hard failures precisely so no ungrounded fallback can be
  mistaken for a grounded answer. Leakage and persistence are stated where they
  bite: `Document.role` explains what `eval_only` holds out and why,
  `evaluate_retrieval` explains why document mode deduplicates and chunk mode does
  not, and `load_rag_bundle` documents that a bundle saved with a custom callable
  embedder reloads with hashing substituted: queries and stored vectors then
  occupy unrelated spaces, and retrieval returns semantically unrelated results.
  `rag_status` reports the absences as plainly as the presences, including that a
  Session checkpoint does not carry the vector index.
- **`buildml.cbr` rewritten to the standard and locked at zero findings.** All 19
  modules: the case base and distance metrics, fit, retrieve, predict, evaluate
  and retain, feature preparation, the result dataclasses, the capability matrix,
  bundle persistence, the history hooks, and the four backend adapters. Case-based
  reasoning promises an explanation alongside every prediction, so the docstrings
  say what that explanation is worth: `pairwise_distances` describes what each of
  the four metrics actually treats as similar, and states that the mixed metric
  weights numeric and categorical features by column count rather than importance;
  `standardize_fit` explains that unscaled features let whichever column has the
  largest units decide every neighbour; and `distance_weights` explains why inverse
  distance falls off sharply enough that one very close case can decide a
  prediction on its own. Leakage discipline is stated wherever memory can absorb a
  label it should not: `fit_cbr` documents that the case base is built from train
  alone and that its `train_score` is in-sample because a row is its own nearest
  neighbour, `retain_cbr` documents that holdout rows are refused outright rather
  than warned about and that identity is the frame index, so a default
  `RangeIndex` silently skips every genuinely new row as a duplicate. The
  approximate and learned backends state their costs: `build_ann_index` documents
  that approximate search can miss a true nearest neighbour, and the torch metric
  encoder documents that a learned space is uninterpretable, which forfeits part
  of why the method was chosen. `pairwise_distances`, `top_k_indices`,
  `distance_weights`, `encode_categoricals`, `standardize_fit`,
  `standardize_apply`, and `numeric_ranges` now carry executable doctests.
- **`buildml.model` rewritten to the standard and locked at zero findings.** The
  classical supervised surface: fit, predict and evaluate, cross-validation and
  the four hyperparameter searches, nested CV, model comparison, the deep
  diagnostics, the evidence records, and both the HTML and plot-board exports.
  This is the package where an honest number and a flattering one look identical,
  so the docstrings say which is which: `fit_estimator` documents that
  `train_score` describes fit on training data rather than holdout performance;
  `cv_score` explains that fold standard deviation describes variation across
  folds rather than statistical significance, and refuses to run when a
  Session split already exists because scoring the whole frame would put test
  rows in a training fold; `nested_cv_score` explains that it estimates the
  *procedure* rather than a model, which is why it returns no single winner; and
  `optuna_search` documents that its early trials are random, so a small budget
  buys a randomized search with extra machinery. The diagnostics state what
  metrics hide: `calibration_report` explains that a well-ranked model can still
  be badly calibrated and that AUC will not show it, `threshold_report` explains
  that 0.5 is a convention rather than a decision, and
  `permutation_importance_report` documents that correlated features split their
  importance and can both look unused. `_infer_task` and
  `fit_kwargs_for_sample_weight` carry executable doctests, the latter showing
  that an estimator which cannot weight refuses rather than ignoring the weights.
- **The persistence and deployment path rewritten to the standard and locked at
  zero findings**: `buildml.core`, `buildml.checkpoint`, `buildml.pipeline`, and
  `buildml.serving`. These are the modules a reader meets first and last, and the
  distinctions they turn on were previously left implicit. `ColumnRole` now
  explains what each role *causes* rather than naming it, including that a group
  column exists so a patient seen in training cannot reappear at test time and
  that an ID is excluded because an identifier correlated with the target is a
  shortcut that will not exist in production. The two artifacts are told apart
  wherever they are confusable: `save_checkpoint` documents that the split is the
  one thing that could not be recomputed, `load_checkpoint` documents that a
  clean load with a `None` split plan is the case to handle, and
  `save_pipeline_bundle` documents that omitting a plan that was used in training
  produces a bundle which silently under-prepares its inputs. The schema contract
  now says why it is loose: that comparing dtype *families* keeps the check
  meaningful across a Parquet round trip, and that a check which cries wolf gets
  turned off: and `coerce_score_frame` documents that numeric coercion turns
  unparseable values into nulls, so a column of mostly-numeric strings converts
  and quietly loses its `'N/A'` entries. `predict_from_pipeline` explains that a
  feature column missing *after* plan replay usually means encoding met a
  category the training data did not contain. Serving states its own limits:
  `serve_bundle` documents that a non-loopback bind without keys is refused
  rather than warned about, and `create_serving_app` documents that the API-key
  middleware is a shared secret with no identities, rotation, or audit trail.
  `coerce_data_mode`, `TableSchema.from_dict`, `validate_role_name`,
  `validate_column_names`, `MissingExtraError`, `normalize_api_keys`,
  `extract_presented_key`, `key_is_authorized`, `dtype_family`, and
  `families_compatible` carry executable doctests.
- **The docstring auditor no longer mis-reports NumPy's comma-grouped
  parameters.** `a, b, c:` on one line is standard NumPy shorthand, but only the
  `a / b / c:` spelling was parsed, so correctly documented sibling arguments
  were reported as undocumented. Fixing the parser removed ten false findings
  from `buildml.session` and two from `buildml.dashboard` without any docstring
  changes in either.
- **`Session` core path rewritten to the standard.** Ingestion, roles, splitting,
  the full preprocessing surface, fit, predict, evaluate, cross-validation,
  nested CV, the four hyperparameter searches, diagnostics, EDA, and the
  persistence and pipeline methods.
- **Sphinx now renders NumPy sections as sections.** `sphinx.ext.napoleon` was
  missing from [`docs/conf.py`](docs/conf.py), so "Parameters" and "Returns"
  headings were being emitted as literal text instead of parameter tables.
  Enabled alongside `intersphinx`, so references to pandas, NumPy, and
  scikit-learn types now link to upstream documentation.

### Added

- **NLP promoted to a first-class Session domain (`buildml.nlp`).** Text was the
  one capability BuildML claimed without holding it to the bar every other domain
  meets: there was no `buildml/nlp/` package, no Session operations, no explain
  coverage, no proof, no benchmark, and no guide. `Session.text_features` wrote
  numeric columns for tabular models and `buildml.rag` retrieved documents for
  generation, but nothing modelled or analysed a text column on its own terms.
  The domain now ships the full standard surface: capability matrix, ops,
  results, explain overlay and concepts, AI tools, bundle, tests, proof,
  benchmark, example, and guides.
  - **Supervised path:** `fit_text_classifier` fits a single-label document
    classifier on train (`tfidf` / `count` / `hashing` × `word` / `char` /
    `char_wb`, with `logistic` / `linear_svm` / `complement_nb` /
    `multinomial_nb` / `sgd` heads), then `predict_text` and
    `evaluate_text_classifier` score holdout partitions with accuracy, balanced
    accuracy, macro/weighted F1, macro precision/recall, log loss, ROC AUC, a
    per-class report, the confusion matrix, and the holdout out-of-vocabulary
    token rate. `log_loss` and `roc_auc` are omitted rather than faked for
    margin-only heads.
  - **`interpret_text_prediction` is exact, or it refuses.** For a linear head on
    an invertible vocabulary a token's contribution is `coefficient × feature
    value`: an identity, not an approximation: and the per-class global tokens
    come straight from the coefficients. Naive Bayes gets centred
    log-likelihoods, and the method string says so. Hashing (no invertible
    vocabulary), dense backends (features are latent dimensions), and heads
    without per-feature weights are refused with the reason.
  - **`profile_text_corpus` screens the split before you quote a number:** empty
    documents, length distribution, vocabulary and hapax rate, duplicate groups,
    train↔holdout exact overlap, near-duplicate overlap at a stated cosine
    threshold, holdout OOV rate, and optional language mix. It **reports**
    contamination in plain-language findings; it never silently drops rows.
  - **Unsupervised description on the same split:** `fit_topics` / `assign_topics`
    (NMF on TF-IDF, LDA on counts, NPMI coherence computed on train and clamped
    to its bounds, with assignment as a pure transform), `extract_keyphrases`
    (TF-IDF / RAKE / TextRank), `summarize_text` (extractive TextRank / LexRank /
    lead: sentences are selected, never generated), `extract_entities`
    (precision-first regex + gazetteer rules with exact character offsets, or
    spaCy), `analyze_sentiment` (lexicon with negation and intensifier handling,
    reusing a fitted classifier, or a transformer), and `detect_language`.
  - **Deterministic normalization, train-only vocabulary.** `buildml/nlp/normalize.py`
    ships a stateless normalizer, tokenizer, and abbreviation-aware sentence
    splitter; `buildml/nlp/lexicons.py` ships stopwords for seven languages, a
    sentiment lexicon with negators and intensifiers, Unicode script ranges,
    conservative English suffix-stem rules, and the entity patterns. Because
    normalization learns no corpus statistics, the fixed plan replays on
    holdout without fitting to those rows: while vocabulary, document frequencies, IDF, topic
    components, and heads are all frozen at fit on train rows only.
  - **`buildml.nlp_bundle.v1`** (`save_nlp_bundle` / `load_nlp_bundle`) carries the
    normalization plan with the fitted representation and head, plus an optional
    topic plan, so a reloaded bundle reproduces the holdout score exactly. The
    proof and the integration smoke both assert that equality rather than
    asserting the file exists.
  - **Optional extras:** `buildml[nlp]` (NLTK morphology, langdetect,
    sentence-transformer embeddings, frozen transformer encoders) and
    `buildml[nlp-industry]` (spaCy statistical NER), both folded into
    `buildml[production]`. Adapters in `buildml/nlp/adapters/` are imported
    lazily, so `import buildml.nlp` stays on the numpy / pandas / scikit-learn
    core. The bag-of-n-grams backend stays the default **even when the extras are
    installed**: it is reproducible, needs no download, and is the only
    representation that can explain its own decisions. A missing extra raises a
    named `MissingExtraError`, never a silent fallback.
  - **Wired across the ecosystem:** `Session` methods and read-only result
    accessors, `buildml/session/nlp_ops.py`, history recording with per-operation
    result summaries, `walkthrough` `nlp_status`, `audit` priority order,
    `dry_run` / `workflow` prerequisites (`nlp-text-plan`, `nlp-topic-plan`,
    `nlp-text-column`, `nlp-extra`), the operation overlay and generated catalog,
    NLP explain concepts, and 15 AI tool specs.
  - **Honesty, stated in the capability matrix and the guides:** single-label
    document classification and analysis: not multi-label, not span/sequence
    labelling, not text generation or abstractive summarization, not machine
    translation, not transformer fine-tuning (the Torch text path owns that), and
    not document retrieval for generation (`buildml.rag` owns that). Sharing a
    text column, or a sentence-transformer, does not merge those surfaces.
  - **Evidence:** Tier A proof [`ticket-routing-nlp`](proofs/ticket-routing-nlp/)
    with a Tier C `Pipeline(TfidfVectorizer + LogisticRegression)` twin on the
    same split indices (Tier A/C now **26/26**); benchmark
    `benchmarks/nlp/representation_tradeoff.py` comparing representations on one
    fixed corpus for accuracy, latency, vocabulary size, and whether attribution
    survives; runnable `examples/nlp_text_classifier_loop.py`; guides
    [`quickstart-nlp`](guides/quickstart-nlp.md) and
    [`nlp-deep`](guides/nlp-deep.md) plus Sphinx pages; and tests
    `tests/unit/test_nlp_slice.py`, `test_nlp_m2_depth.py`,
    `test_nlp_industry_depth.py`, and `tests/integration/test_nlp_alpha_smoke.py`.
    The proof corpus deliberately includes an ambiguous share so the headline
    accuracy lands near its stated ceiling instead of a suspicious 1.0.
- **Tabular TD control: the Q-learning family (`fit_rl(mode="tabular_q")`):**
  closes the value-based gap in `buildml.rl`, which previously shipped
  contextual bandits, REINFORCE-lite policy gradient, and SB3 PPO/DQN/A2C but no
  foundational tabular methods. New `buildml/rl/tabular.py` implements
  `q_learning` (off-policy), `sarsa` and `expected_sarsa` (on-policy), and
  `double_q_learning` (separate estimates reduce maximisation bias) on discrete-action
  Gymnasium envs behind `buildml[rl]`.
  - `ObservationDiscretizer` bins continuous Box observations uniformly
    (`n_bins=`), taking bounds from the declared space where finite and from a
    seeded random-policy probe (1st/99th percentile) where not; `Discrete`
    spaces index directly. Bounds, sources, and state count are recorded in
    `RlPlan.config["discretizer"]`, and tables above 500k states are refused
    with a pointer to function approximation.
  - `TabularValuePolicy` exposes the learned `q_table`, `greedy_policy_table()`,
    `state_value_table()`, and per-state visit counts; `act_rl` returns
    `Q(s, a)` as its scores.
  - Exploration schedule `eps_t = max(epsilon_min, epsilon * epsilon_decay**ep)`
    via new `n_bins` / `epsilon_min` / `epsilon_decay` knobs on `Session.fit_rl`.
  - Honest disclosures: `state_coverage` (fit) and `unseen_state_rate` (eval)
    report how much of the table was actually learned; off-policy TD control is
    explicitly distinguished from batch offline RL (CQL/IQL/DT stay out of scope).
  - Wired end to end: capability matrix (`algorithms_by_mode`), backend/mode
    resolver, `fit_rl` / `act_rl` / `evaluate_rl`, RL bundles, walkthrough
    `rl_status`, operation overlay, AI tool schema + executor, and three concept
    notes (`rl-tabular-q-learning`, `rl-sarsa-on-policy`,
    `rl-state-discretization`) linking Q-learning to DQN.
  - Tests: `tests/unit/test_rl_tabular.py` (discretizer edge cases, tie-breaking,
    hyperparameter guards, all four algorithms on CliffWalking, FrozenLake
    learning floor, Session fit/act/evaluate/bundle round-trip).

## [2.4.0a2]: proof suite / preprocess harden: 2026-08-03

### Summary

Hardens Session preprocess role skipping, FLAML AutoML evaluate/bundle predict
paths, production extras markers (Windows / Py3.13), availability probes, and
benchmark skip discipline; ships the Tier A/B/C proof suite with docs/README
linkage. This **`2.4.0a2`** prerelease was available through GitHub and was
not published to PyPI.

### Added

- **Proof suite (Tier A/B/C complete):** 57 single-domain Tier A projects, 36 Tier B
  cross-domain products, and 57 Tier C same-split industry twins under
  [`proofs/`](proofs/README.md). Harness: `python -m proofs._lib.run_all --tier all`.
  Guides and Sphinx index deep-link to proofs; `buildml[production]` remains
  best-effort on Python 3.13 (environment markers skip broken upstream wheels).
- **README rewrite:** Session 2.x install honesty, extras table, production
  caveats, domain overview, and proof suite (57/57 A, 36/36 B, 57/57 C) with
  `python -m proofs._lib.run_all`.

### Fixed

- **Preprocess role skip:** default `scale` / `encode` / `impute` / outliers /
  binning / text (and related resolvers) transform `feature`-role columns only;
  `ignore` / `id` / `target` / `group` / `time` / `weight` are skipped unless
  `columns=[...]` is passed explicitly: knapsack costs and IDs stay unmutated.
- **FLAML adapter predict path:** Session evaluate/bundle replay wraps the full
  FLAML `AutoML` object (not peeled `.model` / `.estimator`) so string
  categoricals survive modern XGBoost.
- **Production installability (Windows / Py3.13):** `buildml[production]` and nested
  extras use PEP 508 markers so missing wheels no longer hard-fail resolver:
  LightFM split to `recommenders-lightfm` (`python_version < "3.13"` and
  non-Windows); `learn2learn`, `giotto-tda`, `neuralforecast`, and `skope-rules`
  constrained off Py3.13. README honesty: production is best-effort.
- **PyOD 3.x API:** HBOS/COPOD/ECOD no longer pass unsupported `random_state`;
  DeepSVDD imports `pyod.models.deep_svdd` and requires `n_features`.
- **Availability probes:** skope-rules, LightFM, SDMetrics/SDV (and related) use
  real import try/except; capability matrices / default routers use
  `torch_available()` instead of `torch_spec_available()` for graph, online,
  anomaly, semi-supervised, meta-learning, ranking, multitask, active learning,
  CBR, symbolic neuro paths. `metalearning_industry_available()` requires a
  working torch import and discloses native first-order MAML when learn2learn
  is absent.
- **Benchmark hard-fails → skips:** symbolic skips unusable skope; synthetic
  catches torch/sdmetrics `OSError`; CBR runs sklearn/industry before any torch
  probe so AV/DLL faults cannot block the core floor. Active-learning / online /
  multitask / semi-supervised smokes gate torch paths with `torch_available()`
  (not `torch_spec_available()`). CBR sklearn/industry resolve and adapters no
  longer eagerly import torch (lazy `torch_metric` + resolve short-circuits).

### Added

- **Capability matrices:** thin honest `ssl_capability_matrix`,
  `rag_capability_matrix`, `unsupervised_capability_matrix`,
  `forecast_capability_matrix`, `timeseries_capability_matrix` (+ Session
  static accessors).
- **R6 refinement sweep (industry depth):** Each
  domain ships `*_capability_matrix()`, `backend=` auto-routing (sklearn/native
  fallback when extras absent; industry/torch/ssl/rl adapters default when
  installed), per-domain benchmark smoke, guides/explain/AI allowlist updates,
  and its industry extra in `buildml[production]`. Domains:
  - **R6.1 semi-supervised:** XGB/LGBM pseudo-label (`semisupervised-industry`),
    FixMatch/MixMatch tabular (`torch`), HF text pseudo-label (`ssl`); benchmark
    `benchmarks/semisupervised/partial_labels.py`; SSL integration pipeline
    (`fit_ssl_pretext` → `transform_ssl` → `fit_semisupervised`).
  - **R6.2 active learning:** scikit-activeml CoreSet/QBC (`activelearning-industry`),
    Torch BALD/MC-dropout (`torch`); benchmark
    `benchmarks/activelearning/query_efficiency.py`; `label_rows` stays human-only.
  - **R6.3 online / continual:** River streaming + ADWIN/Page-Hinkley drift
    (`online-industry`), lite torch replay/EWC (`torch`); benchmark
    `benchmarks/online/stream_accuracy.py`.
  - **R6.4 multi-task:** XGB/LGBM/CatBoost multi-target (`multitask-industry`),
    shared-trunk multi-head torch (`torch`); benchmark
    `benchmarks/multitask/multi_target_quality.py`.
  - **R6.5 meta-learning:** torch prototypical encoder (`torch`), learn2learn
    MAML/Reptile (`metalearning-industry`); benchmark
    `benchmarks/metalearning/few_shot_adaptation.py`.
  - **R6.6 symbolic / neuro-symbolic:** skope-rules + imodels + optional Z3
    (`symbolic-industry`), lite torch CBN/NAM (`torch`); benchmark
    `benchmarks/symbolic/rule_fidelity.py`.
  - **R6.7 CBR:** hnswlib approximate retrieval (`cbr-industry`), text case
    embedding (`rag|ssl`), lite torch metric encoder (`torch`); benchmark
    `benchmarks/cbr/retrieval_accuracy.py`; CBR≠RAG boundary documented.
  - **R6.8 LTR:** LightGBM LambdaRank / XGB rank:ndcg / CatBoost YetiRank
    (`ranking-industry`), torch listwise-lite (`torch`); benchmark
    `benchmarks/ranking/ndcg_lift.py`; LTR≠RAG≠recommenders boundary documented.
  - **R6.9 optimisation / decisions:** PuLP/OR-Tools knapsack MIP, CVXPY LP
    (`optimize-industry`); benchmark `benchmarks/optimize/policy_value.py`.
  - **R6.10 synthetic data:** SDV CTGAN/TVAE/CopulaGAN + SDMetrics
    (`synthetic-industry`); benchmark `benchmarks/synthetic/tstr_quality.py`; no
    DP claims.
  - **R6.11 imitation + RL:** Gymnasium REINFORCE-lite (`buildml[rl]`), SB3
    PPO/DQN/A2C + imitation BC/GAIL-lite (`buildml[rl-industry]`); benchmark
    `benchmarks/rl/policy_return.py`; offline/batch RL disclosed out of scope.
- **R1–R5 domain refinement (industry depth):** Full refinement sweep across SSL,
  unsupervised, time-series (analysis + forecast), RAG, AutoML, anomaly,
  recommenders, causal, federated, knowledge graphs, probabilistic, graph (PyG),
  and TDA. Each domain ships honest **capability matrices** (installed backends,
  methods, extras) for docs, walkthrough, and AI tools. Industry libraries are
  **defaults when installed** (`backend='auto'` / resolver helpers); core sklearn
  paths remain when extras are absent.
- **Industry optional extras:** `automl-industry` (FLAML + AutoGluon + GBDT
  stack), `anomaly-industry` (PyOD), `recommenders-industry` (implicit +
  LightFM), `causal-industry` (DoWhy + EconML), `federated-industry` (Flower),
  `kg-industry` (PyKEEN), `probabilistic-industry` (MAPIE + NGBoost),
  `semisupervised-industry`, `activelearning-industry`, `online-industry`,
  `multitask-industry`, `metalearning-industry`, `symbolic-industry`,
  `cbr-industry`, `ranking-industry`, `optimize-industry`, `synthetic-industry`,
  `rl-industry`, `tda-industry` (giotto-tda). Depth extras: `ssl`, `unsupervised`,
  `timeseries` (+ `timeseries-prophet`, `timeseries-ml`), `graph-pyg`,
  `rag-advanced` (LangChain retrieve hooks), `rl` (Gymnasium REINFORCE-lite).
- **`buildml[production]` meta-extra:** One-shot optional group aggregating all
  R1–R6 `*-industry` extras plus core depth extras (`torch`, `ssl`, `rag`, `tda`,
  `unsupervised`, `timeseries`, `graph`, `graph-pyg`, `optuna`, `automl`, `rl`).
  Core `import buildml` unchanged.
- **Benchmark smoke scripts:** 25 per-domain runners under `benchmarks/` (R1–R5:
  SSL linear probe, unsupervised cluster quality, TS analysis/forecast, RAG
  retrieval, AutoML tabular search, anomaly detector comparison, recommenders
  ranking, causal ATE, federated FedAvg, KG link prediction, probabilistic
  interval coverage, graph node classification, TDA persistence; R6: semi-supervised,
  active learning, online, multi-task, meta-learning, symbolic, CBR, LTR,
  optimise, synthetic, RL). Discovered by `scripts/run_benchmark_smokes.py`;
  graceful skips when optional extras missing; CI `benchmarks` job on Linux (core
  install).
- **Bundle format bumps (v2, v1 loadable):** `buildml.ssl_bundle.v2`,
  `buildml.unsupervised_bundle.v2`, `buildml.forecast_bundle.v2`,
  `buildml.tda_bundle.v2`: richer plan metadata for refined domains.
- **Guides sync:** Refresh ``guides/`` (and Sphinx includes) so
  tutorials cover new surfaces without inventing APIs: gated multimodal
  fusion + frozen ``multimodal_preprocess`` restore, ``evaluate_asr`` /
  ``SpeechContract``, ``list_pretrained_backbones`` / ``attach_backbone_head``,
  serve ``/metadata`` + ``/predict/batch`` + optional local HTTPS, TorchServe
  compose + K8s ConfigMap/GPU + ``emit_k8s_serve_deployment``, and RAG
  faithfulness hooks. Install honesty (GitHub 2.x vs PyPI 1.x) unchanged.
- **Torch multimodal / RAG depth:** Deepens real library paths for Torch multimodal
  (gated fusion + frozen ``multimodal_preprocess`` restore), speech
  (``evaluate_asr`` WER/CER, ``SpeechContract`` round-trip), pretrained backbones
  (ResNet34/50, ViT-B/32, HuBERT, Whisper-base encoder; ``attach_backbone_head`` /
  ``list_pretrained_backbones``), local serve (``/metadata``, ``/predict/batch``,
  optional local HTTPS), K8s emitters (ConfigMap + GPU requests, serve
  Deployment template), TorchServe compose example, and RAG cheap faithfulness
  hooks. Teaching sync + AI tools for new Session APIs. CI stays mock-safe.
- **Guides depth:** Exhaustive user guide system under ``guides/`` with
  Session-domain → guide map, learning path, and deep tutorials for classical
  E2E, leakage/fold-local recipes/weights/hard-refuse CV, preprocess depth,
  engines, EDA/Teaching Studio, diagnostics/search, artifacts, Torch
  (tabular/text/multimodal/CV/AMP/DDP/export), speech, pretrained backbones,
  RAG, AI safety + tool patterns, and serve/deploy recipes. Sphinx
  ``docs/guides.rst`` / ``usage.rst`` / ``features.rst`` / ``index.rst`` point
  at the expanded set; optional ``examples/`` scripts mirror key snippets.

### Changed

- **SSL defaults:** Torch tabular methods (`simclr_tabular`, `byol_tabular`,
  `vicreg_tabular`, `mae_tabular`, `vae_tabular`) are industry defaults when
  `buildml[torch]` is installed. Legacy sklearn `masked_tabular` remains as
  **deprecated fallback** when Torch is absent.
- **Unsupervised depth:** Extended clustering (GMM, spectral, OPTICS, mean-shift,
  optional HDBSCAN, DEC/IDEC when Torch present) with v2 bundle persistence.
- **Time-series depth:** statsmodels-backed decomposition/diagnostics/changepoints;
  industry forecast backends via Prophet and NeuralForecast behind optional extras.
- **RAG depth:** LangChain adapter behind `buildml[rag-advanced]`; retrieval
  quality benchmark with hashing CI floors and optional semantic/rerank paths.

### Clarified

- **Supported deployment scope clarified.** The library supplies selected
  pretrained-backbone integrations and local serving helpers. Managed cloud
  identity, live multi-cluster orchestration, foundation-model pretraining,
  and hosted vector databases require external systems.

### Fixed

- **Docs install path:** Read the Docs install path now uses
  `.readthedocs.yaml` `path: .` + `extra_requirements: [docs]` (no longer
  relies on RTD installing via `docs/requirements.txt` alone). Guides add
  GitHub-first install honesty; `guides/README.md` no longer implies hosted
  RTD is already 2.x-complete.

### Changed

- **Post-`v2.4.0a1` hygiene + `2.4.0a2` bump:** `e142f0d` removed the private
  `maintainers/` tree and refreshed public doc hygiene. This cut bumps the
  package line to **`2.4.0a2`**, a GitHub prerelease that was not published to PyPI.

## [2.4.0a1]: post-depth / process closure: 2026-08-02

### Summary

The **`2.4.0a1`** prerelease documented installation from GitHub, corrected
repository URLs, expanded pretrained and serving CI, aligned the AI tool
allowlist, and added safeguards for public serving addresses. It was not
published to PyPI.

### Added

- **Release / process closure:** version identity `2.4.0a1`; README +
  `docs/installation.rst` install honesty (GitHub 2.x vs PyPI legacy `1.0.9`);
  `CONTRIBUTING.md` + release checklist pointer; AI tools/executor/
  planner/autonomy wiring; serve non-loopback bind guard
  (`api_keys` or `allow_insecure_public_bind`); CI torch/serve/pretrained matrix
  runs `tests/unit/test_pass_r_pretrained_serve_k8s.py`.
- **Pretrained / serve / K8s depth:** curated vision/audio/speech backbone
  hooks (`load_pretrained_backbone`, extras `vision` / `pretrained`, mock CI
  weights); optional API-key/Bearer serving auth; TorchServe pack + TensorRT
  `trtexec` plan helpers; K8s torchrun Job YAML emitter + `deploy/k8s` example;
  `domain_adapt_speech_torch` + `refuse_speech_foundation_pretrain` honesty.
  Not a managed cloud, not live multi-cluster orchestration, not FM-from-scratch.
- **Speech ASR path:** ASR transcription + classify finetune-lite behind
  `buildml[speech]` / Torch. Session APIs `make_speech_torch_loaders`,
  `fit_speech_torch`, `transcribe_speech` (stub CI-safe backend; optional
  transformers Whisper-class). Honest alpha: integration/finetune, not
  training a foundation model from scratch. Teaching sync + AI tools/executor.
- **Multi-node DDP:** `fit_torch_ddp(..., multi_node=True)` joins
  torchrun env (`WORLD_SIZE` / `RANK` / `LOCAL_RANK` / `MASTER_ADDR` /
  `MASTER_PORT`); clear misconfig errors; CPU multi-process still requires
  `allow_cpu_ddp=True`. Not Kubernetes multi-cluster orchestration.
- **Managed serving:** `buildml[serve]` FastAPI server with `/health` +
  `/predict` for classical pipeline bundles and TorchScript; CLI
  `buildml-serve` / `python -m buildml.serving` and `Session.serve_bundle`.
  Localhost default; no auth product claim.
- **Audio multimodal:** extend multimodal fusion to audio path/waveform
  columns fused with tabular and/or text and/or image. Train-only audio
  amplitude mean/std, built-in small 1D-CNN fusion branch (honest alpha: not a
  speech foundation model), Session facades
  (`make_multimodal_torch_loaders(..., audio_column=)` /
  `make_audio_multimodal_torch_loaders`), `fit_torch` / `export_torch` refuse
  silent tabular rebuilds, AI tools/executor/autonomy allowlist/planner wiring,
  and teaching-surface sync. `soundfile` is included in `buildml[torch]` (also
  via `buildml[audio]`) for path cells; waveform arrays work with Torch alone.
  CI torch job runs audio multimodal tests with `.[torch,onnx]`.

### Changed

- **License:** project license switched from MIT to **Apache-2.0**
  (`LICENSE`, `NOTICE`, `pyproject.toml`, package `__license__`, README / docs
  mentions).

### Fixed

- **Repository URLs:** replace stale `TechLeo-Dev/BuildML` refs with
  `TechLeo-Libraries/BuildML`.
- **Torch label remapping:** Torch classification paths now LabelEncoder-style remap
  sparse/non-contiguous integer class ids to contiguous ``0..K-1`` (speech, text,
  tabular, multimodal, CV/search). ``class_labels`` keeps original ids in index
  order so ``n_classes = len(class_labels)`` matches CrossEntropy targets;
  evaluate confusion matrices decode back to original ids. CI wires label-remap tests.
- **CI torch job expansions:** CI torch job now runs speech/DDP/serve tests;
  extras matrix covers `buildml[serve]`; multi-node DDP requires `LOCAL_RANK`
  (no silent global-rank→device mapping) and writes parsed `MASTER_*` into the
  process env; DDP rank bundles retain speech/multimodal modality metadata;
  executor speech dispatch + serve CLI localhost/no-auth defaults covered by
  tests; teaching overlay honesty for ASR stub (Torch not required) and
  multi-node `LOCAL_RANK` failures.
- **Torch trainer bundles:** torch trainer bundles persist
  `multimodal_preprocess` (audio/image stats, source SR, layout) with load-path
  honesty (meta restored; DataLoaders not auto-rebuilt); ambiguous 2D waveform
  arrays raise instead of silent flatten; repeat-pad kept as alpha pooling
  choice (documented + short-clip pool-signal test) rather than widening the
  forward/export contract with length-masked pooling.
- **Audio clip handling:** short audio clips are
  repeat-padded (not zero-filled) so default `audio_max_samples` does not wipe
  the 1D-CNN pool; train-only amp stats use pre-pad lengths; media path/array
  columns are refused as inferred text without `audio_column=`/`image_column=`;
  `fit_torch` / `evaluate_torch` / `fit_torch_ddp` refuse silent tabular loader
  rebuild after multimodal/text fit (not only `export_torch`); ONNX export uses
  multimodal `input_layout` names; AI tool schemas/executor forward
  `audio_sample_rate` / `audio_max_samples` / `audio_source_sample_rate`;
  `docs/features.rst` no longer falsely claims audio multimodal is deferred.
- **ONNX export on Torch ≥2.9:** ONNX export broke on Torch ≥2.9
  (`dynamo=True` default requires `onnxscript`). Export now uses
  `dynamo=False` and requires `buildml[onnx]` up front. AI registry/executor
  coverage for `make_image_multimodal_torch_loaders` is regression-tested;
  image multimodal ONNX smoke added. CI torch job installs `.[torch,onnx]`.
  Maintainer architecture note no longer claims image multimodal is deferred.

### Added

- **Image multimodal:** extend multimodal fusion beyond tabular⊕text to
  include image path/array columns fused with tabular and/or text. Train-only
  image channel mean/std, built-in CNN fusion branch, Session facades
  (`make_multimodal_torch_loaders(..., image_column=)` /
  `make_image_multimodal_torch_loaders`), `fit_torch` / `export_torch` wiring
  that refuses silent tabular rebuilds, AI tools/executor/autonomy allowlist,
  and teaching-surface sync. Pillow is included in `buildml[torch]` for path
  cells.

### Fixed

- **AI registry / RNG polish:** AI registry listed
  `make_multimodal_torch_loaders` / `search_torch` / `nested_cv_torch` /
  `export_torch` but executor had no dispatch handlers (dead wires). Tool schemas
  now accept `param_grid` / `param_distributions`. Multimodal fusion `forward`
  supports both tuple and dual-arg calling so TorchScript/ONNX export works.
  Randomized Torch search seeds scipy-like `rvs` with an int (not
  ``numpy.Generator``). CI torch/ai jobs run nested-HPO tests. README Torch extra
  row matches shipped depth. Export refuses silent tabular loader rebuild after
  text/multimodal fit.

### Added

- **Nested Torch HPO:** nested Torch HPO (`Session.nested_cv_torch` /
  `search_torch` with fold-local normalize), tabular+text multimodal fusion
  (`make_multimodal_torch_loaders` + built-in fusion module), explicit
  `ai_run_autonomous` operator automation (confirm opt-in, allowlist, max steps,
  blocked sample egress, transcript audit), CUDA AMP via
  `TrainConfig.mixed_precision` / `fit_torch(..., mixed_precision=True)`,
  single-node `fit_torch_ddp`, and TorchScript/ONNX `export_torch`. Optional
  `buildml[onnx]` extra for ONNX checker smoke tests.

### Fixed

- **Soft-leakage docstring fixes:** closed soft-leakage docstring/concept regressions
  that an earlier line-local lint missed (`nested_cv_score` param docs still claimed
  refuse only when no fold-local recipe is provided; Concept Academy still taught
  “without a fold recipe → limited honesty”). Restored UTF-8 in
  `buildml/explain/concepts/*` after cp1252 mojibake (`Â±`, `â†'`, `â€"`, Greek).
  Copy lint now checks adjacent-line windows and rejects mojibake markers.
  Stale DL gate/checklist copy that denied built-in MLP, text loaders, and
  fold-local Torch CV on current HEAD was corrected.
- **Soft-leakage teaching fixes:** corrected soft-leakage teaching regressions that still claimed
  a fold-local `PreprocessRecipe` alone bypasses Session-global CV refuse (README,
  guides, workflow guide, overlays). Added copy-lint rule
  `soft-leakage-false-claim`. Overlay tuple bugs (missing trailing commas) fixed.
- Stale honesty after Phase C: fold-local Torch CV and `rag_generate` are
  documented as shipped; user docs no longer contradict HEAD.

### Added

- Catalog parameter auto-fill from `operation_index.json` plus stricter
  signature↔catalog param parity; dashboard Teaching Studio concept keys are
  gated against `CONCEPT_NOTES`.
- Scoped mypy and Phase C domain tests (`test_dl_phase_c`, `test_rag_generate`,
  `test_ai_phase_c`) in CI.
- Concepts package split (`buildml/explain/concepts/{classical,dl,rag,ai}.py`)
  replacing the single concepts module.
- **Phase D teaching sync:** generated Session operation index
  (`buildml/explain/generated/operation_index.json`), domain catalog overlays
  under `buildml/explain/overlays/`, and CI gate
  `python scripts/sync_teaching_surface.py --check` so Session ↔ catalog ↔ AI
  tools cannot drift silently. `make_text_torch_loaders` added to the default
  AI tool allowlist.
- **Phase C domain depth (RAG / DL / AI):** product names now match shipped
  capability rather than thin retrieve-only / tabular-only / advisor-only slices.
- **RAG generate:** `Session.rag_generate` grounded generation with citations,
  pluggable chat providers (Session AI provider, `MockProvider`, or
  `EchoGroundedProvider` for offline CI), empty-retrieval / missing-index hard
  failures, and `embedder="auto"` (semantic when `buildml[rag]` importable).
- **DL depth:** built-in tabular MLP + text embedding classifier; optional
  `fit_torch()` without a hand-rolled module; classical plan disclosure /
  `apply_plans=` bridge on `make_torch_loaders`; fold-local
  `cross_validate_torch`; `make_text_torch_loaders` sequence/text modality.
- **AI operator depth:** default tool registry covers classical + RAG retrieve /
  generate + Torch train/eval/CV; plan steps carry `parameters`; multi-step
  `ai_run_plan` can orchestrate grounded RAG and Torch tools; MockProvider
  supports queued multi-turn tool calls.
- Cost-sensitive `tune_threshold(fp_cost=..., fn_cost=...)` with recommended
  threshold, expected cost, and structured operating points.
- Stronger `error_slices`: multi-column segments, richer metrics, small-n
  handling, optional HTML export.
- Richer `dry_run` / `summarize_history` audit UX (ranked risks, prerequisite
  graph summary, suggested next ops) surfaced on walkthrough HTML.

### Changed

- Session facade typing for DL/RAG/AI public results and key method signatures
  (still a thin delegate; logic remains in domain/ops packages).
- RAG / DL / AI maintainer locks and quickstarts updated so “no generate” is no
  longer the product ceiling; hashing remains the CI-safe default embedder with
  semantic path first-class behind `buildml[rag]`.

## [2.3.0a1]: AI operator alpha: 2026-08-02

First AI operator alpha on the BuildML 2.x `Session` API. Capabilities and
limitations of that release are listed below. Classical alpha remains at `2.0.0a1`;
DL alpha at `2.1.0a1`; RAG alpha at `2.2.0a1`. This line adds optional
LLM-assisted workflow guidance: **not** autonomous agents or auto-execution.

### Added

- Optional `buildml.ai` domain behind `buildml[ai]` (alias `buildml[llm]`):
  LLM-assisted workflow guidance via typed tool registry, privacy-aware egress,
  and propose-confirm-execute patterns.
- Session delegates: `ai_configure`, `ai_egress_preview`, `ai_dry_run`,
  `ai_advisor`, `ai_plan`, `ai_execute`, `ai_status`, `save_ai_transcript`,
  `load_ai_transcript`; result slots `ai_result` / `ai_transcript`.
- Provider protocol (OpenAI-compatible) with BYO API key; `MockProvider` for
  CI and offline testing.
- Tool registry with typed allowlist: read-only tools (describe, explain,
  workflow_status), write tools (set_roles, impute, split, fit), destructive
  tools (drop_columns) with confirmation policies.
- Egress privacy controls: `STATS_ONLY` default, column allow/deny lists,
  `ai_egress_preview` manifest, `ai_dry_run` payload inspection.
- Multi-step planner with batch approve and budget tracking (token/cost limits).
- Transcript schema `buildml.ai_transcript.v1` (distinct from checkpoint/bundle);
  API keys and raw data never persisted by default.
- Injection hardening: detection of malicious prompts, column names, and RAG
  chunks; tool registry is the trust boundary.
- Explain catalog/concept coverage for AI ops; AI quickstart, alpha gate, and
  release checklist.
- CI `ai` job (Python 3.11–3.12) with unit tests using `MockProvider` only.

### Known limits (AI alpha)

- **Bring-your-own API key.** BuildML never ships, proxies, or embeds keys.
- **Default egress is STATS_ONLY.** Raw rows require explicit opt-in and
  confirmation. Provider sees whatever egress payload the user approved.
- **Propose → confirm → execute.** No autonomous agent or auto-execution.
- **Tool registry is the trust boundary.** Cannot execute arbitrary code or
  tools not in the allowlist.
- **Transcript ≠ checkpoint ≠ bundle.** Three distinct artifacts.
- **Not a replacement for Teaching Studio.** Supplements, not replaces.
- **Not fine-tuning LLMs.** Use DL domain or external tools.
- **Advice must be verified.** Evidence-bound recommendations are not infallible.
- **No local-only provider path.** OpenAI-compatible protocol only; local LLM
  support is later.
- **CI runs with MockProvider only.** No real API keys in tests.
- **Public AI APIs and transcript formats may change before a stable release.**

## [2.2.0a1]: RAG alpha: 2026-08-01

First retrieval (RAG) alpha on the BuildML 2.x `Session` API. Capabilities and
limitations of that release are listed below. Classical alpha remains at `2.0.0a1`;
DL alpha remains at `2.1.0a1`. This line adds optional retrieve / evaluate /
bundle: **not** generate or an LLM operator.

### Added

- Optional `buildml.rag` domain: corpus ingest → chunk → embed/index → retrieve →
  evaluate → upsert/delete → `buildml.rag_bundle.v1` save/load.
- Session delegates: `rag_ingest_corpus`, `rag_chunk`, `rag_embed_and_index`,
  `rag_retrieve`, `rag_evaluate`, `rag_upsert`, `rag_delete`, `save_rag_bundle`,
  `load_rag_bundle`; result slots `rag_index_result` / `rag_retrieve_result` /
  `rag_eval_result`.
- Default embedder `buildml.hashing_embed.v1` (CPU hashing) and NumPy cosine
  store; optional sentence-transformers / cross-encoder behind `buildml[rag]`.
- Hybrid retrieve (dense + BM25, RRF or weighted), metadata filters, eval depth
  (recall@k, MRR, nDCG@k, hit-rate@k, document|chunk relevance,
  `compare_retrieval_configs`), walkthrough `rag_status`.
- Explain catalog/concept coverage; RAG quickstart, glossary terms, alpha gate,
  and release checklist.
- CI `rag` job (Python 3.11–3.12) with unit, integration, and RAG alpha smoke.

### Known limits (RAG alpha)

- Hashing default is lexical/hashed, not semantic retrieval quality.
- Local-first NumPy store; no hosted vector-DB product path.
- **No generate (`rag_generate`) and no LLM operator / agent product
  (`buildml.ai`).**
- No Teaching Studio RAG cockpit redesign; structured results + `rag_status` only.
- CPU merge gate for RAG CI; GPU embed/rerank optional when available.
- Session checkpoints never embed the vector index.
- Public RAG APIs and bundle formats may change before a stable RAG release.

## [2.1.0a1]: DL alpha: 2026-08-01

First deep-learning alpha on the BuildML 2.x `Session` API. Capabilities and
limitations of that release are listed below. Classical alpha remains documented at
`2.0.0a1`; this line adds optional Torch.

### Added

- Optional `buildml.dl` domain behind `buildml[torch]` (alias `buildml[dl]`):
  tabular partition → DataLoaders → train loop → evaluate → trainer bundle.
- Session delegates: `make_torch_loaders`, `fit_torch`, `evaluate_torch`,
  `torch_training_curve`, `save_torch_bundle`, `load_torch_bundle`; result slot
  `dl_train_result`.
- Trainer bundle schema `buildml.torch_bundle.v1` (distinct from Session
  checkpoints and classical pipeline bundles).
- M2 depth: early stopping, grad clip, LR schedulers, group/time split honor,
  resume training, structured training-curve report, walkthrough Torch status.
- Explain catalog/concept coverage for Torch ops; DL quickstart, glossary terms,
  alpha gate, and release checklist.
- CI `torch` job (Python 3.11–3.12) with unit, integration, and DL alpha smoke.

### Known limits (DL alpha)

- CPU merge gate; no GPU CI on every PR. Tabular numeric features first.
- No built-in model collection; caller supplies `nn.Module`.
- Materialized Pandas/NumPy tensors; no Polars/DuckDB zero-copy into loaders.
- Classical preprocess is not auto-applied before loaders.
- No fold-local Torch CV, DDP, mixed precision, or ONNX/TorchScript product path.
- RAG / LLM operator remain out of scope.
- Public Torch APIs and bundle formats may change before a stable DL release.

## [2.0.0a1]: classical alpha: 2026-08-01

First classical-ML alpha of the BuildML 2.x `Session` API. Capabilities and
limitations of that release are listed below.

### Added

- Stateful `Session` workflow: ingest → roles → EDA → split → train-fitted
  preprocess → CV/search → fit → evaluate → checkpoint / pipeline persist.
- Fold-local `PreprocessRecipe` for CV/search (dates, text, outliers, impute,
  encode, binning, scale, PCA reduce, select) with Session-global-only custom
  transforms and resample documented and disclosed.
- Optional Polars/DuckDB engines for project/filter/sample/aggregate before
  Pandas materialization (sklearn still needs an in-memory design matrix).
- `Dataset.aggregate` supports `median` and integer percentiles `q0`..`q100`
  (continuous/linear interpolation; DuckDB `quantile_cont`).
- Teaching Studio / walkthrough disclosures for engine/lazy-native status,
  nested-CV `warm_start_studies`, and fold-local vs Session-global preprocess
  scope (text/PCA, custom, resample).
- Offline HTML reports, explain catalog/concepts, classical alpha-gate smoke
  test, and CI matrix (core, engines, Optuna, extras).

### Known limits (alpha)

- Custom transforms and resample stay Session-global (not in `PreprocessRecipe`).
- Native engines do not enable out-of-core sklearn fitting.
- Hashing text features are not invertible; PCA explained variance is unsupervised.
- Deep learning / RAG / LLM operator, fairness, and SHAP-style explainability
  are out of classical alpha scope.
- Public APIs and checkpoint/pipeline formats may change before stable 2.0.

## [1.x]

Archival release notes for the retired 1.x line live in
[`docs/history.rst`](docs/history.rst). The 1.x `SupervisedLearning` facade is
not part of the 2.x public API.
