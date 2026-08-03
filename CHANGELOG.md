# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with pre-release tags for alpha (`aN`) builds.

## [Unreleased]

### Added

- **CI proof smoke gate.** `python -m proofs._lib.run_all --smoke` runs a fixed
  Tier A subset without `--skip-existing`; GitHub Actions job `proofs-smoke`.
- **Coverage ratchet process.** Full-suite measure via
  `scripts/run_full_coverage.py` (per-module isolation on Windows + combine):
  **70.7%** (39539/55914). `fail_under` raised 25 → 60; next planned 70
  (`scripts/coverage_ratchet.json`).
- **Runtime stability probe.** `scripts/verify_runtime_stability.py`
  subprocess-isolates core/optional use cases (ok/fail/crash/skip) so native
  access violations cannot kill the parent; see `docs/stability.md`.
- **Safe install guide.** `guides/safe-install-and-runtime.md`: staged venv
  install on Python 3.11/3.12, `PYTHONNOUSERSITE=1` on Windows, classical first,
  then Torch/industry extras one group at a time with re-probe.
- **User-doc voice cleanup.** Guides and user-facing docs drop internal Phase /
  R5–R6 / PASS trackers; copy addresses readers directly (developer → user).
- **Lazy/cached walkthrough capability probes.** Default
  `walkthrough(capability_probe="lazy")` skips inactive domain industry imports;
  process-wide matrix + subprocess import caches; `eager` / `skip` modes.
- **Fairness disparity reporting.** `Session.evaluate_fairness` /
  `fairness_capability_matrix` (observational DP / DI / equalized odds).
- **Optional SHAP attribution.** `Session.explain_shap` behind `buildml[shap]`.
- **Release workflow.** `.github/workflows/release.yml` (manual/tag build +
  optional PyPI publish).
- **Surface stability policy.** `docs/stability.md` for alpha churn control.
- **Harder RAG proof corpora.** Adversarial distractors / paraphrases in
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
- **Session monolith split (critical maintainability fix).** Public
  ``buildml.Session`` API unchanged (474 public attributes preserved). Domain method
  signatures/docstrings moved into ``buildml/session/mixins/`` (34 domain
  mixins + ``_shared`` annotation bag); ``session.py`` is now the thin
  assembler (~840 LOC) owning ``__init__``, context manager, and state glue.
  Orchestration remains in ``*_ops.py``. Eliminated all
  ``from buildml.session._imports import *`` star-imports in ops (explicit
  imports; ``Literal``/``Any`` from ``typing``). Coverage
  ``fail_under`` raised 20 → 25 (classical+checkpoint smoke ~26.5%).

- **Quality / hygiene ratchet pass (critical-evaluation follow-up).** Deleted
  stale root ``audit_session2.txt`` (outdated docstring residue). Added
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

- **Fourth adversarial trust-recovery pass (remaining High/Medium polish).**
  Graph beginner examples aligned to real ``fit_graph`` kwargs (``mode``,
  ``pyg_model``, ``include_graph_metrics``, ``classical_estimator``); added
  ``Session.export_round_history`` facade wired to
  ``buildml.federated.results.export_round_history``; all proof
  ``industry_comparison`` stubs replaced with honest ``filled`` Tier C notes
  (baseline_industry.py companions); ``nlp_capability_matrix`` task entries
  now expose ``backends_available`` gating matching runtime refusal; thin
  per-detector anomaly concept notes and RL bridge notes (Monte Carlo, n-step
  TD, actor-critic) linking to REINFORCE/SB3/tabular_q paths.

- **Docstring ratchet (mid-size domain batch).** Completed and enforced at 0:
  ``online`` (114), ``federated`` (107), ``automl`` (99), ``multitask`` (96),
  ``activelearning`` (95), ``anomaly`` (95), ``synthetic`` (93),
  ``semisupervised`` (90), ``unsupervised`` (83).
- **Docstring mission complete.** Every audited package: including
  ``buildml/session/session.py`` and all ``*_ops.py`` facades: is at 0
  findings. ``buildml/session/`` is now in ``ENFORCED_PREFIXES``; repo-wide
  ``scripts/audit_docstrings.py --check`` total is 0.
- **Session ops depth pass.** All 34 ``buildml/session/*_ops.py`` modules plus
  ``audit``, ``state``, and ``walkthrough`` documented to 0 findings.
  ``session`` budget ratcheted 1987 → 0.
- **Docstring budget:** total findings 2859 → 0.
- **Prior round:** ``kg``, ``causal``, ``forecasting``, ``ranking``,
  ``selfsupervised``, ``graph``, ``optimize``, ``recommenders``, ``metalearning``.

### Added

- **Third adversarial capability pass (foundational gaps).**
  Forecasting refuses ``exog_columns`` on univariate-only methods (ETS,
  auto_arima, Prophet, N-BEATS) via ``method_supports_exog``; AI agents get
  ``time_split`` tool + executor dispatch; ranking accepts ``lambdarank`` →
  ``lambdarank_lgbm`` alias; DoWhy holdout evaluate re-estimates ATE on the
  partition; KG evaluate adds relation-prediction ranks; ``dl_capability_matrix``
  + Session/AI introspection; unsupervised/synthetic explain status embed live
  matrices; scikit-activeml industry path tries import with native fallback;
  teaching notes for k-means vs density, ETS/ARIMA, River/torch online paths.

- **Capability-matrix walkthrough / audit hooks (second pass).**
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
    so it cannot drift from the expert sections it fronts: and any operation can
    override any section with authored prose.
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
  overstate, `create_time_split` documents that a chronological cut is the only
  honest evaluation of forecasting, and `guard_fit_partition` explains that its
  refusal is deliberate friction rather than a missing convenience. The second
  is memory: every method now says whether it defers work or forces it, so
  `select_columns` and `filter_expr` are documented as the operations that keep
  data off disk and out of memory, while `sample_rows`, `filter_rows`, and
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
  the history hooks, and the LangChain adapter. Retrieval always returns
  something, so the docstrings say what that something is worth: `retrieve`
  documents that there is no relevance threshold and that a question the corpus
  cannot answer still produces `k` confidently ranked passages; `rrf_fuse`
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
  occupy unrelated spaces, and retrieval returns confident nonsense.
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
  `train_score` is in-sample and therefore not evidence of anything;
  `cv_score` explains that a fold standard deviation is the number that says
  whether a difference between two models is real, and refuses to run when a
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
    normalization learns nothing it cannot leak, so the plan replays it on
    holdout freely: while vocabulary, document frequencies, IDF, topic
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
  `double_q_learning` (cross-evaluated, no maximisation bias) on discrete-action
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
linkage. Bumps the package line to **`2.4.0a2`**. **Not published to PyPI** :
GitHub prerelease / honesty banner only.

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
- **R6 refinement sweep (Phase 2 industry depth: complete, R6.1–R6.11):** Each
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
- **Pass X guides sync:** Refresh ``guides/`` (and Sphinx includes) so Pass W
  tutorials cover Pass V surfaces without inventing APIs: gated multimodal
  fusion + frozen ``multimodal_preprocess`` restore, ``evaluate_asr`` /
  ``SpeechContract``, ``list_pretrained_backbones`` / ``attach_backbone_head``,
  serve ``/metadata`` + ``/predict/batch`` + optional local HTTPS, TorchServe
  compose + K8s ConfigMap/GPU + ``emit_k8s_serve_deployment``, and RAG
  faithfulness hooks. Install honesty (GitHub 2.x vs PyPI 1.x) unchanged.
- **Pass V capability depth:** Deepens real library paths for Torch multimodal
  (gated fusion + frozen ``multimodal_preprocess`` restore), speech
  (``evaluate_asr`` WER/CER, ``SpeechContract`` round-trip), pretrained zoo
  (ResNet34/50, ViT-B/32, HuBERT, Whisper-base encoder; ``attach_backbone_head`` /
  ``list_pretrained_backbones``), local serve (``/metadata``, ``/predict/batch``,
  optional local HTTPS), K8s emitters (ConfigMap + GPU requests, serve
  Deployment template), TorchServe compose example, and RAG cheap faithfulness
  hooks. Teaching sync + AI tools for new Session APIs. CI stays mock-safe.
- **Pass W guides depth:** Exhaustive user guide system under ``guides/`` with
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

- **Honesty limits = product scope, not stubs.** Docs “not a full zoo / not
  managed cloud IAM / not live multi-cluster / not FM-from-scratch / not hosted
  vector DB” statements describe intentional product boundaries around shipped
  library paths: not unfinished placeholder APIs.

### Fixed

- **Pass U process residuals:** Read the Docs install path now uses
  `.readthedocs.yaml` `path: .` + `extra_requirements: [docs]` (no longer
  relies on RTD installing via `docs/requirements.txt` alone). Guides add
  GitHub-first install honesty; `guides/README.md` no longer implies hosted
  RTD is already 2.x-complete.

### Changed

- **Post-`v2.4.0a1` hygiene + `2.4.0a2` bump:** `e142f0d` removed the private
  `maintainers/` tree and refreshed public doc hygiene. This cut bumps the
  package line to **`2.4.0a2`**. GitHub release is **prerelease**. Still not
  published to PyPI: honesty banner only.

## [2.4.0a1]: post-depth / process closure: 2026-08-02

### Summary

Closes release/process gaps after the depth loop (Passes L–R): bumps the package
line to **`2.4.0a1`**, documents GitHub-first install until PyPI carries 2.x,
fixes stale org URLs, wires Pass R CI + AI allowlist parity, hardens public
serve binds, and refreshes maintainer / contributor process docs. **Not published
to PyPI in this cut**: honesty banner only.

### Added

- **Pass T release/process closure:** version identity `2.4.0a1`; README +
  `docs/installation.rst` install honesty (GitHub 2.x vs PyPI legacy `1.0.9`);
  `CONTRIBUTING.md` + release checklist pointer; Pass R AI tools/executor/
  planner/autonomy wiring; serve non-loopback bind guard
  (`api_keys` or `allow_insecure_public_bind`); CI torch/serve/pretrained matrix
  runs `tests/unit/test_pass_r_pretrained_serve_k8s.py`.
- **Pass R pretrained / serve / K8s depth:** curated vision/audio/speech backbone
  hooks (`load_pretrained_backbone`, extras `vision` / `pretrained`, mock CI
  weights); optional API-key/Bearer serving auth; TorchServe pack + TensorRT
  `trtexec` plan helpers; K8s torchrun Job YAML emitter + `deploy/k8s` example;
  `domain_adapt_speech_torch` + `refuse_speech_foundation_pretrain` honesty.
  Not a managed cloud, not live multi-cluster orchestration, not FM-from-scratch.
- **Pass O speech FM path:** ASR transcription + classify finetune-lite behind
  `buildml[speech]` / Torch. Session APIs `make_speech_torch_loaders`,
  `fit_speech_torch`, `transcribe_speech` (stub CI-safe backend; optional
  transformers Whisper-class). Honest alpha: integration/finetune, not
  training a foundation model from scratch. Teaching sync + AI tools/executor.
- **Pass O multi-node DDP:** `fit_torch_ddp(..., multi_node=True)` joins
  torchrun env (`WORLD_SIZE` / `RANK` / `LOCAL_RANK` / `MASTER_ADDR` /
  `MASTER_PORT`); clear misconfig errors; CPU multi-process still requires
  `allow_cpu_ddp=True`. Not Kubernetes multi-cluster orchestration.
- **Pass O managed serving:** `buildml[serve]` FastAPI server with `/health` +
  `/predict` for classical pipeline bundles and TorchScript; CLI
  `buildml-serve` / `python -m buildml.serving` and `Session.serve_bundle`.
  Localhost default; no auth product claim.
- **Pass L audio multimodal:** extend multimodal fusion to audio path/waveform
  columns fused with tabular and/or text and/or image. Train-only audio
  amplitude mean/std, built-in small 1D-CNN fusion branch (honest alpha: not a
  speech foundation model), Session facades
  (`make_multimodal_torch_loaders(..., audio_column=)` /
  `make_audio_multimodal_torch_loaders`), `fit_torch` / `export_torch` refuse
  silent tabular rebuilds, AI tools/executor/autonomy allowlist/planner wiring,
  and teaching-surface sync. `soundfile` is included in `buildml[torch]` (also
  via `buildml[audio]`) for path cells; waveform arrays work with Torch alone.
  CI torch job runs Pass L tests with `.[torch,onnx]`.

### Changed

- **Pass O license:** project license switched from MIT to **Apache-2.0**
  (`LICENSE`, `NOTICE`, `pyproject.toml`, package `__license__`, README / docs
  mentions).

### Fixed

- **Pass T URLs:** replace stale `TechLeo-Dev/BuildML` refs with
  `TechLeo-Libraries/BuildML`.
- **Pass Q after Pass P:** Torch classification paths now LabelEncoder-style remap
  sparse/non-contiguous integer class ids to contiguous ``0..K-1`` (speech, text,
  tabular, multimodal, CV/search). ``class_labels`` keeps original ids in index
  order so ``n_classes = len(class_labels)`` matches CrossEntropy targets;
  evaluate confusion matrices decode back to original ids. CI wires Pass Q tests.
- **Pass P after Pass O:** CI torch job now runs Pass O speech/DDP/serve tests;
  extras matrix covers `buildml[serve]`; multi-node DDP requires `LOCAL_RANK`
  (no silent global-rank→device mapping) and writes parsed `MASTER_*` into the
  process env; DDP rank bundles retain speech/multimodal modality metadata;
  executor speech dispatch + serve CLI localhost/no-auth defaults covered by
  tests; teaching overlay honesty for ASR stub (Torch not required) and
  multi-node `LOCAL_RANK` failures.
- **Pass N leftovers after Pass M:** torch trainer bundles persist
  `multimodal_preprocess` (audio/image stats, source SR, layout) with load-path
  honesty (meta restored; DataLoaders not auto-rebuilt); ambiguous 2D waveform
  arrays raise instead of silent flatten; repeat-pad kept as alpha pooling
  choice (documented + short-clip pool-signal test) rather than widening the
  forward/export contract with length-masked pooling.
- **Pass M adversarial re-audit after Pass L:** short audio clips are
  repeat-padded (not zero-filled) so default `audio_max_samples` does not wipe
  the 1D-CNN pool; train-only amp stats use pre-pad lengths; media path/array
  columns are refused as inferred text without `audio_column=`/`image_column=`;
  `fit_torch` / `evaluate_torch` / `fit_torch_ddp` refuse silent tabular loader
  rebuild after multimodal/text fit (not only `export_torch`); ONNX export uses
  multimodal `input_layout` names; AI tool schemas/executor forward
  `audio_sample_rate` / `audio_max_samples` / `audio_source_sample_rate`;
  `docs/features.rst` no longer falsely claims audio multimodal is deferred.
- **Pass K adversarial re-audit after Pass J:** ONNX export broke on Torch ≥2.9
  (`dynamo=True` default requires `onnxscript`). Export now uses
  `dynamo=False` and requires `buildml[onnx]` up front. AI registry/executor
  coverage for `make_image_multimodal_torch_loaders` is regression-tested;
  image multimodal ONNX smoke added. CI torch job installs `.[torch,onnx]`.
  Maintainer architecture note no longer claims image multimodal is deferred.

### Added

- **Pass J image multimodal:** extend multimodal fusion beyond tabular⊕text to
  include image path/array columns fused with tabular and/or text. Train-only
  image channel mean/std, built-in CNN fusion branch, Session facades
  (`make_multimodal_torch_loaders(..., image_column=)` /
  `make_image_multimodal_torch_loaders`), `fit_torch` / `export_torch` wiring
  that refuses silent tabular rebuilds, AI tools/executor/autonomy allowlist,
  and teaching-surface sync. Pillow is included in `buildml[torch]` for path
  cells.

### Fixed

- **Pass H adversarial re-audit after Pass G:** AI registry listed
  `make_multimodal_torch_loaders` / `search_torch` / `nested_cv_torch` /
  `export_torch` but executor had no dispatch handlers (dead wires). Tool schemas
  now accept `param_grid` / `param_distributions`. Multimodal fusion `forward`
  supports both tuple and dual-arg calling so TorchScript/ONNX export works.
  Randomized Torch search seeds scipy-like `rvs` with an int (not
  ``numpy.Generator``). CI torch/ai jobs run Pass G tests. README Torch extra
  row matches shipped depth. Export refuses silent tabular loader rebuild after
  text/multimodal fit.

### Added

- **Pass G deferred depth:** nested Torch HPO (`Session.nested_cv_torch` /
  `search_torch` with fold-local normalize), tabular+text multimodal fusion
  (`make_multimodal_torch_loaders` + built-in fusion module), explicit
  `ai_run_autonomous` operator automation (confirm opt-in, allowlist, max steps,
  blocked sample egress, transcript audit), CUDA AMP via
  `TrainConfig.mixed_precision` / `fit_torch(..., mixed_precision=True)`,
  single-node `fit_torch_ddp`, and TorchScript/ONNX `export_torch`. Optional
  `buildml[onnx]` extra for ONNX checker smoke tests.

### Fixed

- **Pass F adversarial re-audit:** closed soft-leakage docstring/concept regressions
  that Pass E’s line-local lint missed (`nested_cv_score` param docs still claimed
  refuse only when no fold-local recipe is provided; Concept Academy still taught
  “without a fold recipe → limited honesty”). Restored UTF-8 in
  `buildml/explain/concepts/*` after cp1252 mojibake (`Â±`, `â†'`, `â€"`, Greek).
  Copy lint now checks adjacent-line windows and rejects mojibake markers.
  Stale DL gate/checklist copy that denied built-in MLP, text loaders, and
  fold-local Torch CV on current HEAD was corrected.
- **Pass E re-audit:** corrected soft-leakage teaching regressions that still claimed
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
  replacing the monolithic hand blob.
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

First AI operator alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are listed in this section. Classical alpha remains at `2.0.0a1`;
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

### Verification

- Confirm known limits above still match shipped behavior.
- Tag only after remote CI is green (see `CONTRIBUTING.md`).

## [2.2.0a1]: RAG alpha: 2026-08-01

First retrieval (RAG) alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are listed in this section. Classical alpha remains at `2.0.0a1`;
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

### Verification

- Confirm known limits above still match shipped behavior.
- Tag only after remote CI is green (see `CONTRIBUTING.md`).

## [2.1.0a1]: DL alpha: 2026-08-01

First deep-learning alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are listed in this section. Classical alpha remains documented at
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
- No built-in model zoo; caller supplies `nn.Module`.
- Materialized Pandas/NumPy tensors; no Polars/DuckDB zero-copy into loaders.
- Classical preprocess is not auto-applied before loaders.
- No fold-local Torch CV, DDP, mixed precision, or ONNX/TorchScript product path.
- RAG / LLM operator remain out of scope.
- Public Torch APIs and bundle formats may change before a stable DL release.

### Verification

- Confirm known limits above still match shipped behavior.
- Tag only after remote CI is green (see `CONTRIBUTING.md`).

## [2.0.0a1]: classical alpha: 2026-08-01

First classical-ML alpha of the BuildML 2.x `Session` API. Exit criteria and
known limits are listed in this section.

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

### Verification

- Confirm known limits above still match shipped behavior.
- Tag only after remote CI is green on the push that includes this release
  candidate (see `CONTRIBUTING.md`).

## [1.x]

Archival release notes for the retired 1.x line live in
[`docs/history.rst`](docs/history.rst). The 1.x `SupervisedLearning` facade is
not part of the 2.x public API.
