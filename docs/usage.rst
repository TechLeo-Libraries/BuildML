Quickstart
==========

BuildML enforces a deliberate order: ingest, assign roles, define partitions,
fit preprocessing on training rows only, fit a model, then evaluate on a
partition whose purpose you declare. Skipping a step or calling ``fit`` before
``split`` raises a clear error rather than leaking statistics across holdouts.

This page shows several realistic patterns. For a chapter-style walkthrough,
see :doc:`quickstart-classical`. For the encyclopedic classical path (many use
cases, failure modes, and persistence), see :doc:`classical-end-to-end`. The
full guide map lives in :doc:`guide-index` / :doc:`guides`.

Loan approval (classification)
------------------------------

A small binary classification loop with missing values and mixed numeric
features:

.. code-block:: python

   import pandas as pd
   from sklearn.linear_model import LogisticRegression

   from buildml import Session

   frame = pd.DataFrame(
       {
           "age": [21, None, 35, 40, 29, 33, 52, 47],
           "income": [40, 55, 60, 80, 50, 70, 90, 65],
           "approved": [0, 1, 0, 1, 0, 1, 1, 0],
       }
   )

   session = Session.ingest(frame)
   session.set_roles(
       {"age": "feature", "income": "feature", "approved": "target"}
   )
   session.split(test_size=0.25, stratify=True, random_state=42)

   session.impute(strategy="median")
   session.scale(method="standard")
   session.fit(LogisticRegression(max_iter=500), task="classification")

   result = session.evaluate(partition="test")
   print(result.metrics)

Use a validation partition when model, feature, calibration, or threshold
choices will be repeated:

.. code-block:: python

   session.split(
       test_size=0.2,
       validation_size=0.2,
       stratify=True,
       random_state=42,
   )

Imbalanced fraud detection
--------------------------

When the positive class is rare, read prevalence on the training partition
before trusting accuracy. Resample **train only** after the split:

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier

   from buildml import Session

   # Synthetic fraud-like imbalance (5% positive)
   rng = pd.Series(range(200))
   frame = pd.DataFrame(
       {
           "amount": rng * 1.5 + 10,
           "velocity": (rng % 7).astype(float),
           "is_fraud": (rng % 20 == 0).astype(int),
       }
   )

   session = (
       Session.ingest(frame)
       .set_roles({"amount": "feature", "velocity": "feature", "is_fraud": "target"})
       .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
   )

   # Requires: pip install "buildml[imbalanced]"
   session.resample(sampler="smote", random_state=0)
   session.fit(RandomForestClassifier(n_estimators=50, random_state=0))

   val = session.evaluate(partition="validation")
   test = session.evaluate(partition="test")
   print("validation f1:", val.metrics.get("f1"))
   print("test f1:", test.metrics.get("f1"))

Resampling changes training prevalence. Validation and test rows are never
altered. Compare against a baseline that does not resample before claiming
improvement.

House price regression
----------------------

Regression uses the same Session spine with a different task and metrics:

.. code-block:: python

   import pandas as pd
   from sklearn.linear_model import Ridge

   from buildml import Session

   frame = pd.DataFrame(
       {
           "sqft": [850, 920, 1100, 1400, 1600, 1800, 2100, 2400],
           "beds": [2, 2, 3, 3, 4, 4, 4, 5],
           "price_k": [210, 235, 290, 360, 410, 455, 520, 610],
       }
   )

   session = (
       Session.ingest(frame)
       .set_roles({"sqft": "feature", "beds": "feature", "price_k": "target"})
       .split(test_size=0.25, random_state=42)
       .impute(strategy="median")
       .scale(method="standard")
       .fit(Ridge(alpha=1.0), task="regression")
   )

   print(session.evaluate(partition="test").metrics)

Group and time partitions
-------------------------

Random ``split`` assumes independent, exchangeable rows. When rows share an
entity or time ordering, use ``group_split`` or ``time_split`` instead:

.. code-block:: python

   import pandas as pd

   from buildml import Session

   # Multiple visits per customer — random row split would leak customers
   visits = pd.DataFrame(
       {
           "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
           "spend": [10, 12, 15, 8, 9, 20, 22, 25, 5, 6],
           "churned": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
       }
   )

   session = (
       Session.ingest(visits)
       .set_roles(
           {
               "customer_id": "group",
               "spend": "feature",
               "churned": "target",
           }
       )
       .group_split(test_size=0.25, random_state=0)
   )

For temporal data, assign a ``time`` role and call ``time_split``. When an
external system already defined memberships, pass positional indices to
``inject_split``.

Why Session enforces order
--------------------------

Fit-capable steps — imputation, encoding, scaling, resampling, and ``fit`` —
require a split and learn from training rows. That guard prevents the most
common partition leakage: computing holdout statistics during preparation.

BuildML does **not** infer valid group or time boundaries, detect target
proxies, or prove that externally supplied indices match your deployment
assumptions. Roles and splits are explicit because those judgments belong to
the project.

Typical failure modes:

* ``ValidationError: No split exists`` — call ``split``, ``group_split``,
  ``time_split``, or ``inject_split`` before ``impute`` or ``fit``.
* ``LeakageError`` — attempting to fit on validation or test, or resampling
  outside train.
* Missing extra — ``optuna_search``, ``resample``, ``eda_app``, and engine
  adapters name the install group when an optional dependency is absent.

``session.explain("impute", moment="before")`` lists prerequisites, leakage
risks, and alternatives from the operation catalog before you mutate state.

Teaching surfaces: explain, learn, workflow, walkthrough, dry_run
-----------------------------------------------------------------

BuildML ships a versioned operation catalog (``buildml.explain``) linked to
every public Session method. These APIs expose what the library knows; they
do not certify that your split or model suits the domain.

.. code-block:: python

   # Prerequisites, assumptions, leakage risks, alternatives
   before = session.explain("feature_importance", moment="before")

   # Every cataloged operation: done, available, blocked, or skipped
   steps = session.workflow()
   for step in steps:
       if step.status == "blocked":
           print(step.operation, step.reasons or step.blockers)

   importance = session.feature_importance(partition="validation", n_repeats=8)
   after = session.explain("feature_importance", moment="after")

``explain(..., moment="after")`` joins catalog text to the latest recorded
call and its state transition. ``workflow()`` resolves prerequisites for all
public operations. ``available`` means API prerequisites pass, not that the
step is recommended.

Explanations are written for three reading levels, and ``beginner`` is the
default. It assumes no prior machine-learning vocabulary:

.. code-block:: python

   primer = session.explain("feature_importance").beginner
   print(primer.plain_summary, primer.analogy, sep="\n")
   for knob in primer.key_parameters:
       print(knob.name, knob.plain_meaning, knob.typical_choice)
   primer.common_pitfalls
   primer.glossary          # the jargon this answer used, defined in place

   session.explain("feature_importance", level="advanced")   # no scaffolding

When the question is conceptual rather than about the current session, use
``learn``. It accepts a concept key, an operation name, or a term, and returns
a reading order rather than an index:

.. code-block:: python

   session.learn()                       # foundation concepts, in order
   brief = session.learn("leakage")      # a term resolves to its concept
   brief.concept.misconceptions          # wrong belief → correction
   [note.key for note in brief.read_first]
   [note.key for note in brief.read_next]

The level changes how much is shown, never what is true; assumptions, leakage
risks, and failure modes are present at every level.

Preview without mutation:

.. code-block:: python

   preview = session.dry_run(["impute", "scale", "fit"])
   summary = session.summarize_history()
   print(summary.unresolved_risks)

``dry_run`` does not append history. ``summarize_history()`` counts operations,
lists heuristic unresolved risks, and suggests next steps from the prerequisite
graph.

``walkthrough()`` joins workflow status, history, and catalog risks into one
report and can export offline HTML:

.. code-block:: python

   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

Findings, recommendations, and EDA
----------------------------------

``session.eda()`` returns structured findings (observations with severity),
evidence tables, and read-only recommendations. Recommendations name a Session
operation but do not run it.

.. code-block:: python

   report = session.eda(include_plots=False)
   for finding in report.findings[:5]:
       print(finding.severity, finding.title)

Reports and walkthroughs
------------------------

.. code-block:: python

   # Offline dashboard snapshot (default; needs buildml[dashboard])
   eda = session.eda(export_html="artifacts/eda_studio.html", html_format="studio")
   # Optional layered research shell with matplotlib embeds
   research = session.eda(
       include_plots=True,
       export_html="artifacts/eda_research.html",
       html_format="research",
       export_figures="artifacts/eda-figures",
   )
   # Live dashboard (requires: pip install "buildml[dashboard]")
   handle = session.eda_app(port=8765)  # or session.open_eda_dashboard()
   # If port 8765 is busy: session.eda_app(port=8766)
   evaluation = session.evaluate(
       partition="test",
       include_plots=True,
       export_html="artifacts/evaluation.html",
   )
   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

``eda_app()`` opens interactive Plotly domain boards with teaching notes and
concept references. CSV downloads cover major evidence tables. Offline HTML
downloads a self-contained snapshot of the same dashboard SPA. HTML artifacts
embed required styles and assets so they open without a network connection.

Engines: pandas, Polars, DuckDB
-------------------------------

Pandas is the canonical sklearn-facing materialization path. Polars and DuckDB
are optional engines for ingest, filtering, projection, and aggregation:

.. code-block:: python

   from buildml import Session
   from buildml.data import portable_filter_expr

   with Session.ingest("data.csv", engine="duckdb") as session:
       narrowed = session.dataset.filter_expr(
           portable_filter_expr("amount", ">", 100)
       )

``with session:`` calls ``close_native`` on exit so owned DuckDB connections
are released. ``portable_filter_expr`` builds simple quoted comparisons for
Polars and DuckDB; complex SQL remains engine-specific. Lazy Polars frames
collect on ``to_pandas()`` / sklearn materialization — that is not out-of-core
training.

Checkpoint and pipeline round-trip
----------------------------------

.. code-block:: python

   session.checkpoint_save(
       "artifacts/checkpoint",
       sidecar_layout="auto",
       sidecar_partition_rows=25_000,
       sidecar_compression="zstd",
   )
   restored = Session.checkpoint_load("artifacts/checkpoint")
   print(restored.reattach_result.status)

A checkpoint restores data, roles, partitions, history, and optional preprocess
plan objects. It does not restore a fitted model. Use ``save_model`` /
``load_model`` for estimator-only artifacts, or ``save_pipeline`` /
``load_pipeline`` for plans plus estimator and a model card. Pipeline bundles
and checkpoints are complementary: neither embeds the other.

Replay restored plans with ``session.apply_preprocess_plans()`` or score new
rows with ``predict_from_pipeline``. Resample plans are lineage-only at score
time.

Leakage-safe CV and search
--------------------------

Prefer ``PreprocessRecipe`` inside ``cv_score`` / ``grid_search`` /
``randomized_search`` / ``optuna_search`` / ``evolutionary_search`` /
``nested_cv_score`` on data that has
**not** already been Session-globally prepared. Session-global prep then CV is
hard-refused by default — see :doc:`leakage-cv-recipes`.

Optional paths on the same Session
----------------------------------

Torch, RAG, and AI operator features attach to the same Session without
replacing classical APIs:

* :doc:`quickstart-torch` / :doc:`torch-deep` — tabular, text, multimodal,
  CV/search/nested, AMP/DDP, export
* :doc:`speech-asr-finetune` / :doc:`pretrained-backbones` — ASR + classify
  finetune-lite; curated backbone hooks
* :doc:`serve-deploy` — local FastAPI serve, TorchServe/TRT/K8s recipes
* :doc:`quickstart-rag` / :doc:`rag-deep` — retrieve, grounded generate, eval,
  bundle
* :doc:`quickstart-ai` / :doc:`ai-operator-safety` /
  :doc:`ai-tools-operator-patterns` — advisor, confirmed execute, autonomy caps,
  tool allowlist patterns
* :doc:`artifacts-checkpoints-bundles` — checkpoint vs pipeline vs Torch/RAG/AI
  artifacts
* :doc:`eda-teaching-studio` / :doc:`engines-polars-duckdb` /
  :doc:`classical-diagnostics-search` / :doc:`preprocess-depth` — explore, prep
  engines, diagnostics, and preprocess depth

Teaching copy for every public Session method is kept in sync by CI
(``scripts/sync_teaching_surface.py``). Prefer ``session.explain(...)`` over
hand-maintained method lists when exploring the surface.

See :doc:`guides` for the full Markdown tutorials and :doc:`guide-index` for
the learning path.
