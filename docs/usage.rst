Quickstart
==========

The order matters: ingest, assign semantic roles, define evaluation
partitions, fit preprocessing on training rows, fit a baseline, and then
evaluate.

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

   # These plans are learned from train and then applied to every partition.
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

Do not use a random split for related entities or future prediction when it
would mix groups or time periods. Compute valid positional memberships and
call ``session.inject_split(...)``.

Explain choices before and after
--------------------------------

.. code-block:: python

   before = session.explain("feature_importance", moment="before")
   workflow = session.workflow()

   importance = session.feature_importance(partition="test", n_repeats=8)
   after = session.explain("feature_importance", moment="after")

``before`` reports prerequisites, alternatives, assumptions, and leakage
risks. ``after`` joins those notes to the latest recorded call and its state
transition. An explanation reports what BuildML knows; it cannot prove that
the chosen partition represents deployment.

Use ``session.dry_run(...)`` to preview intended operations without mutating
state, and ``session.summarize_history()`` for operation counts and unresolved
risk cues. Train-fitted text features, PCA, and registered custom transforms
are available as ``text_features``, ``reduce_dimensions``, and
``apply_custom_transform``. For CV selection, prefer fold-local
``PreprocessRecipe(text=..., reduce='pca')``; custom transforms remain
Session-global. ``Dataset.project`` / ``Dataset.aggregate`` prefer native
Polars/DuckDB ops when an engine handle is attached.

Reports and walkthroughs
------------------------

.. code-block:: python

   # Offline Teaching Studio snapshot (default; needs buildml[dashboard])
   eda = session.eda(export_html="artifacts/eda_studio.html", html_format="studio")
   # Optional layered research shell with matplotlib embeds
   research = session.eda(
       include_plots=True,
       export_html="artifacts/eda_research.html",
       html_format="research",
       export_figures="artifacts/eda-figures",
   )
   # Live Teaching Studio (requires: pip install "buildml[dashboard]")
   handle = session.eda_app(port=8765)  # or session.open_eda_dashboard()
   # If port 8765 is busy: session.eda_app(port=8766)
   # PDF briefing embeds static Plotly PNG stills (kaleido); interactive charts stay in Studio.
   evaluation = session.evaluate(
       partition="test",
       include_plots=True,
       export_html="artifacts/evaluation.html",
   )
   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

``eda_app()`` opens interactive Plotly domain boards with Teaching Studio pages
and Concept Academy notes. The SPA light/dark theme restyles Plotly ink, grids,
series, heatmaps, and annotations. CSV downloads cover major evidence tables;
**Offline HTML** downloads a self-contained snapshot of the same Studio SPA;
the PDF briefing includes metrics, findings, teaching notes, and static chart
stills (not interactive Plotly). ``session.eda(export_html=...)`` defaults to
that Studio offline snapshot; use ``html_format="research"`` for the layered
research shell.

HTML artifacts embed required styles and assets so they can be opened without a
network connection. EDA recommendations do not mutate the Session. Inspect the
evidence, population, sample size, and stated limits before acting.

Checkpoint and resume
---------------------

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
plan objects. Native sidecars default to ``zstd`` compression and ``auto``
layout (single-file below 50k rows; partitioned at or above). Force
``sidecar_layout='single'`` or ``'partitioned'`` when needed. It does not
restore a fitted model; use ``save_model`` / ``load_model`` for estimator-only
artifacts, or ``save_pipeline`` / ``load_pipeline`` for plans plus estimator and
a model card. Pipeline bundles and checkpoints are complementary: neither
embeds the other. Bundle metadata uses ``buildml.pipeline_bundle.v2`` /
``buildml.plans.v2`` (older flat plan dicts still load). Replay restored plans
with ``session.apply_preprocess_plans()`` or
``buildml.preprocess.apply_preprocess_plans``; resample plans are lineage-only
and are not reapplied at score time.

DuckDB connection ownership and portable filters
------------------------------------------------

.. code-block:: python

   from buildml import Session
   from buildml.data import portable_filter_expr

   with Session.ingest("data.csv", engine="duckdb") as session:
       narrowed = session.dataset.filter_expr(
           portable_filter_expr("amount", ">", 100)
       )

``with session:`` / ``with dataset:`` call ``close_native`` on exit so owned
DuckDB connections are not leaked. ``portable_filter_expr`` builds simple
quoted comparisons for Polars and DuckDB; complex SQL remains engine-specific.
