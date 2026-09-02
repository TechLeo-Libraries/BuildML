Installation
============

BuildML supports Python 3.10 through 3.13.

.. important::

   **Default install:** ``pip install buildml`` installs Session **2.5.x**
   (Apache-2.0). You are done when this works::

      python -c "from buildml import Session; print(Session)"

   Legacy **1.x** remains available only if you pin
   ``buildml==1.0.9``.

Install from PyPI
-----------------

.. code-block:: console

   pip install buildml

That is the current stable Session line. Optional extras append the same
way, for example ``pip install "buildml[torch]"``.

GitHub tip of main, or an editable checkout:

.. code-block:: console

   pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
   pip install -e ".[dev]"

Extras by job
-------------

The core install is NumPy, Pandas, PyArrow, and scikit-learn. Install
only what the workflow uses.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - If you need
     - Install
     - Adds
   * - Plots and static EDA
     - ``buildml[viz]``, ``[reports]``, ``[eda]``
     - matplotlib / seaborn; Sweetviz / profiling
   * - Local EDA app
     - ``buildml[dashboard]``
     - Industry EDA App on localhost
   * - Polars / DuckDB
     - ``buildml[engines]``
     - engine adapters for ingest and prep
   * - Search / AutoML
     - ``buildml[optuna]``, ``[automl]``
     - Optuna; native AutoML
   * - Imbalance resample
     - ``buildml[imbalanced]``
     - imbalanced-learn
   * - Excel input
     - ``buildml[excel]``
     - spreadsheet ingest
   * - Time-series analysis depth
     - ``buildml[timeseries]``, ``[timeseries-prophet]``, ``[timeseries-ml]``
     - statsmodels / Prophet / neuralforecast when wheels resolve
   * - Torch / speech / vision
     - ``buildml[torch]``, ``[speech]``, ``[vision]``, ``[pretrained]``
     - tabular + multimodal DL; ASR; backbones
   * - RAG
     - ``buildml[rag]``, ``[rag-advanced]``
     - dense / rerank backends; LangChain hooks
   * - AI operator
     - ``buildml[ai]``
     - LLM operator (bring your own key)
   * - Serve / ONNX
     - ``buildml[serve]``, ``[onnx]``
     - local FastAPI serve; ONNX checker
   * - Graph / RL / TDA
     - ``buildml[graph]``, ``[graph-pyg]``, ``[rl]``, ``[tda]``
     - NetworkX / PyG; Gymnasium; ripser / persim
   * - NLP encoders
     - ``buildml[nlp]``, ``[nlp-industry]``
     - sentence-transformers / langdetect / NLTK; spaCy NER
   * - SHAP
     - ``buildml[shap]``
     - ``explain_shap``
   * - Classical bundle
     - ``buildml[all-classical]``
     - engines + imbalanced + eda + excel + dashboard + optuna + automl
   * - Industry meta
     - ``buildml[production]``
     - best-effort R1–R6 industry extras (see below)

Methods name the missing extra when an optional dependency is absent
(for example ``pip install 'buildml[optuna]'``).

If an extra fails to install
----------------------------

``buildml[production]`` is best-effort. It is not a guarantee that every
nested industry wheel installs on every platform.

On **Python 3.13**, especially Windows, some nested pins are skipped
with environment markers when upstream wheels are missing or broken
(LightFM, learn2learn / qpth, giotto-tda, neuralforecast, skope-rules,
and similar). Core sklearn paths still install. Check the domain
capability matrix (``session.automl.capability_matrix()``, and the same
pattern on other facades).

Prefer **Python 3.11 or 3.12** for Torch and heavy industry extras.
Always use a project virtual environment. The staged-install guide is
``guides/safe-install-and-runtime.md``.

From a source checkout:

.. code-block:: console

   python scripts/probe_industry_extras.py
   python scripts/verify_runtime_stability.py

``probe_industry_extras.py`` reports what imports. It never hard-fails.
It does not include dashboard, serve, or AI operator extras.
``verify_runtime_stability.py`` runs subprocess probes (``ok`` /
``fail`` / ``crash`` / ``skip``) because some native stacks can
hard-crash a process even after ``pip install`` succeeded.

Engines
-------

Path ingest with ``engine="polars"`` or ``engine="duckdb"`` loads
through the engine without a Pandas-first pass. With Polars and
``mode="lazy"``, ``Dataset.native`` may be a LazyFrame that collects on
``to_pandas()`` / sklearn materialization. That is not out-of-core
sklearn training. DuckDB Arrow / IPC paths use PyArrow when they can.
Checkpoint load rebuilds an eager native handle from the Parquet payload
when engine metadata allows.

Loading saved artifacts
-----------------------

Checkpoint, pipeline, and domain bundle loaders that deserialize pickle,
joblib, or torch default to ``trusted=False`` and raise until you pass
``trusted=True`` for a file you created or fully trust. A SHA-256 in the
manifest can detect tampering after save. It does not make a malicious
author safe. Prefer JSON sidecars, parquet, or
``Session.checkpoint_load(..., data_only=True)`` when provenance is
unclear.

See :doc:`artifacts-checkpoints-bundles` and
:doc:`ai-operator-safety`.

Source checkout
---------------

.. code-block:: console

   git clone https://github.com/TechLeo-Libraries/BuildML.git
   cd BuildML
   pip install -e ".[dev]"

The development extra includes pytest, Ruff, mypy, build, and coverage
tools.

Legacy 1.x
----------

.. code-block:: console

   pip install "buildml==1.0.9"

That pins the old MIT line. Prefer unpinned ``pip install buildml`` for
Session 2.x. See :doc:`legacy`.
