Installation
============

BuildML supports Python 3.10 through 3.13.

.. important::

   **Install honesty (2.x):** PyPI ``buildml`` currently publishes the legacy
   **1.x** line (``1.0.9``, MIT). It does **not** install Session 2.x.
   Until a 2.x wheel is published, install from GitHub (or a source checkout).

Install BuildML 2.x (GitHub)
----------------------------

.. code-block:: console

   pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"

Optional extras append the same way as for a local checkout, for example
``pip install "buildml[torch] @ git+https://github.com/TechLeo-Libraries/BuildML.git"``
or install extras after an editable checkout (below).

Core installation from PyPI (legacy 1.x only)
---------------------------------------------

.. code-block:: console

   pip install buildml

That command installs **BuildML 1.x** from PyPI. Prefer the GitHub install above
for Session 2.x.

Optional dependencies (2.x)
---------------------------

Install only the capabilities the workflow uses (after a GitHub / editable
install of 2.x):

.. code-block:: console

   pip install "buildml[viz]"          # matplotlib and seaborn
   pip install "buildml[reports]"      # Sweetviz and ydata-profiling
   pip install "buildml[eda]"          # viz + reports
   pip install "buildml[dashboard]"    # local interactive EDA dashboard
   pip install "buildml[engines]"      # Polars and DuckDB
   pip install "buildml[optuna]"       # Optuna hyperparameter search
   pip install "buildml[automl]"       # native AutoML (+ Optuna method)
   pip install "buildml[imbalanced]"   # imbalanced-learn
   pip install "buildml[excel]"        # Excel input
   pip install "buildml[timeseries]"   # statsmodels / ruptures TS analysis depth
   pip install "buildml[timeseries-prophet]"
   pip install "buildml[timeseries-ml]"  # neuralforecast when wheels resolve
   pip install "buildml[graph]"        # NetworkX graph ML
   pip install "buildml[graph-pyg]"    # PyG GCN/SAGE/GAT (needs torch)
   pip install "buildml[nlp]"          # text encoders / langdetect / NLTK helpers
   pip install "buildml[torch]"        # Torch DL path (alias: buildml[dl])
   pip install "buildml[speech]"       # ASR + speech finetune-lite (+ transformers)
   pip install "buildml[vision]"       # torchvision pretrained vision hooks
   pip install "buildml[pretrained]"   # vision + speech pretrained extras
   pip install "buildml[serve]"        # managed local FastAPI model serving
   pip install "buildml[onnx]"         # optional ONNX checker for export_torch
   pip install "buildml[rag]"          # optional dense/rerank backends
   pip install "buildml[rl]"           # optional Gymnasium tabular Q-learning + REINFORCE-lite
   pip install "buildml[tda]"          # ripser + persim persistent homology
   pip install "buildml[ai]"           # LLM operator (alias: buildml[llm])
   pip install "buildml[shap]"         # optional SHAP attribution (explain_shap)
   pip install "buildml[all-classical]"
   pip install "buildml[production]"   # best-effort R1-R6 industry meta-extra

The core install includes NumPy, Pandas, PyArrow, and scikit-learn. Plotting
methods, ``session.eda_app()``, engine adapters, ``session.optuna_search()``,
and ``nested_cv_score(..., inner_search="optuna")`` name the missing extra when
an optional dependency is unavailable (for example
``pip install 'buildml[optuna]'``). ``buildml[production]`` is best-effort: some
nested industry wheels are skipped by environment markers on Python 3.13 /
Windows. Check each domain capability matrix and
``python scripts/probe_industry_extras.py`` for what actually imports.

Path ingest with ``engine="polars"`` or ``engine="duckdb"`` loads through the
engine without a Pandas-first pass. With Polars and ``mode="lazy"``,
``Dataset.native`` may be a LazyFrame that collects on ``to_pandas()`` /
sklearn materialization: not out-of-core sklearn training. DuckDB Arrow/IPC
paths use PyArrow rather than a Pandas Feather bridge when feasible. Checkpoint
load rebuilds an eager native handle from the Parquet payload when engine
metadata allows.

Source checkout
---------------

.. code-block:: console

   git clone https://github.com/TechLeo-Libraries/BuildML.git
   cd BuildML
   pip install -e ".[dev]"

The development extra includes pytest, Ruff, mypy, build, and coverage tools.
