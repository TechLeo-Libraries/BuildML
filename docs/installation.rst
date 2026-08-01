Installation
============

BuildML supports Python 3.10 through 3.13.

Core installation
-----------------

.. code-block:: console

   pip install buildml

Optional dependencies
---------------------

Install only the capabilities the workflow uses:

.. code-block:: console

   pip install "buildml[viz]"          # matplotlib and seaborn
   pip install "buildml[reports]"      # Sweetviz and ydata-profiling
   pip install "buildml[eda]"          # viz + reports
   pip install "buildml[dashboard]"    # Teaching Studio (FastAPI/Plotly/ReportLab/kaleido)
   pip install "buildml[engines]"      # Polars and DuckDB
   pip install "buildml[optuna]"       # Optuna hyperparameter search
   pip install "buildml[imbalanced]"   # imbalanced-learn
   pip install "buildml[excel]"        # Excel input
   pip install "buildml[all-classical]"

The core install includes NumPy, Pandas, PyArrow, and scikit-learn. Plotting
methods, ``session.eda_app()``, engine adapters, ``session.optuna_search()``,
and ``nested_cv_score(..., inner_search="optuna")`` name the missing extra when
an optional dependency is unavailable (for example
``pip install 'buildml[optuna]'``).

Path ingest with ``engine="polars"`` or ``engine="duckdb"`` loads through the
engine without a Pandas-first pass. With Polars and ``mode="lazy"``,
``Dataset.native`` may be a LazyFrame that collects on ``to_pandas()`` /
sklearn materialization — not out-of-core sklearn training. DuckDB Arrow/IPC
paths use PyArrow rather than a Pandas Feather bridge when feasible. Checkpoint
load rebuilds an eager native handle from the Parquet payload when engine
metadata allows.

Source checkout
---------------

.. code-block:: console

   git clone https://github.com/TechLeo-Dev/BuildML.git
   cd BuildML
   pip install -e ".[dev]"

The development extra includes pytest, Ruff, mypy, build, and coverage tools.
