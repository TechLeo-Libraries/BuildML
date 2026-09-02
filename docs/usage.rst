A first Session
===============

You have a table. You want a holdout number you can trust. BuildML
enforces a deliberate order: ingest, assign roles, split, prepare on
training rows only, fit, then evaluate on a partition you name. Skip a
step or call ``fit`` before ``split`` and you get a clear error instead
of a leaked score.

This page is a few realistic loops. For a chapter-style walkthrough see
:doc:`quickstart-classical`. For the long classical path (many cases,
failure modes, persistence) see :doc:`classical-end-to-end`. The full
map is :doc:`guide-index`.

Loan approval
-------------

A small binary classification loop with a missing value and two numeric
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

Add a validation partition when you will repeat model, feature,
calibration, or threshold choices:

.. code-block:: python

   session.split(
       test_size=0.2,
       validation_size=0.2,
       stratify=True,
       random_state=42,
   )

Imbalanced fraud detection
--------------------------

When the positive class is rare, read prevalence on the training
partition before you trust accuracy. Resample **train only** after the
split:

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier

   from buildml import Session

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
       .set_roles(
           {"amount": "feature", "velocity": "feature", "is_fraud": "target"}
       )
       .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
   )

   # Requires: pip install "buildml[imbalanced]"
   session.resample(sampler="smote", random_state=0)
   session.fit(RandomForestClassifier(n_estimators=50, random_state=0))

   val = session.evaluate(partition="validation")
   test = session.evaluate(partition="test")
   print("validation f1:", val.metrics.get("f1"))
   print("test f1:", test.metrics.get("f1"))

Resampling changes training prevalence. Validation and test rows are
never altered. Compare against a baseline that does not resample before
you claim an improvement.

House price regression
----------------------

Same spine, different task and metrics:

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

Random ``split`` assumes independent, exchangeable rows. When rows share
an entity or a time order, use ``group_split`` or ``time_split``.
``group_split``'s ``test_size`` counts *groups*, not rows: partitions
will not land on an exact row fraction. ``time_split`` sorts by the
``time`` role and holds out the most recent rows.

.. code-block:: python

   import pandas as pd

   from buildml import Session

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

For temporal data, assign a ``time`` role and call ``time_split``. When
another system already defined memberships, pass positional indices to
``inject_split``.

Why the order exists
--------------------

Imputation, encoding, scaling, resampling, and ``fit`` require a split
and learn from training rows. That guard stops the most common leak:
computing holdout statistics during preparation.

BuildML does not infer valid group or time boundaries, detect target
proxies, or prove that indices you injected match deployment. Roles and
splits are explicit because those judgments belong to the project.

Typical failures:

* ``ValidationError: No split exists``: call ``split``, ``group_split``,
  ``time_split``, or ``inject_split`` before ``impute`` or ``fit``.
* ``LeakageError``: fitting on validation or test, or resampling outside
  train.
* Missing extra: ``optuna_search``, ``resample``, ``eda_app``, and
  engine adapters name the install group when a dependency is absent.

``session.explain("impute", moment="before")`` lists prerequisites,
leakage risks, and alternatives before you mutate state.

Ask the Session
---------------

Every public method has a catalog entry. These APIs tell you what the
library knows. They do not certify that your split or model suits the
domain.

.. code-block:: python

   session.explain("split")
   session.learn("leakage")
   steps = session.workflow()
   preview = session.dry_run(["impute", "scale", "fit"])
   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

``explain`` is about this Session right now. ``learn`` is the idea, in
reading order. ``workflow`` marks operations done, available, blocked, or
skipped from API prerequisites; available is not a recommendation.
``dry_run`` does not append history.

The default reading level is ``beginner``. It assumes no prior
machine-learning vocabulary. The same facts are available at
``intermediate`` and ``advanced`` with less scaffolding.

For the teaching studio, findings, and the local EDA app, see
:doc:`eda-teaching-studio`.

Save and reload
---------------

.. code-block:: python

   session.checkpoint_save("artifacts/checkpoint")
   restored = Session.checkpoint_load("artifacts/checkpoint")

   session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
   loaded = Session.ingest(frame).load_pipeline("artifacts/pipeline")
   loaded.apply_preprocess_plans()

A checkpoint restores data, roles, partitions, history, and optional
preprocess plans. It does not restore a fitted model. Use
``save_pipeline`` for plans plus estimator. Neither artifact embeds the
other. Loaders that deserialize pickle default to ``trusted=False``;
pass ``trusted=True`` only for files you made or fully trust.

Cross-validation should use a ``PreprocessRecipe`` on unpoisoned data.
Session-global prep then CV is refused by default. See
:doc:`leakage-cv-recipes`.

What to read next
-----------------

* :doc:`concepts` for roles, leakage, and partitions
* :doc:`workflow-guide` for the decision path
* :doc:`quickstart-classical` for the full classical tutorial
* :doc:`guide-index` for every other domain

Torch, RAG, and the AI operator attach to the same Session without
replacing classical APIs. Start from the matching quickstart when you
need them.
