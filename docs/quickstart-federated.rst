Federated learning quickstart
=============================

.. note::

   Install with ``pip install buildml`` (Session 2.4.x).
   Install 2.x from GitHub (or an editable checkout). Federated learning uses
   core sklearn façades: no optional extra. See :doc:`installation`.

Local FedAvg-style simulation on Session data partitioned by a client/group
column: ``session.federated.fit`` runs train-only local updates, aggregates
``coef_`` / ``intercept_``, then ``session.federated.evaluate`` /
``session.federated.predict`` on holdout. Persist via ``buildml.federated_bundle.v1``.
Honesty: **not** a distributed FL platform. Optional Flower
(``backend='flower'``) is still an in-process **local simulation** on Session
partitions (not a networked ServerApp). **Not** cryptographic secure
aggregation.

**Go deeper:** :doc:`federated-deep`.

.. code-block:: bash

   pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"

.. code-block:: python

   import numpy as np
   import pandas as pd
   from buildml import Session

   rng = np.random.default_rng(0)
   rows = []
   for client in range(8):
       shift = rng.normal(0, 0.8, size=2)
       for i in range(40):
           label = i % 2
           center = shift + (1.1 if label else -1.1)
           x = rng.normal(center, 0.35, size=2)
           rows.append(
               {
                   "x": float(x[0]),
                   "y": float(x[1]),
                   "label": int(label),
                   "client_id": f"c{client}",
               }
           )
   frame = pd.DataFrame(rows)

   session = (
       Session.ingest(frame)
       .set_roles(
           {
               "x": "feature",
               "y": "feature",
               "label": "target",
               "client_id": "group",
           }
       )
       .split(test_size=0.2, validation_size=0.2, random_state=0)
       .scale(method="standard")
   )

   fit = session.federated.fit(
       method="fedavg",
       estimator="sgd_classifier",
       n_rounds=5,
       local_epochs=2,
   )
   ev = session.federated.evaluate(partition="validation", per_client=True)
   session.federated.predict(partition="test")
   session.federated.save_bundle("artifacts/federated_bundle")

What you get
------------

* Client partitioning via ``role='group'`` or ``client_column=``.
* Local train-only updates; holdout evaluation never trains.
* ``buildml.federated_bundle.v1``.

* Holdout metrics: accuracy / f1_macro / balanced_accuracy (+ roc_auc when binary).
* Bundle roundtrip: ``save_bundle`` → ``load_bundle(..., trusted=True)`` → re-evaluate.

* **Out of scope:** Flower/OpenFL networking; cryptographic secure aggregation;
  non-linear FedAvg zoo. Flower remains disclosed as local-sim when installed.

Related next: Causal ML
(assumption-declared path; EDA stays associational).
