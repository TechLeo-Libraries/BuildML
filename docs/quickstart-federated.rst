Federated learning quickstart
=============================

.. note::

   PyPI ``buildml`` is still legacy 1.x and does **not** install Session 2.x.
   Install 2.x from GitHub (or an editable checkout). Federated learning uses
   core sklearn façades: no optional extra. See :doc:`installation`.

Local FedAvg-style simulation on Session data partitioned by a client/group
column: ``fit_federated`` runs train-only local updates, aggregates
``coef_`` / ``intercept_``, then ``evaluate_federated`` /
``predict_federated`` on holdout. Persist via ``buildml.federated_bundle.v1``.
Honesty: **not** a distributed FL platform (Flower/OpenFL); **not**
cryptographic secure aggregation.

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

   fit = session.fit_federated(
       method="fedavg",
       estimator="sgd_classifier",
       n_rounds=5,
       local_epochs=2,
   )
   ev = session.evaluate_federated(partition="validation", per_client=True)
   session.predict_federated(partition="test")
   session.save_federated_bundle("artifacts/federated_bundle")

What you get
------------

* Client partitioning via ``role='group'`` or ``client_column=``.
* Local train-only updates; holdout evaluation never trains.
* ``buildml.federated_bundle.v1``.

* **Out of scope:** Flower/OpenFL networking; cryptographic secure aggregation;
  non-linear FedAvg zoo.

Next Phase 2 item after Bayesian / probabilistic: **Causal ML**
(assumption-declared path; EDA stays associational).
