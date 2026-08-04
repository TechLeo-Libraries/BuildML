Federated learning deep guide
=============================

Practical Session-facing federated learning simulation for research, teaching,
and workflows. This is **not** a distributed FL network stack (Flower/OpenFL)
and does **not** implement cryptographic secure aggregation.

Mental model
------------

1. A **client** is the set of rows sharing a client/group id.
2. Each round samples clients (``client_fraction``), clones global
   ``coef_`` / ``intercept_``, and runs ``local_epochs`` on that client's
   **train** rows only.
3. The server aggregates with sample-size weights (FedAvg). Optional FedProx
   proximal pull (``method='fedprox'``, ``mu > 0``) after each local epoch.
4. Validation/test partitions are evaluation-only.

Algorithms
----------

``fedavg``
  Weighted-by-n coefficient averaging for sklearn linear / SGD families.

``fedprox``
  FedAvg plus proximal pull toward the round's global weights
  (``coef ← coef − mu·(coef − global)``).

Supported estimators: ``sgd_classifier``, ``logistic_regression``,
``sgd_regressor``, ``ridge``, ``linear_regression``.

Session API
-----------

* ``session.federated.fit``: train-only federated rounds
* ``session.federated.evaluate``: global and optional per-client holdout metrics
* ``session.federated.predict``: global predictions (no update)
* ``session.federated.save_bundle`` / ``session.federated.load_bundle`` :
  ``buildml.federated_bundle.v1``

Properties: ``session.federated.plan``, ``session.federated.fit_result``,
``session.federated.eval_result``, ``session.federated.predict_result``.

Leakage and privacy
-------------------

* Local updates use the train partition only.
* Clients see only their own train rows during local updates.
* Aggregation is in-process; the orchestrator sees client coefficient updates.
* Do not claim differential privacy or cryptographic secure aggregation.

Bundle boundary
---------------

``buildml.federated_bundle.v1`` stores ``FederatedPlan``. Session checkpoints
do not embed the federated model.

Explicit non-goals
------------------

* No Flower / OpenFL / gRPC client runtime.
* No cryptographic secure aggregation.
* No FedOpt / SCAFFOLD / neural FedAvg zoo (unless later implemented for real).
* No causal APIs in this module (see the separate assumption-declared causal path).

Next after Bayesian / probabilistic: **Causal ML**.
