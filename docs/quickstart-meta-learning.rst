Meta-learning quickstart
========================

.. note::

   Install with ``pip install buildml`` (Session 2.5.x).
   Install 2.x from GitHub (or an editable checkout). Meta-learning uses core
   sklearn façades: no optional extra. See :doc:`installation`.

Practical tabular few-shot / episodic meta-learning: assign a
``role="group"`` task column (or pass ``task_column=``),
``session.metalearning.fit`` on train tasks only, then ``session.metalearning.adapt`` /
``session.metalearning.evaluate`` on holdout episodes, and save a distinct bundle.
Honesty: **not** foundation-model meta-learning or MAML-at-scale.

**Go deeper:** :doc:`meta-learning-deep`.

.. code-block:: bash

   pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"

.. code-block:: python

   import numpy as np
   import pandas as pd

   from buildml import Session

   rng = np.random.default_rng(0)
   rows = []
   for task in range(8):
       shift = rng.normal(0, 1.0, size=2)
       for i in range(40):
           label = i % 2
           center = shift + (1.2 if label else -1.2)
           x = rng.normal(center, 0.45, size=2)
           rows.append(
               {
                   "x": float(x[0]),
                   "y": float(x[1]),
                   "label": label,
                   "task_id": f"t{task}",
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
               "task_id": "group",
           }
       )
       .split(test_size=0.2, validation_size=0.2, random_state=0)
       .scale(method="standard")
   )

   fit = session.metalearning.fit(
       method="prototypical",
       k_shot=3,
       n_query=6,
       n_episodes=20,
       task_holdout_fraction=0.25,
   )
   print(fit.n_meta_train_tasks, fit.meta_train_accuracy)

   adapt = session.metalearning.adapt(
       task_id=session.metalearning.plan.train_task_ids[0],
       partition="train",
       max_support_per_class=3,
   )
   print(adapt.n_support, adapt.n_classes_adapted)

   ev = session.metalearning.evaluate(partition="train", k_shot=3)
   print(ev.metrics)

   session.metalearning.save_bundle("artifacts/metalearning_bundle")

Honest boundaries
-----------------

* **In scope:** episodic few-shot via task/group column; ``prototypical``
  nearest-centroid; ``warm_start`` pooled sklearn init + support adapt;
  train-only meta-train; ``buildml.metalearning_bundle.v1``.
* **Out of scope:** foundation-model meta-learning; learned ProtoNet
  embeddings; full MAML/Reptile; meta-training on holdout.

Related next: federated learning
(:doc:`quickstart-federated`).
