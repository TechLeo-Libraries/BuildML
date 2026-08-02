Meta-learning quickstart
========================

.. note::

   PyPI ``buildml`` is still legacy 1.x and does **not** install Session 2.x.
   Install 2.x from GitHub (or an editable checkout). Meta-learning uses core
   sklearn façades — no optional extra. See :doc:`installation`.

Practical tabular few-shot / episodic meta-learning: assign a
``role="group"`` task column (or pass ``task_column=``),
``fit_metalearning`` on train tasks only, then ``adapt_to_task`` /
``evaluate_metalearning`` on holdout episodes, and save a distinct bundle.
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

   fit = session.fit_metalearning(
       method="prototypical",
       k_shot=3,
       n_query=6,
       n_episodes=20,
       task_holdout_fraction=0.25,
   )
   print(fit.n_meta_train_tasks, fit.meta_train_accuracy)

   adapt = session.adapt_to_task(
       task_id=session.metalearning_plan.train_task_ids[0],
       partition="train",
       max_support_per_class=3,
   )
   print(adapt.n_support, adapt.n_classes_adapted)

   ev = session.evaluate_metalearning(partition="train", k_shot=3)
   print(ev.metrics)

   session.save_metalearning_bundle("artifacts/metalearning_bundle")

Honest boundaries
-----------------

* **In scope:** episodic few-shot via task/group column; ``prototypical``
  nearest-centroid; ``warm_start`` pooled sklearn init + support adapt;
  train-only meta-train; ``buildml.metalearning_bundle.v1``.
* **Out of scope:** foundation-model meta-learning; learned ProtoNet
  embeddings; full MAML/Reptile; meta-training on holdout.

Next Phase 2 item after meta-learning (now shipped): **federated learning**
(:doc:`quickstart-federated`).
