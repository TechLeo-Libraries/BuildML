Overview
========

BuildML is a Python machine-learning library. One object holds your
data, the train / validation / test split, preprocessing, the model, and
the history of what you ran. That object is :class:`buildml.Session`.
A Session is a unified, stateful ML lifecycle. Enforced leakage
safeguards, fold-local preprocessing, contextual teaching, workflow
guidance, checkpointing, and auditable export live on that same object.

You say which columns are features, which one is the target, and how to
split. After that, preparation and fitting learn from the training rows
only. Validation and test get the frozen version. Skip the split and the
call fails instead of leaking statistics into the holdout.

A Session also keeps the story of the run: roles, membership, fitted
plans, the optional estimator, and every operation you called. That is
why you can ask it what a step means (``session.explain``), learn the
idea behind a word (``session.learn``), see what is blocked
(``session.workflow``), or write a local HTML walkthrough
(``session.walkthrough``). Those surfaces teach the contract. They do
not inspect your data or certify that a split matches the real world.

The same Session hosts optional domains when you install the extra:
forecasting, NLP, graph and knowledge graphs, RAG, Torch, and others.
Classical ``fit`` / ``evaluate`` stay first-class. Domain work uses
``session.<domain>.*``.

What it will not pretend
------------------------

BuildML requires a split before fit-capable preprocessing and fits those
plans on training rows only. Those checks do not prove that a random
split matches your domain, detect target proxies, or validate memberships
you injected from outside.

Pandas is the sklearn-facing materialization path. Polars and DuckDB
help with ingest and engine-aware prep. They do not make every Session
operation lazy or out-of-core.

Version 2.6.1 is the current Session 2.x line. PyPI serves
`2.6.1 <https://pypi.org/project/buildml/2.6.1/>`_. See :doc:`stability` and
:doc:`pypi-2x-publish` for the public-surface policy.

Start with :doc:`usage`. The ideas sit in :doc:`concepts`. Tutorials
live in :doc:`guides`.

Author
------

**Leonard Onyiriuba**

* Email: leonard.c.onyiriuba@gmail.com
* LinkedIn: `Leonard Onyiriuba
  <https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/>`_

BuildML is distributed under the Apache License, Version 2.0.
