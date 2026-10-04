Overview
========

BuildML is a Python machine-learning library. One object holds your
data, the train / validation / test split, preprocessing, the model, and
the history of what you ran. That object is :class:`buildml.Session`.

You say which columns are features, which one is the target, and how to
split. Preprocessing steps that estimate parameters and model fitting use
training rows. The resulting plans are then applied to validation and test
rows. These fitting operations require a split.

A Session also records the run's state: roles, membership, fitted
plans, the optional estimator, and recorded Session operations. That is
why you can ask it what a step means (``session.explain``), learn the
idea behind a word (``session.learn``), see what is blocked
(``session.workflow``), or write a local HTML walkthrough
(``session.walkthrough``). These methods describe recorded state and prerequisites; selecting a split suitable
for the prediction task requires knowledge of the data.

The same Session hosts forecasting, NLP, graph and knowledge graphs, RAG,
Torch, and other domains. Some paths work with the core installation;
others require extras. See :doc:`installation` for each backend's requirements.
Classical ``fit`` / ``evaluate`` stay first-class. Domain work uses
``session.<domain>.*``.

Scope and limitations
---------------------

BuildML requires a split before fit-capable preprocessing and fits those
plans on training rows only. Those checks do not prove that a random
split matches your domain, detect target proxies, or validate memberships
you injected from outside.

Pandas is the sklearn-facing materialization path. Polars and DuckDB
help with ingest and engine-aware prep. They do not make every Session
operation lazy or out-of-core.

This documentation covers version 2.6.4 of the Session 2.x line. Published
releases are available on `PyPI <https://pypi.org/project/buildml/>`_. See :doc:`stability` and
:doc:`session-facade-migration` for the public API policy.

Start with :doc:`usage`. The ideas sit in :doc:`concepts`. Tutorials
live in :doc:`guides`.

Author
------

**Leonard Onyiriuba**

* Email: leonard.c.onyiriuba@gmail.com
* LinkedIn: `Leonard Onyiriuba
  <https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/>`_

BuildML is distributed under the Apache License, Version 2.0.
