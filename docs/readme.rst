Overview
========

BuildML is a stateful Python library for tabular classification and regression.
The public entry point is :class:`buildml.Session`.

A Session records:

* the canonical dataset and semantic column roles;
* train, optional validation, and test membership;
* fitted preprocessing plans and the active estimator;
* operation parameters and state transitions.

``Session.explain`` connects an operation's static catalog entry to live Session
state. ``Session.workflow`` resolves prerequisites for every cataloged operation.
``Session.walkthrough`` combines that state with history and can write a local,
self-contained HTML report.

BuildML requires a split before fit-capable preprocessing and fits those plans
on training rows only. Those checks do not establish that a random split matches
your domain, detect target proxies, or validate externally supplied
memberships.

Alpha status
------------

Version 2.3 is under active development. Checkpoint layouts, report schemas,
and method signatures may change. Pandas remains the canonical sklearn-facing
materialization path. Polars and DuckDB support conversion and engine-aware
paths; they do not make every Session operation lazy or out-of-core.

Optional Torch, RAG, and AI operator paths install as extras and attach to the
same Session. Follow the :doc:`index` learning path, the :doc:`guides`
quickstarts, and the Markdown tutorials in ``guides/`` for runnable depth.

Author
------

**Leonard Onyiriuba**

* Email: leonard.c.onyiriuba@gmail.com
* LinkedIn: `Leonard Onyiriuba
  <https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/>`_

BuildML is distributed under the MIT License.
