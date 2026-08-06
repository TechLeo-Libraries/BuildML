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
state, leading with a plain-language primer unless a higher reading level is
requested. ``Session.learn`` teaches the concept, operation, or term behind the
call and says what to read first. ``Session.workflow`` resolves prerequisites for
every cataloged operation. ``Session.walkthrough`` combines that state with
history and can write a local, self-contained HTML report.

BuildML requires a split before fit-capable preprocessing and fits those plans
on training rows only. Those checks do not establish that a random split matches
your domain, detect target proxies, or validate externally supplied
memberships.

Stability
---------

Version ``2.5.0`` is the current stable Session 2.x line on PyPI
(https://pypi.org/project/buildml/2.5.0/) and GitHub Release ``v2.5.0``.
See :doc:`stability` for the public-surface freeze policy. Pandas remains the
canonical sklearn-facing materialization path. Polars and DuckDB support
conversion and engine-aware paths; they do not make every Session operation
lazy or out-of-core.

Optional Torch, RAG, and AI operator paths install as extras and attach to the
same Session. Follow the :doc:`index` learning path, the :doc:`guides`
quickstarts, and the Markdown tutorials in ``guides/`` for runnable depth.

Author
------

**Leonard Onyiriuba**

* Email: leonard.c.onyiriuba@gmail.com
* LinkedIn: `Leonard Onyiriuba
  <https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/>`_

BuildML is distributed under the Apache License, Version 2.0.
