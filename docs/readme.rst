Overview
========

BuildML 2.0 provides a stateful API for classical classification and
regression. The public entry point is :class:`buildml.Session`.

A Session records:

* the canonical dataset and semantic column roles;
* train, optional validation, and test membership;
* fitted preprocessing plans and the active estimator;
* operation parameters, decision origins, and state transitions.

``Session.explain`` connects an operation's static catalog entry to live
Session state. ``Session.workflow`` resolves prerequisites for every cataloged
operation. ``Session.walkthrough`` combines that state with history and can
write a local, self-contained HTML report.

BuildML guards fit-capable built-in operations by requiring an existing split
and fitting only on training rows. Those checks do not establish that a random
split is scientifically valid, detect target proxies, or inspect arbitrary
external arrays.

Alpha status
------------

Version 2.0 is under active development. Checkpoint layouts, report schemas,
and methods may change. Pandas remains the canonical sklearn-facing
materialization path. Polars and DuckDB are optional conversion and
engine-aware paths, not a promise that every operation runs lazily.

Author
------

**Leonard Onyiriuba**

* Email: leonard.c.onyiriuba@gmail.com
* LinkedIn: `Leonard Onyiriuba
  <https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/>`_

BuildML is distributed under the MIT License.
