BuildML 1.x archive
===================

BuildML 1.x shipped a monolithic facade with module names and call patterns
that differ from the current public API. That code remains under
``buildml/_legacy/`` as implementation history. It is excluded from package
discovery and is not imported from the package root.

There is no automatic compatibility shim. Projects that depend on 1.x can keep
their pinned environment, or migrate explicitly to :class:`buildml.Session`.
Both choices are valid; the archive exists so you can compare behavior when
porting, not to suggest the old release was unusable.

What changed in 2.x
-------------------

The 2.x design makes workflow stages explicit instead of folding them into one
object:

#. ingest and read ``session.ingest_report``;
#. assign semantic roles with ``set_roles``;
#. create or inject partitions before any fit-capable step;
#. apply train-fitted preparation;
#. fit an estimator;
#. evaluate on a partition whose purpose you declare;
#. persist checkpoints, pipeline bundles, and model artifacts separately.

Historical changelog and architecture notes describe 1.x releases. Treat them
as migration context, not as instructions for new projects.

Where legacy code lives
-----------------------

``buildml/_legacy/`` holds the old implementation. It is useful when you need
to trace how a 1.x behavior was implemented or when comparing test fixtures
during migration. Do not import it in new application code.

Old public method names are omitted from current docs so search results point
to supported Session APIs. If you maintain a 1.x codebase, keep a local copy of
the 1.x documentation or tag that matches your pinned version.

Migration checklist
-------------------

When moving a notebook or service to 2.x:

* Replace implicit preparation with explicit ``impute``, ``encode``, ``scale``,
  and related Session calls after ``split``.
* Declare column roles; dtype alone is not enough.
* Reserve ``test`` for fixed choices; use ``validation`` while iterating.
* Use ``explain`` and ``workflow`` to see prerequisites instead of guessing
  call order from stack traces.
* Store data workflow state in checkpoints and fitted artifacts in pipeline or
  model bundles; neither embeds the other.

If migration is not urgent, staying on 1.x with a pinned environment remains
a reasonable option until you need 2.x features.
