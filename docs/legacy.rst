BuildML 1.x archive
===================

BuildML 1.x used a monolithic facade and module names that are not public in
2.x. Its source remains under ``buildml/_legacy/`` as implementation history
and is excluded from package discovery.

There is no compatibility shim. Existing 1.x projects must either pin their
working 1.x environment or migrate explicitly to :class:`buildml.Session`.
Historical changelog and architecture files describe the old release and are
not usage instructions for BuildML 2.x.

Migration shape
---------------

Replace one object that performed implicit preparation and model selection
with explicit Session stages:

#. ingest and inspect the ingest report;
#. assign semantic roles;
#. create or inject partitions;
#. apply train-fitted preparation;
#. fit an estimator;
#. evaluate on a partition whose purpose is declared;
#. persist workflow and model artifacts separately.

Old method names are intentionally omitted here so that search results for
current documentation lead to supported APIs.
