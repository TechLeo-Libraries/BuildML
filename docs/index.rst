BuildML 2.1 alpha
=================

BuildML organizes a machine-learning workflow around
:class:`buildml.Session`. A Session owns data, column roles, partitions,
train-fitted preparation plans, an optional classical estimator or Torch
trainer, and operation history.

The 2.1 line is alpha software (DL alpha ``2.1.0a1`` on the classical
``2.0.0a1`` spine). Public methods and serialized formats may change before a
stable release. The 1.x API is archival and is not imported from the package
root.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   readme
   installation
   usage
   concepts
   workflow-guide
   features
   legacy
   modules
   authors
   history
   sponsor

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
