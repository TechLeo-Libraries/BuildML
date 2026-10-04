BuildML
=======

BuildML is a Python machine-learning library. One object holds your
data, the train / validation / test split, preprocessing, the model, and
the history of what you ran. That object is :class:`buildml.Session`.
The core path is classification and regression. The same object also
runs forecasting, AutoML, fairness, recommenders, RAG, graphs, NLP,
Torch, and other documented domains. Train-fitted preprocessing and model
fitting require a split.

Cross-validation can fit preprocessing separately within each fold. Teaching
methods and reports explain the operations recorded in the Session.

Install with ``pip install buildml`` (Python 3.10 through 3.13). This repo
contains BuildML 2.6.4. The install command selects the latest published
stable release from PyPI.

Start here
==========

#. **Install and check the import**: :doc:`installation`
#. **Run a first Session**: :doc:`usage`
#. **Understand the main concepts**: :doc:`concepts`
#. **Plan a modeling workflow**: :doc:`workflow-guide`

After that, open the :doc:`guide-index` when you need a domain tutorial, or
:doc:`features` when you need a map of what is shipped. The Markdown files
under ``guides/`` are the source for those tutorials. Sphinx includes them
so this site and GitHub stay on the same text.

Legacy 1.x remains available under ``pip install "buildml==1.0.9"`` and as
``buildml/_legacy/`` for reference. New work uses Session. See :doc:`legacy`.

.. toctree::
   :maxdepth: 1
   :caption: Get started

   readme
   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Ideas

   concepts
   workflow-guide

.. toctree::
   :maxdepth: 1
   :caption: Guides and reference

   guides
   features
   modules

.. toctree::
   :maxdepth: 1
   :caption: Project

   stability
   legacy
   authors
   history
   sponsor

.. toctree::
   :hidden:

   pypi-2x-publish
   session-facade-migration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
