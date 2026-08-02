BuildML
=======

BuildML is a Python library for tabular machine-learning workflows built
around :class:`buildml.Session`. A Session holds the dataset, column roles,
partition membership, train-fitted preprocessing plans, an optional estimator,
and operation history.

Version ``2.4.0a1`` is alpha software. Public methods and serialized formats may
change before a stable 2.x release. Install 2.x from GitHub until a PyPI 2.x
wheel exists (PyPI ``buildml`` is still legacy 1.x). The 1.x API remains available
under ``buildml/_legacy/`` for reference; new work should use Session.

Learning path
=============

Use this sequence to move from first import to competent use of BuildML as it
exists today. Each step links to pages that go deeper than a single toy example.

#. **Install and orient** — :doc:`installation`, :doc:`readme`
#. **Run one honest classical loop** — :doc:`usage` (loan-style classification)
#. **Learn the vocabulary** — :doc:`concepts`, :doc:`glossary`
#. **Follow the decision framework** — :doc:`workflow-guide`
#. **Work through the classical tutorial** — :doc:`quickstart-classical`,
   then :doc:`classical-end-to-end`
#. **Leakage-safe selection** — :doc:`leakage-cv-recipes`,
   :doc:`classical-diagnostics-search`
#. **Open the teaching machinery** — :doc:`eda-teaching-studio`, :doc:`usage`
   (explain / workflow / walkthrough / dry_run / dashboard sections)
#. **Optional extras on the same Session** — :doc:`torch-deep`,
   :doc:`rag-deep`, :doc:`ai-operator-safety` (quickstarts remain short on-ramps)
#. **Capability inventory and guide map** — :doc:`features`, :doc:`guide-index`
#. **1.x migration** — :doc:`legacy`

The Markdown files under ``guides/`` are the canonical source for quickstarts,
deep guides, and the glossary. Sphinx includes them via MyST so Read the Docs
and GitHub stay aligned.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   readme
   installation
   usage
   concepts
   workflow-guide
   guides
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
