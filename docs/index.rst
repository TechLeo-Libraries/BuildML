BuildML
=======

BuildML is a Python library for machine-learning workflows built around
:class:`buildml.Session`. Classical tabular supervised learning is the core
path; the same Session also hosts forecasting, NLP, graph/KG, RAG, Torch, and
other optional domains. A Session holds the dataset, column roles, partition
membership, train-fitted preprocessing plans, optional fitted artifacts, and
operation history.

Version ``2.4.0`` is the stable Session 2.x line. Install with
``pip install buildml``. Legacy 1.x remains available under a pin
(``buildml==1.0.9``) and as ``buildml/_legacy/`` for reference; new work should
use Session. See :doc:`stability`.

Learning path
=============

Use this sequence to move from first import to competent use of BuildML as it
exists today. Each step links to pages that go deeper than a single toy example.

#. **Install and orient**: :doc:`installation`, :doc:`readme`
#. **Run one honest classical loop**: :doc:`usage` (loan-style classification)
#. **Learn the vocabulary**: :doc:`concepts`, :doc:`glossary`
#. **Follow the decision framework**: :doc:`workflow-guide`
#. **Work through the classical tutorial**: :doc:`quickstart-classical`,
   then :doc:`classical-end-to-end`
#. **Leakage-safe selection**: :doc:`leakage-cv-recipes`,
   :doc:`classical-diagnostics-search`
#. **Open the teaching machinery**: :doc:`eda-teaching-studio`, :doc:`usage`
   (explain / workflow / walkthrough / dry_run / dashboard sections)
#. **Optional extras on the same Session**: :doc:`torch-deep`,
   :doc:`rag-deep`, :doc:`ai-operator-safety` (quickstarts remain short on-ramps)
#. **Capability inventory and guide map**: :doc:`features`, :doc:`guide-index`
#. **Proof suite (Tier A/B/C)**: deep end-to-end projects under
   `proofs/` on GitHub (`python -m proofs._lib.run_all --tier all`);
   see :doc:`features` and the Markdown guide index for domain → proof maps
#. **1.x migration**: :doc:`legacy`

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
