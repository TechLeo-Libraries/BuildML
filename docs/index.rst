BuildML
=======

BuildML keeps a machine-learning job in one :class:`buildml.Session`: the
table, what each column is for, the split, the preparation that learned from
train only, and the model. If you try to prepare or fit before a split, it
stops you.

Install with ``pip install buildml`` (Python 3.10 through 3.13). That is
BuildML 2.5.0, the current stable Session line.

Start here
==========

#. **Install and check the import**: :doc:`installation`
#. **Run a first Session**: :doc:`usage`
#. **Learn the few ideas that matter**: :doc:`concepts`
#. **Use the order as a decision path**: :doc:`workflow-guide`

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
