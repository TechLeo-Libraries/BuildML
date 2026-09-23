API reference
=============

Session
-------

.. currentmodule:: buildml

.. autoclass:: buildml.Session
   :members:
   :undoc-members:

Explanation catalog
-------------------

.. automodule:: buildml.explain
   :members:
   :undoc-members:

Result and type objects
-----------------------

.. automodule:: buildml.core.results
   :members:

.. automodule:: buildml.core.types
   :members:

Core data and validation
------------------------

.. autoclass:: buildml.data.dataset.Dataset
   :no-members:

.. autoclass:: buildml.data.splits.SplitPlan
   :no-members:

.. autoclass:: buildml.checkpoint.validate.ReattachResult
   :no-members:

.. autoexception:: buildml.core.errors.ValidationError
   :no-members:

Explanation serialization
-------------------------

.. currentmodule:: buildml.explain.schemas

.. autoclass:: buildml.explain.schemas.SerializableSchema
   :members: to_dict

.. py:data:: buildml.explain.schemas.JsonValue

   Recursive JSON-compatible value: null, boolean, integer, float, string,
   a list of JSON values, or a dictionary with string keys and JSON values.

Session workflow methods
------------------------

.. currentmodule:: buildml

The following inherited methods are exposed directly on :class:`buildml.Session`.

.. automethod:: buildml.Session.ingest

.. automethod:: buildml.Session.set_roles

.. automethod:: buildml.Session.split

.. automethod:: buildml.Session.group_split

.. automethod:: buildml.Session.time_split

.. automethod:: buildml.Session.inject_split

.. automethod:: buildml.Session.summarize_history

.. automethod:: buildml.Session.walkthrough

.. automethod:: buildml.Session.workflow

.. automethod:: buildml.Session.assert_can_fit

.. automethod:: buildml.Session.close_native

.. automethod:: buildml.Session.checkpoint_load

.. automethod:: buildml.Session.with_engine

.. automethod:: buildml.Session.explain
