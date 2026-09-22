"""
BuildML 2.x
===========

Manage machine-learning data, training, evaluation, and saved artifacts.

The public entry point is the :class:`~buildml.session.Session` API. Legacy 1.x
modules are retained under ``buildml._legacy`` and are not part of the supported
import graph.
"""

from buildml._version import __version__
from buildml.session import Session

__author__ = "Leonard Onyiriuba"
__email__ = "leonard.c.onyiriuba@gmail.com"
__copyright__ = "Copyright (c) 2023-2026 Leonard Onyiriuba"
__license__ = "Apache-2.0"

__all__ = [
    "Session",
    "__version__",
]
