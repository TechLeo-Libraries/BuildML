"""
BuildML 2.x
===========

A flexible, depth-first toolkit for building machine-learning workflows.

Phase-1 public surface is the :class:`~buildml.session.Session` API. Legacy 1.x
modules are parked under ``buildml._legacy`` and are not part of the supported
import graph.
"""

from buildml._version import __version__
from buildml.session import Session

__author__ = "Leonard Onyiriuba"
__email__ = "leonard.c.onyiriuba@gmail.com"
__copyright__ = "Copyright (c) 2023-2026 Leonard Onyiriuba"
__license__ = "MIT"

__all__ = [
    "Session",
    "__version__",
]
