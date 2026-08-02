"""
Handles simple date complexities in your data like extracting date features and turning categorical data to datetime.
"""

from ._date import categorical_to_datetime, extract_date_features

__author__ = "Leonard Onyiriuba"
__email__ = "leonard.c.onyiriuba@gmail.com"
__copyright__ = "Copyright (c) 2023-2026 Leonard Onyiriuba"
__license__ = "Apache-2.0"


__all__ = [
    "categorical_to_datetime",
    "extract_date_features",
    ]
