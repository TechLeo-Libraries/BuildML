"""
Handles simple date complexities in your data like extracting date features and turning categorical data to datetime.
"""

from ._date import categorical_to_datetime, extract_date_features

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"


__all__ = [
    "categorical_to_datetime",
    "extract_date_features",
    ]
