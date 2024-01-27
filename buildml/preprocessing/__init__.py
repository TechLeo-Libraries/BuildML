"""
BuildML's preprocessing module for data cleaning, transformation, handling data types and more.
"""

from ._preprocessing import (
    column_binning,
    categorical_to_numerical,
    count_column_categories,
    drop_columns,
    filter_data,
    fix_missing_values,
    fix_unbalanced_dataset,
    group_data,
    load_large_dataset,
    numerical_to_categorical,
    remove_duplicates,
    remove_outlier,
    rename_columns,
    replace_values,
    reset_index,
    scale_independent_variables,
    select_datatype,
    set_index,
    sort_index,
    sort_values,
    )

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"


__all__ = [
    "column_binning",
    "categorical_to_numerical",
    "count_column_categories",
    "drop_columns",
    "filter_data",
    "fix_missing_values",
    "fix_unbalanced_dataset",
    "group_data",
    "load_large_dataset",
    "numerical_to_categorical",
    "remove_duplicates",
    "remove_outlier",
    "rename_columns",
    "replace_values",
    "reset_index",
    "scale_independent_variables",
    "select_datatype",
    "set_index",
    "sort_index",
    "sort_values",
    "replace_values",
]