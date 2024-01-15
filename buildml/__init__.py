# -*- coding: utf-8 -*-

from .automate import SupervisedLearning
from .model import select_features, split_data, build_regressor_model, classifier_model_testing, regressor_model_testing, build_classifier_model, build_multiple_regressors, build_multiple_classifiers, build_single_regressor_from_features, build_single_classifier_from_features, build_multiple_regressors_from_features, build_multiple_classifiers_from_features, classifier_graph, FindK_KNN_Classifier, FindK_KNN_Regressor, simple_linregres_graph
from .date import categorical_to_datetime, extract_date_features
from .eda import eda, eda_visual, sweet_viz, pandas_profiling
from .output_dataset import output_dataset_as_csv, output_dataset_as_excel
from .preprocessing import column_binning, categorical_to_numerical, count_column_categories, drop_columns, filter_data, fix_missing_values, fix_unbalanced_dataset, group_data, load_large_dataset, numerical_to_categorical, remove_duplicates, remove_outlier, rename_columns, replace_values, reset_index, scale_independent_variables, select_datatype, set_index, sort_index, sort_values, replace_values

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"
__version__ = "1.0.3"
