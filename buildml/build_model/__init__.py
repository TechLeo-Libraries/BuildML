"""
Create multiple regressor and classifiers based on your approach to the problem. This module provides functions for model splitting, training, prediction, evaluation, etc.
"""

from ._model import (
    select_features,
    split_data,
    build_regressor_model,
    classifier_model_testing,
    regressor_model_testing,
    build_classifier_model,
    build_multiple_regressors,
    build_multiple_classifiers,
    build_single_regressor_from_features,
    build_single_classifier_from_features,
    build_multiple_regressors_from_features,
    build_multiple_classifiers_from_features,
    classifier_graph,
    FindK_KNN_Classifier,
    FindK_KNN_Regressor,
    simple_linregres_graph,
    )

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"


__all__ = [
    "select_features",
    "split_data",
    "build_regressor_model",
    "classifier_model_testing",
    "regressor_model_testing",
    "build_classifier_model",
    "build_multiple_regressors",
    "build_multiple_classifiers",
    "build_single_regressor_from_features",
    "build_single_classifier_from_features",
    "build_multiple_regressors_from_features",
    "build_multiple_classifiers_from_features",
    "classifier_graph",
    "FindK_KNN_Classifier",
    "FindK_KNN_Regressor",
    "simple_linregres_graph",
    ]
