# -*- coding: utf-8 -*-

from .automate import SupervisedLearning
from .build_model import select_features, split_data, build_regressor_model, classifier_model_testing, regressor_model_testing, build_classifier_model, build_multiple_regressors, build_multiple_classifiers, build_single_regressor_from_features, build_single_classifier_from_features, build_multiple_regressors_from_features, build_multiple_classifiers_from_features, classifier_graph, FindK_KNN_Classifier, FindK_KNN_Regressor, simple_linregres_graph
from .date import categorical_to_datetime, extract_date_features
from .eda import eda, eda_visual, sweet_viz, pandas_profiling
from .output_dataset import output_dataset_as_csv, output_dataset_as_excel
from .preprocessing import column_binning, categorical_to_numerical, count_column_categories, drop_columns, filter_data, fix_missing_values, fix_unbalanced_dataset, group_data, load_large_dataset, numerical_to_categorical, remove_duplicates, remove_outlier, rename_columns, replace_values, reset_index, scale_independent_variables, select_datatype, set_index, sort_index, sort_values, replace_values


"""
Automated Supervised Learning module designed for end-to-end data handling,
preprocessing, model development, and evaluation in the context of supervised
machine learning.

Parameters
----------
dataset : Union[pd.DataFrame, dt.Frame]
    The input dataset for supervised learning.
user_guide : bool, optional
    If True, provide user guides and warnings. Default is False.
show_warnings : bool, optional
    If True, display warnings. Default is False.

Examples
--------
>>> # Create an instance of the SupervisedLearning class
>>> df = SupervisedLearning(dataset)

Notes
-----
This class encapsulates various functionalities for data handling and model development.
It leverages popular libraries such as pandas, numpy, matplotlib, seaborn, ydata_profiling,
sweetviz, imbalanced-learn, scikit-learn, warnings, feature-engine, and datatable.

The workflow involves steps like loading and handling data, cleaning and manipulation,
formatting and transformation, exploratory data analysis, feature engineering, data preprocessing,
model building and evaluation, data aggregation and summarization, and data type handling.


References
----------
- pandas: https://pandas.pydata.org/
- numpy: https://numpy.org/
- matplotlib: https://matplotlib.org/
- seaborn: https://seaborn.pydata.org/
- ydata_profiling: https://github.com/dexplo/ydata_profiling
- sweetviz: https://github.com/fbdesignpro/sweetviz
- imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn
- scikit-learn: https://scikit-learn.org/
- warnings: https://docs.python.org/3/library/warnings.html
- feature-engine: https://github.com/solegalli/feature_engine
- datatable: https://datatable.readthedocs.io/en/latest/

"""

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"
