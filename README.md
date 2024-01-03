# MLWizard

MLWizard is a Python machine learning library designed to simplify the process of data preparation, feature engineering, model building, and evaluation. It provides a collection of tools for both classification and regression tasks, as well as functionalities for data exploration and manipulation. MLWizard is a distribution of the TechLeo community with the aim of making the complex, easy.

## Features

### Data Loading and Handling
- `get_dataset`: Load a dataset.
- `get_training_test_data`: Split the dataset into training and test sets.
- `load_large_dataset`: Load a large dataset efficiently.
- `reduce_data_memory_useage`: Reduce memory usage of the dataset.

### Data Cleaning and Manipulation
- `drop_columns`: Drop specified columns from the dataset.
- `fix_missing_values`: Handle missing values in the dataset.
- `fix_unbalanced_dataset`: Address class imbalance in a classification dataset.
- `filter_data`: Filter data based on specified conditions.
- `remove_duplicates`: Remove duplicate rows from the dataset.
- `rename_columns`: Rename columns in the dataset.
- `replace_values`: Replace specified values in the dataset.
- `reset_index`: Reset the index of the dataset.
- `set_index`: Set a specific column as the index.
- `sort_index`: Sort the index of the dataset.
- `sort_values`: Sort the values of the dataset.

### Data Formatting and Transformation
- `categorical_to_datetime`: Convert categorical columns to datetime format.
- `categorical_to_numerical`: Convert categorical columns to numerical format.
- `numerical_to_categorical`: Convert numerical columns to categorical format.
- `column_binning`: Bin values in a column into specified bins.

### Exploratory Data Analysis
- `eda`: Perform exploratory data analysis on the dataset.
- `eda_visual`: Visualize exploratory data analysis results.
- `pandas_profiling`: Generate a Pandas Profiling report for the dataset.
- `sweetviz_profile_report`: Generate a Sweetviz Profile Report for the dataset.
- `count_column_categories`: Count the categories in a categorical column.

### Feature Engineering
- `extract_date_features`: Extract date-related features from a datetime column.
- `select_features`: Select relevant features for modeling.
- `select_dependent_and_independent`: Select dependent and independent variables.

### Data Preprocessing
- `scale_independent_variables`: Scale independent variables in the dataset.
- `remove_outlier`: Remove outliers from the dataset.
- `split_data`: Split the dataset into training and test sets.

### Model Building and Evaluation
- `get_bestK_KNNregressor`: Find the best K value for KNN regression.
- `train_model_regressor`: Train a regression model.
- `regressor_predict`: Make predictions using a regression model.
- `regressor_evaluation`: Evaluate the performance of a regression model.
- `regressor_model_testing`: Test a regression model.
- `build_multiple_regressors`: Build multiple regression models.
- `build_multiple_regressors_from_features`: Build regression models using selected features.
- `build_single_regressor_from_features`: Build a single regression model using selected features.
- `get_bestK_KNNclassifier`: Find the best K value for KNN classification.
- `train_model_classifier`: Train a classification model.
- `classifier_predict`: Make predictions using a classification model.
- `classifier_evaluation`: Evaluate the performance of a classification model.
- `classifier_model_testing`: Test a classification model.
- `classifier_graph`: Visualize the KNN classification graph.
- `build_multiple_classifiers`: Build multiple classification models.
- `build_multiple_classifiers_from_features`: Build classification models using selected features.
- `build_single_classifier_from_features`: Build a single classification model using selected features.

### Data Aggregation and Summarization
- `group_data`: Group and summarize data based on specified conditions.

### Data Type Handling
- `select_datatype`: Select columns of a specific datatype in the dataset.

## Installation

You can install MLWizard using pip:

```bash
pip install mlwizard
```


## Useage
from mlwizard import SupervisedLearning

# Example usage
```bash
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.snm import SVC


dataset = ...  # Load your dataset(e.g Pandas DataFrame)
data = SupervisedLearning(dataset)

eda = data.eda()

classifiers = ["LogisticRegression(random_state = 0)", "RandomForestClassifier(random_state = 0)", "DecisionTreeClassifier(random_state = 0)", "SVC()"]
build_model = data.build_multiple_classifiers()
```

## Acknowledgments
MLWizard relies on several open-source libraries to provide its functionality. We would like to express our gratitude to the developers and contributors of the following libraries:

NumPy
Pandas
Matplotlib
Seaborn
yData Profiling
Sweetviz
Imbalanced-Learn (imblearn)
Scikit-learn
Warnings
Datatable
The MLWizard library builds upon the functionality provided by these excellent tools, and we appreciate the effort and dedication of their respective communities.

## License
MLWizard is distributed under the MIT License. Feel free to use, modify, and distribute it according to the terms of the license.
