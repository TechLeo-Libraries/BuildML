import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pp
import sweetviz as sv
import imblearn.over_sampling as ios
import imblearn.under_sampling as ius
import sklearn.impute as si
import sklearn.linear_model as slm
import sklearn.preprocessing as sp
import sklearn.model_selection as sms
import sklearn.metrics as sm
import sklearn.neighbors as sn
import warnings
import sklearn.feature_selection as sfs
import datatable as dt


__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"



class SupervisedLearning: 
    """
    Automated Supervised Learning module designed for end-to-end data handling,
    preprocessing, model development, and evaluation in the context of supervised
    machine learning.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataset for supervised learning.
    show_warnings : bool, optional
        If True, display warnings. Default is False.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
    >>> from sklearn.snm import SVC
    >>> from buildml.automate import SupervisedLearning
    >>>
    >>>
    >>> dataset = pd.read_csv("Your_file_path")  # Load your dataset(e.g Pandas DataFrame)
    >>> data = SupervisedLearning(dataset)
    >>>
    >>> # Exploratory Data Analysis
    >>> eda = data.eda()
    >>> 
    >>> # Build and Evaluate Classifier
    >>> classifiers = ["LogisticRegression(random_state = 0)", 
    >>>                "RandomForestClassifier(random_state = 0)", 
    >>>                "DecisionTreeClassifier(random_state = 0)", 
    >>>                "SVC()"]
    >>> build_model = data.build_multiple_classifiers(classifiers)

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
    __all__ = [
        # Data Loading and Handling
        "get_dataset",
        "get_training_test_data",
        "load_large_dataset",
        "reduce_data_memory_useage"
    
        # Data Cleaning and Manipulation
        "drop_columns",
        "fix_missing_values",
        "fix_unbalanced_dataset",
        "filter_data",
        "remove_duplicates",
        "rename_columns",
        "replace_values",
        "reset_index",
        "set_index",
        "sort_index",
        "sort_values",
    
        # Data Formatting and Transformation
        "categorical_to_datetime",
        "categorical_to_numerical",
        "numerical_to_categorical",
        "column_binning",
    
        # Exploratory Data Analysis
        "eda",
        "eda_visual",
        "pandas_profiling",
        "sweetviz_profile_report",
        "count_column_categories",
        "unique_elements_in_columns",
    
        # Feature Engineering
        "polyreg_x",
        "extract_date_features",
        "select_features",
        "select_dependent_and_independent",
    
        # Data Preprocessing
        "scale_independent_variables",
        "remove_outlier",
        "split_data",
    
        # Model Building and Evaluation
        "poly_get_optimal_degree"
        "get_bestK_KNNregressor",
        "train_model_regressor",
        "regressor_predict",
        "regressor_evaluation",
        "regressor_model_testing",
        "polyreg_graph",
        "simple_linregres_graph",
        "build_multiple_regressors",
        "build_multiple_regressors_from_features",
        "build_single_regressor_from_features",
        "get_bestK_KNNclassifier"
        "train_model_classifier",
        "classifier_predict",
        "classifier_evaluation",
        "classifier_model_testing",
        "classifier_graph",
        "build_multiple_classifiers",
        "build_multiple_classifiers_from_features",
        "build_single_classifier_from_features",
    
        # Data Aggregation and Summarization
        "group_data",
    
        # Data Type Handling
        "select_datatype"
    ]


       
    def __init__(self, dataset, show_warnings: bool = False):
        if isinstance(show_warnings, bool):
            if show_warnings == False:
                self.warnings = show_warnings
                warnings.filterwarnings("ignore")
            else:
                self.warnings = warnings
        
        else:
            raise ValueError("Warnings must be boolean of True or False.")
        
        self.__dataset = dataset
        self.__data = dataset
        self.__scaled = False
        self.__fixed_missing = False
        self.__eda = False
        self.__model_training = False
        self.__dropped_column = False
        self.__get_dataset = False
        self.__data_transformation = False
        self.__model_prediction = False
        self.__model_evaluation = False
        self.__model_testing = False
        self.__remove_outlier = False
        self.__eda_visual = False
        self.__split_data = False
        self.__dependent_independent = False
        
        
        self.classification_problem = False
        self.regression_problem = False
        
    def drop_columns(self, columns: list or str):
        """
        Drop specified columns from the dataset.
    
        Parameters
        ----------
        columns : str or list of str
            A single column name (string) or a list of column names to be dropped.
    
        Returns
        -------
        pd.DataFrame
            A new DataFrame with the specified columns dropped.
        
        Notes
        -----
        This method utilizes the `pandas` library for DataFrame manipulation.
    
        See Also
        --------
        - pandas.DataFrame.drop : Drop specified labels from rows or columns.
        
        Examples
        --------
        >>> # Drop a single column
        >>> df = SupervisedLearning(dataset)
        >>> df.drop_columns('column_name')
    
        >>> # Drop multiple columns
        >>> df = SupervisedLearning(dataset)
        >>> df.drop_columns(['column1', 'column2'])

        """
        self.__data = self.__data.drop(columns, axis = 1)
        return self.__data
    
    
    def get_training_test_data(self):
        """
        Get the training and test data splits.
    
        Returns
        -------
        Tuple
            A tuple containing X_train, X_test, y_train, and y_test.
    
    
        Notes
        -----
        This method uses the `sklearn.model_selection` library for splitting the data into training and test sets.
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.
        
        Examples
        --------
        >>> # Get training and test data splits
        >>> df = SupervisedLearning(dataset)
        >>> X_train, X_test, y_train, y_test = df.get_training_test_data()
        """
        return (self.__x_train, self.__x_test, self.__y_train, self.__y_test)
        
    
    def get_dataset(self):
        """
        Retrieve the original dataset and the processed data.
        
        Returns
        -------
        Tuple
            A tuple containing the original dataset and the processed data.
        
        Notes
        -----
        This method provides access to both the original and processed datasets.
        
        See Also
        --------
        pandas.DataFrame : Data structure for handling tabular data.
        
        Examples
        --------
        >>> # Get the original and processed datasets
        >>> df = SupervisedLearning(dataset)
        >>> original_data, processed_data = df.get_dataset()
        """
        return (self.__data, self.__dataset)
    
    
    def fix_missing_values(self, strategy: str = None):
        """
        Fix missing values in the dataset.
    
        Parameters
        ----------
        strategy : str, optional
            The strategy to use for imputation. If not specified, it defaults to "mean".
            Options: "mean", "median", "mode".
    
        Returns
        -------
        pd.DataFrame
            The dataset with missing values imputed.
        
        Notes
        -----
        This method uses the `sklearn.impute` library for handling missing values.
    
        See Also
        --------
        sklearn.impute.SimpleImputer : Imputation transformer for completing missing values.

        Examples
        --------
        >>> # Fix missing values using the default strategy ("mean")
        >>> df = SupervisedLearning(dataset)
        >>> df.fix_missing_values()
    
        >>> # Fix missing values using a specific strategy (e.g., "median")
        >>> df.fix_missing_values(strategy="median")
        """
        self.__strategy = strategy
        if self.__strategy == None:
            imputer = si.SimpleImputer(strategy = "mean")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "mean":
            imputer = si.SimpleImputer(strategy = "mean")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "median":
            imputer = si.SimpleImputer(strategy = "median")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "mode":
            imputer = si.SimpleImputer(strategy = "most_frequent")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_)
            self.__fixed_missing = True
            return self.__data
        
        
    def categorical_to_numerical(self, columns: list = None):
        """
        Convert categorical columns to numerical using one-hot encoding.
    
        Parameters
        ----------
        columns : list, optional
            A list of column names to apply one-hot encoding. If not provided, one-hot encoding is applied to all categorical columns.
    
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with categorical columns converted to numerical using one-hot encoding.
    
        Notes
        -----
        This method uses the `pandas` library for one-hot encoding.
    
        See Also
        --------
        pandas.get_dummies : Convert categorical variable(s) into dummy/indicator variables.
        
        Examples
        --------
        >>> # Convert all categorical columns to numerical using one-hot encoding
        >>> df = SupervisedLearning(dataset)
        >>> df.categorical_to_numerical()
    
        >>> # Convert specific columns to numerical using one-hot encoding
        >>> df.categorical_to_numerical(columns=['Category1', 'Category2'])
        """
        self.__columns = columns
        
        if self.__columns == None:
            self.__data = pd.get_dummies(self.__data, drop_first = True, dtype = int)
            self.__data_transformation = True
            return self.__data
            
        else:
            self.__data = pd.get_dummies(self.__data, columns = self.__columns, drop_first = True, dtype = int)
            self.__data_transformation = True
            return self.__data
    
  
    def remove_outlier(self, drop_na: bool):
        """
        Remove outliers from the dataset.
    
        This method uses the `sklearn.preprocessing` library for standard scaling and outlier removal.
    
        Parameters
        ----------
        drop_na : bool
            If False, outliers are replaced with NaN values. If True, rows with NaN values are dropped.
    
        Returns
        -------
        pd.DataFrame
            The dataset with outliers removed.
    
        Notes
        -----
        The method applies standard scaling using `sklearn.preprocessing.StandardScaler` and removes outliers
        based on the range of -3 to 3 standard deviations.
    
        See Also
        --------
        sklearn.preprocessing.StandardScaler : Standardize features by removing the mean and scaling to unit variance.

        Examples
        --------
        >>> # Remove outliers, replace with NaN
        >>> df = SupervisedLearning(dataset)
        >>> df.remove_outlier(drop_na=False)
    
        >>> # Remove outliers and drop rows with NaN values
        >>> df.remove_outlier(drop_na=True)
        """
        if drop_na == False:
            scaler = sp.StandardScaler()
            self.__data = scaler.fit_transform(self.__data)
            self.__data = pd.DataFrame(self.__data, columns = scaler.feature_names_in_)
            self.__data = self.__data[(self.__data >= -3) & (self.__data <= 3)]
            self.__data = pd.DataFrame(scaler.inverse_transform(self.__data), columns = scaler.feature_names_in_)
            self.__remove_outlier = True
            
        elif drop_na == True:
            scaler = sp.StandardScaler()
            self.__data = scaler.fit_transform(self.__data)
            self.__data = pd.DataFrame(self.__data, columns = scaler.feature_names_in_)
            self.__data = self.__data[(self.__data >= -3) & (self.__data <= 3)].dropna()
            self.__data = pd.DataFrame(scaler.inverse_transform(self.__data), columns = scaler.feature_names_in_)
            self.__remove_outlier = True
            
        return self.__data
        
    
    def scale_independent_variables(self):
        """
        Standardize independent variables using sklearn.preprocessing.StandardScaler.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame with scaled independent variables.
    
        Notes
        -----
        This method uses the `sklearn.preprocessing` library for standardization.
    
        Examples
        --------
        >>> # Create an instance of SupervisedLearning
        >>> df = SupervisedLearning(dataset)
        >>> # Scale independent variables
        >>> scaled_data = df.scale_independent_variables()
    
        See Also
        --------
        sklearn.preprocessing.StandardScaler : Standardize features by removing the mean and scaling to unit variance.

        """
        self.__scaler = sp.StandardScaler()
        self.__x = self.__scaler.fit_transform(self.__x)
        self.__x = pd.DataFrame(self.__x, columns = self.__scaler.feature_names_in_)
        self.__scaled = True
        return self.__x
  
    
    def eda(self):
        """
        Perform Exploratory Data Analysis (EDA) on the dataset.
    
        Returns
        -------
        Dict
            A dictionary containing various EDA results, including data head, data tail, descriptive statistics, mode, distinct count, null count, total null count, and correlation matrix.
    
        Notes
        -----
        This method utilizes functionalities from pandas for data analysis.
    
        Examples
        --------
        >>> # Perform Exploratory Data Analysis
        >>> df = SupervisedLearning(dataset)
        >>> eda_results = df.eda()
    
        See Also
        --------
        - pandas.DataFrame.info : Get a concise summary of a DataFrame.
        - pandas.DataFrame.head : Return the first n rows.
        - pandas.DataFrame.tail : Return the last n rows.
        - pandas.DataFrame.describe : Generate descriptive statistics.
        - pandas.DataFrame.mode : Get the mode(s) of each element.
        - pandas.DataFrame.nunique : Count distinct observations.
        - pandas.DataFrame.isnull : Detect missing values.
        - pandas.DataFrame.corr : Compute pairwise correlation of columns.
        """
        self.__data.info()
        print("\n\n")
        data_head = self.__data.head()
        data_tail = self.__data.tail()
        data_descriptive_statistic = self.__data.describe()
        data_more_descriptive_statistics = self.__data.describe(include = "all")
        data_mode = self.__data.mode()
        data_unique = SupervisedLearning.unique_elements_in_columns(self)
        data_distinct_count = self.__data.nunique()
        data_null_count = self.__data.isnull().sum()
        data_total_null_count = self.__data.isnull().sum().sum()
        data_correlation_matrix = self.__data.corr()
        self.__eda = True
        return {"Data": self.__data, "Data_Head": data_head, "Data_Tail": data_tail, "Data_Descriptive_Statistic": data_descriptive_statistic, "Data_More_Descriptive_Statistic": data_more_descriptive_statistics, "Data_Mode": data_mode, "Data_Distinct_Count": data_distinct_count, "Unique_Elements_in_Data": data_unique, "Data_Null_Count": data_null_count, "Data_Total_Null_Count": data_total_null_count, "Data_Correlation_Matrix": data_correlation_matrix}
    
    
    def eda_visual(self, y: str, histogram_bins: int = 10, figsize_heatmap: tuple = (15, 10), figsize_histogram: tuple = (15, 10), figsize_barchart: tuple = (15, 10), before_data_cleaning: bool = True):
        """
        Generate visualizations for exploratory data analysis (EDA).
    
        Parameters
        ----------
        y : str
            The target variable for visualization.
        histogram_bins: int
            The number of bins for each instogram.
        figsize_heatmap: tuple
            The length and breadth for the frame of the heatmap.
        figsize_histogram: tuple
            The length and breadth for the frame of the histogram.
        figsize_barchart: tuple
            The length and breadth for the frame of the barchart.
        before_data_cleaning : bool, default True
            If True, visualizes data before cleaning. If False, visualizes cleaned data.
    
        Returns
        -------
        None
            The method generates and displays various visualizations based on the data distribution and correlation.
    
        Notes
        -----
        This method utilizes the following libraries for visualization:
        - `matplotlib.pyplot` for creating histograms and heatmaps.
        - `seaborn` for creating count plots and box plots.
    
        Examples
        --------
        >>> # Generate EDA visualizations before data cleaning
        >>> df = SupervisedLearning(dataset)
        >>> df.eda_visual(y='target_variable', before_data_cleaning=True)
    
        >>> # Generate EDA visualizations after data cleaning
        >>> df.eda_visual(y='target_variable', before_data_cleaning=False)

        """
        if before_data_cleaning == False:
            data_histogram = self.__data.hist(figsize = figsize_histogram, bins = histogram_bins)
            plt.figure(figsize = figsize_heatmap)
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        
        elif before_data_cleaning == True:
            # Visualize the distribution of categorical features
            categorical_features = self.__data.select_dtypes(include = "object").columns
            for feature in categorical_features:
                plt.figure(figsize=figsize_barchart)
                sns.countplot(x=feature, data = self.__data)
                plt.title(f'Distribution of {feature}')
                plt.show()
                
            data_histogram = self.__data.hist(figsize = figsize_histogram, bins = histogram_bins)
            plt.figure(figsize = figsize_heatmap)
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        self.__eda_visual = True
      
    
    def select_dependent_and_independent(self, predict: str):
        """
        Select the dependent and independent variables for the supervised learning model.
    
        Parameters
        ----------
        predict : str
            The name of the column to be used as the dependent variable.
    
        Returns
        -------
        Dict
            A dictionary containing the dependent variable and independent variables.
    
        Notes
        -----
        This method uses the `pandas` library for data manipulation.
    
        Examples
        --------
        >>> # Select dependent and independent variables
        >>> df = SupervisedLearning(dataset)
        >>> variables = df.select_dependent_and_independent("target_column")
    
        See Also
        --------
        - pandas.DataFrame.drop : Drop specified labels from rows or columns.
        - pandas.Series : One-dimensional ndarray with axis labels.

        """
        self.__x = self.__data.drop(predict, axis = 1)
        self.__y = self.__data[f"{predict}"]
        self.__dependent_independent = True
        return {"Dependent Variable": self.__y, "Independent Variables": self.__x}
    
      
    def split_data(self, test_size: float = 0.2):
        """
        Split the dataset into training and test sets.
        
        Parameters
        ----------
        test_size: float, optional, default=0.2
            Specifies the size of the data to split as test data.
        
        Returns
        -------
        Dict
            A dictionary containing the training and test sets for independent (X) and dependent (y) variables.
    
        Notes
        -----
        This method uses the `sklearn.model_selection.train_test_split` function for data splitting.
    
        Examples
        --------
        >>> # Split the data into training and test sets
        >>> df = SupervisedLearning(dataset)
        >>> data_splits = df.split_data()
        >>>
        >>> X_train = data_splits["Training X"]
        >>> X_test = data_splits["Test X"]
        >>> y_train = data_splits["Training Y"]
        >>> y_test = data_splits["Test Y"]
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.

        """
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = sms.train_test_split(self.__x, self.__y, test_size = test_size, random_state = 0)
        self.__split_data = True
        
        if self.__polynomial_regression == True:
            self.__x_train1, self.__x_test1, self.__y_train1, self.__y_test1 = sms.train_test_split(self.__x1, self.__y, test_size = test_size, random_state = 0)
        
        return {"Training X": self.__x_train, "Test X": self.__x_test, "Training Y": self.__y_train, "Test Y": self.__y_test}
    

    def train_model_regressor(self, regressor):
        """
        Train a regressor model.
    
        Parameters
        ----------
        regressor : Any
            A regressor model object compatible with scikit-learn's regressor interface.
    
        Returns
        -------
        Any
            The trained regressor model.
    
        Notes
        -----
        - This method uses the `sklearn.model_selection` and `sklearn.metrics` libraries for training and evaluation. 
        - All required steps before model training should have been completed before running this function.
    
        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> # Train a regressor model
        >>> df = SupervisedLearning(dataset)
        >>> regressor = LinearRegression()
        >>> trained_regressor = df.train_model_regressor(regressor)
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.
        - sklearn.metrics.r2_score : R^2 (coefficient of determination) regression score function.

        """
        if self.__split_data == True:
            self.regression_problem = True
            self.regressor = regressor
            self.model_regressor = self.regressor.fit(self.__x_train, self.__y_train)
            score = self.model_regressor.score(self.__x_train, self.__y_train)
            print(f"{self.regressor.__class__.__name__}'s amount of variation in Y predicted by your features X after training: (Rsquared) ----> {score}")
            self.__model_training = True
            
        else:
            self.regression_problem = True
            self.regressor = regressor
            self.model_regressor = self.regressor.fit(self.__x, self.__y)
            score = self.model_regressor.score(self.__x, self.__y)
            print(f"{self.regressor.__class__.__name__}'s amount of variation in Y predicted by your features X after training: (Rsquared) ----> {score}")
            self.__model_training = True
            
        return self.model_regressor 

      
    def train_model_classifier(self, classifier):
        """
        Train a classifier on the provided data.
    
        Parameters
        ----------
        classifier : Any
            The classifier object to be trained.
    
        Returns
        -------
        Any
            The trained classifier.
    
        Notes
        -----
        This method uses the `sklearn.model_selection` and `sklearn.metrics` libraries for training and evaluating the classifier.
    
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> # Train a classifier
        >>> df = SupervisedLearning(dataset)
        >>> classifier = RandomForestClassifier(random_state = 0)
        >>> trained_classifier = df.train_model_classifier(classifier)
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.
        - sklearn.metrics.accuracy_score : Accuracy classification score.

        """
        if self.__split_data == True:
            self.classification_problem = True
            self.classifier = classifier
            self.model_classifier = self.classifier.fit(self.__x_train, self.__y_train)
            score = self.model_classifier.score(self.__x_train, self.__y_train)
            print(f"{self.classifier.__class__.__name__} accuracy in prediction after training: (Accuracy) ---> {score}")
            self.__model_training = True
            
        else:
            self.classification_problem = True
            self.classifier = classifier
            self.model_classifier = self.classifier.fit(self.__x, self.__y)
            score = self.model_classifier.score(self.__x, self.__y)
            print(f"{self.classifier.__class__.__name__} accuracy in prediction after training: (Accuracy) ---> {score}")
            self.__model_training = True
            
        return self.model_classifier
        
    
    def regressor_predict(self):
        """
        Predict the target variable for regression models.
    
        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing actual training and test targets along with predicted values,
            or None if the model is set for classification.
    
        Raises
        ------
        AssertionError
            If the training phase of the model is set to classification, as regression models
            cannot predict a classification model.
    
        Notes
        -----
        This method uses the `sklearn` library for regression model prediction.
    
        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> # Train a regressor model
        >>> df = SupervisedLearning(dataset)
        >>> regressor = LinearRegression()
        >>> trained_regressor = df.train_model_regressor(regressor)
        >>>
        >>> # Predict for regression model
        >>> predictions = df.regressor_predict()
        >>>
        >>> print(predictions)
        {'Actual Training Y': array([...]), 'Actual Test Y': array([...]),
         'Predicted Training Y': array([...]), 'Predicted Test Y': array([...])}
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.

        """
        if self.__split_data == True:
            if self.regression_problem == True:
                self.__y_pred = self.model_regressor.predict(self.__x_train)
                self.__y_pred1 = self.model_regressor.predict(self.__x_test)
                self.__model_prediction = True
                return {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1}
             
            else:
                raise AssertionError("The training phase of the model has been set to classification. Can not predict a classification model with a regression model.")
        else:
            if self.regression_problem == True:
                self.__y_pred = self.model_regressor.predict(self.__x)
                self.__model_prediction = True
                return {"Actual Y": self.__y, "Predicted Y": self.__y_pred}
             
            else:
                raise AssertionError("The training phase of the model has been set to classification. Can not predict a classification model with a regression model.")
        
     
    def classifier_predict(self):
        """
        Predict the target variable using the trained classifier.
    
        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the actual and predicted values for training and test sets.
            Keys include 'Actual Training Y', 'Actual Test Y', 'Predicted Training Y', and 'Predicted Test Y'.
    
        Raises
        ------
        AssertionError
            If the model is set for regression, not classification.
    
        Notes
        -----
        This method uses the `sklearn` library for classification model prediction.
    
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> # Train a regressor model
        >>> df = SupervisedLearning(dataset)
        >>> classifier = RandomForestClassifier(random_state = 0)
        >>> trained_classifier = df.train_model_classifier(classifier)
        >>>
        >>> # Predict for regression model
        >>> predictions = df.classifier_predict()
        >>>
        >>> print(predictions)
        {'Actual Training Y': array([...]), 'Actual Test Y': array([...]),
         'Predicted Training Y': array([...]), 'Predicted Test Y': array([...])}
    
        See Also
        --------
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.

        """
        if self.__split_data == True:
            if self.classification_problem == True:
                self.__y_pred = self.model_classifier.predict(self.__x_train)
                self.__y_pred1 = self.model_classifier.predict(self.__x_test)
                self.__model_prediction = True
                return {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1}
            
            else:
                raise AssertionError("The training phase of the model has been set to regression. Can not predict a regression model with a classification model.")
        
        else:
            if self.classification_problem == True:
                self.__y_pred = self.model_classifier.predict(self.__x)
                self.__model_prediction = True
                return {"Actual Y": self.__y, "Predicted Y": self.__y_pred}
            
            else:
                raise AssertionError("The training phase of the model has been set to regression. Can not predict a regression model with a classification model.")
        
        
    def regressor_model_testing(self, variables_values: list, scaling: bool = False):
        """
        Test the trained regressor model with given input variables.
    
        Parameters
        ----------
        variables_values : list
            A list containing values for each independent variable.
        scaling : bool, default False
            Whether to scale the input variables. If True, the method expects
            scaled input using the same scaler used during training.
    
        Returns
        -------
        np.ndarray
            The predicted values from the regressor model.
    
        Raises
        ------
        AssertionError
            If the problem type is not regression.
    
        Notes
        -----
        - This method tests a pre-trained regressor model. 
        - If scaling is set to True, the input variables are expected to be scaled using the same scaler used during training.
    
        Examples
        --------
        >>> # Assuming df is an instance of SupervisedLearning class with a trained regressor model
        >>> df.regressor_model_testing([1.5, 0.7, 2.0], scaling=True)
        array([42.0])
    
        See Also
        --------
        - sklearn.preprocessing.StandardScaler : Standardize features by removing the mean and scaling to unit variance.

        """
        if self.regression_problem == True:
            if scaling == False:
                prediction = self.model_regressor.predict([variables_values])
                self.__model_testing = True
                return prediction
            
            elif scaling == True:
                variables_values = self.__scaler.transform([variables_values])
                prediction = self.model_regressor.predict(variables_values)
                self.__model_testing = True
                return prediction
            
        else:
            raise AssertionError("You can't test a classification problem with this function.")
        
            
    def regressor_evaluation(self, kfold: int = None, cross_validation: bool = False):
        """
        Evaluate the performance of the regression model.
    
        Parameters
        ----------
        kfold : int, optional
            Number of folds for cross-validation. If provided, cross-validation will be performed.
        cross_validation : bool, default False
            If True, perform cross-validation; otherwise, perform a simple train-test split evaluation.
    
        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
    
        Raises
        ------
        ValueError
            If invalid combination of parameters is provided.
    
        Notes
        -----
        This method uses the `sklearn.metrics` and `sklearn.model_selection` libraries for regression evaluation.
    
        Examples
        --------
        >>> # Evaluate regression model performance using simple train-test split
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> # Train a regressor model
        >>> df = SupervisedLearning(dataset)
        >>> regressor = LinearRegression()
        >>> trained_regressor = df.train_model_regressor(regressor)
        >>>
        >>> # Predict for regression model
        >>> predictions = df.regressor_predict()
        >>> evaluation_results = df.regressor_evaluation()
    
        >>> # Evaluate regression model performance using 10-fold cross-validation
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> # Train a regressor model
        >>> df = SupervisedLearning(dataset)
        >>> regressor = LinearRegression()
        >>> trained_regressor = df.train_model_regressor(regressor)
        >>>
        >>> # Predict for regression model
        >>> predictions = df.regressor_predict()
        >>> evaluation_results = df.regressor_evaluation(kfold=10, cross_validation=True)
    
        See Also
        --------
        - sklearn.metrics.r2_score : R-squared (coefficient of determination) regression score function.
        - sklearn.metrics.mean_squared_error : Mean squared error regression loss.
        - sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.

        """
        if self.__split_data == True:
            if self.regression_problem == True:
                if kfold == None and cross_validation == False:
                    training_rsquared = sm.r2_score(self.__y_train, self.__y_pred)
                    test_rsquared = sm.r2_score(self.__y_test, self.__y_pred1)
                    
                    training_rmse = np.sqrt(sm.mean_squared_error(self.__y_train, self.__y_pred))
                    test_rmse = np.sqrt(sm.mean_squared_error(self.__y_test, self.__y_pred1))
                    self.__model_evaluation = True
                    return {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}}
                
                elif kfold != None and cross_validation == False:
                    raise ValueError
                    
                elif kfold == None and cross_validation == True:
                    training_rsquared = sm.r2_score(self.__y_train, self.__y_pred)
                    test_rsquared = sm.r2_score(self.__y_test, self.__y_pred1)
                    
                    training_rmse = np.sqrt(sm.mean_squared_error(self.__y_train, self.__y_pred))
                    test_rmse = np.sqrt(sm.mean_squared_error(self.__y_test, self.__y_pred1))
                    
                    cross_val = sms.cross_val_score(self.model_regressor, self.__x_train, self.__y_train, cv = 10)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
                
                elif kfold != None and cross_validation == True:
                    training_rsquared = sm.r2_score(self.__y_train, self.__y_pred)
                    test_rsquared = sm.r2_score(self.__y_test, self.__y_pred1)
                    
                    training_rmse = np.sqrt(sm.mean_squared_error(self.__y_train, self.__y_pred))
                    test_rmse = np.sqrt(sm.mean_squared_error(self.__y_test, self.__y_pred1))
                    
                    cross_val = sms.cross_val_score(self.model_regressor, self.__x_train, self.__y_train, cv = kfold)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
            
            else:
                raise AssertionError("You can not use a regression evaluation function for a classification problem.")
        else:
            if self.regression_problem == True:
                if kfold == None and cross_validation == False:
                    rsquared = sm.r2_score(self.__y, self.__y_pred)
                    rmse = np.sqrt(sm.mean_squared_error(self.__y, self.__y_pred))
                    self.__model_evaluation = True
                    return {"R2": rsquared, "RMSE": rmse}
                
                elif kfold != None and cross_validation == False:
                    raise ValueError
                    
                elif kfold == None and cross_validation == True:
                    rsquared = sm.r2_score(self.__y, self.__y_pred)
                    rmse = np.sqrt(sm.mean_squared_error(self.__y, self.__y_pred))
                    
                    cross_val = sms.cross_val_score(self.model_regressor, self.__x, self.__y, cv = 10)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
                
                elif kfold != None and cross_validation == True:
                    rsquared = sm.r2_score(self.__y, self.__y_pred)
                    rmse = np.sqrt(sm.mean_squared_error(self.__y, self.__y_pred))
                    
                    cross_val = sms.cross_val_score(self.model_regressor, self.__x, self.__y, cv = kfold)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {"R2": rsquared, "RMSE": rmse, "Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}
            
            else:
                raise AssertionError("You can not use a regression evaluation function for a classification problem.")
            
         
    def build_multiple_regressors(self, regressors: list or tuple, kfold: int = None, cross_validation: bool = False, graph: bool = False, length: int = 20, width: int = 10, linestyle: str = 'dashed', marker: str = 'o', markersize: int = 12, fontweight: int = 80, fontstretch: int = 50):
        """
        Build, evaluate, and optionally graph multiple regression models.
        
        This method facilitates the construction and assessment of multiple regression models using a variety of algorithms.
        It supports both single train-test split and k-fold cross-validation approaches. The generated models are evaluated
        based on key regression metrics, providing insights into their performance on both training and test datasets.
    
        Parameters
        ----------
        regressors : list or tuple
            List of regression models to build and evaluate.
        kfold : int, optional
            Number of folds for cross-validation. Default is None.
        cross_validation : bool, default False
            If True, perform cross-validation; otherwise, use a simple train-test split.
        graph : bool, default False
            If True, plot evaluation metrics for each regression model.
        length : int, optional, default=None
            Length of the graph (if graph=True).
        width : int, optional, default=None
            Width of the graph (if graph=True).
    
        Returns
        -------
        dict
            A dictionary containing regression metrics and additional information.
    
        Notes
        -----
        This method uses the following libraries:
        - `sklearn.model_selection` for train-test splitting and cross-validation.
        - `matplotlib.pyplot` for plotting if graph=True.
    
        Examples
        --------
        >>> # Build and evaluate multiple regression models
        >>> df = SupervisedLearning(dataset)
        >>> models = [LinearRegression(), 
        >>>           RandomForestRegressor(), 
        >>>           GradientBoostingRegressor()]
        >>> results = df.build_multiple_regressors(regressors=models, 
        >>>                                        kfold=5, 
        >>>                                        cross_validation=True, 
        >>>                                        graph=True)
    
        See Also
        --------
        - SupervisedLearning.train_model_regressor : Train a single regression model.
        - SupervisedLearning.regressor_predict : Make predictions using a trained regression model.
        - SupervisedLearning.regressor_evaluation : Evaluate the performance of a regression model.

        """
        if self.__split_data == True:
            if (isinstance(regressors, list) or isinstance(regressors, tuple)) and cross_validation == False:
                self.__multiple_regressor_models = {}
                store = []
                for algorithms in regressors:
                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                    info = [
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"]
                        ]
                    store.append(info)
                    
                dataset_regressors = pd.DataFrame(store, columns = ["Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                
                if graph == True:
                    # Training R2
                    plt.figure(figsize = (length, width))
                    plt.title("Training R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training R2", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("Training RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training RMSE", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test R2
                    plt.figure(figsize = (length, width))
                    plt.title("Training R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training R2", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("Training RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training RMSE", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                   
            elif (isinstance(regressors, list) or isinstance(regressors, tuple)) and cross_validation == True:
                self.__multiple_regressor_models = {}
                store = []
                for algorithms in regressors:
                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                    info = [
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"],
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"],
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"],
                        ]
                    store.append(info)
                    
                dataset_regressors = pd.DataFrame(store, columns = ["Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                if graph == True:
                    # Training R2
                    plt.figure(figsize = (length, width))
                    plt.title("Training R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training R2", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("Training RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training RMSE", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test R2
                    plt.figure(figsize = (length, width))
                    plt.title("Training R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training R2", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("Training RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training RMSE", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Mean
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Mean", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Mean"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Mean"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Mean", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Standard Deviation
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Standard Deviation", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Standard Deviation"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Standard Deviation"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
            
                
            return {"Regressor Metrics": dataset_regressors, "More Info": self.__multiple_regressor_models}
        
        else:
            if (isinstance(regressors, list) or isinstance(regressors, tuple)) and cross_validation == False:
                self.__multiple_regressor_models = {}
                store = []
                for algorithms in regressors:
                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                    info = [
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"], 
                        ]
                    store.append(info)
                    
                dataset_regressors = pd.DataFrame(store, columns = ["Algorithm", "R2", "RMSE"])
                
                if graph == True:
                    # R2
                    plt.figure(figsize = (length, width))
                    plt.title("R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("R2", labelpad = 20)
                    plt.show()
                    
                    
                    # RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("RMSE", labelpad = 20)
                    plt.show()
                    
                   
            elif (isinstance(regressors, list) or isinstance(regressors, tuple)) and cross_validation == True:
                self.__multiple_regressor_models = {}
                store = []
                for algorithms in regressors:
                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                    info = [
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"], 
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"],
                        self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"],
                        ]
                    store.append(info)
                    
                dataset_regressors = pd.DataFrame(store, columns = ["Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                if graph == True:
                    # R2
                    plt.figure(figsize = (length, width))
                    plt.title("R2", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["R2"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["R2"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("R2", labelpad = 20)
                    plt.show()
                    
                    
                    # RMSE
                    plt.figure(figsize = (length, width))
                    plt.title("RMSE", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["RMSE"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["RMSE"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("RMSE", labelpad = 20)
                    plt.show()
                    
                    
                    # Cross Validation Mean
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Mean", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Mean"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Mean"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Mean", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Standard Deviation
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Standard Deviation", pad = 10)
                    plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Standard Deviation"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Standard Deviation"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()

                
            return {"Regressor Metrics": dataset_regressors, "More Info": self.__multiple_regressor_models}
    
    
    def build_multiple_classifiers(self, classifiers: list or tuple, kfold: int = None, cross_validation: bool = False, graph: bool = False, length: int = 20, width: int = 10, linestyle: str = 'dashed', marker: str = 'o', markersize: int = 12, fontweight: int = 80, fontstretch: int = 50):
        """
        Build and evaluate multiple classifiers.
    
        Parameters
        ----------
        classifiers : list or tuple
            A list or tuple of classifier objects to be trained and evaluated.
        kfold : int, optional, default=None
            Number of folds for cross-validation. If None, cross-validation is not performed.
        cross_validation : bool, optional, default=False
            Perform cross-validation if True.
        graph : bool, optional, default=False
            Whether to display performance metrics as graphs.
        length : int, optional, default=None
            Length of the graph (if graph=True).
        width : int, optional, default=None
            Width of the graph (if graph=True).
    
        Returns
        -------
        dict
            A dictionary containing classifier metrics and additional information.
    
        Notes
        -----
        This method builds and evaluates multiple classifiers on the provided dataset. It supports both traditional
        training/testing evaluation and cross-validation.
    
        If `graph` is True, the method also displays graphs showing performance metrics for training and testing datasets.
    
        References
        ----------
        - `scikit-learn: Machine Learning in Python <https://scikit-learn.org/stable/>`_
        - `matplotlib: Python plotting <https://matplotlib.org/>`_
        - `numpy: The fundamental package for scientific computing with Python <https://numpy.org/>`_
        - `pandas: Powerful data structures for data analysis <https://pandas.pydata.org/>`_
    
    
        Example:
        --------
        >>> classifiers = [LogisticRegression(random_state = 0), 
        >>>                RandomForestClassifier(random_state = 0), 
        >>>                SVC(random_state = 0)]
        >>> results = build_multiple_classifiers(classifiers, 
        >>>                                      kfold=5, 
        >>>                                      cross_validation=True, 
        >>>                                      graph=True, 
        >>>                                      length=8, 
        >>>                                      width=12)
    
        Note: Ensure that the classifiers provided are compatible with scikit-learn's classification API.

        """
        if self.__split_data == True:
            if (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cross_validation == False:
                self.__multiple_classifier_models = {}
                store = []
                for algorithms in classifiers:
                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cross_validation)}
                    info = [
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                        ]
                    store.append(info)
                  
                dataset_classifiers = pd.DataFrame(store, columns = ["Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                
                if graph == True:
                    # Training Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Training Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Training Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Training Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("Training F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Test Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Test Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Test Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("Test F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
            
            elif (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cross_validation == True:
                self.__multiple_classifier_models = {}
                store = []
                for algorithms in classifiers:
                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cross_validation)}
                    name = self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__
                    info = [
                        name, 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"],  
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                        ]
                    store.append(info)
                    
                dataset_classifiers = pd.DataFrame(store, columns = ["Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test Model F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if graph == True:
                    # Training Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Training Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Training Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Training Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("Training F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Training F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Test Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Test Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Test Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Test F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("Test F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Model F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Model F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Test F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Mean
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Mean", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Mean"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Mean"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Mean", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Standard Deviation
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Standard Deviation", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Standard Deviation"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Standard Deviation"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                
            return {"Classifier Metrics": dataset_classifiers, "More Info": self.__multiple_classifier_models}
        
        else:
            if (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cross_validation == False:
                self.__multiple_classifier_models = {}
                store = []
                for algorithms in classifiers:
                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cross_validation)}
                    info = [
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"], 
                        ]
                    store.append(info)
                  
                dataset_classifiers = pd.DataFrame(store, columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score",])
                
                if graph == True:
                    # Training Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()

            elif (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cross_validation == True:
                self.__multiple_classifier_models = {}
                store = []
                for algorithms in classifiers:
                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cross_validation)}
                    name = self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__
                    info = [
                        name, 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"], 
                        self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"], 
                        ]
                    store.append(info)
                    
                dataset_classifiers = pd.DataFrame(store, columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if graph == True:
                    # Training Accuracy
                    plt.figure(figsize = (length, width))
                    plt.title("Accuracy", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Accuracy"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Accuracy"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Accuracy", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Precision
                    plt.figure(figsize = (length, width))
                    plt.title("Precision", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Precision"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Precision"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Precision", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training Recall
                    plt.figure(figsize = (length, width))
                    plt.title("Recall", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Recall"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Recall"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Recall", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Training F1 Score
                    plt.figure(figsize = (length, width))
                    plt.title("F1 Score", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["F1 Score"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["F1 Score"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("F1 Score", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Mean
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Mean", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Mean"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Mean"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Mean", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                    
                    
                    # Cross Validation Standard Deviation
                    plt.figure(figsize = (length, width))
                    plt.title("Cross Validation Standard Deviation", pad = 10)
                    plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Standard Deviation"], 'go--', linestyle = linestyle, marker = marker, markersize = markersize)
                    for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Standard Deviation"], 4)):
                        plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = fontweight, fontstretch = fontstretch)
                    plt.xlabel("Algorithm", labelpad = 20)
                    plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                    # plt.yticks(np.arange(0.0, 1.0, 0.1))
                    plt.show()
                
            return {"Classifier Metrics": dataset_classifiers, "More Info": self.__multiple_classifier_models}
    
    
    def build_single_regressor_from_features(self, strategy: str, estimator: str, regressor, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
        """
        Build and evaluate a single regression model using feature selection.
    
        Parameters
        ----------
        strategy : str
            Feature selection strategy. Should be one of ["selectkbest", "selectpercentile", "rfe", "selectfrommodel"].
        estimator : str
            Estimator used for feature selection, applicable for "rfe" and "selectfrommodel" strategies. Is set to a regressor that implements 'fit'. Should be one of ["f_regression", "f_oneway", "chi2"] if strategy is set to: ["selectkbest", "selectpercentile"].
        regressor : object
            Regression model object to be trained.
        max_num_features : int, optional
            Maximum number of features to consider, by default None.
        min_num_features : int, optional
            Minimum number of features to consider, by default None.
        kfold : int, optional
            Number of folds for cross-validation, by default None. Needs cv to be set to True to work.
        cv : bool, optional
            Whether to perform cross-validation, by default False.
    
        Returns
        -------
        dict
            A dictionary containing feature metrics and additional information about the models.
    
        Notes
        -----
        - This method builds a regression model using feature selection techniques and evaluates its performance. 
        - The feature selection strategies include "selectkbest", "selectpercentile", "rfe", and "selectfrommodel". 
        - The estimator parameter is required for "rfe" and "selectfrommodel" strategies.
        - This method assumes that the dataset and labels are already set in the class instance.
    
        See Also
        --------
        - `scikit-learn.feature_selection` for feature selection techniques.
        - `scikit-learn.linear_model` for regression models.
        - `scikit-learn.model_selection` for cross-validation techniques.
        - `scikit-learn.metrics` for regression performance metrics.
        - Other libraries used in this method: `numpy`, `pandas`, `matplotlib`, `seaborn`, `ydata_profiling`, `sweetviz`, 
          `imblearn`, `sklearn`, `warnings`, `datatable`.
    
        References
        -----------
        - scikit-learn documentation for feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
        - scikit-learn documentation for regression models: https://scikit-learn.org/stable/supervised_learning.html#regression
    
        Example
        --------
        >>> learn = SupervisedLearning(dataset)
        >>> results = learn.build_single_regressor_from_features(strategy='selectkbest', 
        >>>                                                      estimator='f_regression', 
        >>>                                                      regressor=LinearRegression())
        >>> print(results)
        """
        if self.__split_data == True:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]
            
            if not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        
                        self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                        info = [
                            num,
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"]
                            ]
                        store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                        
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_regressor_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"]
                                ]
                            store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                            
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_regressor_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
                    
            elif not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        
                        self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                        info = [
                            num,
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"],
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"],
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"],
                            ]
                        store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_regressor_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"],
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"],
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"],
                                ]
                            store_data.append(info)
                            
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_regressor_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                   
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
                
                
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
        
        else:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]

            if not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        
                        self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                        info = [
                            num,
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["RMSE"], 
                            ]
                        store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                        
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_regressor_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["RMSE"],
                                ]
                            store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                            
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_regressor_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                    
            elif not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        
                        self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                        info = [
                            num,
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["R2"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["RMSE"], 
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"],
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"],
                            ]
                        store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_regressor_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            
                            self.__multiple_regressor_models[f"{regressor.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = regressor), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["R2"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["RMSE"], 
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"],
                                self.__multiple_regressor_models[f"{regressor.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"],
                                ]
                            store_data.append(info)
                            
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_regressor_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                   
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                
                
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
    
    
    def build_single_classifier_from_features(self, strategy: str, estimator: str, classifier, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
        """
        Build and evaluate a single classification model using feature selection.
    
        Parameters
        ----------
        strategy : str
            Feature selection strategy. Should be one of ["selectkbest", "selectpercentile", "rfe", "selectfrommodel"].
        estimator : str
            Estimator used for feature selection, applicable for "rfe" and "selectfrommodel" strategies. Is set to a classifier that implements 'fit'. Should be one of ["f_classif", "chi2", "mutual_info_classif"] if strategy is set to: ["selectkbest", "selectpercentile"].
        classifier : object
            Classification model object to be trained.
        max_num_features : int, optional
            Maximum number of features to consider, by default None.
        min_num_features : int, optional
            Minimum number of features to consider, by default None.
        kfold : int, optional
            Number of folds for cross-validation, by default None. Needs cv to be set to True to work.
        cv : bool, optional
            Whether to perform cross-validation, by default False.
    
        Returns
        -------
        dict
            A dictionary containing feature metrics and additional information about the models.
    
        Notes
        -----
        - This method builds a classification model using feature selection techniques and evaluates its performance. 
        - The feature selection strategies include "selectkbest", "selectpercentile", "rfe", and "selectfrommodel". 
        - The estimator parameter is required for "rfe" and "selectfrommodel" strategies.
        - This method assumes that the dataset and labels are already set in the class instance.
    
        See Also
        --------
        - `scikit-learn.feature_selection` for feature selection techniques.
        - `scikit-learn.linear_model` for classification models.
        - `scikit-learn.model_selection` for cross-validation techniques.
        - `scikit-learn.metrics` for classification performance metrics.
        - Other libraries used in this method: `numpy`, `pandas`, `matplotlib`, `seaborn`.
    
        References
        -----------
        - scikit-learn documentation for feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
        - scikit-learn documentation for classification models: https://scikit-learn.org/stable/supervised_learning.html#classification
    
        Example
        --------
        >>> learn = SupervisedLearning(dataset)
        >>> results = learn.build_single_classifier_from_features(strategy='selectkbest', 
        >>>                                                       estimator='f_classif', 
        >>>                                                       classifier=RandomForestClassifier(random_state = 0))
        >>> print(results)
        """
        if self.__split_data == True:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]
            
            if not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        
                        self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                        info = [
                            num,
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                            ]
                        store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_classifier_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                ]
                            store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_classifier_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
            
            
            elif not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range((length_col - 1), 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        
                        self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                        info = [
                            num,
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                            ]
                        store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_classifier_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                                ]
                            store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_classifier_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
            
                    
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
        
        else:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]

            if not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        
                        self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                        info = [
                            num,
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                            ]
                        store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_classifier_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                                ]
                            store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_classifier_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")


            elif not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range((length_col - 1), 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        
                        self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                        info = [
                            num,
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"], 
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"], 
                            ]
                        store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"] = self.__multiple_classifier_models
                        store[f"{num}"]["Feature Info"] = feature_info
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            
                            self.__multiple_classifier_models[f"{classifier.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = classifier), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"], 
                                self.__multiple_classifier_models[f"{classifier.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"], 
                                ]
                            store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"] = self.__multiple_classifier_models
                            store[f"{num}"]["Feature Info"] = feature_info
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                    
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
        
    
    def build_multiple_regressors_from_features(self, strategy: str, estimator: str, regressors: list or tuple, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
        """
        Build and evaluate multiple regression models with varying numbers of features.
    
        Parameters:
        -----------
        strategy : str
            The feature selection strategy. Supported values are 'selectkbest', 'selectpercentile',
            'rfe', and 'selectfrommodel'.
        estimator : str
            The estimator to use for feature selection. Choose from 'f_regression', 'f_classif', 'mutual_info_regression',
            'mutual_info_classif', 'linear', 'lasso', 'tree', etc., based on the chosen strategy.
        regressors : list or tuple
            List of regression models to build and evaluate.
        max_num_features : int, optional
            Maximum number of features to consider during the feature selection process.
        min_num_features : int, optional
            Minimum number of features to consider during the feature selection process.
        kfold : int, optional
            Number of folds for cross-validation. If provided, cross-validation metrics will be calculated.
        cv : bool, default False
            If True, perform cross-validation; otherwise, use a single train-test split.
    
        Returns:
        --------
        dict
            A dictionary containing feature metrics and additional information for each model.
    
        Example:
        --------
        >>> from sklearn.ensemble import RandomForestRegressor, DecisionTreeRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>>
        >>> data = SupervisedLearning(dataset)
        >>> results = data.build_multiple_regressors_from_features(
        >>>        strategy='selectkbest',
        >>>        estimator='f_regression',
        >>>        regressors=[LinearRegression(), 
        >>>                    RandomForestRegressor(random_state = 0), 
        >>>                    DecisionTreeRegressor(random_state = 0)],
        >>>        max_num_features=10,
        >>>        kfold=5,
        >>>        cv=True
        >>>        )
    
        See Also:
        ---------
        - `sklearn.feature_selection.SelectKBest`
        - `sklearn.feature_selection.SelectPercentile`
        - `sklearn.feature_selection.RFE`
        - `sklearn.feature_selection.SelectFromModel`
        - `sklearn.linear_model`
        - `sklearn.ensemble.RandomForestRegressor`
        - `sklearn.model_selection.cross_val_score`
        - `sklearn.metrics.mean_squared_error`
        - `sklearn.metrics.r2_score`
    
        """
        if self.__split_data == True:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]
            
            if (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        for algorithms in regressors:
                            self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"]
                                ]
                            store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                        
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
    
    
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            for algorithms in regressors:
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                                info = [
                                    num,
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"]
                                    ]
                                store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                            
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
                    
            elif (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        for algorithms in regressors:
                            self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"],
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"],
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"],
                                ]
                            store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            for algorithms in regressors:
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                                info = [
                                    num,
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Training RMSE"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Test RMSE"],
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"],
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"],
                                    ]
                                store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                   
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
                
                
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
        
        else:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]

            if (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        for algorithms in regressors:
                            self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"],
                                ]
                            store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                        
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)


                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            for algorithms in regressors:
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = False)}
                                info = [
                                    num,
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"],
                                    ]
                                store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE"])
                            
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                    
            elif (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_regressor_models = {}
                        store_data = []
                        for algorithms in regressors:
                            self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                            info = [
                                num,
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"],
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"],
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"],
                                ]
                            store_data.append(info)
                            
                        dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                        
                        dataset2 = dataset_regressors
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Regression Model with {num} Feature(s): \n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_regressor_models = {}
                            store_data = []
                            for algorithms in regressors:
                                self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_regressor(self, regressor = algorithms), "Prediction": SupervisedLearning.regressor_predict(self), "Evaluation": SupervisedLearning.regressor_evaluation(self, kfold = kfold, cross_validation = True)}
                                info = [
                                    num,
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["R2"], 
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["RMSE"],
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"],
                                    self.__multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"],
                                    ]
                                store_data.append(info)
                                
                            dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "R2", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_regressor_models
                            
                            dataset2 = dataset_regressors
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                   
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                
                
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
    
    
    def build_multiple_classifiers_from_features(self, strategy: str, estimator: str, classifiers: list or tuple, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
        """
        Build multiple classifiers using different feature selection strategies and machine learning algorithms.
    
        This method performs feature selection, trains multiple classifiers, and evaluates their performance.
    
        Parameters:
        -----------
        - strategy (str): Feature selection strategy. Should be one of 'selectkbest', 'selectpercentile', 'rfe', or 'selectfrommodel'.
        - estimator (str): The estimator to use for feature selection. It could be the name of an estimator (e.g., 'RandomForestClassifier') or a string specifying the strategy (e.g., 'mean' for 'selectpercentile').
        - classifiers (list or tuple): List of classifier instances to be trained and evaluated.
        - max_num_features (int, optional): Maximum number of features to consider. If None, all features are considered.
        - min_num_features (int, optional): Minimum number of features to consider. If None, the process starts with max_num_features and decreases the count until 1.
        - kfold (int, optional): Number of folds for cross-validation. If None, regular train-test split is used.
        - cv (bool, optional): If True, perform cross-validation. If False, use a train-test split.
    
        Returns:
        --------
        - dict: A dictionary containing feature metrics and additional information about the trained models.
    
        See Also:
        ---------
        - `scikit-learn.feature_selection` for feature selection strategies: https://scikit-learn.org/stable/modules/feature_selection.html
        - `scikit-learn.ensemble` for ensemble classifiers: https://scikit-learn.org/stable/modules/ensemble.html
        - `scikit-learn.neighbors` for k-nearest neighbors classifiers: https://scikit-learn.org/stable/modules/neighbors.html
        - `pandas` for data manipulation: https://pandas.pydata.org/docs/
        - `numpy` for numerical operations: https://numpy.org/doc/stable/
        - `sklearn` for various machine learning utilities: https://scikit-learn.org/stable/documentation.html
    
        Example:
        --------
        >>> from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
        >>>
        >>>
        >>> data = SupervisedLearning(dataset)
        >>> classifiers = [RandomForestClassifier(random_state = 0), 
        >>>                DecisionTreeClassifier(random_state = 0)]
        >>> result = data.build_multiple_classifiers_from_features(strategy='selectkbest', 
        >>>                                                        estimator='f_classif', 
        >>>                                                        classifiers=classifiers, 
        >>>                                                        max_num_features=10, 
        >>>                                                        kfold=5)
        """
        if self.__split_data == True:        
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]
            
            if (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        for algorithms in classifiers:
                            self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                ]
                            store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            for algorithms in classifiers:
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                                info = [
                                    num,
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                    ]
                                store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
            
            
            elif (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range((length_col - 1), 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        for algorithms in classifiers:
                            self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                                ]
                            store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            for algorithms in classifiers:
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                                info = [
                                    num,
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Training Evaluation"]["Model F1 Score"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Test Evaluation"]["Model F1 Score"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Mean"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                                    ]
                                store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
            
                    
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}
        
        else:
            types1 = ["selectkbest", "selectpercentile"]
            types2 = ["rfe", "selectfrommodel"]

            if (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == False:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range(length_col, 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        for algorithms in classifiers:
                            self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                                ]
                            store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            for algorithms in classifiers:
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                                info = [
                                    num,
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"],
                                    ]
                                store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")


            elif (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == True:
                data_columns = [col for col in self.__x.columns]
                length_col = len(data_columns)
                store = {}
                dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                if (max_num_features != None) and isinstance(max_num_features, int):
                    length_col = max_num_features
                
                if (min_num_features == None):
                    for num in range((length_col - 1), 0, -1):
                        print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                        feature_info = {}
                        features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                        
                        strategy = strategy.lower()
                        if strategy in types2: 
                            self.__x = features
                            
                        elif strategy in types1: 
                            self.__x = features["Dataset ---> Features Selected"]
                           
                            
                        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                        
                        self.__multiple_classifier_models = {}
                        store_data = []
                        for algorithms in classifiers:
                            self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                            info = [
                                num,
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"], 
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"], 
                                ]
                            store_data.append(info)
                          
                        dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                        
                        feature_info[f"{num} Feature(s) Selected"] = features
                        feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                        store[f"{num}"] = {}
                        store[f"{num}"]["Feature Info"] = feature_info
                        store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                        
                        dataset2 = dataset_classifiers
                        dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                        
                        
                elif (min_num_features != None) and isinstance(min_num_features, int):
                    if (min_num_features <= length_col):
                        for num in range(length_col, (min_num_features - 1), -1):
                            print(f"\n\n\nBuilding Classification Model with {num} Feature(s):\n")
                            feature_info = {}
                            features = SupervisedLearning.select_features(self, strategy = strategy, estimator = estimator, number_of_features = num)
                            
                            strategy = strategy.lower()
                            if strategy in types2: 
                                self.__x = features
                                
                            elif strategy in types1: 
                                self.__x = features["Dataset ---> Features Selected"]
                               
                                
                            # self.__x_train, self.__x_test, self.__y_train, self.__y_test = SupervisedLearning.split_data(self).values()
                            
                            self.__multiple_classifier_models = {}
                            store_data = []
                            for algorithms in classifiers:
                                self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"] = {"Built Model": SupervisedLearning.train_model_classifier(self, classifier = algorithms), "Prediction": SupervisedLearning.classifier_predict(self), "Evaluation": SupervisedLearning.classifier_evaluation(self, kfold = kfold, cross_validation = cv)}
                                info = [
                                    num,
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Accuracy"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Precision"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model Recall"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Model F1 Score"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Mean"], 
                                    self.__multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Evaluation"]["Cross Validation Standard Deviation"], 
                                    ]
                                store_data.append(info)
                              
                            dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                            
                            feature_info[f"{num} Feature(s) Selected"] = features
                            feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                            store[f"{num}"] = {}
                            store[f"{num}"]["Feature Info"] = feature_info
                            store[f"{num}"]["More Info"] = self.__multiple_classifier_models
                            
                            dataset2 = dataset_classifiers
                            dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                            
                    else:
                        raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

                    
            dataset_features = dataset_features.reset_index(drop = True)
            return {"Feature Metrics": dataset_features, "More Info": store}

    
    def classifier_evaluation(self, kfold: int = None, cross_validation: bool = False):
        """
        Evaluate the performance of a classification model.
    
        Parameters
        ----------
        kfold : int, optional
            Number of folds for cross-validation. If not provided, default is None.
        cross_validation : bool, default False
            Flag to indicate whether to perform cross-validation.
    
        Returns
        -------
        dict
            A dictionary containing evaluation metrics for both training and test sets.
    
        Raises
        ------
        ValueError
            - If `kfold` is provided without enabling cross-validation.
        
        AssertionError
            - If called for a regression problem.
    
        Notes
        -----
        - This method evaluates the performance of a classification model using metrics such as confusion matrix, classification report, accuracy, precision, recall, and F1 score.
        - If `kfold` is not provided, it evaluates the model on the training and test sets.
        - If `cross_validation` is set to True, cross-validation scores are also included in the result.
    
        Examples
        --------
        >>> # Create a supervised learning instance and train a classification model
        >>> model = SupervisedLearning(dataset)
        >>> model.train_model_classifier()
        >>>
        >>> # Evaluate the model
        >>> evaluation_results = model.classifier_evaluation(kfold=5, cross_validation=True)
        >>> print(evaluation_results)
    
        See Also
        --------
        - sklearn.metrics.confusion_matrix : Compute confusion matrix.
        - sklearn.metrics.classification_report : Build a text report showing the main classification metrics.
        - sklearn.metrics.accuracy_score : Accuracy classification score.
        - sklearn.metrics.precision_score : Compute the precision.
        - sklearn.metrics.recall_score : Compute the recall.
        - sklearn.metrics.f1_score : Compute the F1 score.
        - sklearn.model_selection.cross_val_score : Evaluate a score by cross-validation.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html
        - NumPy: https://numpy.org/doc/stable/
        - Matplotlib: https://matplotlib.org/stable/contents.html

        """
        if self.__split_data == True: 
            if self.classification_problem == True:
                if kfold == None and cross_validation == False:
                    training_analysis = sm.confusion_matrix(self.__y_train, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y_train, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y_train, self.__y_pred)
                    training_precision = sm.precision_score(self.__y_train, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y_train, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y_train, self.__y_pred, average='weighted')
        
                    test_analysis = sm.confusion_matrix(self.__y_test, self.__y_pred1)
                    test_class_report = sm.classification_report(self.__y_test, self.__y_pred1)
                    test_accuracy = sm.accuracy_score(self.__y_test, self.__y_pred1)
                    test_precision = sm.precision_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_recall = sm.recall_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_f1score = sm.f1_score(self.__y_test, self.__y_pred1, average='weighted')
                    self.__model_evaluation = True
                    return {
                        "Training Evaluation": {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                            },
                        "Test Evaluation": {
                            "Confusion Matrix": test_analysis,
                            "Classification Report": test_class_report,
                            "Model Accuracy": test_accuracy,
                            "Model Precision": test_precision,
                            "Model Recall": test_recall,
                            "Model F1 Score": test_f1score,
                            },
                        }
                
                elif kfold != None and cross_validation == False:
                    raise ValueError("Cross Validation must be set to True if kfold is specified.")
                    
                elif kfold == None and cross_validation == True:
                    training_analysis = sm.confusion_matrix(self.__y_train, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y_train, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y_train, self.__y_pred)
                    training_precision = sm.precision_score(self.__y_train, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y_train, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y_train, self.__y_pred, average='weighted')
            
                    test_analysis = sm.confusion_matrix(self.__y_test, self.__y_pred1)
                    test_class_report = sm.classification_report(self.__y_test, self.__y_pred1)
                    test_accuracy = sm.accuracy_score(self.__y_test, self.__y_pred1)
                    test_precision = sm.precision_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_recall = sm.recall_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_f1score = sm.f1_score(self.__y_test, self.__y_pred1, average='weighted')
                    
                    cross_val = sms.cross_val_score(self.model_classifier, self.__x_train, self.__y_train, cv = 10)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {
                        "Training Evaluation": {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                            },
                        "Test Evaluation": {
                            "Confusion Matrix": test_analysis,
                            "Classification Report": test_class_report,
                            "Model Accuracy": test_accuracy,
                            "Model Precision": test_precision,
                            "Model Recall": test_recall,
                            "Model F1 Score": test_f1score,
                            },
                        "Cross Validation": {
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }
                        }
                
                elif kfold != None and cross_validation == True:
                    training_analysis = sm.confusion_matrix(self.__y_train, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y_train, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y_train, self.__y_pred)
                    training_precision = sm.precision_score(self.__y_train, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y_train, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y_train, self.__y_pred, average='weighted')
            
                    test_analysis = sm.confusion_matrix(self.__y_test, self.__y_pred1)
                    test_class_report = sm.classification_report(self.__y_test, self.__y_pred1)
                    test_accuracy = sm.accuracy_score(self.__y_test, self.__y_pred1)
                    test_precision = sm.precision_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_recall = sm.recall_score(self.__y_test, self.__y_pred1, average='weighted')
                    test_f1score = sm.f1_score(self.__y_test, self.__y_pred1, average='weighted')
                    
                    cross_val = sms.cross_val_score(self.model_classifier, self.__x_train, self.__y_train, cv = kfold)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {
                        "Training Evaluation": {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                            },
                        "Test Evaluation": {
                            "Confusion Matrix": test_analysis,
                            "Classification Report": test_class_report,
                            "Model Accuracy": test_accuracy,
                            "Model Precision": test_precision,
                            "Model Recall": test_recall,
                            "Model F1 Score": test_f1score,
                            },
                        "Cross Validation": {
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }
                        }
            
            else:
                raise AssertionError("You can not use a classification evaluation function for a regression problem.")
        
        else:
            if self.classification_problem == True:
                if kfold == None and cross_validation == False:
                    training_analysis = sm.confusion_matrix(self.__y, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y, self.__y_pred)
                    training_precision = sm.precision_score(self.__y, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y, self.__y_pred, average='weighted')
                    self.__model_evaluation = True
                    return {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                           }
                
                elif kfold != None and cross_validation == False:
                    raise ValueError("Cross Validation must be set to True if kfold is specified.")
                    
                elif kfold == None and cross_validation == True:
                    training_analysis = sm.confusion_matrix(self.__y, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y, self.__y_pred)
                    training_precision = sm.precision_score(self.__y, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y, self.__y_pred, average='weighted')
                    
                    cross_val = sms.cross_val_score(self.model_classifier, self.__x_train, self.__y, cv = 10)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }
                
                elif kfold != None and cross_validation == True:
                    training_analysis = sm.confusion_matrix(self.__y, self.__y_pred)
                    training_class_report = sm.classification_report(self.__y, self.__y_pred)
                    training_accuracy = sm.accuracy_score(self.__y, self.__y_pred)
                    training_precision = sm.precision_score(self.__y, self.__y_pred, average='weighted')
                    training_recall = sm.recall_score(self.__y, self.__y_pred, average='weighted')
                    training_f1score = sm.f1_score(self.__y, self.__y_pred, average='weighted')
                    
                    cross_val = sms.cross_val_score(self.model_classifier, self.__x_train, self.__y, cv = kfold)    
                    score_mean = round((cross_val.mean() * 100), 2)
                    score_std_dev = round((cross_val.std() * 100), 2)
                    self.__model_evaluation = True
                    return {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1score,
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }

            else:
                raise AssertionError("You can not use a classification evaluation function for a regression problem.")
                
        
    def classifier_model_testing(self, variables_values: list, scaling: bool = False):
        """
        Test a classification model with given input variables.
    
        Parameters
        ----------
        variables_values : list
            A list containing values for input variables used to make predictions.
        scaling : bool, default False
            Flag to indicate whether to scale input variables. If True, the method
            assumes that the model was trained on scaled data.
    
        Returns
        -------
        array
            Predicted labels for the given input variables.
    
        Raises
        ------
        AssertionError
            If called for a regression problem.
    
        Notes
        -----
        - This method is used to test a classification model by providing values for the input variables and obtaining predicted labels. 
        - If scaling is required, it is important to set the `scaling` parameter to True.
    
        Examples
        --------
        >>> # Create a supervised learning instance and train a classification model
        >>> model = SupervisedLearning(dataset)
        >>> model.train_model_classifier()
        >>>
        >>> # Provide input variables for testing
        >>> input_data = [value1, value2, value3]
        >>>
        >>> # Test the model
        >>> predicted_labels = model.classifier_model_testing(input_data, scaling=True)
        >>> print(predicted_labels)
    
        See Also
        --------
        - sklearn.preprocessing.StandardScaler : Standardize features by removing the mean and scaling to unit variance.
        - sklearn.neighbors.KNeighborsClassifier : K-nearest neighbors classifier.
        - sklearn.ensemble.RandomForestClassifier : A meta-estimator that fits a number of decision tree classifiers on various sub-samples of the dataset.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        """
        if self.classification_problem == True:
            if scaling == False:
                prediction = self.model_classifier.predict([variables_values])
                self.__model_testing = True
                return prediction
            
            elif scaling == True:
                variables_values = self.__scaler.transform([variables_values])
                prediction = self.model_classifier.predict(variables_values)
                self.__model_testing = True
                return prediction
        
        else:
            raise AssertionError("You can't test a classification problem with this function.")
    
    
    def classifier_graph(self, classifier, cmap_train = "viridis", cmap_test = "viridis", size_train_marker: float = 10, size_test_marker: float = 10, resolution=100):
        """
        Visualize the decision boundaries of a classification model.
    
        Parameters
        ----------
        classifier : scikit-learn classifier object
            The trained classification model.
        cmap_train : str, default "viridis"
            Colormap for the training set.
        cmap_test : str, default "viridis"
            Colormap for the test set.
        size_train_marker : float, default 10
            Marker size for training set points.
        size_test_marker : float, default 10
            Marker size for test set points.
        resolution : int, default 100
            Resolution of the decision boundary plot.
    
        Raises
        ------
        AssertionError
            If called for a regression problem.
        ValueError
            If the number of features is not 2.
    
        Notes
        -----
        - This method visualizes the decision boundaries of a classification model by plotting the regions where the model predicts different classes. 
        - It supports both training and test sets, with different markers and colormaps for each.
    
        Examples
        --------
        >>> # Create a supervised learning instance and train a classification model
        >>> model = SupervisedLearning(dataset)
        >>> model.train_model_classifier()
        >>>
        >>> # Visualize the decision boundaries
        >>> model.classifier_graph(classifier=model.model_classifier)
    
        See Also
        --------
        - sklearn.preprocessing.LabelEncoder : Encode target labels.
        - sklearn.linear_model.LogisticRegression : Logistic Regression classifier.
        - sklearn.svm.SVC : Support Vector Classification.
        - sklearn.tree.DecisionTreeClassifier : Decision Tree classifier.
        - sklearn.ensemble.RandomForestClassifier : Random Forest classifier.
        - sklearn.neighbors.KNeighborsClassifier : K-Nearest Neighbors classifier.
        - matplotlib.pyplot.scatter : Plot scatter plots.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/supervised_learning.html
        - Matplotlib: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill.html

        """
        if self.__split_data == True:
            if self.classification_problem == True:
                columns = [col for col in self.__x_train.columns]
                
                if len(columns) == 2:
                    feature1 = self.__x_train.iloc[:, 0].name
                    feature2 = self.__x_train.iloc[:, 1].name
                    
                    le = sp.LabelEncoder()
                    self.__y_train_encoded = le.fit_transform(self.__y_train)
    
                    if isinstance(self.__x_train, pd.DataFrame):
                        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((self.__x_train.iloc[:, 0].min() - (self.__x_train.iloc[:, 0].min() / 8)), (self.__x_train.iloc[:, 0].max() + (self.__x_train.iloc[:, 0].max() / 8)), resolution),
                                                                    np.linspace((self.__x_train.iloc[:, 1].min() - (self.__x_train.iloc[:, 1].min() / 8)), (self.__x_train.iloc[:, 1].max() + (self.__x_train.iloc[:, 1].max() / 8)), resolution))
                    elif isinstance(self.__x_train, np.ndarray):
                        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((self.__x_train.iloc[:, 0].min() - (self.__x_train.iloc[:, 0].min() / 8)), (self.__x_train.iloc[:, 0].max() + (self.__x_train.iloc[:, 0].max() / 8)), resolution),
                                                                    np.linspace((self.__x_train.iloc[:, 1].min() - (self.__x_train.iloc[:, 1].min() / 8)), (self.__x_train.iloc[:, 1].max() + (self.__x_train.iloc[:, 1].max() / 8)), resolution))
                    else:
                        raise ValueError("Unsupported input type for self.__x_train. Use either Pandas DataFrame or NumPy array.")
    
                    grid_points_train = np.c_[x1_vals_train.ravel(), x2_vals_train.ravel()]
                    predictions_train = classifier.predict(grid_points_train)
                    predictions_train = le.inverse_transform(predictions_train)
    
                    plt.figure(figsize = (15, 10))
                    
                    plt.contourf(x1_vals_train, x2_vals_train, le.transform(predictions_train).reshape(x1_vals_train.shape), alpha=0.3, cmap = cmap_train)
                    if isinstance(self.__x_train, pd.DataFrame):
                        plt.scatter(self.__x_train.iloc[:, 0], self.__x_train.iloc[:, 1], c=self.__y_train_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
                    elif isinstance(self.__x_train, np.ndarray):
                        plt.scatter(self.__x_train[:, 0], self.__x_train[:, 1], c=self.__y_train_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
                    plt.title(f"{classifier.__class__.__name__} Training Classification Graph")
                    plt.xlabel(feature1)
                    plt.ylabel(feature2)
                    plt.tight_layout()
                    plt.show()
    
                    if self.__x_test is not None and self.__y_test is not None:
                        plt.figure(figsize = (15, 10))
                        x1_vals_test, x2_vals_test = np.meshgrid(np.linspace((self.__x_test.iloc[:, 0].min() - (self.__x_test.iloc[:, 0].min() / 8)), (self.__x_test.iloc[:, 0].max() + (self.__x_test.iloc[:, 0].max() / 8)), resolution),
                                                                  np.linspace((self.__x_test.iloc[:, 1].min() - (self.__x_test.iloc[:, 1].min() / 8)), (self.__x_test.iloc[:, 1].max() + (self.__x_test.iloc[:, 1].max() / 8)), resolution))
    
                        grid_points_test = np.c_[x1_vals_test.ravel(), x2_vals_test.ravel()]
                        predictions_test = classifier.predict(grid_points_test)
                        predictions_test = le.inverse_transform(predictions_test)
    
                        plt.contourf(x1_vals_test, x2_vals_test, le.transform(predictions_test).reshape(x1_vals_test.shape), alpha=0.3, cmap=cmap_test)
    
                        if isinstance(self.__x_test, pd.DataFrame):
                            plt.scatter(self.__x_test.iloc[:, 0], self.__x_test.iloc[:, 1], c=le.transform(self.__y_test), cmap=cmap_test, edgecolors='k', s=size_test_marker, marker='o')
                        elif isinstance(self.__x_test, np.ndarray):
                            plt.scatter(self.__x_test[:, 0], self.__x_test[:, 1], c=le.transform(self.__y_test), cmap=cmap_test, edgecolors='k', s=size_test_marker, marker='o')
    
                        plt.title(f"{classifier.__class__.__name__} Test Classification Graph")
                        plt.xlabel(feature1)
                        plt.ylabel(feature2)
                        plt.tight_layout()
                        plt.show()
                    
                else:
                    raise ValueError(f"Visualization needs a maximum of 2 features for the independent variables. {len(self.__x_train.columns)} given.")
    
            else:
                raise AssertionError("You can not use a classification graph for a regression problem.")
                
        else:
            if self.classification_problem == True:
                columns = [col for col in self.__x.columns]
                
                if len(columns) == 2:
                    feature1 = self.__x.iloc[:, 0].name
                    feature2 = self.__x.iloc[:, 1].name
                    
                    le = sp.LabelEncoder()
                    self.__y_encoded = le.fit_transform(self.__y)

                    if isinstance(self.__x, pd.DataFrame):
                        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((self.__x.iloc[:, 0].min() - (self.__x.iloc[:, 0].min() / 8)), (self.__x.iloc[:, 0].max() + (self.__x.iloc[:, 0].max() / 8)), resolution),
                                                                    np.linspace((self.__x.iloc[:, 1].min() - (self.__x.iloc[:, 1].min() / 8)), (self.__x.iloc[:, 1].max() + (self.__x.iloc[:, 1].max() / 8)), resolution))
                    elif isinstance(self.__x, np.ndarray):
                        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((self.__x.iloc[:, 0].min() - (self.__x.iloc[:, 0].min() / 8)), (self.__x.iloc[:, 0].max() + (self.__x.iloc[:, 0].max() / 8)), resolution),
                                                                    np.linspace((self.__x.iloc[:, 1].min() - (self.__x.iloc[:, 1].min() / 8)), (self.__x.iloc[:, 1].max() + (self.__x.iloc[:, 1].max() / 8)), resolution))
                    else:
                        raise ValueError("Unsupported input type for self.__x. Use either Pandas DataFrame or NumPy array.")

                    grid_points_train = np.c_[x1_vals_train.ravel(), x2_vals_train.ravel()]
                    predictions_train = classifier.predict(grid_points_train)
                    predictions_train = le.inverse_transform(predictions_train)

                    plt.figure(figsize = (15, 10))
                    
                    plt.contourf(x1_vals_train, x2_vals_train, le.transform(predictions_train).reshape(x1_vals_train.shape), alpha=0.3, cmap = cmap_train)
                    if isinstance(self.__x, pd.DataFrame):
                        plt.scatter(self.__x.iloc[:, 0], self.__x.iloc[:, 1], c=self.__y_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
                    elif isinstance(self.__x, np.ndarray):
                        plt.scatter(self.__x[:, 0], self.__x[:, 1], c=self.__y_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
                    plt.title(f"{classifier.__class__.__name__} Training Classification Graph")
                    plt.xlabel(feature1)
                    plt.ylabel(feature2)
                    plt.tight_layout()
                    plt.show()
                    
                else:
                    raise ValueError(f"Visualization needs a maximum of 2 features for the independent variables. {len(self.__x.columns)} given.")

            else:
                raise AssertionError("You can not use a classification graph for a regression problem.")

    def numerical_to_categorical(self, column):
        """
        Convert numerical columns to categorical in the dataset.
    
        Parameters
        ----------
        column : str, list, or tuple
            The name of the column or a list/tuple of column names to be converted.
    
        Returns
        -------
        DataFrame
            A new DataFrame with specified numerical columns converted to categorical.
    
        Notes
        -----
        - This method converts numerical columns in the dataset to categorical type. 
        - It is useful when dealing with features that represent categories or labels but are encoded as numerical values.
    
        Examples
        --------
        >>> # Create a supervised learning instance and load a dataset
        >>> data = SupervisedLearning(dataset)
        >>>
        >>> # Convert a single numerical column to categorical
        >>> data.numerical_to_categorical("numeric_column")
        >>>
        >>> # Convert multiple numerical columns to categorical
        >>> data.numerical_to_categorical(["numeric_col1", "numeric_col2"])
    
        See Also
        --------
        - pandas.DataFrame.astype : Cast a pandas object to a specified dtype.
        - NumPy: https://numpy.org/doc/stable/
        
        References
        ----------
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
        - NumPy: https://numpy.org/doc/stable/user/basics.types.html

        """
        if isinstance(column, list):
            for items in column:
                self.__data[items] = self.__data[items].astype("object")
        
        elif isinstance(column, str):
            self.__data[column] = self.__data[column].astype("object")
            
        elif isinstance(column, tuple):
            for items in column:
                self.__data[items] = self.__data[items].astype("object")
        
        return self.__data
    
    
    def categorical_to_datetime(self, column):
        """
        Convert specified categorical columns to datetime format.
    
        Parameters
        ----------
        column : str, list, or tuple
            The column or columns to be converted to datetime format.
    
        Returns
        -------
        DataFrame
            The DataFrame with specified columns converted to datetime.
    
        Notes
        -----
        This method allows for the conversion of categorical columns containing date or time information to the datetime format.
    
        Examples
        --------
        >>> # Create a supervised learning instance and convert a single column
        >>> model = SupervisedLearning(dataset)
        >>> model.categorical_to_datetime('date_column')
    
        >>> # Convert multiple columns
        >>> model.categorical_to_datetime(['start_date', 'end_date'])
    
        >>> # Convert a combination of columns using a tuple
        >>> model.categorical_to_datetime(('start_date', 'end_date'))
    
        See Also
        --------
        - pandas.to_datetime : Convert argument to datetime.
    
        References
        ----------
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
        - NumPy: https://numpy.org/doc/stable/

        """
        if isinstance(column, list):
            for items in column:
                self.__data[items] = pd.to_datetime(self.__data[items])
        
        elif isinstance(column, str):
            self.__data[column] = pd.to_datetime(self.__data[column])
            
        elif isinstance(column, tuple):
            for items in column:
                self.__data[items] = pd.to_datetime(self.__data[items])
        
        return self.__data
    
    
    def extract_date_features(self, datetime_column, hrs_mins_sec: bool = False):
        """
        Extracts date-related features from a datetime column.
    
        Parameters
        ----------
        datetime_column : str, list, or tuple
            The name of the datetime column or a list/tuple of datetime columns.
        hrs_mins_sec : bool, default False
            Flag indicating whether to include hour, minute, and second features.
    
        Returns
        -------
        DataFrame
            A DataFrame with additional columns containing extracted date features.
    
        Notes
        -----
        - This method extracts date-related features such as day, month, year, quarter, and day of the week from the specified datetime column(s). 
        - If `hrs_mins_sec` is set to True, it also includes hour, minute, and second features.
    
        Examples
        --------
        >>> # Create a supervised learning instance and extract date features
        >>> model = SupervisedLearning(dataset)
        >>> date_columns = ['DateOfBirth', 'TransactionDate']
        >>> model.extract_date_features(date_columns, hrs_mins_sec=True)
        >>>
        >>> # Access the DataFrame with additional date-related columns
        >>> processed_data = model.get_dataset()
    
        See Also
        --------
        - pandas.Series.dt : Accessor object for datetime properties.
        - sklearn.preprocessing.OneHotEncoder : Encode categorical integer features using a one-hot encoding.
        - sklearn.model_selection.train_test_split : Split arrays or matrices into random train and test subsets.
        - matplotlib.pyplot : Plotting library for creating visualizations.
    
        References
        ----------
        - Pandas Documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        - Scikit-learn Documentation: https://scikit-learn.org/stable/modules/preprocessing.html
        - Matplotlib Documentation: https://matplotlib.org/stable/contents.html

        """
        
        if hrs_mins_sec == False:
            if isinstance(datetime_column, list):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = self.__data[items].day
                    self.__data[f"Month_{items}"] = self.__data[items].month
                    self.__data[f"Year_{items}"] = self.__data[items].year
                    self.__data[f"Quarter_{items}"] = self.__data[items].quarter
                    self.__data[f"Day_of_Week_{items}"] = self.__data[items].day_of_week
            
            elif isinstance(datetime_column, str):
                self.__data["Day"] = self.__data[datetime_column].day
                self.__data["Month"] = self.__data[datetime_column].month
                self.__data["Year"] = self.__data[datetime_column].year
                self.__data["Quarter"] = self.__data[datetime_column].quarter
                self.__data["Day_of_Week"] = self.__data[datetime_column].day_of_week
                
            elif isinstance(datetime_column, tuple):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = self.__data[items].day
                    self.__data[f"Month_{items}"] = self.__data[items].month
                    self.__data[f"Year_{items}"] = self.__data[items].year
                    self.__data[f"Quarter_{items}"] = self.__data[items].quarter
                    self.__data[f"Day_of_Week_{items}"] = self.__data[items].day_of_week
            
            return self.__data
        
        elif hrs_mins_sec == False:
            if isinstance(datetime_column, list):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = self.__data[items].day
                    self.__data[f"Month_{items}"] = self.__data[items].month
                    self.__data[f"Year_{items}"] = self.__data[items].year
                    self.__data[f"Quarter_{items}"] = self.__data[items].quarter
                    self.__data[f"Day_of_Week_{items}"] = self.__data[items].day_of_week
                    self.__data[f"Hour_{items}"] = self.__data[items].hour
                    self.__data[f"Minutes_{items}"] = self.__data[items].minute
                    self.__data[f"Seconds_{items}"] = self.__data[items].second                   
                    
            elif isinstance(datetime_column, str):
                self.__data["Day"] = self.__data[datetime_column].day
                self.__data["Month"] = self.__data[datetime_column].month
                self.__data["Year"] = self.__data[datetime_column].year
                self.__data["Quarter"] = self.__data[datetime_column].quarter
                self.__data["Day_of_Week"] = self.__data[datetime_column].day_of_week
                self.__data[f"Hour_{items}"] = self.__data[items].hour
                self.__data[f"Minutes_{items}"] = self.__data[items].minute
                self.__data[f"Seconds_{items}"] = self.__data[items].second  
                
            elif isinstance(datetime_column, tuple):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = self.__data[items].day
                    self.__data[f"Month_{items}"] = self.__data[items].month
                    self.__data[f"Year_{items}"] = self.__data[items].year
                    self.__data[f"Quarter_{items}"] = self.__data[items].quarter
                    self.__data[f"Day_of_Week_{items}"] = self.__data[items].day_of_week
                    self.__data[f"Hour_{items}"] = self.__data[items].hour
                    self.__data[f"Minutes_{items}"] = self.__data[items].minute
                    self.__data[f"Seconds_{items}"] = self.__data[items].second  
            
            return self.__data
                
    
    def column_binning(self, column, number_of_bins: int = 10, labels: list or tuple = None):
        """
        Apply binning to specified columns in the dataset.
    
        Parameters
        ----------
        column : str or list or tuple
            The column(s) to apply binning to.
        number_of_bins : int, default 10
            The number of bins to use.
        labels: list or tuple, default = None
            Name the categorical columns created by giving them labels.
    
        Returns
        -------
        DataFrame
            The dataset with specified column(s) binned.
    
        Notes
        -----
        - This method uses the `pd.cut` function to apply binning to the specified column(s).
        - Binning is a process of converting numerical data into categorical data.
    
        Examples
        --------
        >>> # Create a supervised learning instance and perform column binning
        >>> model = SupervisedLearning(dataset)
        >>> model.column_binning(column="Age", number_of_bins=5)
        >>> model.column_binning(column=["Salary", "Experience"], number_of_bins=10)
    
        See Also
        --------
        - pandas.cut : Bin values into discrete intervals.
        - pandas.DataFrame : Data structure for handling the dataset.
    
        References
        ----------
        - Pandas Documentation: https://pandas.pydata.org/pandas-docs/stable/index.html

        """
        if isinstance(column, list):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = labels)
        
        elif isinstance(column, str):
            self.__data[column] = pd.cut(self.__data[column], bins = number_of_bins, labels = labels)
            
        elif isinstance(column, tuple):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = labels)
        
        return self.__data
    
    
    def fix_unbalanced_dataset(self, sampler: str, k_neighbors: int = None, random_state: int = None):
        """
        Apply techniques to address class imbalance in the dataset.
    
        Parameters
        ----------
        sampler : str
            The resampling technique. Options: "SMOTE", "RandomOverSampler", "RandomUnderSampler".
        k_neighbors : int, optional
            The number of nearest neighbors to use in the SMOTE algorithm.
        random_state : int, optional
            Seed for reproducibility.
    
        Returns
        -------
        dict
            A dictionary containing the resampled training data.
    
        Raises
        ------
        ValueError
            - If `k_neighbors` is specified for a sampler other than "SMOTE".
    
        Notes
        -----
        - This method addresses class imbalance in the dataset using various resampling techniques.
        - Supported samplers include SMOTE, RandomOverSampler, and RandomUnderSampler.
    
        Examples
        --------
        >>> # Create a supervised learning instance and fix unbalanced dataset
        >>> model = SupervisedLearning(dataset)
        >>> model.fix_unbalanced_dataset(sampler="SMOTE", k_neighbors=5)
    
        See Also
        --------
        - imblearn.over_sampling.SMOTE : Synthetic Minority Over-sampling Technique.
        - imblearn.over_sampling.RandomOverSampler : Random over-sampling.
        - imblearn.under_sampling.RandomUnderSampler : Random under-sampling.
        - sklearn.impute.SimpleImputer : Simple imputation for handling missing values.
    
        References
        ----------
        - Imbalanced-learn Documentation: https://imbalanced-learn.org/stable/
        - Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
    
        """
        if self.__split_data == True: 
            if random_state == None:
                if sampler == "SMOTE" and k_neighbors != None:
                    technique = ios.SMOTE(random_state = 0, k_neighbors = k_neighbors)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
                elif sampler == "SMOTE" and k_neighbors == None:
                    technique = ios.SMOTE(random_state = 0)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                elif sampler == "RandomOverSampler" and k_neighbors == None:
                    technique = ios.RandomOverSampler(random_state = 0)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                elif sampler == "RandomUnderSampler" and k_neighbors == None:
                    technique = ius.RandomUnderSampler(random_state = 0)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                else:
                    raise ValueError("k_neighbors works with only the SMOTE algorithm.")
                
                return {"Training X": self.__x_train, "Training Y": self.__y_train}
            
            elif random_state != None:
                if sampler == "SMOTE" and k_neighbors != None:
                    technique = ios.SMOTE(random_state = random_state, k_neighbors = k_neighbors)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
                elif sampler == "SMOTE" and k_neighbors == None:
                    technique = ios.SMOTE(random_state = random_state)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                elif sampler == "RandomOverSampler" and k_neighbors == None:
                    technique = ios.RandomOverSampler(random_state = random_state)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                elif sampler == "RandomUnderSampler" and k_neighbors == None:
                    technique = ius.RandomUnderSampler(random_state = random_state)
                    self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                    
                else:
                    raise ValueError("k_neighbors works with only the SMOTE algorithm.")
                
                return {"Training X": self.__x_train, "Training Y": self.__y_train}
        else:
            if random_state == None:
                if sampler == "SMOTE" and k_neighbors != None:
                    technique = ios.SMOTE(random_state = 0, k_neighbors = k_neighbors)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                
                elif sampler == "SMOTE" and k_neighbors == None:
                    technique = ios.SMOTE(random_state = 0)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                elif sampler == "RandomOverSampler" and k_neighbors == None:
                    technique = ios.RandomOverSampler(random_state = 0)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                elif sampler == "RandomUnderSampler" and k_neighbors == None:
                    technique = ius.RandomUnderSampler(random_state = 0)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                else:
                    raise ValueError("k_neighbors works with only the SMOTE algorithm.")
                
                return {"X": self.__x, "Y": self.__y}

            elif random_state != None:
                if sampler == "SMOTE" and k_neighbors != None:
                    technique = ios.SMOTE(random_state = random_state, k_neighbors = k_neighbors)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                
                elif sampler == "SMOTE" and k_neighbors == None:
                    technique = ios.SMOTE(random_state = random_state)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                elif sampler == "RandomOverSampler" and k_neighbors == None:
                    technique = ios.RandomOverSampler(random_state = random_state)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                elif sampler == "RandomUnderSampler" and k_neighbors == None:
                    technique = ius.RandomUnderSampler(random_state = random_state)
                    self.__x, self.__y = technique.fit_resample(self.__x, self.__y)
                    
                else:
                    raise ValueError("k_neighbors works with only the SMOTE algorithm.")
                
                return {"X": self.__x, "Y": self.__y}

    
    def replace_values(self, replace: int or float or str or list or tuple or dict, new_value: int or float or str or list or tuple):
        """
        Replace specified values in the dataset.
    
        Parameters
        ----------
        replace : int or float or str or list or tuple or dict
            The value or set of values to be replaced.
        new_value : int or float or str or list or tuple
            The new value or set of values to replace the existing ones.
    
        Returns
        -------
        pd.DataFrame
            A dataframe containing the modified dataset with replaced values.
    
        Raises
        ------
        ValueError
            - If `replace` is a string, integer, or float, and `new_value` is not a string, integer, or float.
            - If `replace` is a list or tuple, and `new_value` is not a string, integer, float, list, or tuple.
            - If `replace` is a dictionary, and `new_value` is not specified.
    
        Notes
        -----
        This method replaces specified values in the dataset. The replacement can be done for a single value, a list of values, or using a dictionary for multiple replacements.
    
        Examples
        --------
        >>> # Create a supervised learning instance and replace values
        >>> data = SupervisedLearning(dataset)
        >>>
        >>> # Replace a single value
        >>> replaced_values = data.replace_values(0, -1)
        >>>
        >>> # Replace multiple values using a dictionary
        >>> replaced_values = data.replace_values({'Male': 1, 'Female': 0})
    
        See Also
        --------
        - pandas.DataFrame.replace : Replace values in a DataFrame.
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        - Matplotlib: https://matplotlib.org/stable/contents.html
    
        References
        ----------
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html
        - NumPy: https://numpy.org/doc/stable/
    
        """
    
        if isinstance(replace, str) or isinstance(new_value, int) or isinstance(new_value, float):
            if isinstance(new_value, str) or isinstance(new_value, int) or isinstance(new_value, float):
                self.__data.replace(to_replace = replace, value = new_value, inplace = True)
            else:
                raise ValueError("If replace is a string, integer, or float, then new value must be either a string, integer, or float.")
        
        elif isinstance(replace, list) or isinstance(replace, tuple):
            if isinstance(new_value, str) or isinstance(new_value, int) or isinstance(new_value, float):
                self.__data.replace(to_replace = replace, value = new_value, inplace = True)
            elif isinstance(new_value, list) or isinstance(new_value, tuple):
                for word, new in zip(replace, new_value):
                    self.__data.replace(to_replace = word, value = new, inplace = True)
            else:
                raise ValueError("If replace is a list or tuple, then value can be any of int, str, float, list, or tuple.")
                    
        elif isinstance(replace, dict):
            self.__data.replace(to_replace = replace, value = new_value, inplace = True)
            
        else:
            raise ValueError("Check your input arguments for the parameters: replace and new_value")
        
        return self.__data
     
    
    def sort_values(self, column: str or list, ascending: bool = True, reset_index: bool = False):
        """
        Sort the dataset based on specified columns.

        Parameters
        ----------
        column : str or list
            The column(s) to sort the dataset.
        ascending : bool, optional
            Whether to sort in ascending order. Default is True.
        reset_index : bool, optional
            Whether to reset the index after sorting. Default is False.

        Returns
        -------
        pd.DataFrame
            The dataset sorted according to the specified column or columns.

        Examples
        --------
        >>> # Create a supervised learning instance and sort the dataset
        >>> data = SupervisedLearning(dataset)
        >>> sorted_data = data.sort_values("column_name", 
        >>>                                ascending=False, 
        >>>                                reset_index=True)
        >>> print(sorted_data)
        
        See Also
        --------
        - pandas.DataFrame.sort_values
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        """
        
        if isinstance(column, str) or isinstance(column, list):
            self.__data.sort_values(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
        
        return self.__data
    
    
    def set_index(self, column: str or list):
        """
        Set the index of the dataset.

        Parameters
        ----------
        column : str or list
            The column(s) to set as the index.

        Returns
        -------
        pd.DataFrame
            The dataset with the index set to the specified column or columns.

        Examples
        --------
        >>> # Create a supervised learning instance and set the index
        >>> data = SupervisedLearning(dataset)
        >>> index_set_data = data.set_index("column_name")
        >>> print(index_set_data)
        
        See Also
        --------
        - pandas.DataFrame.set_index
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        """
        if isinstance(column, str) or isinstance(column, list):
            self.__data = self.__data.set_index(column)
        
        return self.__data
    
      
    def sort_index(self, column: str or list, ascending: bool = True, reset_index: bool = False):
        """
        Sort the dataset based on the index.

        Parameters
        ----------
        column : str or list
            The index column(s) to sort the dataset.
        ascending : bool, optional
            Whether to sort in ascending order. Default is True.
        reset_index : bool, optional
            Whether to reset the index after sorting. Default is False.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the sorted dataset.

        Examples
        --------
        >>> # Create a supervised learning instance and sort the dataset based on index
        >>> data = SupervisedLearning(dataset)
        >>> sorted_index_data = data.sort_index("index_column", 
        >>>                                     ascending=False, 
        >>>                                     reset_index=True)
        >>> print(sorted_index_data)
        
        See Also
        --------
        - pandas.DataFrame.sort_index
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        """
        if isinstance(column, str) or isinstance(column, list):
            self.__data.sort_index(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
        
        return self.__data
     
    
    def rename_columns(self, old_column: str or list, new_column: str or list):
        """
        Rename columns in the dataset.

        Parameters
        ----------
        old_column : str or list
            The old column name(s) to be renamed.
        new_column : str or list
            The new column name(s).

        Returns
        -------
        pd.DataFrame
            A dataframe containing the modified dataset with the column name(s) changed.

        Examples
        --------
        >>> # Create a supervised learning instance and rename columns
        >>> data = SupervisedLearning(dataset)
        >>> renamed_data = data.rename_columns("old_column_name", "new_column_name")
        >>> print(renamed_data)
        
        See Also
        --------
        - pandas.DataFrame.rename_columns
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        """
        if isinstance(old_column, str) and isinstance(new_column, str):
            self.__data.rename({old_column: new_column}, axis = 1, inplace = True)
            
        elif isinstance(old_column, list) and isinstance(new_column, list):
            self.__data.rename({key:value for key, value in zip(old_column, new_column)}, axis = 1, inplace = True)
        
        return self.__data
       
    
    def reset_index(self, drop_index_after_reset: bool = False):
        """
        Reset the index of the dataset.

        Parameters
        ----------
        drop_index_after_reset : bool, optional
            Whether to drop the old index after resetting. Default is False.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the modified dataset with the index reset.

        Examples
        --------
        >>> # Create a supervised learning instance and reset the index
        >>> data = SupervisedLearning(dataset)
        >>> reset_index_data = data.reset_index(drop_index_after_reset=True)
        >>> print(reset_index_data)

        See Also
        --------
        - pandas.DataFrame.reset_index
        - NumPy: https://numpy.org/doc/stable/
        - Scikit-learn: https://scikit-learn.org/stable/modules/preprocessing.html#imputation
        """
        self.__data.reset_index(drop = drop_index_after_reset, inplace = True)
        
        return self.__data
    
      
    def filter_data(self, column: str or list or tuple, operation: str or list or tuple = None, value: int or float or str or list or tuple = None):
        """
        Filter data based on specified conditions.
    
        Parameters
        ----------
        - column: str or list or tuple
            The column or columns to filter.
        - operation: str or list or tuple, optional
            The operation or list of operations to perform. Supported operations: 
            'greater than', 'less than', 'equal to', 'greater than or equal to', 
            'less than or equal to', 'not equal to', '>', '<', '==', '>=', '<=', '!='.
            Default is None.
        - value: int or float or str or list or tuple, optional
            The value or list of values to compare against. Default is None.
    
        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
    
        Raises
        ------
        - ValueError: If input parameters are invalid or inconsistent.
    
        Example
        -------
        >>> # Create a supervised learning instance and sort the dataset
        >>> data = SupervisedLearning(dataset)
        >>>
        >>> # Filter data where 'column' is greater than 5
        >>> filter_data = data.filter_data(column='column', 
        >>>                                operation='>', 
        >>>                                value=5)
        >>>
        >>> # Filter data where 'column1' is less than or equal to 10 and 'column2' is not equal to 'value'
        >>> filter_data = data.filter_data(column=['column1', 'column2'], 
        >>>                                operation=['<=', '!='], 
        >>>                               value=[10, 'value'])
    
        References
        ----------
        - Pandas library: https://pandas.pydata.org
    
        """
        
        possible_operations = ['greater than', 'less than', 'equal to', 'greater than or equal to', 'less than or equal to', 'not equal to', '>', '<', '==', '>=', '<=', '!=']
        if column != None:
            if isinstance(column, str):
                if isinstance(value, int) or isinstance(value, float):
                    if isinstance(operation, str):
                        if operation.lower() not in possible_operations:
                            raise ValueError(f"This operation is not supported. Please use the following: {possible_operations}")
                            
                        elif (operation.lower() == 'greater than' or operation == '>'):
                            condition = self.__data[column] > value
                            self.__data = self.__data[condition]
                            
                        elif (operation.lower() == 'less than' or operation == '<'):
                            condition = self.__data[column] < value
                            self.__data = self.__data[condition]
                            
                        elif (operation.lower() == 'equal to' or operation == '=='):
                            condition = self.__data[column] == value
                            self.__data = self.__data[condition]
                            
                        elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                            condition = self.__data[column] >= value
                            self.__data = self.__data[condition]
                            
                        elif (operation.lower() == 'less than or equal to' or operation == '<='):
                            condition = self.__data[column] <= value
                            self.__data = self.__data[condition]
                            
                        elif (operation.lower() == 'not equal to' or operation == '!='):
                            condition = self.__data[column] != value
                            self.__data = self.__data[condition]
                            
                    elif isinstance(operation, list) or isinstance(operation, int) or isinstance(operation, float) or isinstance(operation, tuple):
                        raise ValueError("When column is set to string and value is set to either float, int, or string. Operation can not be a list or tuple. Must be set to string")
                       
                        
                       
                elif isinstance(value, str):
                    if (operation.lower() == 'equal to' or operation == '=='):
                        condition = self.__data[column] == value
                        self.__data = self.__data[condition]
                        
                    elif (operation.lower() == 'not equal to' or operation == '!='):
                        condition = self.__data[column] != value
                        self.__data = self.__data[condition]  
                        
                    else:
                        raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                        
                        
                        
                elif isinstance(value, list) or isinstance(value, tuple):
                    if isinstance(operation, str):
                        raise ValueError("Length of values should be same as length of available operations to perform")
                                
                    elif isinstance(operation, list) or isinstance(operation, tuple):
                        for item, symbol in zip(value, operation):
                            if isinstance(item, str):
                                if (symbol.lower() == 'equal to' or symbol == '=='):
                                    condition = self.__data[column] == item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                    condition = self.__data[column] != item
                                    self.__data = self.__data[condition]  
                                    
                                else:
                                    raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                                    
                            
                            elif isinstance(item, int) or isinstance(item, float):
                                if (symbol.lower() == 'greater than' or symbol == '>'):
                                    condition = self.__data[column] > item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'less than' or symbol == '<'):
                                    condition = self.__data[column] < item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'equal to' or symbol == '=='):
                                    condition = self.__data[column] == item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                                    condition = self.__data[column] >= item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                                    condition = self.__data[column] <= item
                                    self.__data = self.__data[condition]
                                    
                                elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                    condition = self.__data[column] != item
                                    self.__data = self.__data[condition]
            
            
            
            elif isinstance(column, list) or isinstance(column, tuple):
                if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                    raise ValueError("If column is a list or tuple, then value must assume form of a list or tuple with same length.")
                    
                elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) == len(column)):
                    if isinstance(operation, str):
                        for col, item in zip(column, value):
                            if isinstance(item, str):
                                if (operation.lower() == 'equal to' or operation == '=='):
                                    condition = self.__data[col] == item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'not equal to' or operation == '!='):
                                    condition = self.__data[col] != item
                                    self.__data = self.__data[condition]  
                                    
                                else:
                                    raise ValueError("When value is a string, comparison of greater than or less than cannot be made. Consider switching operation to a list or tuple for more control.")
                                    
                            
                            elif isinstance(item, int) or isinstance(item, float):
                                if (operation.lower() == 'greater than' or operation == '>'):
                                    condition = self.__data[col] > item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'less than' or operation == '<'):
                                    condition = self.__data[col] < item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'equal to' or operation == '=='):
                                    condition = self.__data[col] == item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                                    condition = self.__data[col] >= item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'less than or equal to' or operation == '<='):
                                    condition = self.__data[col] <= item
                                    self.__data = self.__data[condition]
                                    
                                elif (operation.lower() == 'not equal to' or operation == '!='):
                                    condition = self.__data[col] != item
                                    self.__data = self.__data[condition]
                                    
                                
                    elif isinstance(operation, list) or isinstance(operation, tuple):
                        if len(operation) == len(value) == len(column):
                            for col, item, symbol in zip(column, value, operation):
                                if isinstance(item, str):
                                    if (symbol.lower() == 'equal to' or symbol == '=='):
                                        condition = self.__data[col] == item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                        condition = self.__data[col] != item
                                        self.__data = self.__data[condition]  
                                        
                                    else:
                                        raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                                        
                                
                                elif isinstance(item, int) or isinstance(item, float):
                                    if (symbol.lower() == 'greater than' or symbol == '>'):
                                        condition = self.__data[col] > item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'less than' or symbol == '<'):
                                        condition = self.__data[col] < item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'equal to' or symbol == '=='):
                                        condition = self.__data[col] == item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                                        condition = self.__data[col] >= item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                                        condition = self.__data[col] <= item
                                        self.__data = self.__data[condition]
                                        
                                    elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                        condition = self.__data[col] != item
                                        self.__data = self.__data[condition]
                        
                        
                        else:
                            raise ValueError("When arguments in column, value, and operation are a list or tuple, they must all have same size.")
            
            
                elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) != len(column)):
                    raise ValueError("The parameters column and value must have the same length when both set to either list or tuple.")
            
            else:
                raise ValueError("Column must be either a string, list, tuple, or dictionary.")
        
        
        
        elif column == None:
            if isinstance(operation, str):
                if isinstance(value, int) or isinstance(value, float):
                   if operation.lower() not in possible_operations:
                       raise ValueError(f"This operation is not supported. Please use the following: {possible_operations}")
                       
                   elif (operation.lower() == 'greater than' or operation == '>'):
                       condition = self.__data > value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'less than' or operation == '<'):
                       condition = self.__data < value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'equal to' or operation == '=='):
                       condition = self.__data == value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                       condition = self.__data >= value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'less than or equal to' or operation == '<='):
                       condition = self.__data <= value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'not equal to' or operation == '!='):
                       condition = self.__data != value
                       self.__data = self.__data[condition]
                       
                
                elif isinstance(value, str):
                   if (operation.lower() == 'equal to' or operation == '=='):
                       condition = self.__data == value
                       self.__data = self.__data[condition]
                       
                   elif (operation.lower() == 'not equal to' or operation == '!='):
                       condition = self.__data != value
                       self.__data = self.__data[condition]  
                       
                   else:
                       raise ValueError("When column is set to NONE and value is a string, comparison of greater than or less than cannot be made.")
                       
                       
                       
                elif isinstance(value, list) or isinstance(value, tuple):
                    raise ValueError("Length of values should be same as length of available operations to perform")
                               
            elif isinstance(operation, list) or isinstance(operation, tuple):
                if isinstance(value, int) or isinstance(value, float):
                    raise ValueError("If operation is list or tuple, then value must be list or tuple of same size.")
                    
                elif isinstance(value, str):
                    raise ValueError("If operation is list or tuple, then value must be list or tuple of same size.")
                
                elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) == len(operation)):
                    for item, symbol in zip(value, operation):
                        if isinstance(item, str):
                            if (symbol.lower() == 'equal to' or symbol == '=='):
                                condition = self.__data == item
                                self.__data = self.__data[condition]
                               
                            elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                condition = self.__data != item
                                self.__data = self.__data[condition]  
                               
                            else:
                                raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                               
                       
                        elif isinstance(item, int) or isinstance(item, float):
                            if (symbol.lower() == 'greater than' or symbol == '>'):
                                condition = self.__data > item
                                self.__data = self.__data[condition]
                               
                            elif (symbol.lower() == 'less than' or symbol == '<'):
                                condition = self.__data < item
                                self.__data = self.__data[condition]
                                
                            elif (symbol.lower() == 'equal to' or symbol == '=='):
                                condition = self.__data == item
                                self.__data = self.__data[condition]
                                
                            elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                                condition = self.__data >= item
                                self.__data = self.__data[condition]
                                
                            elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                                condition = self.__data <= item
                                self.__data = self.__data[condition]
                                
                            elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                condition = self.__data != item
                                self.__data = self.__data[condition]
                                
                elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) != len(operation)):
                    raise ValueError("If operation is list or tuple, then value must be list or tuple of same size.")
            
        return self.__data
    
    
    def remove_duplicates(self, which_columns: str or list or tuple = None):
        """
        Remove duplicate rows from the dataset based on specified columns.
    
        Parameters
        ----------
        which_columns : str or list or tuple, optional
            Column(s) to consider when identifying duplicate rows. If specified,
            the method will drop rows that have the same values in the specified column(s).
    
        Returns
        -------
        DataFrame
            A new DataFrame with duplicate rows removed.
    
        Raises
        ------
        ValueError
            If `which_columns` is not a valid string, list, or tuple.
    
        Notes
        -----
        - This method removes duplicate rows from the dataset based on the specified column(s).
        - If no columns are specified, it considers all columns when identifying duplicates.
    
        Examples
        --------
        >>> # Create a supervised learning instance and load a dataset
        >>> model = SupervisedLearning(dataset)
        >>> # Remove duplicate rows based on a specific column
        >>> model.remove_duplicates(which_columns='column_name')
    
        See Also
        --------
        - pandas.DataFrame.drop_duplicates : Drop duplicate rows.
        - pandas.DataFrame : Pandas DataFrame class for handling tabular data.
        - ydata_profiling : Data profiling library for understanding and analyzing datasets.
        - sweetviz : Visualize and compare datasets for exploratory data analysis.
    
        References
        ----------
        - Pandas Documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
        """
        

        if isinstance(which_columns, str) or isinstance(which_columns, list) or isinstance(which_columns, tuple):
            self.__data.drop_duplicates(inplace = True, subset = which_columns)
            
        else:
            raise ValueError("Removing duplicates from your dataset must be done by indicating the column as either a string, list, or tuple.")
        
        return self.__data
     
    
    def select_features(self, strategy: str, estimator: str, number_of_features: int):
        """
        Select features using different techniques.
    
        Parameters
        ----------
        strategy : str
            The feature selection strategy. Options include "rfe", "selectkbest", 
            "selectfrommodel", and "selectpercentile".
        estimator : str
            The estimator or score function used for feature selection.
        number_of_features : int
            The number of features to select.
    
        Returns
        -------
        DataFrame or dict
            DataFrame with selected features or a dictionary with selected features
            and selection metrics.
    
        Raises
        ------
        ValueError
            If the strategy or estimator is not recognized.
    
        Notes
        -----
        - This method allows feature selection using different techniques such as Recursive Feature Elimination (RFE), SelectKBest, SelectFromModel, and SelectPercentile.
    
        Examples
        --------
        >>> # Create a supervised learning instance and load a dataset
        >>> model = SupervisedLearning(dataset)
        >>>
        >>> # Select features using Recursive Feature Elimination (RFE)
        >>> selected_features = model.select_features(strategy='rfe', 
        >>>                                           estimator=RandomForestRegressor(), 
        >>>                                           number_of_features=5)
        >>>
        >>> print(selected_features)
    
        See Also
        --------
        - sklearn.feature_selection.RFE : Recursive Feature Elimination.
        - sklearn.feature_selection.SelectKBest : Select features based on k highest scores.
        - sklearn.feature_selection.SelectFromModel : Feature selection using an external estimator.
        - sklearn.feature_selection.SelectPercentile : Select features based on a percentile of the highest scores.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
    
        """
        types = ["rfe", "selectkbest", "selectfrommodel", "selectpercentile"]
        rfe_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
        kbest_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2"]
        frommodel_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
        percentile_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2", "mutual_info_classif"]
        
        strategy = strategy.lower()
        if strategy in types: 
            if strategy == "rfe" and estimator != None:
                technique = sfs.RFE(estimator = estimator, n_features_to_select = number_of_features)
                self.__x = technique.fit_transform(self.__x, self.__y)
                
                self.__x = pd.DataFrame(self.__x, columns = technique.get_feature_names_out())
                return self.__x
                
            elif strategy == "selectkbest" and estimator != None:
                technique = sfs.SelectKBest(score_func = estimator, k = number_of_features)
                self.__x = technique.fit_transform(self.__x, self.__y)
                
                self.__x = pd.DataFrame(self.__x, columns = technique.get_feature_names_out())
                best_features = pd.DataFrame({"Features": technique.feature_names_in_, "Scores": technique.scores_, "P_Values": technique.pvalues_})
                return {"Dataset ---> Features Selected": self.__x, "Selection Metrics": best_features}
                
            elif strategy == "selectfrommodel" and estimator != None:
                technique = sfs.SelectFromModel(estimator = estimator, max_features = number_of_features)
                self.__x = technique.fit_transform(self.__x, self.__y)
                
                self.__x = pd.DataFrame(self.__x, columns = technique.get_feature_names_out())
                return self.__x
                
            elif strategy == "selectpercentile" and estimator != None:
                technique = sfs.SelectPercentile(score_func = estimator, percentile = number_of_features)
                self.__x = technique.fit_transform(self.__x, self.__y)
                
                self.__x = pd.DataFrame(self.__x, columns = technique.get_feature_names_out())
                best_features = pd.DataFrame({"Features": technique.feature_names_in_, "Scores": technique.scores_, "P_Values": technique.pvalues_})
                return {"Dataset ---> Features Selected": self.__x, "Selection Metrics": best_features}
            
            elif estimator == None:
                raise ValueError("You must specify an estimator or score function to use feature selection processes")
                
        else:
            raise ValueError(f"Select a feature selection technique from the following: {types}. \n\nRFE Estimator = {rfe_possible_estimator} e.g XGBoost, RandomForest, SVM etc\nSelectKBest Score Function = {kbest_possible_score_functions}\nSelectFromModel Estimator = {frommodel_possible_estimator} e.g XGBoost, RandomForest, SVM etc.\nSelectPercentile Score Function = {percentile_possible_score_functions}")
           
    
    def group_data(self, columns: list or tuple, column_to_groupby: str or list or tuple, aggregate_function: str, reset_index: bool = False, inplace: bool = False):
        """
        Group data by specified columns and apply an aggregate function.
    
        Parameters
        ----------
        columns : list or tuple
            Columns to be grouped and aggregated.
        column_to_groupby : str or list or tuple
            Column or columns to be used for grouping.
        aggregate_function : str
            The aggregate function to apply (e.g., 'mean', 'count', 'min', 'max', 'std', 'var', 'median').
        reset_index : bool, default False
            Whether to reset the index after grouping.
        inplace : bool, default False
            Replace the original dataset with this groupby operation.
    
        Returns
        -------
        DataFrame
            Grouped and aggregated data.
    
        Raises
        ------
        ValueError
            If the column types or aggregate function are not recognized.
            
        
        Examples
        --------
        >>> # Create a supervised learning instance and load a dataset
        >>> model = SupervisedLearning(dataset)
        >>>
        >>> # Group data by 'Category' and calculate the mean for 'Value'
        >>> grouped_data = model.group_data(columns=['Value'], 
        >>>                                 column_to_groupby='Category', 
        >>>                                 aggregate_function='mean')
        >>>
        >>> print(grouped_data)
    
        See Also
        --------
        - pandas.DataFrame.groupby : Group DataFrame using a mapper or by a Series of columns.
        - pandas.DataFrame.agg : Aggregate using one or more operations over the specified axis.
    
        References
        ----------
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html

        """
        
        agg = ["mean", "count", "min", "max", "std", "var", "median"]
        
        if reset_index == False:
            if isinstance(columns, list) or isinstance(columns, tuple):
                if isinstance(column_to_groupby, str) or isinstance(column_to_groupby, list) or isinstance(column_to_groupby, tuple):
                    aggregate_function = aggregate_function.lower().strip()
                    if aggregate_function in agg and isinstance(aggregate_function, str):
                        if aggregate_function == "mean":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).mean()
                        
                        elif aggregate_function == "count":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).count()
                            
                        elif aggregate_function == "min":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).min()
                            
                        elif aggregate_function == "max":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).max()
                            
                        elif aggregate_function == "std":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).std()
                            
                        elif aggregate_function == "var":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).var()
                            
                        elif aggregate_function == "median":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).median()
                            
                    else:
                        raise ValueError(f"Specify the right aggregate function from the following: {agg}")
                            
            else:
                raise ValueError("You need to select more than one column as a list or tuple to perform a groupby operation.")
        
        elif reset_index == True:
            if isinstance(columns, list) or isinstance(columns, tuple):
                if isinstance(column_to_groupby, str) or isinstance(column_to_groupby, list) or isinstance(column_to_groupby, tuple):
                    aggregate_function = aggregate_function.lower().strip()
                    if aggregate_function in agg and isinstance(aggregate_function, str):
                        if aggregate_function == "mean":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).mean()
                            grouped_columns = grouped_columns.reset_index()
                        
                        elif aggregate_function == "count":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).count()
                            grouped_columns = grouped_columns.reset_index()
                            
                        elif aggregate_function == "min":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).min()
                            grouped_columns = grouped_columns.reset_index()
                            
                        elif aggregate_function == "max":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).max()
                            grouped_columns = grouped_columns.reset_index()
                            
                        elif aggregate_function == "std":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).std()
                            grouped_columns = grouped_columns.reset_index()
                            
                        elif aggregate_function == "var":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).var()
                            grouped_columns = grouped_columns.reset_index()
                            
                        elif aggregate_function == "median":
                            grouped_columns = self.__data[columns].groupby(column_to_groupby).median()
                            grouped_columns = grouped_columns.reset_index()
                            
                    else:
                        raise ValueError(f"Specify the right aggregate function from the following: {agg}")
                            
            else:
                raise ValueError("You need to select more than one column as a list or tuple to perform a groupby operation.")
        
        
        else:
            raise ValueError("The arguments for 'reset_index' must be boolean of TRUE or FALSE.")
        
        
        if inplace == True:
            self.__data = grouped_columns
            return self.__data
        
        return grouped_columns               
             
    
    def count_column_categories(self, column: str or list or tuple, reset_index: bool = False, inplace: bool = False, test_data: bool = False):
        """
        Count the occurrences of categories in a categorical column.
    
        Parameters
        ----------
        column : str or list or tuple
            Categorical column or columns to count categories.
        reset_index : bool, default False
            Whether to reset the index after counting.
        inplace : bool, default False
            Replace the original dataset with this groupby operation.
        test_data : bool, default False
            Include the categories count for the test data.
    
        Returns
        -------
        DataFrame
            Count of occurrences of each category in the specified column.
    
        Raises
        ------
        ValueError
            If the column type is not recognized.
    
        Examples
        --------
        >>> # Create a supervised learning instance and load a dataset
        >>> model = SupervisedLearning(dataset)
        >>>
        >>> # Count the occurrences of each category in the 'Category' column
        >>> category_counts = model.count_column_categories(column='Category')
        >>>
        >>> print(category_counts)
    
        See Also
        --------
        - pandas.Series.value_counts : Return a Series containing counts of unique values.
        - pandas.DataFrame.reset_index : Reset the index of a DataFrame.
    
        References
        ----------
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html
        - Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

        """
        if self.__split_data == False:
            if reset_index == False:
                if isinstance(column, str) or isinstance(column, list) or isinstance(column, tuple):
                    categories_count = self.__data[column].value_counts()
                    
                else:
                    raise ValueError("Column inserted must be a string, list, or tuple.")
                    
            elif reset_index == True:
                if isinstance(column, str) or isinstance(column, list) or isinstance(column, tuple):
                    categories_count = self.__data[column].value_counts()
                    categories_count = categories_count.reset_index()
                    
                else:
                    raise ValueError("Column inserted must be a string, list, or tuple.")
                    
            else:
                raise ValueError("The arguments for 'reset_index' must be boolean TRUE or FALSE.")
                
            
            if inplace == True:
                self.__data = categories_count
                return self.__data
                
            return categories_count
        
        else:
            x_train_col = [col for col in self.__x_train.columns]
            y_train_col = [self.__y_train.name]

            if test_data == False:
                if isinstance(column, str):
                    if column in x_train_col:
                        categories_count = self.__x_train[column].value_counts()
                    elif column in y_train_col:
                        categories_count = self.__y_train.value_counts()
                        
                elif isinstance(column, list) or isinstance(column, tuple):
                    a = set(x_train_col)
                    b = set(column)
                    if b.issubset(a):
                        categories_count = self.__x_train[column].value_counts()
                    
                    else:
                        raise ValueError("One or more of the columns specified cannot be found among your independent variables columns (x_train). Also, check that your dependent variable column (y_train) is not among the columns. To get the value counts of the y_train, it must be specified seperately as a string.")
                
                if reset_index == True:
                    categories_count = categories_count.reset_index()

                if inplace == True:
                    self.__data = categories_count
                    return self.__data
                    
                return categories_count
                
            elif test_data == True:
                if isinstance(column, str):
                    if column in x_train_col:
                        categories_count_train = self.__x_train[column].value_counts()
                        categories_count_test = self.__x_test[column].value_counts()
                        
                    elif column in y_train_col:
                        categories_count_train = self.__y_train.value_counts()
                        categories_count_test = self.__y_test.value_counts()
                        
                elif isinstance(column, list) or isinstance(column, tuple):
                    a = set(x_train_col)
                    b = set(column)
                    if b.issubset(a):
                        categories_count_train = self.__x_train[column].value_counts()
                        categories_count_test = self.__x_test[column].value_counts()
                        
                    else:
                        raise ValueError("One or more of the columns specified cannot be found among your independent variables columns (x_train). Also, check that your dependent variable column (y_train) is not among the columns. To get the value counts of the y_train, it must be specified seperately as a string.")    

                if reset_index == True:
                    categories_count_train = categories_count_train.reset_index()
                    categories_count_test = categories_count_test.reset_index()
                    
                return {"Training Data": categories_count_train, "Test Data": categories_count_test}
    
    def sweetviz_profile_report(self, filename: str = "Pandas Profile Report.html", auto_open: bool = False):
        """
        Generate a Sweetviz profile report for the dataset.

        Parameters
        ----------
        filename : str, default "Pandas Profile Report.html"
            The name of the HTML file to save the Sweetviz report.
        auto_open : bool, default False
            If True, open the generated HTML report in a web browser.

        Returns
        -------
        None

        See Also
        --------
        - sweetviz.analyze : Generate and analyze a Sweetviz data comparison report.

        References
        ----------
        - Sweetviz Documentation: https://github.com/fbdesignpro/sweetviz

        """
        report1 = sv.analyze(self.__data)
        report1.show_html(filepath = filename, open_browser = auto_open)
           
        
    def pandas_profiling(self, output_file: str = "Pandas Profile Report.html", dark_mode: bool = False, title: str = "Report"):
        """
        Generate a Pandas profiling report for the dataset.

        Parameters
        ----------
        output_file : str, default "Pandas Profile Report.html"
            The name of the HTML file to save the Pandas profiling report.
        dark_mode : bool, default False
            If True, use a dark mode theme for the generated report.
        title : str, default "Report"
            The title of the Pandas profiling report.

        Returns
        -------
        None

        See Also
        --------
        - pandas_profiling.ProfileReport : Generate a profile report from a DataFrame.

        References
        ----------
        - Pandas Profiling Documentation: https://github.com/pandas-profiling/pandas-profiling

        """
        
        report = pp(df = self.__data, dark_mode = dark_mode, explorative = True, title = title)
        report.to_widgets()
        report.to_file(output_file = output_file)
    
    
    def select_datatype(self, datatype_to_select: str = None, datatype_to_exclude: str = None, inplace: bool = False):
        """
        Select columns of specific data types from the dataset.

        Parameters
        ----------
        datatype_to_select : str, optional
            Data type(s) to include. All data types are included by default.
        datatype_to_exclude : str, optional
            Data type(s) to exclude. None are excluded by default.
        inplace : bool, default False
            Replace the original dataset with this groupby operation.

        Returns
        -------
        DataFrame
            Subset of the dataset containing columns of the specified data types.

        See Also
        --------
        - pandas.DataFrame.select_dtypes : Select columns based on data type.

        References
        ----------
        - Pandas Documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html

        """
        
        selected_data = self.__data.select_dtypes(include = datatype_to_select, exclude = datatype_to_exclude)
        
        if inplace == True:
            self.__data = selected_data
            return self.__data
        
        return selected_data
    
    
    def load_large_dataset(self, dataset):
        """
        Load a large dataset using the Datatable library.

        Parameters
        ----------
        dataset : str
            The path or URL of the dataset.

        Returns
        -------
        DataFrame
            Pandas DataFrame containing the loaded data.

        See Also
        --------
        - datatable.fread : Read a DataTable from a file.

        References
        ----------
        - Datatable Documentation: https://github.com/h2oai/datatable

        """
        
        self.__data = dt.fread(dataset).to_pandas()
        return self.__data
    
    
    def reduce_data_memory_useage(self, verbose: bool = True):
        """
        Reduce memory usage of the dataset by converting data types.

        Parameters
        ----------
        verbose : bool, default True
            If True, print information about the memory reduction.

        Returns
        -------
        DataFrame
            Pandas DataFrame with reduced memory usage.

        See Also
        --------
        - pandas.DataFrame.memory_usage : Return the memory usage of each column.

        References
        ----------
        - NumPy Documentation: https://numpy.org/doc/stable/user/basics.types.html

        """
        
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = self.__data.memory_usage().sum() / 1024 ** 2
        for col in self.__data.columns:
            col_type = self.__data[col].dtypes
            if col_type in numerics:
                c_min = self.__data[col].min()
                c_max = self.__data[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.__data[col] = self.__data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.__data[col] = self.__data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.__data[col] = self.__data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.__data[col] = self.__data[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        self.__data[col] = self.__data[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        self.__data[col] = self.__data[col].astype(np.float32)
                    else:
                        self.__data[col] = self.__data[col].astype(np.float64)
        end_mem = self.__data.memory_usage().sum() / 1024 ** 2
        if verbose:
            print(
                "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return self.__data
    
    
    def get_bestK_KNNclassifier(self, weight = "uniform", algorithm = "auto", metric = "minkowski", max_k_range: int = 31, figsize: tuple = (15, 10)):
        """
        Find the best value of k for K-Nearest Neighbors (KNN) classifier.
    
        Parameters
        ----------
        weight : str, default 'uniform'
            Weight function used in prediction. Possible values: 'uniform' or 'distance'.
        algorithm : str, default 'auto'
            Algorithm used to compute the nearest neighbors. Possible values: 'auto', 'ball_tree', 'kd_tree', 'brute'.
        metric : str, default 'minkowski'
            Distance metric for the tree. Refer to the documentation of sklearn.neighbors.DistanceMetric for more options.
        max_k_range : int, default 31
            Maximum range of k values to consider.
        figsize: tuple
            A tuple containing the frame length and breadth for the graph to be plotted.
    
        Returns
        -------
        Int
            An integer indicating the best k value for the KNN Classifier.
    
        Raises
        ------
        ValueError
            If invalid values are provided for 'algorithm' or 'weight'.
    
        Notes
        -----
        This method evaluates the KNN classifier with different values of k and plots a graph to help identify the best k.
        The best k-value is determined based on the highest accuracy score.
    
        Examples
        --------
        >>> data = SupervisedLearning(dataset)
        >>> data.get_bestK_KNNclassifier(weight='distance', 
        >>>                              algorithm='kd_tree')
    
        See Also
        --------
        - sklearn.neighbors.KNeighborsClassifier : K-nearest neighbors classifier.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        """
        if self.__split_data == True: 
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            weights = ['uniform', 'distance']
            
            a = algorithm.lower()
            b = weight.lower()
            
            if (a in algorithms) or (b in weights):    
                k = [num for num in range(1, max_k_range)]
                scores_knn = []
                scores_store = {}
                for num in k:
                    classifier = sn.KNeighborsClassifier(n_neighbors = num, weights = weight, algorithm = algorithm, metric = metric)
                    model = classifier.fit(self.__x_train, self.__y_train)
                    
                    # Model Evaluation
                    scores_knn.append(model.score(self.__x_train, self.__y_train))
                    scores_store[num] = (model.score(self.__x_train, self.__y_train))
                
                # Plotting a graph
                plt.figure(figsize = figsize)
                plt.plot(k, scores_knn)
                plt.title('KNN graph for values of K and their scores')
                plt.xlabel('Ranges of K values')
                plt.ylabel('Scores')
                plt.show()
                
                # Getting the best score
                b = (0, 0)
                for key, value in scores_store.items():    
                    if value > b[1]:
                        b = (key, value)
                print(f'\n\nKNN CLASSIFIER ------> Finding the besk K value:\nThe best k-value is {b[0]} with a score of {b[1]}.')
                
        
            else:
                raise ValueError(f"Check that the parameter 'algorithm' is one of the following: {algorithms}. Also, check that the parameter 'weight' is one of the following: {weights}")
                
            return b[0]
        
        else:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            weights = ['uniform', 'distance']

            a = algorithm.lower()
            b = weight.lower()

            if (a in algorithms) or (b in weights):    
                k = [num for num in range(1, max_k_range)]
                scores_knn = []
                scores_store = {}
                for num in k:
                    classifier = sn.KNeighborsClassifier(n_neighbors = num, weights = weight, algorithm = algorithm, metric = metric)
                    model = classifier.fit(self.__x, self.__y)
                    
                    # Model Evaluation
                    scores_knn.append(model.score(self.__x, self.__y))
                    scores_store[num] = (model.score(self.__x, self.__y))
                
                # Plotting a graph
                plt.figure(figsize = figsize)
                plt.plot(k, scores_knn)
                plt.title('KNN graph for values of K and their scores')
                plt.xlabel('Ranges of K values')
                plt.ylabel('Scores')
                plt.show()
                
                # Getting the best score
                b = (0, 0)
                for key, value in scores_store.items():    
                    if value > b[1]:
                        b = (key, value)
                print(f'\n\nKNN CLASSIFIER ------> Finding the besk K value:\nThe best k-value is {b[0]} with a score of {b[1]}.')
                

            else:
                raise ValueError(f"Check that the parameter 'algorithm' is one of the following: {algorithms}. Also, check that the parameter 'weight' is one of the following: {weights}")
                
            return b[0]
    
    
    def get_bestK_KNNregressor(self, weight = "uniform", algorithm = "auto", metric = "minkowski", max_k_range: int = 31, figsize: tuple = (15, 10)):
        """
        Find the best value of k for K-Nearest Neighbors (KNN) regressor.
    
        Parameters
        ----------
        weight : str, default 'uniform'
            Weight function used in prediction. Possible values: 'uniform' or 'distance'.
        algorithm : str, default 'auto'
            Algorithm used to compute the nearest neighbors. Possible values: 'auto', 'ball_tree', 'kd_tree', 'brute'.
        metric : str, default 'minkowski'
            Distance metric for the tree. Refer to the documentation of sklearn.neighbors.DistanceMetric for more options.
        max_k_range : int, default 31
            Maximum range of k values to consider.
        figsize: tuple
            A tuple containing the frame length and breadth for the graph to be plotted.
    
        Returns
        -------
        Int
            An integer indicating the best k value for the KNN Regressor.
    
        Raises
        ------
        ValueError
            If invalid values are provided for 'algorithm' or 'weight'.
    
        Notes
        -----
        This method evaluates the KNN regressor with different values of k and plots a graph to help identify the best k.
        The best k-value is determined based on the highest R-squared score.
    
        Examples
        --------
        >>> data = SupervisedLearning(dataset)
        >>> data.get_bestK_KNNregressor(weight='distance', algorithm='kd_tree')
    
        See Also
        --------
        - sklearn.neighbors.KNeighborsRegressor : K-nearest neighbors regressor.
    
        References
        ----------
        - Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

        """
        if self.__split_data == True: 
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            weights = ['uniform', 'distance']
            
            a = algorithm.lower()
            b = weight.lower()
            
            if (a in algorithms) or (b in weights):    
                k = [num for num in range(1, max_k_range)]
                scores_knn = []
                scores_store = {}
                for num in k:
                    regressor = sn.KNeighborsRegressor(n_neighbors = num, weights = weight, algorithm = algorithm, metric = metric)
                    model = regressor.fit(self.__x_train, self.__y_train)
                    
                    # Model Evaluation
                    scores_knn.append(model.score(self.__x_train, self.__y_train))
                    scores_store[num] = (model.score(self.__x_train, self.__y_train))
                
                # Plotting a graph
                plt.figure(figsize = figsize)
                plt.plot(k, scores_knn)
                plt.title('KNN graph for values of K and their scores')
                plt.xlabel('Ranges of K values')
                plt.ylabel('Scores')
                plt.show()
                
                # Getting the best score
                b = (0, 0)
                for key, value in scores_store.items():    
                    if value > b[1]:
                        b = (key, value)
                print(f'\n\nKNN Regressor ------> Finding the besk K value:\nThe best k-value is {b[0]} with a score of {b[1]}.')
                
        
            else:
                raise ValueError(f"Check that the parameter 'algorithm' is one of the following: {algorithms}. Also, check that the parameter 'weight' is one of the following: {weights}")
            
            return b[0]
        
        else:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            weights = ['uniform', 'distance']

            a = algorithm.lower()
            b = weight.lower()

            if (a in algorithms) or (b in weights):    
                k = [num for num in range(1, max_k_range)]
                scores_knn = []
                scores_store = {}
                for num in k:
                    regressor = sn.KNeighborsRegressor(n_neighbors = num, weights = weight, algorithm = algorithm, metric = metric)
                    model = regressor.fit(self.__x, self.__y)
                    
                    # Model Evaluation
                    scores_knn.append(model.score(self.__x, self.__y))
                    scores_store[num] = (model.score(self.__x, self.__y))
                
                # Plotting a graph
                plt.figure(figsize = figsize)
                plt.plot(k, scores_knn)
                plt.title('KNN graph for values of K and their scores')
                plt.xlabel('Ranges of K values')
                plt.ylabel('Scores')
                plt.show()
                
                # Getting the best score
                b = (0, 0)
                for key, value in scores_store.items():    
                    if value > b[1]:
                        b = (key, value)
                print(f'\n\nKNN Regressor ------> Finding the besk K value:\nThe best k-value is {b[0]} with a score of {b[1]}.')
                

            else:
                raise ValueError(f"Check that the parameter 'algorithm' is one of the following: {algorithms}. Also, check that the parameter 'weight' is one of the following: {weights}")

            return b[0]

   
    def unique_elements_in_columns(self, count: bool = False):
        """
        Extracts unique elements in each column of the dataset.
    
        This method generates a DataFrame containing unique elements in each column. If specified, it can also provide the count of unique elements in each column.
    
        Parameters
        ----------
        count : bool, optional, default=False
            If True, returns the count of unique elements in each column.
    
        Returns
        -------
        pd.DataFrame or pd.Series
            If count is False, a DataFrame with unique elements. If count is True, a Series with counts.
    
        Notes
        -----
        This method utilizes Pandas for extracting unique elements and their counts.
    
        Examples
        --------
        >>> from buildml import SupervisedLearning
        >>> model = SupervisedLearning(dataset)
        >>> unique_elements = model.unique_elements_in_columns(count=True)
    
        References
        ----------
        - Pandas unique: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html
        - Pandas nunique: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.nunique.html

        """
        column = [col for col in self.__data.columns]
        
        dataframe = pd.DataFrame()
        for element in column:
            data = self.__data[element].unique()
            data = pd.Series(data, name = element)
            dataframe = pd.concat([dataframe, data], axis = 1)
            
        if count == True:
            dataframe = dataframe.nunique()           
        
        return dataframe
    

    def simple_linregres_graph(self, regressor, title: str, xlabel: str, ylabel: str, figsize: tuple = (15, 10), line_style: str = "dashed", line_width: float = 2, line_marker: str = "o", line_marker_size: float = 12, train_color_marker: str = "red", test_color_marker: str = "red", line_color: str = "green", size_train_marker: float = 10, size_test_marker: float = 10, whole_dataset: bool = False):
        """
        Generate a simple linear regression graph with optional visualization of training and test datasets.
    
        Parameters:
        -----------
        regressor (object or list): 
            Single or list of regression models (e.g., sklearn.linear_model.LinearRegression) to visualize.
        title (str): 
            Title of the graph.
        xlabel : str
            A title for the xaxis.
        ylabel : str
            A title for the yaxis. 
        figsize : str, optional, default: (15, 10)
            The size(length, breadth) of the figure frame where we plot our graph.
        line_style : str, optional, default: "dashed"
            Style of the regression line ("solid", "dashed", "dashdot", etc.).
        line_width (float, optional): 
            Width of the regression line. Default is 2.
        line_marker (str, optional): 
            Marker style for data points on the regression line. Default is "o".
        line_marker_size (float, optional): 
            Size of the marker for data points on the regression line. Default is 12.
        train_color_marker (str, optional): 
            Color of markers for the training dataset. Default is "red".
        test_color_marker (str, optional): 
            Color of markers for the test dataset. Default is "red".
        line_color (str, optional): 
            Color of the regression line. Default is "green".
        size_train_marker (float, optional): 
            Size of markers for the training dataset. Default is 10.
        size_test_marker (float, optional): 
            Size of markers for the test dataset. Default is 10.
        whole_dataset (bool, optional): 
            If True, visualize the entire dataset with the regression line. If False, visualize training and test datasets separately. Default is False.
    
        Returns:
        --------
        None
            Displays a simple linear regression graph.
    
        Examples:
        ---------
        >>> # Example 1: Visualize a simple linear regression model
        >>> model.simple_linregres_graph(regressor=LinearRegression(), 
                                         title="Simple Linear Regression",
                                         xlabel="Specify your title for xaxis",
                                         ylabel="Specify your title for yaxis")
    
        >>> # Example 2: Visualize multiple linear regression models
        >>> regressors = [LinearRegression(), Ridge(), Lasso()]
        >>> model.simple_linregres_graph(regressor=regressors, 
                                         title="Analyzing Impact of Expenditure on Growth",
                                         xlabel="Expenditure",
                                         ylabel="Growth")
        
        References:
        -----------
        - `matplotlib.pyplot`: Matplotlib's plotting interface. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
        - `scikit-learn`: Scikit-learn for machine learning tools. [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
        - `numpy`: NumPy for numerical operations. [NumPy Documentation](https://numpy.org/doc/stable/)
        - `pandas`: Pandas for data manipulation. [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html)
        - `matplotlib`: Matplotlib library for plotting. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
        """
        
        name_x = [col for col in self.__x.columns]
        if not (isinstance(regressor, list) or isinstance(regressor, tuple)):
            if len(name_x) == 1:
                
                if not whole_dataset:
                    # Visualising the Training set results
                    plt.figure(figsize = figsize)
                    plt.scatter(self.__x_train, self.__y_train, color = train_color_marker, s=size_train_marker)
                    plt.plot(self.__x_train, self.model_regressor.predict(self.__x_train), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                    plt.title(f"{title} (Training Dataset)")
                    plt.xlabel(f"{xlabel}")
                    plt.ylabel(f"{ylabel}")
                    plt.show()
            
                    # Visualising the Test set results
                    plt.figure(figsize = figsize)
                    plt.scatter(self.__x_test, self.__y_test, color = test_color_marker, s=size_test_marker)
                    plt.plot(self.__x_train, self.model_regressor.predict(self.__x_train), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                    plt.title(f"{title} (Test Dataset)")
                    plt.xlabel(f"{xlabel}")
                    plt.ylabel(f"{ylabel}")
                    plt.show()
                    
                else:
                    plt.figure(figsize = figsize)
                    plt.scatter(self.__x, self.__y, color = train_color_marker, s=size_train_marker)
                    plt.plot(self.__x, self.model_regressor.predict(self.__x), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                    plt.title(title)
                    plt.xlabel(f"{xlabel}")
                    plt.ylabel(f"{ylabel}")
                    plt.show()
            
            else:
                raise ValueError("Simple Linear Regression involves only one independent variable. Ensure that your dataframe for x has just one column.")
    
    
        else:
            for each_regressor in regressor:
                if len(name_x) == 1:
                    
                    if not whole_dataset:
                        # Visualising the Training set results
                        plt.figure(figsize = figsize)
                        plt.scatter(self.__x_train, self.__y_train, color = train_color_marker, s=size_train_marker)
                        plt.plot(self.__x_train, each_regressor.fit(self.__x_train, self.__y_train).predict(self.__x_train), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                        plt.title(f"{title} (Training Dataset) for {each_regressor.__class__.__name__}")
                        plt.xlabel(f"{xlabel}")
                        plt.ylabel(f"{ylabel}")
                        plt.show()
                
                        # Visualising the Test set results
                        plt.figure(figsize = figsize)
                        plt.scatter(self.__x_test, self.__y_test, color = test_color_marker, s=size_test_marker)
                        plt.plot(self.__x_train, each_regressor.fit(self.__x_train, self.__y_train).predict(self.__x_train), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                        plt.title(f"{title} (Test Dataset) for {each_regressor.__class__.__name__}")
                        plt.xlabel(f"{xlabel}")
                        plt.ylabel(f"{ylabel}")
                        plt.show()
                        
                    else:
                        plt.figure(figsize = figsize)
                        plt.scatter(self.__x, self.__y, color = train_color_marker, s=size_train_marker)
                        plt.plot(self.__x, each_regressor.fit(self.__x, self.__y).predict(self.__x), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
                        plt.title(f"{title} for {each_regressor.__class__.__name__}")
                        plt.xlabel(f"{xlabel}")
                        plt.ylabel(f"{ylabel}")
                        plt.show()
                
                else:
                    raise ValueError("Simple Linear Regression involves only one independent variable. Ensure that your dataframe for x has just one column.")
        
                
    def polyreg_graph(self, title: str, xlabel: str, ylabel: str, figsize: tuple = (15, 10), line_style: str = "dashed", line_width: float = 2, line_marker: str = "o", line_marker_size: float = 12, train_color_marker: str = "red", test_color_marker: str = "red", line_color: str = "green", size_train_marker: float = 10, size_test_marker: float = 10, whole_dataset: bool = False):        
        """
        Generate a polynomial regression graph for visualization.
    
        Parameters
        ----------
        title : str
            The title of the graph.
        xlabel : str
            A title for the xaxis.
        ylabel : str
            A title for the yaxis. 
        figsize : str, optional, default: (15, 10)
            The size(length, breadth) of the figure frame where we plot our graph.
        line_style : str, optional, default: "dashed"
            Style of the regression line ("solid", "dashed", "dashdot", etc.).
        line_width : float, optional, default: 2
            Width of the regression line.
        line_marker : str, optional, default: "o"
            Marker style for data points on the regression line.
        line_marker_size : float, optional, default: 12
            Size of the marker on the regression line.
        train_color_marker : str, optional, default: "red"
            Color of markers for training data.
        test_color_marker : str, optional, default: "red"
            Color of markers for test data.
        line_color : str, optional, default: "green"
            Color of the regression line.
        size_train_marker : float, optional, default: 10
            Size of markers for training data.
        size_test_marker : float, optional, default: 10
            Size of markers for test data.
        whole_dataset : bool, optional, default: False
            If True, visualize the regression line on the entire dataset.
            If False, visualize on training and test datasets separately.
    
        Returns
        -------
        None
            Displays the polynomial regression graph.
    
        See Also
        --------
        - `matplotlib.pyplot.scatter`: Plot a scatter plot using Matplotlib.
        - `matplotlib.pyplot.plot`: Plot lines and/or markers using Matplotlib.
        - `numpy`: Fundamental package for scientific computing with Python.
        - `scikit-learn`: Simple and efficient tools for predictive data analysis.
    
        Example
        -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from buildml.automate import SupervisedLearning
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> # Get the Dataset
        >>> dataset = pd.read_csv("Your dataset/path")
        >>> 
        >>> # Assuming `automate` is an instance of the SupervisedLearning class
        >>> automate = SupervisedLearning(dataset)
        >>> regressor = LinearRegression()
        >>>
        >>> # Further Data Preparation and Segregation
        >>> select_variables = automate.select_dependent_and_independent(predict = "Salary")
        >>> poly_x = automate.polyreg_x(degree = 5)
        >>>
        >>> # Model Building
        >>> training = automate.train_model_regressor(regressor)
        >>> prediction = automate.regressor_predict()
        >>> evaluation = automate.regressor_evaluation()
        >>> poly_reg = automate.polyreg_graph(title = "Analyzing salary across different levels",  
        >>>                                   xlabel = "Levels", 
        >>>                                   ylabel = "Salary", 
        >>>                                   whole_dataset = True, 
        >>>                                   line_marker = None, 
        >>>                                   line_style = "solid")
        """
        
        possible_line_styles = ["dashed", "solid"]
        possible_line_styles_symbols = ["-", "--", "-.", ":"]
        
        model = self.model_regressor
        if not whole_dataset:
            # Visualising the Training set results
            plt.figure(figsize = figsize)
            plt.scatter(self.__x_train1, self.__y_train, color = train_color_marker, s=size_train_marker)
            plt.plot(self.__x_train1, model.predict(self.__x_train), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
            plt.title(f"{title} (Training Dataset)")
            plt.xlabel(f"{xlabel}")
            plt.ylabel(f"{ylabel}")
            plt.show()

            # Visualising the Test set results
            plt.figure(figsize = figsize)
            plt.scatter(self.__x_test1, self.__y_test, color = test_color_marker, s=size_test_marker)
            plt.plot(self.__x_test1, model.predict(self.__x_test), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
            plt.title(f"{title} (Test Dataset)")
            plt.xlabel(f"{xlabel}")
            plt.ylabel(f"{ylabel}")
            plt.show()

        else:
            plt.figure(figsize = figsize)
            plt.title(title)
            plt.scatter(self.__x1, self.__y, color = train_color_marker, s=size_train_marker)
            plt.plot(self.__x1, model.predict(self.__x), color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_marker_size)
            plt.xlabel(f"{xlabel}")
            plt.ylabel(f"{ylabel}")
            plt.show()
    
                
    def polyreg_x(self, degree: int, include_bias: bool = False):
        """
        Polynomial Regression Feature Expansion.
    
        This method performs polynomial regression feature expansion on the independent variables (features).
        It uses scikit-learn's PolynomialFeatures to generate polynomial and interaction features up to a specified degree.
    
        Parameters
        ----------
        degree : int
            The degree of the polynomial features.
        include_bias : bool, optional, default=False
            If True, the polynomial features include a bias column (intercept).
    
        Returns
        -------
        pd.DataFrame
            DataFrame with polynomial features.
    
        Notes
        -----
        This method utilizes scikit-learn's PolynomialFeatures for feature expansion.
    
        Examples
        --------
        >>> from buildml import SupervisedLearning
        >>> model = SupervisedLearning(dataset)
        >>> model.polyreg_x(degree=2, include_bias=True)
    
        References
        ----------
        - Scikit-learn PolynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        - Numpy: https://numpy.org/
        - Pandas: https://pandas.pydata.org/
        - Scikit-learn: https://scikit-learn.org/
        """
        self.__poly_features = sp.PolynomialFeatures(degree = degree, include_bias = include_bias)
        poly_x = self.__poly_features.fit_transform(self.__x)
        
        a, b = poly_x.shape
        column = [f"x{num}" for num in range(1, (b + 1))]
        self.__poly_x = pd.DataFrame(poly_x, columns = column)
        
        self.__x1 = self.__x
        self.__polynomial_regression = True
        self.__x = self.__poly_x
        return self.__x
    
    
    def poly_get_optimal_degree(self, max_degree: int = 10, whole_dataset: bool = False, test_size: float = 0.2, random_state: int = 0, include_bias: bool = True, cross_validation: bool = False):
        x = self.__x
        y = self.__y
        
        if cross_validation == False:
            data = []
            regressor = slm.LinearRegression()
            if whole_dataset == False:  
                for num in range(1, (max_degree + 1)):
                    info = []
                    poly_features = sp.PolynomialFeatures(degree = num, include_bias = include_bias)
                    poly_x = poly_features.fit_transform(x)
                    
                    x_train, x_test, y_train, y_test = sms.train_test_split(poly_x, y, test_size = test_size, random_state = random_state)
                    
                    # Model Training
                    model = regressor.fit(x_train, y_train)
                    
                    # Model Prediction
                    y_pred = model.predict(x_train)
                    y_pred1 = model.predict(x_test)
                    
                    # Model Evaluation
                    train_r2 = sm.r2_score(y_train, y_pred)
                    train_rmse = np.sqrt(sm.mean_squared_error(y_train, y_pred))
                
                    test_r2 = sm.r2_score(y_test, y_pred1)
                    test_rmse = np.sqrt(sm.mean_squared_error(y_test, y_pred1))
                    
                    info.append(num)
                    info.append(regressor.__class__.__name__)
                    info.append(train_r2)
                    info.append(train_rmse)
                    info.append(test_r2)
                    info.append(test_rmse)
                    data.append(info)
                
                data = pd.DataFrame(data, columns = ["Degree", "Base Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
            
            else:
                for num in range(1, (max_degree + 1)):
                    info = []
                    poly_features = sp.PolynomialFeatures(degree = num, include_bias = include_bias)
                    poly_x = poly_features.fit_transform(x)
                    
                    # Model Training
                    model = regressor.fit(poly_x, y)
                    
                    # Model Prediction
                    y_pred = model.predict(poly_x)
                    
                    # Model Evaluation
                    r2 = sm.r2_score(y, y_pred)
                    rmse = np.sqrt(sm.mean_squared_error(y, y_pred))
                    
                    info.append(num)
                    info.append(regressor.__class__.__name__)
                    info.append(r2)
                    info.append(rmse)
                    data.append(info)
                
                data = pd.DataFrame(data, columns = ["Degree", "Base Algorithm", "R-Squared", "RMSE"])
            
            return data
            
        else:
            data = []
            regressor = slm.LinearRegression()
            if whole_dataset == False:  
                for num in range(1, (max_degree + 1)):
                    info = []
                    poly_features = sp.PolynomialFeatures(degree = num, include_bias = include_bias)
                    poly_x = poly_features.fit_transform(x)
                    
                    x_train, x_test, y_train, y_test = sms.train_test_split(poly_x, y, test_size = test_size, random_state = random_state)
                    
                    # Model Training
                    model = regressor.fit(x_train, y_train)
                    
                    # Model Prediction
                    y_pred = model.predict(x_train)
                    y_pred1 = model.predict(x_test)
                    
                    # Model Evaluation
                    train_r2 = sm.r2_score(y_train, y_pred)
                    train_rmse = np.sqrt(sm.mean_squared_error(y_train, y_pred))
                
                    test_r2 = sm.r2_score(y_test, y_pred1)
                    test_rmse = np.sqrt(sm.mean_squared_error(y_test, y_pred1))
                    
                    cross_validation = sms.cross_val_score(estimator = regressor, X = x, y = y)
                    cross_val_mean = np.mean(cross_validation)
                    cross_val_standard_deviation = np.std(cross_validation)
                    
                    info.append(num)
                    info.append(regressor.__class__.__name__)
                    info.append(train_r2)
                    info.append(train_rmse)
                    info.append(test_r2)
                    info.append(test_rmse)
                    info.append(cross_val_mean)
                    info.append(cross_val_standard_deviation)
                    data.append(info)
                
                data = pd.DataFrame(data, columns = ["Degree", "Base Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
    
            else:
                for num in range(1, (max_degree + 1)):
                    info = []
                    poly_features = sp.PolynomialFeatures(degree = num, include_bias = include_bias)
                    poly_x = poly_features.fit_transform(x)
                    
                    # Model Training
                    model = regressor.fit(poly_x, y)
                    
                    # Model Prediction
                    y_pred = model.predict(poly_x)
                    
                    # Model Evaluation
                    r2 = sm.r2_score(y, y_pred)
                    rmse = np.sqrt(sm.mean_squared_error(y, y_pred))
                    
                    cross_validation = sms.cross_val_score(estimator = regressor, X = x, y = y)
                    cross_val_mean = np.mean(cross_validation)
                    cross_val_standard_deviation = np.std(cross_validation)
                    
                    info.append(num)
                    info.append(regressor.__class__.__name__)
                    info.append(r2)
                    info.append(rmse)
                    info.append(cross_val_mean)
                    info.append(cross_val_standard_deviation)
                    data.append(info)
                
                data = pd.DataFrame(data, columns = ["Degree", "Base Algorithm", "R-Squared", "RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
            
            return {"Degree Metrics": data, "Cross Validation Info": cross_validation}          