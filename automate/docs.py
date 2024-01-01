def drop_columns():
    """
        Drop specified columns from the dataset.
    
        This function removes the specified columns from the dataset, allowing you to exclude irrelevant or unnecessary features
        before training your machine learning model.
    
        Parameters
        ----------
        columns : list
            Column or list of columns to drop from the dataset.
    
        Returns
        -------
        pd.DataFrame
            A new DataFrame with the specified columns removed.
    
        Notes
        -----
        This function is part of the data cleaning stage in the machine learning workflow. It is optional and should be used when
        there are columns in the dataset that are not relevant for the machine learning model.
    
        Example
        -------
        >>> # Assuming 'data' is an instance of SupervisedLearning
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.drop_columns(columns=["column1", "column2"])
        """
        
        if self.__user_guide == True:
            print(
                f"""
                \n
                DATA CLEANING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
def get_dataset():
        """
        Get the original dataset and provides a user guide if set to TRUE for the machine learning workflow.
    
        Returns
        -------
        pd.DataFrame
            The original dataset loaded from the provided CSV file.
    
        Notes
        -----
        This method initializes the machine learning workflow and provides a user guide if set to TRUE to help you navigate through the different stages.
        Follow the guide step by step, marking each completed task as TRUE. Tasks denoted as FALSE are incomplete and can be revisited.
    
        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                Welcome to your machine learning guide by TechLeo.

Please follow these stages step by step. When a task has been completed, move on to the next. A completed task is denoted as TRUE, and an incomplete task is denoted as FALSE. These tasks refer to things you have done in your machin learning process. Please note, tasks set to optional are dataset specific. 

1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
def fix_missing_column():
        """
        Impute missing values in the dataset using the univariate imputer from sklearn.impute.

        Replace missing values using a descriptive statistic (e.g., mean, median, or most frequent) along each column,
        or using a constant value.
    
        Parameters
        ----------
        strategy : str, optional
            The imputation strategy.
            - If "mean", replace missing values using the mean along each column (numeric data).
            - If "median", replace missing values using the median along each column (numeric data).
            - If "most_frequent", replace missing values using the most frequent value along each column.
              Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
            The default is "mean".
    
        Returns
        -------
        pd.DataFrame
            The dataset with missing values imputed.
    
        Notes
        -----
        This method is part of the data cleaning stage in the machine learning workflow. Follow the guide step by step,
        marking each completed task as TRUE. Tasks denoted as FALSE are incomplete and can be revisited.
    
        Examples
        --------
        1. Impute missing values using the mean:
            
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.fix_missing_values(strategy="mean")
    
        2. Impute missing values using the median:
            
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.fix_missing_values(strategy="median")
        
        3. Impute missing values using the most frequent value:
            
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.fix_missing_values(strategy="most_frequent")
        
        4. Impute missing values without specifying a strategy (default is mean):
            
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.fix_missing_values()
        
        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                DATA CLEANING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
        
    


def categorical_to_numerical():
        """
        Convert categorical columns to numerical representations using the pandas get_dummies function.

        Parameters
        ----------
        columns : list, optional
            List of column names to apply one-hot encoding. If None, apply one-hot encoding to all categorical columns.
            The default is None.
    
        Returns
        -------
        pd.DataFrame
            The dataset with categorical columns converted to numerical representations.
    
        Notes
        -----
        This method is part of the data cleaning stage in the machine learning workflow. Follow the guide step by step,
        marking each completed task as TRUE. Tasks denoted as FALSE are incomplete and can be revisited.
    
        Examples
        --------
        1. Convert all categorical columns to numerical using the pandas get_dummies function:
        
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.categorical_to_numerical()
    
        2. Convert specific categorical columns to numerical using the pandas get_dummies function:
        
        >>> # python
        >>> data = SupervisedLearning("your_dataset.csv")
        >>> data.categorical_to_numerical(columns=["column1", "column2"])
        
        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                DATA CLEANING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    

def remove_outlier():
        """
        Remove outliers from the dataset using StandardScaler and filtering within a range.
    
        Returns
        -------
        pd.DataFrame
            The dataset with outliers removed.
    
        Notes
        -----
        This method is part of the data cleaning stage in the machine learning workflow. Follow the guide step by step,
        marking each completed task as TRUE. Tasks denoted as FALSE are incomplete and can be revisited.
    
        Examples
        --------
        1. Remove outliers from the entire dataset:
       
        >>> # python
        >>> dataset = SupervisedLearning("your_dataset.csv")
        >>> dataset.remove_outlier()
    
        2. Remove outliers after categorical-to-numerical transformation:
        
        >>> # python
        >>> dataset = SupervisedLearning("your_dataset.csv")
        >>> dataset.categorical_to_numerical()
        >>> dataset.remove_outlier()
    
        3. Remove outliers after scaling independent variables:
            
        >>> # python
        >>> dataset = SupervisedLearning("your_dataset.csv")
        >>> dataset.scale_independent_variables()
        >>> dataset.remove_outlier()
        
        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                DATA CLEANING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
def scale_independent_variables():
        """
        

        Returns
        -------
        None.

        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                FURTHER DATA PREPARATION STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    


def eda():
        """
        

        Returns
        -------
        None.

        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                EXPLORATORY DATA ANALYSIS
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
def eda_visual():
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        before_data_cleaning : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if self.__user_guide == True:
            print(
                f"""
                \n
                EXPLORATORY DATA ANALYSIS
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
        
    
    

def select_dependent_and_independent():
    if self.__user_guide == True:
            print(
                f"""
                \n
                FURTHER DATA PREPARATION STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )

    
    

def train_model_regressor():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
def train_model_classifier():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )

    

    
def regressor_predict():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )

    

    

def classifier_predict():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
def regressor_model_testing():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
    
def regressor_evaluation():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
    
def classifier_evaluation():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
  
    
    
    
def classifier_model_testing():
    if self.__user_guide == True:
            print(
                f"""
                \n
                MODEL BUILDING STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
    
def split_data():
    if self.__user_guide == True:
            print(
                f"""
                \n
                FURTHER DATA PREPARATION STAGE
                
1) Get the dataset ---> True.
2) Initial eda ---> {self.__eda}.

    
DATA CLEANING STEPS:
    
1) Drop columns (OPTIONAL: Do this if you have irrelevant columns) ---> {self.__dropped_column}.
2) Categorical to numerical (OPTIONAL: Do this if you have categorical data in the columns) ---> {self.__data_transformation}.
3) Fix missing values (OPTIONAL: Do this if you have missing values in your columns) ---> {self.__fixed_missing}.
4) Removing outliers ---> {self.__remove_outlier}.
5) EDA ---> {self.__eda}.
6) EDA vizualizations ---> {self.__eda_visual}.

    
FURTHER DATA PREPARATION:
    
1) Select dependent and independent variables ---> {self.__dependent_independent}.
2) Scaling independent variables ---> {self.__scaled}.
3) Splitting the data into train and test dataset ---> {self.__split_data}.
    
    
BUILDING YOUR MODEL:
1) Model training ---> {self.__model_training}.
2) Model prediction ---> {self.__model_prediction}.
3) Model evaluation ---> {self.__model_evaluation}.
4) Model testing ---> {self.__model_testing}.
    
                This process highlights what a typical machine learning workflow should look like. For beginners, this should assist you getting your model ready. 
\n\n\n
                """
                )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    