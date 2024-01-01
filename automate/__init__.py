import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pp
import sweetviz as sv
import imblearn.over_sampling as ios
import imblearn.under_sampling as ius
import sklearn.impute as si
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
__version__ = "0.0.1"



class SupervisedLearning: 
    """
    Allows you perform some quick data cleaning process on the training and test dataset. 
    
    Perform processes like:
        a) Fixing missing values
        b) Data transformation
        c) Data scaling
        d) Exploratory data analysis
        e) Dropping columns, etc.
    
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
    
        # Feature Engineering
        "select_features",
        "select_dependent_and_independent",
    
        # Data Preprocessing
        "scale_independent_variables",
        "remove_outlier",
        "split_data",
    
        # Model Building and Evaluation
        "k_knn_classifier",
        "train_model_regressor",
        "regressor_predict",
        "regressor_evaluation",
        "regressor_model_testing",
        "build_multiple_regressors",
        "build_multiple_regressors_from_features",
        "build_single_regressor_from_features",
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


       
    def __init__(self, dataset, user_guide: bool =  False, show_warnings: bool = False):
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
        self.__user_guide = user_guide
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
        
    def drop_columns(self, columns: list):
        self.__data = self.__data.drop(columns, axis = 1)
        return self.__data
    
    
    def get_training_test_data(self):
        return (self.__x_train, self.__x_test, self.__y_train, self.__y_test)
        
    def get_dataset(self):
        return {"Working Dataset": self.__data, "Initial Dataset": self.__dataset}
    
    def fix_missing_values(self, strategy: str = None):
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
        self.__scaler = sp.StandardScaler()
        self.__x = self.__scaler.fit_transform(self.__x)
        self.__x = pd.DataFrame(self.__x, columns = self.__scaler.feature_names_in_)
        self.__scaled = True
        return {"Dependent Variable": self.__y, "Scaled Independent Variables": self.__x}
  
    
  
    def eda(self):
        self.__data.info()
        print("\n\n")
        data_head = self.__data.head()
        data_tail = self.__data.tail()
        data_descriptive_statistic = self.__data.describe()
        data_more_descriptive_statistics = self.__data.describe(include = "all")
        data_mode = self.__data.mode()
        data_distinct_count = self.__data.nunique()
        data_null_count = self.__data.isnull().sum()
        data_total_null_count = self.__data.isnull().sum().sum()
        data_correlation_matrix = self.__data.corr()
        self.__eda = True
        return {"Dataset": self.__data, "Data_Head": data_head, "Data_Tail": data_tail, "Data_Descriptive_Statistic": data_descriptive_statistic, "Data_More_Descriptive_Statistic": data_more_descriptive_statistics, "Data_Mode": data_mode, "Data_Distinct_Count": data_distinct_count, "Data_Null_Count": data_null_count, "Data_Total_Null_Count": data_total_null_count, "Data_Correlation_Matrix": data_correlation_matrix}
    
    
    
    def eda_visual(self, y: str, before_data_cleaning: bool = True):
        if before_data_cleaning == False:
            data_histogram = self.__data.hist(figsize = (15, 10), bins = 10)
            plt.figure(figsize = (15, 10))
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        
        elif before_data_cleaning == True:
            # Visualize the distribution of categorical features
            categorical_features = self.__data.select_dtypes(include = "object").columns
            for feature in categorical_features:
                plt.figure(figsize=(8, 5))
                sns.countplot(x=feature, data = self.__data)
                plt.title(f'Distribution of {feature}')
                plt.show()
            
            # Box plots for numerical features by categorical features
            for feature in categorical_features:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=feature, y = y, data = self.__data)
                plt.title(f'Box Plot of {feature} vs. {y}')
                plt.show()
                
            plt.figure(figsize = (15, 10))
            data_histogram = self.__data.hist(figsize = (15, 10), bins = 10)
            plt.figure(figsize = (15, 10))
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        self.__eda_visual = True
    
    
    
    def select_dependent_and_independent(self, predict: str):
        self.__x = self.__data.drop(predict, axis = 1)
        self.__y = self.__data[f"{predict}"]
        self.__dependent_independent = True
        return {"Dependent Variable": self.__y, "Independent Variables": self.__x}
  
        
      
    def split_data(self):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = sms.train_test_split(self.__x, self.__y, test_size = 0.2, random_state = 0)
        self.__split_data = True
        return {"Training X": self.__x_train, "Test X": self.__x_test, "Training Y": self.__y_train, "Test Y": self.__y_test}
    
    
    
    def train_model_regressor(self, regressor):
        self.regression_problem = True
        self.regressor = regressor
        self.model_regressor = self.regressor.fit(self.__x_train, self.__y_train)
        score = self.model_regressor.score(self.__x_train, self.__y_train)
        print(f"{self.regressor.__class__.__name__}'s amount of variation in Y predicted by your features X after training: (Rsquared) ----> {score}")
        self.__model_training = True
        return self.model_regressor
      

      
    def train_model_classifier(self, classifier):
        self.classification_problem = True
        self.classifier = classifier
        self.model_classifier = self.classifier.fit(self.__x_train, self.__y_train)
        score = self.model_classifier.score(self.__x_train, self.__y_train)
        print(f"{self.classifier.__class__.__name__} accuracy in prediction after training: (Accuracy) ---> {score}")
        self.__model_training = True
        return self.model_classifier
        
    
    
    def regressor_predict(self):
        if self.regression_problem == True:
            self.__y_pred = self.model_regressor.predict(self.__x_train)
            self.__y_pred1 = self.model_regressor.predict(self.__x_test)
            self.__model_prediction = True
            return {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1}
         
        else:
            raise AssertionError("The training phase of the model has been set to classification. Can not predict a classification model with a regression model.")
        
        
     
    def classifier_predict(self):
        if self.classification_problem == True:
            self.__y_pred = self.model_classifier.predict(self.__x_train)
            self.__y_pred1 = self.model_classifier.predict(self.__x_test)
            self.__model_prediction = True
            return {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1}
        
        else:
            raise AssertionError("The training phase of the model has been set to regression. Can not predict a regression model with a classification model.")
        
    
    
    def regressor_model_testing(self, variables_values: list, scaling: bool = False):
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
        
        
        
    def build_multiple_regressors(self, regressors: list or tuple, kfold: int = None, cross_validation: bool = False, graph: bool = False, length: int = None, width: int = None):
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
                plt.figure(figsize = (width, length))
                plt.title("Training R2", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training R2"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training R2", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training RMSE
                plt.figure(figsize = (width, length))
                plt.title("Training RMSE", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training RMSE"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training RMSE", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test R2
                plt.figure(figsize = (width, length))
                plt.title("Training R2", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test R2"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training R2", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test RMSE
                plt.figure(figsize = (width, length))
                plt.title("Training RMSE", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test RMSE"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
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
                plt.figure(figsize = (width, length))
                plt.title("Training R2", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training R2"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training R2", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training RMSE
                plt.figure(figsize = (width, length))
                plt.title("Training RMSE", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Training RMSE"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training RMSE", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test R2
                plt.figure(figsize = (width, length))
                plt.title("Training R2", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test R2"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training R2", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test RMSE
                plt.figure(figsize = (width, length))
                plt.title("Training RMSE", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Test RMSE"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training RMSE", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Cross Validation Mean
                plt.figure(figsize = (width, length))
                plt.title("Cross Validation Mean", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Mean"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Mean"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Cross Validation Mean", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Cross Validation Standard Deviation
                plt.figure(figsize = (width, length))
                plt.title("Cross Validation Standard Deviation", pad = 10)
                plt.plot(dataset_regressors["Algorithm"], dataset_regressors["Cross Validation Standard Deviation"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_regressors.Algorithm, round(dataset_regressors["Cross Validation Standard Deviation"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
        
            
        return {"Regressor Metrics": dataset_regressors, "More Info": self.__multiple_regressor_models}
    
    
    
    def build_multiple_classifiers(self, classifiers: list or tuple, kfold: int = None, cross_validation: bool = False, graph: bool = False, length: int = None, width: int = None):
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
                plt.figure(figsize = (width, length))
                plt.title("Training Accuracy", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Accuracy"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Accuracy", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training Precision
                plt.figure(figsize = (width, length))
                plt.title("Training Precision", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Precision"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Precision"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Precision", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training Recall
                plt.figure(figsize = (width, length))
                plt.title("Training Recall", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Recall"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Recall"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Recall", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training F1 Score
                plt.figure(figsize = (width, length))
                plt.title("Training F1 Score", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training F1 Score"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training F1 Score"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training F1 Score", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Accuracy
                plt.figure(figsize = (width, length))
                plt.title("Test Accuracy", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Accuracy"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Accuracy", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Precision
                plt.figure(figsize = (width, length))
                plt.title("Test Precision", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Precision"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Precision"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Precision", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Recall
                plt.figure(figsize = (width, length))
                plt.title("Test Recall", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Recall"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Recall"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Recall", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test F1 Score
                plt.figure(figsize = (width, length))
                plt.title("Test F1 Score", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test F1 Score"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test F1 Score"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
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
                plt.figure(figsize = (width, length))
                plt.title("Training Accuracy", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Accuracy"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Accuracy", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training Precision
                plt.figure(figsize = (width, length))
                plt.title("Training Precision", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Precision"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Precision"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Precision", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training Recall
                plt.figure(figsize = (width, length))
                plt.title("Training Recall", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training Recall"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training Recall"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training Recall", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Training F1 Score
                plt.figure(figsize = (width, length))
                plt.title("Training F1 Score", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Training F1 Score"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Training F1 Score"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Training F1 Score", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Accuracy
                plt.figure(figsize = (width, length))
                plt.title("Test Accuracy", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Accuracy"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Accuracy"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Accuracy", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Precision
                plt.figure(figsize = (width, length))
                plt.title("Test Precision", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Precision"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Precision"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Precision", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test Recall
                plt.figure(figsize = (width, length))
                plt.title("Test Recall", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Recall"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Recall"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test Recall", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Test F1 Score
                plt.figure(figsize = (width, length))
                plt.title("Test F1 Score", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Test Model F1 Score"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Test Model F1 Score"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Test F1 Score", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Cross Validation Mean
                plt.figure(figsize = (width, length))
                plt.title("Cross Validation Mean", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Mean"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Mean"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Cross Validation Mean", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
                
                
                # Cross Validation Standard Deviation
                plt.figure(figsize = (width, length))
                plt.title("Cross Validation Standard Deviation", pad = 10)
                plt.plot(dataset_classifiers["Algorithm"], dataset_classifiers["Cross Validation Standard Deviation"], 'go--', linestyle = 'dashed', marker = 'o', markersize = 12)
                for x1, y1 in zip(dataset_classifiers.Algorithm, round(dataset_classifiers["Cross Validation Standard Deviation"], 4)):
                    plt.text(x = x1, y = y1 + 0.0002, s = str(y1), horizontalalignment = "center", verticalalignment = "bottom", size = "large", fontweight = 80, fontstretch = 50)
                plt.xlabel("Algorithm", labelpad = 20)
                plt.ylabel("Cross Validation Standard Deviation", labelpad = 20)
                # plt.yticks(np.arange(0.0, 1.0, 0.1))
                plt.show()
            
        return {"Classifier Metrics": dataset_classifiers, "More Info": self.__multiple_classifier_models}
    
    
    def build_single_regressor_from_features(self, strategy: str, estimator: str, regressor, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
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
    
    def build_single_classifier_from_features(self, strategy: str, estimator: str, classifier, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
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
    
    
    def build_multiple_regressors_from_features(self, strategy: str, estimator: str, regressors: list or tuple, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
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
            
    
    
    def build_multiple_classifiers_from_features(self, strategy: str, estimator: str, classifiers: list or tuple, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False):
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
    
    
    
    def classifier_evaluation(self, kfold: int = None, cross_validation: bool = False):
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
                raise ValueError
                
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
        
        
        
    def classifier_model_testing(self, variables_values: list, scaling: bool = False):
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


    def numerical_to_categorical(self, column):
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
                
    
    
    def column_binning(self, column, number_of_bins: int = 10):
        if isinstance(column, list):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = False)
        
        elif isinstance(column, str):
            self.__data[column] = pd.cut(self.__data[column], bins = number_of_bins, labels = False)
            
        elif isinstance(column, tuple):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = False)
        
        return self.__data
    
    
    
    def fix_unbalanced_dataset(self, sampler: str, k_neighbors: int = None, random_state: int = None):
        if random_state == None:
            if sampler == "SMOTE" and k_neighbors != None:
                technique = ios.SMOTE(random_state = 0, k_neighbors = k_neighbors)
                self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
            
            elif sampler == "SMOTE" and k_neighbors == None:
                technique = ios.SMOTE(random_state = 0)
                self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
            elif sampler == "Random over sampler" and k_neighbors == None:
                technique = ios.RandomOverSampler(random_state = 0)
                self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
            elif sampler == "Random under sampler" and k_neighbors == None:
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
                
            elif sampler == "Random over sampler" and k_neighbors == None:
                technique = ios.RandomOverSampler(random_state = random_state)
                self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
            elif sampler == "Random under sampler" and k_neighbors == None:
                technique = ius.RandomUnderSampler(random_state = random_state)
                self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
                
            else:
                raise ValueError("k_neighbors works with only the SMOTE algorithm.")
            
            return {"Training X": self.__x_train, "Training Y": self.__y_train}
        
    
    

    def replace_values(self, replace: int or float or str or list or tuple or dict, new_value: int or float or str or list or tuple):
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
        
        return {"Dataset ---> Dataset with Replaced Values": self.__data}
    
    
    
    def sort_values(self, column: str or list, ascending: bool = True, reset_index: bool = False):
        if isinstance(column, str) or isinstance(column, list):
            self.__data.sort_values(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
        
        return {"Dataset ---> Sorted Dataset": self.__data}
    
    
    
    def set_index(self, column: str or list):
        if isinstance(column, str) or isinstance(column, list):
            self.__data = self.__data.set_index(column)
        
        return {"Dataset ---> Index Set": self.__data}
    
    
    
    def sort_index(self, column: str or list, ascending: bool = True, reset_index: bool = False):
        if isinstance(column, str) or isinstance(column, list):
            self.__data.sort_index(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
        
        return {"Dataset ---> Sorted Dataset": self.__data}
    
    
    
    def rename_columns(self, old_column: str or list, new_column: str or list):
        if isinstance(old_column, str) and isinstance(new_column, str):
            self.__data.rename({old_column: new_column}, axis = 1, inplace = True)
            
        elif isinstance(old_column, list) and isinstance(new_column, list):
            self.__data.rename({key:value for key, value in zip(old_column, new_column)}, axis = 1, inplace = True)
        
        return {"Dataset ---> Column Changed": self.__data}
    
    
    
    def reset_index(self, drop_index_after_reset: bool = False):
        self.__data.reset_index(drop = drop_index_after_reset, inplace = True)
        
        return {"Dataset ---> Column Changed": self.__data}
    
    
    
    def filter_data(self, column: str or list or tuple, operation: str or list or tuple = None, value: int or float or str or list or tuple = None):
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
        if isinstance(which_columns, str) or isinstance(which_columns, list) or isinstance(which_columns, tuple):
            self.__data.drop_duplicates(inplace = True, subset = which_columns)
            
        else:
            raise ValueError("Removing duplicates from your dataset must be done by indicating the column as either a string, list, or tuple.")
        
        return {"Dataset ---> Removed Duplicates": self.__data}
    
    
    
    def select_features(self, strategy: str, estimator: str, number_of_features: int):
        types = ["rfe", "selectkbest", "selectfrommodel", "selectpercentile"]
        rfe_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
        kbest_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2"]
        frommodel_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
        percentile_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2"]
        
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
        
        
    
    def group_data(self, columns: list or tuple, column_to_groupby: str or list or tuple, aggregate_function: str, reset_index: bool = False):
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
        
        return grouped_columns               
             
    
    def count_column_categories(self, column: str or list or tuple, reset_index: bool = False):
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
            raise ValueError("The arguments for 'reset_index' must be boolean of TRUE or FALSE.")
            
        return categories_count
    
    
    
    def sweetviz_profile_report(self, filename: str = "Pandas Profile Report.html", auto_open: bool = False):
        report1 = sv.analyze(self.__data)
        report1.show_html(filepath = filename, open_browser = auto_open)
        
        
        
    def pandas_profiling(self, output_file: str = "Pandas Profile Report.html", dark_mode: bool = False, title: str = "Report"):
        report = pp(df = self.__data, dark_mode = dark_mode, explorative = True, title = title)
        report.to_widgets()
        report.to_file(output_file = output_file)
    
    
    def select_datatype(self, datatype_to_select: str = None, datatype_to_exclude: str = None):
        selected_data = self.__data.select_dtypes(include = datatype_to_select, exclude = datatype_to_exclude)
        return selected_data
    
    def load_large_dataset(self, dataset):
        self.__data = dt.fread(dataset).to_pandas()
        return self.__data
    
    def reduce_data_memory_useage(self, verbose: bool = True):
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
    
    

def k_knn_classifier(self, weight = "uniform", algorithm = "auto", metric = "minkowski", max_k_range: int = 31):
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
        plt.figure(figsize = (15, 10))
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
    
    
    
