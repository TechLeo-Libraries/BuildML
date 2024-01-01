import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from ydata_profiling import ProfileReport
import sweetviz as sv
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")


__version__ = "0.0.1"


"""
THINGS TO DO:
    1) Data binning
    2) Create conditions for only when functions can work after certain steps have been taken in our pipeline
    3) Numerical to categorical
"""




class MultiPartSupervisedLearning: 
    
    __all__ = [
        "drop_columns", 
        "get_dataset", 
        "fix_missing_values", 
        "categorical_to_numerical",
        "remove_outlier",
        "scale_independent_variables",
        "eda",
        "eda_visual",
        "select_dependent_and_independent",
        "split_data",
        "train_model_regressor",
        "train_model_classifier",
        "regressor_predict",
        "classifier_predict",
        "regressor_model_testing",
        "regressor_evaluation",
        "select_datatype",
        "classifier_evaluation",
        "classifier_model_testing"
        ]
    
    datasets = []    
    def __init__(self, dataset: list, user_guide: bool =  False):
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
        
        
        
    def get_dataset(self):
        return self.__data
        
    
    
    def drop_columns(self, columns: list):
        self.__data = self.__data.drop(columns, axis = 1, errors = "ignore")
        return self.__data
    
    
    
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
    
    
    
    def eda_visual(self, num: int, before_data_cleaning: bool = True):    
        if before_data_cleaning == False:
            data_histogram = self.__data.hist(figsize = (15, 10), bins = 10)
            plt.show()
            
            plt.figure(figsize = (15, 10))
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title(f'Correlation Matrix for Dataset {num}')
            plt.show()
        
        elif before_data_cleaning == True:
            # Visualize the distribution of categorical features
            categorical_features = self.__data.select_dtypes(include = "object").columns
            for feature in categorical_features:
                plt.figure(figsize=(8, 5))
                sns.countplot(x=feature, data = self.__data)
                plt.title(f'Distribution of {feature} for Dataset {num}')
                plt.show()
              
            data_histogram = self.__data.hist(figsize = (15, 10), bins = 10)
            plt.show()
            
            plt.figure(figsize = (15, 10))
            data_heatmap = sns.heatmap(self.__data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
            plt.title(f'Correlation Matrix for Dataset {num}')
            plt.show()
        self.__eda_visual = True
        
        
    
    def fix_missing_values(self, strategy: str = None):
        self.__strategy = strategy
        if self.__strategy == None:
            imputer = SimpleImputer(strategy = "mean")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_, dtype = np.int64)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "mean":
            imputer = SimpleImputer(strategy = "mean")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_, dtype = np.int64)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "median":
            imputer = SimpleImputer(strategy = "median")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_, dtype = np.int64)
            self.__fixed_missing = True
            return self.__data
            
        elif self.__strategy.lower().strip() == "mode":
            imputer = SimpleImputer(strategy = "most_frequent")
            self.__data = pd.DataFrame(imputer.fit_transform(self.__data), columns = imputer.feature_names_in_, dtype = np.int64)
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
            scaler = StandardScaler()
            self.__data = scaler.fit_transform(self.__data)
            self.__data = pd.DataFrame(self.__data, columns = scaler.feature_names_in_)
            self.__data = self.__data[(self.__data >= -3) & (self.__data <= 3)]
            self.__data = pd.DataFrame(scaler.inverse_transform(self.__data), columns = scaler.feature_names_in_)
            self.__remove_outlier = True
            
        elif drop_na == True:
            scaler = StandardScaler()
            self.__data = scaler.fit_transform(self.__data)
            self.__data = pd.DataFrame(self.__data, columns = scaler.feature_names_in_)
            self.__data = self.__data[(self.__data >= -3) & (self.__data <= 3)].dropna()
            self.__data = pd.DataFrame(scaler.inverse_transform(self.__data), columns = scaler.feature_names_in_)
            self.__remove_outlier = True
            
        return self.__data
        
    
    
    def scale_independent_variables(self):
        self.__scaler = StandardScaler()
        self.__x = self.__scaler.fit_transform(self.__x)
        self.__x = pd.DataFrame(self.__x, columns = self.__scaler.feature_names_in_)
        self.__scaled = True
        return {"Dependent Variable": self.__y, "Scaled Independent Variables": self.__x}   
    
    
    
    def split_data(self):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y, test_size = 0.2, random_state = 0)
        self.__split_data = True
        return {"Training X": self.__x_train, "Test X": self.__x_test, "Training Y": self.__y_train, "Test Y": self.__y_test}
    
    
    
    def select_dependent_and_independent(self, predict: str):
        self.__x = self.__data.drop(predict, axis = 1,)
        self.__y = self.__data[f"{predict}"]
        self.__dependent_independent = True
        return {"Dependent Variable": self.__y, "Independent Variables": self.__x}
    
    
    
    def build_model_regressor(self, regressor, kfold: int = None, cross_validation: bool = False):
        self.regressor = regressor
        self.model_regressor = self.regressor.fit(self.__x_train, self.__y_train)
        score = self.model_regressor.score(self.__x_train, self.__y_train)
        print(f"Amount of variation in Y predicted by your features X: (Rsquared) ----> {score}")
        self.__model_training = True
        
        self.__y_pred = self.model_regressor.predict(self.__x_train)
        self.__y_pred1 = self.model_regressor.predict(self.__x_test)
        self.__model_prediction = True
        
        if kfold == None and cross_validation == False:
            training_rsquared = r2_score(self.__y_train, self.__y_pred)
            test_rsquared = r2_score(self.__y_test, self.__y_pred1)
            
            training_rmse = np.sqrt(mean_squared_error(self.__y_train, self.__y_pred))
            test_rmse = np.sqrt(mean_squared_error(self.__y_test, self.__y_pred1))
            self.__model_evaluation = True
            return {
                "Built Model": self.model_regressor,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation": {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}}
                }
            
            
        elif kfold != None and cross_validation == False:
            raise ValueError
            
        elif kfold == None and cross_validation == True:
            training_rsquared = r2_score(self.__y_train, self.__y_pred)
            test_rsquared = r2_score(self.__y_test, self.__y_pred1)
            
            training_rmse = np.sqrt(mean_squared_error(self.__y_train, self.__y_pred))
            test_rmse = np.sqrt(mean_squared_error(self.__y_test, self.__y_pred1))
            
            cross_val = cross_val_score(self.model_regressor, self.__x_train, self.__y_train, cv = 10)    
            score_mean = round((cross_val.mean() * 100), 2)
            score_std_dev = round((cross_val.std() * 100), 2)
            self.__model_evaluation = True
            return {
                "Built Model": self.model_regressor,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation": {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
                }
            
            
        elif kfold != None and cross_validation == True:
            training_rsquared = r2_score(self.__y_train, self.__y_pred)
            test_rsquared = r2_score(self.__y_test, self.__y_pred1)
            
            training_rmse = np.sqrt(mean_squared_error(self.__y_train, self.__y_pred))
            test_rmse = np.sqrt(mean_squared_error(self.__y_test, self.__y_pred1))
            
            cross_val = cross_val_score(self.model_regressor, self.__x_train, self.__y_train, cv = kfold)    
            score_mean = round((cross_val.mean() * 100), 2)
            score_std_dev = round((cross_val.std() * 100), 2)
            self.__model_evaluation = True
            return {
                "Built Model": self.model_regressor,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation":  {"Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
                }
            
        
        
    def regressor_model_testing(self, variables_values: list, scaling: bool = False):
        if scaling == False:
            prediction = self.model_regressor.predict([variables_values])
            self.__model_testing = True
            return prediction
        
        elif scaling == True:
            variables_values = self.__scaler.transform([variables_values])
            prediction = self.model_regressor.predict(variables_values)
            self.__model_testing = True
            return prediction   
        
    
    
    def build_model_classifier(self, classifier, kfold: int = None, cross_validation: bool = False):
        self.classifier = classifier
        self.model_classifier = self.classifier.fit(self.__x_train, self.__y_train)
        score = self.model_classifier.score(self.__x_train, self.__y_train)
        print(f"Accuracy in model prediction: (Accuracy) ----> {score}")
        self.__model_training = True
        
        self.__y_pred = self.model_classifier.predict(self.__x_train)
        self.__y_pred1 = self.model_classifier.predict(self.__x_test)
        self.__model_prediction = True
    
        if kfold == None and cross_validation == False:
            training_analysis = confusion_matrix(self.__y_train, self.__y_pred)
            training_class_report = classification_report(self.__y_train, self.__y_pred)
            training_accuracy = accuracy_score(self.__y_train, self.__y_pred)
            training_precision = precision_score(self.__y_train, self.__y_pred, average='weighted')
            training_recall = recall_score(self.__y_train, self.__y_pred, average='weighted')
            training_f1_score = f1_score(self.__y_train, self.__y_pred, average='weighted')

            test_analysis = confusion_matrix(self.__y_test, self.__y_pred1)
            test_class_report = classification_report(self.__y_test, self.__y_pred1)
            test_accuracy = accuracy_score(self.__y_test, self.__y_pred1)
            test_precision = precision_score(self.__y_test, self.__y_pred1, average='weighted')
            test_recall = recall_score(self.__y_test, self.__y_pred1, average='weighted')
            test_f1_score = f1_score(self.__y_test, self.__y_pred1, average='weighted')
            self.__model_evaluation = True
            return {
                "Built Model": self.model_classifier,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation": {
                       "Training Evaluation": {
                           "Confusion Matrix": training_analysis,
                           "Classification Report": training_class_report,
                           "Model Accuracy": training_accuracy,
                           "Model Precision": training_precision,
                           "Model Recall": training_recall,
                           "Model F1 Score": training_f1_score,
                           },
                       "Test Evaluation": {
                           "Confusion Matrix": test_analysis,
                           "Classification Report": test_class_report,
                           "Model Accuracy": test_accuracy,
                           "Model Precision": test_precision,
                           "Model Recall": test_recall,
                           "Model F1 Score": test_f1_score,
                           }, 
                    }
                }
        
        elif kfold != None and cross_validation == False:
            raise ValueError
            
        elif kfold == None and cross_validation == True:
            training_analysis = confusion_matrix(self.__y_train, self.__y_pred)
            training_class_report = classification_report(self.__y_train, self.__y_pred)
            training_accuracy = accuracy_score(self.__y_train, self.__y_pred)
            training_precision = precision_score(self.__y_train, self.__y_pred, average='weighted')
            training_recall = recall_score(self.__y_train, self.__y_pred, average='weighted')
            training_f1_score = f1_score(self.__y_train, self.__y_pred, average='weighted')
    
            test_analysis = confusion_matrix(self.__y_test, self.__y_pred1)
            test_class_report = classification_report(self.__y_test, self.__y_pred1)
            test_accuracy = accuracy_score(self.__y_test, self.__y_pred1)
            test_precision = precision_score(self.__y_test, self.__y_pred1, average='weighted')
            test_recall = recall_score(self.__y_test, self.__y_pred1, average='weighted')
            test_f1_score = f1_score(self.__y_test, self.__y_pred1, average='weighted')
            
            cross_val = cross_val_score(self.model_classifier, self.__x_train, self.__y_train, cv = 10)    
            score_mean = round((cross_val.mean() * 100), 2)
            score_std_dev = round((cross_val.std() * 100), 2)
            self.__model_evaluation = True
            return {
                "Built Model": self.model_classifier,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation": {
                        "Training Evaluation": {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1_score,
                            },
                        "Test Evaluation": {
                            "Confusion Matrix": test_analysis,
                            "Classification Report": test_class_report,
                            "Model Accuracy": test_accuracy,
                            "Model Precision": test_precision,
                            "Model Recall": test_recall,
                            "Model F1 Score": test_f1_score,
                            },
                        "Cross Validation": {
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }
                    }
                }
        
        elif kfold != None and cross_validation == True:
            training_analysis = confusion_matrix(self.__y_train, self.__y_pred)
            training_class_report = classification_report(self.__y_train, self.__y_pred)
            training_accuracy = accuracy_score(self.__y_train, self.__y_pred)
            training_precision = precision_score(self.__y_train, self.__y_pred, average='weighted')
            training_recall = recall_score(self.__y_train, self.__y_pred, average='weighted')
            training_f1_score = f1_score(self.__y_train, self.__y_pred, average='weighted')
    
            test_analysis = confusion_matrix(self.__y_test, self.__y_pred1)
            test_class_report = classification_report(self.__y_test, self.__y_pred1)
            test_accuracy = accuracy_score(self.__y_test, self.__y_pred1)
            test_precision = precision_score(self.__y_test, self.__y_pred1, average='weighted')
            test_recall = recall_score(self.__y_test, self.__y_pred1, average='weighted')
            test_f1_score = f1_score(self.__y_test, self.__y_pred1, average='weighted')
            
            cross_val = cross_val_score(self.model_classifier, self.__x_train, self.__y_train, cv = kfold)    
            score_mean = round((cross_val.mean() * 100), 2)
            score_std_dev = round((cross_val.std() * 100), 2)
            self.__model_evaluation = True
            return {
                "Built Model": self.model_classifier,
                "Predictions": {"Actual Training Y": self.__y_train, "Actual Test Y": self.__y_test, "Predicted Training Y": self.__y_pred, "Predicted Test Y": self.__y_pred1},
                "Evaluation": {
                        "Training Evaluation": {
                            "Confusion Matrix": training_analysis,
                            "Classification Report": training_class_report,
                            "Model Accuracy": training_accuracy,
                            "Model Precision": training_precision,
                            "Model Recall": training_recall,
                            "Model F1 Score": training_f1_score,
                            },
                        "Test Evaluation": {
                            "Confusion Matrix": test_analysis,
                            "Classification Report": test_class_report,
                            "Model Accuracy": test_accuracy,
                            "Model Precision": test_precision,
                            "Model Recall": test_recall,
                            "Model F1 Score": test_f1_score,
                            },
                        "Cross Validation": {
                            "Cross Validation Mean": score_mean, 
                            "Cross Validation Standard Deviation": score_std_dev
                            }
                    }            
                }
      
        
        
    def classifier_model_testing(self, variables_values: list, scaling: bool = False):
        if scaling == False:
            prediction = self.model_classifier.predict([variables_values])
            self.__model_testing = True
            return prediction
        
        elif scaling == True:
            variables_values = self.__scaler.transform([variables_values])
            prediction = self.model_classifier.predict(variables_values)
            self.__model_testing = True
            return prediction
        
    
    
    def classifier_graph(self, num: int, classifier, xlabel: str, ylabel: str, color1: str, color2: str):
        # Visualising the Training set results
        X_set, y_set = self.__x_train, self.__y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap((f'{color1}', '{color2}')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap((f'{color1}', '{color2}'))(i), label = j)
        plt.title(f'{classifier.__class__.__name__} (Training set) for Dataset {num}')
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.legend()
        plt.show()
        
        
        # Visualising the Test set results
        X_set, y_set = self.__x_test, self.__y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap((f'{color1}', '{color2}')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap((f'{color1}', '{color2}'))(i), label = j)
        plt.title(f'{classifier.__class__.__name__} (Test set) for Dataset {num}')
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.legend()
        plt.show()
        
    
    
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
    
    def extract_date_features(self, datetime_column, convert_without_extract: bool = False, hrs_mins_sec: bool = False, day_first: bool = False, yearfirst: bool = False, date_format: str = None):
        if hrs_mins_sec == False and convert_without_extract == False:
            if isinstance(datetime_column, list):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day
                    self.__data[f"Month_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.month
                    self.__data[f"Year_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.year
                    self.__data[f"Quarter_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.quarter
                    self.__data[f"Day_of_Week_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day_of_week
            
            elif isinstance(datetime_column, str):
                self.__data["Day"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.day
                self.__data["Month"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.month
                self.__data["Year"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.year
                self.__data["Quarter"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.quarter
                self.__data["Day_of_Week"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.day_of_week
                
            elif isinstance(datetime_column, tuple):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day
                    self.__data[f"Month_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.month
                    self.__data[f"Year_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.year
                    self.__data[f"Quarter_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.quarter
                    self.__data[f"Day_of_Week_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day_of_week
            
            return self.__data
        
        elif hrs_mins_sec == True and convert_without_extract == False:
            if isinstance(datetime_column, list):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day
                    self.__data[f"Month_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.month
                    self.__data[f"Year_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.year
                    self.__data[f"Quarter_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.quarter
                    self.__data[f"Day_of_Week_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day_of_week
                    self.__data[f"Hour_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).hour
                    self.__data[f"Minutes_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).minute
                    self.__data[f"Seconds_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).second                   
                    
            elif isinstance(datetime_column, str):
                self.__data["Day"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.day
                self.__data["Month"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.month
                self.__data["Year"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.year
                self.__data["Quarter"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.quarter
                self.__data["Day_of_Week"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).dt.day_of_week
                self.__data[f"Hour_{items}"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).hour
                self.__data[f"Minutes_{items}"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).minute
                self.__data[f"Seconds_{items}"] = pd.to_datetime(self.__data[datetime_column], yearfirst = yearfirst, dayfirst = day_first, format = date_format).second  
                
            elif isinstance(datetime_column, tuple):
                for items in datetime_column:
                    self.__data[f"Day_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day
                    self.__data[f"Month_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.month
                    self.__data[f"Year_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.year
                    self.__data[f"Quarter_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.quarter
                    self.__data[f"Day_of_Week_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).dt.day_of_week
                    self.__data[f"Hour_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).hour
                    self.__data[f"Minutes_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).minute
                    self.__data[f"Seconds_{items}"] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format).second   
            
            return self.__data
        
        if convert_without_extract == True:
            if isinstance(datetime_column, list):
                for items in datetime_column:
                    self.__data[items] = pd.to_datetime(self.__data[items], dayfirst = day_first, yearfirst = yearfirst, format = date_format)
                
            elif isinstance(datetime_column, str):
                self.__data[datetime_column] = pd.to_datetime(self.__data[datetime_column], dayfirst = day_first, yearfirst = yearfirst, format = date_format)
                
            return self.__data
                
    
    
    def column_binning(self, column: str or list or tuple, number_of_bins: int = 10):
        if isinstance(column, list):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = False)
        
        elif isinstance(column, str):
            self.__data[column] = pd.cut(self.__data[column], bins = number_of_bins, labels = False)
            
        elif isinstance(column, tuple):
            for items in column:
                self.__data[items] = pd.cut(self.__data[items], bins = number_of_bins, labels = False)
        
        return self.__data
    
    
    
    def fix_unbalanced_dataset(self, sampler: str, k_neighbors: int = None):
        if sampler.upper().strip() == "SMOTE" and k_neighbors != None:
            technique = SMOTE(random_state = 0, k_neighbors = k_neighbors)
            self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
        
        elif sampler.upper().strip() == "SMOTE" and k_neighbors == None:
            technique = SMOTE(random_state = 0)
            self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
            
        elif sampler.lower().strip() == "over sampler" and k_neighbors == None:
            technique = RandomOverSampler(random_state = 0)
            self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
            
        elif sampler.lower().strip() == "under sampler" and k_neighbors == None:
            technique = RandomUnderSampler(random_state = 0)
            self.__x_train, self.__y_train = technique.fit_resample(self.__x_train, self.__y_train)
            
        else:
            ValueError("k_neighbors works with only the SMOTE algorithm.")
        
        return {"Training X": self.__x_train, "Training Y": self.__y_train, "Training Y ---> Value Counts": self.__y_train.value_counts()}  
    
    
    
    def get_training_test_data(self):
        return {"Training X": self.__x_train, "Training Y": self.__y_train, "Test X": self.__x_test, "Test Y": self.__y_test}
 
    
 
    def get_trained_models(self, model: str):
        types_regressor = ["regress", "reg", "regression", "regressor", "r", "regres"]
        types_classifier = ["classifier", "class", "classification", "clas", "c", "classif"]
        
        if model in types_regressor:
            return self.model.regressor
        
        elif model in types_classifier:
            return self.model.classifier
        