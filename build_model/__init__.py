import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import sklearn.model_selection as sms
import sklearn.metrics as sm
import sklearn.feature_selection as sfs
import sklearn.neighbors as sn
import warnings


__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"


def select_features(x, y, strategy: str, estimator: str, number_of_features: int, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    types = ["rfe", "selectkbest", "selectfrommodel", "selectpercentile"]
    rfe_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
    kbest_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2"]
    frommodel_possible_estimator = "A regression or classification algorithm that can implement 'fit'."
    percentile_possible_score_functions = ["f_regression", "f_classif", "f_oneway", "chi2"]
    
    strategy = strategy.lower()
    if strategy in types: 
        if strategy == "rfe" and estimator != None:
            technique = sfs.RFE(estimator = estimator, n_features_to_select = number_of_features)
            x = technique.fit_transform(x, y)
            
            x = pd.DataFrame(x, columns = technique.get_feature_names_out())
            return x
            
        elif strategy == "selectkbest" and estimator != None:
            technique = sfs.SelectKBest(score_func = estimator, k = number_of_features)
            x = technique.fit_transform(x, y)
            
            x = pd.DataFrame(x, columns = technique.get_feature_names_out())
            best_features = pd.DataFrame({"Features": technique.feature_names_in_, "Scores": technique.scores_, "P_Values": technique.pvalues_})
            return {"Dataset ---> Features Selected": x, "Selection Metrics": best_features}
            
        elif strategy == "selectfrommodel" and estimator != None:
            technique = sfs.SelectFromModel(estimator = estimator, max_features = number_of_features)
            x = technique.fit_transform(x, y)
            
            x = pd.DataFrame(x, columns = technique.get_feature_names_out())
            return x
            
        elif strategy == "selectpercentile" and estimator != None:
            technique = sfs.SelectPercentile(score_func = estimator, percentile = number_of_features)
            x = technique.fit_transform(x, y)
            
            x = pd.DataFrame(x, columns = technique.get_feature_names_out())
            best_features = pd.DataFrame({"Features": technique.feature_names_in_, "Scores": technique.scores_, "P_Values": technique.pvalues_})
            return {"Dataset ---> Features Selected": x, "Selection Metrics": best_features}
        
        elif estimator == None:
            raise ValueError("You must specify an estimator or score function to use feature selection processes")
            
    else:
        raise ValueError(f"Select a feature selection technique from the following: {types}. \n\nRFE Estimator = {rfe_possible_estimator} e.g XGBoost, RandomForest, SVM etc\nSelectKBest Score Function = {kbest_possible_score_functions}\nSelectFromModel Estimator = {frommodel_possible_estimator} e.g XGBoost, RandomForest, SVM etc.\nSelectPercentile Score Function = {percentile_possible_score_functions}")
    
    

def split_data(x, y, test_size, random_state, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = test_size, random_state = random_state)
    return {"Training X": x_train, "Test X": x_test, "Training Y": y_train, "Test Y": y_test}

def build_regressor_model(regressor, x_train, y_train, x_test, y_test, kfold: int = None, cross_validation: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    model = regressor.fit(x_train, y_train)
    
    y_pred = model.predict(x_train)
    y_pred1 = model.predict(x_test)
    
    if kfold == None and cross_validation == False:
        training_rsquared = sm.r2_score(y_train, y_pred)
        test_rsquared = sm.r2_score(y_test, y_pred1)
        
        training_rmse = np.sqrt(sm.mean_squared_error(y_train, y_pred))
        test_rmse = np.sqrt(sm.mean_squared_error(y_test, y_pred1))
        return {"Model": model, "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1}, "Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}}
    
    elif kfold != None and cross_validation == False:
        raise ValueError("KFold cannot work when cross validation is set to FALSE")
        
    elif kfold == None and cross_validation == True:
        training_rsquared = sm.r2_score(y_train, y_pred)
        test_rsquared = sm.r2_score(y_test, y_pred1)
        
        training_rmse = np.sqrt(sm.mean_squared_error(y_train, y_pred))
        test_rmse = np.sqrt(sm.mean_squared_error(y_test, y_pred1))
        
        cross_val = sms.cross_val_score(model, x_train, y_train, cv = 10)    
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {"Model": model, "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1}, "Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}
    
    elif kfold != None and cross_validation == True:
        training_rsquared = sm.r2_score(y_train, y_pred)
        test_rsquared = sm.r2_score(y_test, y_pred1)
        
        training_rmse = np.sqrt(sm.mean_squared_error(y_train, y_pred))
        test_rmse = np.sqrt(sm.mean_squared_error(y_test, y_pred1))
        
        cross_val = sms.cross_val_score(model, x_train, y_train, cv = kfold)    
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {"Model": model, "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1}, "Training Evaluation": {"Training R2": training_rsquared, "Training RMSE": training_rmse}, "Test Evaluation": {"Test R2": test_rsquared, "Test RMSE": test_rmse}, "Cross Validation": {"Cross Validation Mean": score_mean, "Cross Validation Standard Deviation": score_std_dev}}


def classifier_model_testing(classifier_model, variables_values: list, scaling: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    scaler = sp.StandardScaler()
    if scaling == False:
        prediction = classifier_model.predict([variables_values])
        return prediction
    
    elif scaling == True:
        variables_values = scaler.transform([variables_values])
        prediction = classifier_model.predict(variables_values)
        return prediction
    

def regressor_model_testing(regressor_model, variables_values: list, scaling: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    scaler = sp.StandardScaler()
    if scaling == False:
        prediction = regressor_model.predict([variables_values])
        return prediction
    
    elif scaling == True:
        variables_values = scaler.transform([variables_values])
        prediction = regressor_model.predict(variables_values)
        return prediction
   

def build_classifier_model(classifier, x_train, y_train, x_test, y_test, kfold: int = None, cross_validation: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore") 
    
    model = classifier.fit(x_train, y_train)    
    
    y_pred = model.predict(x_train)
    y_pred1 = model.predict(x_test)
    
    if kfold == None and cross_validation == False:
        training_analysis = sm.confusion_matrix(y_train, y_pred)
        training_class_report = sm.classification_report(y_train, y_pred)
        training_accuracy = sm.accuracy_score(y_train, y_pred)
        training_precision = sm.precision_score(y_train, y_pred, average='weighted')
        training_recall = sm.recall_score(y_train, y_pred, average='weighted')
        training_f1_score = sm.f1_score(y_train, y_pred, average='weighted')

        test_analysis = sm.confusion_matrix(y_test, y_pred1)
        test_class_report = sm.classification_report(y_test, y_pred1)
        test_accuracy = sm.accuracy_score(y_test, y_pred1)
        test_precision = sm.precision_score(y_test, y_pred1, average='weighted')
        test_recall = sm.recall_score(y_test, y_pred1, average='weighted')
        test_f1_score = sm.f1_score(y_test, y_pred1, average='weighted')
        return {
            "Model": model,
            "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1},
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
    
    elif kfold != None and cross_validation == False:
        raise ValueError("KFold cannot work when cross validation is set to FALSE")
        
    elif kfold == None and cross_validation == True:
        training_analysis = sm.confusion_matrix(y_train, y_pred)
        training_class_report = sm.classification_report(y_train, y_pred)
        training_accuracy = sm.accuracy_score(y_train, y_pred)
        training_precision = sm.precision_score(y_train, y_pred, average='weighted')
        training_recall = sm.recall_score(y_train, y_pred, average='weighted')
        training_f1_score = sm.f1_score(y_train, y_pred, average='weighted')

        test_analysis = sm.confusion_matrix(y_test, y_pred1)
        test_class_report = sm.classification_report(y_test, y_pred1)
        test_accuracy = sm.accuracy_score(y_test, y_pred1)
        test_precision = sm.precision_score(y_test, y_pred1, average='weighted')
        test_recall = sm.recall_score(y_test, y_pred1, average='weighted')
        test_f1_score = sm.f1_score(y_test, y_pred1, average='weighted')
        
        cross_val = sms.cross_val_score(model, x_train, y_train, cv = 10)    
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {
            "Model": model,
            "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1},
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
    
    elif kfold != None and cross_validation == True:
        training_analysis = sm.confusion_matrix(y_train, y_pred)
        training_class_report = sm.classification_report(y_train, y_pred)
        training_accuracy = sm.accuracy_score(y_train, y_pred)
        training_precision = sm.precision_score(y_train, y_pred, average='weighted')
        training_recall = sm.recall_score(y_train, y_pred, average='weighted')
        training_f1_score = sm.f1_score(y_train, y_pred, average='weighted')

        test_analysis = sm.confusion_matrix(y_test, y_pred1)
        test_class_report = sm.classification_report(y_test, y_pred1)
        test_accuracy = sm.accuracy_score(y_test, y_pred1)
        test_precision = sm.precision_score(y_test, y_pred1, average='weighted')
        test_recall = sm.recall_score(y_test, y_pred1, average='weighted')
        test_f1_score = sm.f1_score(y_test, y_pred1, average='weighted')
        
        cross_val = sms.cross_val_score(model, x_train, y_train, cv = kfold)    
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {
            "Model": model,
            "Predictions": {"Actual Training Y": y_train, "Actual Test Y": y_test, "Predicted Training Y": y_pred, "Predicted Test Y": y_pred1},
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


def build_multiple_regressors(regressors: list or tuple, x_train, y_train, x_test, y_test, kfold: int = None, cross_validation: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    if isinstance(regressors, list) or isinstance(regressors, tuple):
        multiple_regressor_models = {}
        for algorithms in regressors:
            multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(regressor = algorithms, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold , cross_validation = cross_validation)
            
    return multiple_regressor_models


def build_multiple_classifiers(classifiers: list or tuple, x_train, y_train, x_test, y_test, kfold: int = None, cross_validation: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    if isinstance(classifiers, list) or isinstance(classifiers, tuple):
        multiple_classifier_models = {}
        for algorithms in classifiers:
            multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(classifier = algorithms, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold , cross_validation = cross_validation)
        
    return multiple_classifier_models


def build_single_regressor_from_features(x, y, regressor, test_size: float, random_state: int, strategy: str, estimator: str, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    types1 = ["selectkbest", "selectpercentile"]
    types2 = ["rfe", "selectfrommodel"]
    
    if not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == False:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_regressor_models = {}
                store_data = []
                
                multiple_regressor_models[f"{regressor.__class__.__name__}"] = build_regressor_model(regressor, x_train, y_train, x_test, y_test, kfold = kfold, cross_validation = cv)
                info = [
                    num,
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Model"].__class__.__name__, 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test RMSE"]
                    ]
                store_data.append(info)
                    
                dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                store[f"{num}"] = {}
                store[f"{num}"] = multiple_regressor_models
                store[f"{num}"]["Feature Info"] = feature_info
                
                dataset2 = dataset_regressors
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_regressor_models = {}
                    store_data = []
                    
                    multiple_regressor_models[f"{regressor.__class__.__name__}"] = build_regressor_model(regressor, x_train, y_train, x_test, y_test, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Model"].__class__.__name__, 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test RMSE"]
                        ]
                    store_data.append(info)
                        
                    dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                    
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                    store[f"{num}"] = {}
                    store[f"{num}"] = multiple_regressor_models
                    store[f"{num}"]["Feature Info"] = feature_info
                    
                    dataset2 = dataset_regressors
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

            
    elif not (isinstance(regressor, list) or isinstance(regressor, tuple)) and cv == True:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_regressor_models = {}
                store_data = []
                
                multiple_regressor_models[f"{regressor.__class__.__name__}"] = build_regressor_model(regressor, x_train, y_train, x_test, y_test, kfold = kfold, cross_validation = cv)
                info = [
                    num,
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Model"].__class__.__name__, 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test RMSE"],
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"],
                    multiple_regressor_models[f"{regressor.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"],
                    ]
                store_data.append(info)
                    
                dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                store[f"{num}"] = {}
                store[f"{num}"] = multiple_regressor_models
                store[f"{num}"]["Feature Info"] = feature_info
                
                dataset2 = dataset_regressors
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_regressor_models = {}
                    store_data = []
                    
                    multiple_regressor_models[f"{regressor.__class__.__name__}"] = build_regressor_model(regressor, x_train, y_train, x_test, y_test, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Model"].__class__.__name__, 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Test Evaluation"]["Test RMSE"],
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"],
                        multiple_regressor_models[f"{regressor.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"],
                        ]
                    store_data.append(info)
                    
                    dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                    store[f"{num}"] = {}
                    store[f"{num}"] = multiple_regressor_models
                    store[f"{num}"]["Feature Info"] = feature_info
                    
                    
                    dataset2 = dataset_regressors
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
           
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

        
        
    dataset_features = dataset_features.reset_index(drop = True)
    return {"Feature Metrics": dataset_features, "More Info": store}

def build_single_classifier_from_features(x, y, classifier, test_size: float, random_state: int, strategy: str, estimator: str, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    types1 = ["selectkbest", "selectpercentile"]
    types2 = ["rfe", "selectfrommodel"]
    
    if not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == False:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_classifier_models = {}
                store_data = []
                
                multiple_classifier_models[f"{classifier.__class__.__name__}"] = build_classifier_model(classifier = classifier, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold, cross_validation = cv)
                info = [
                    num,
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Model"].__class__.__name__, 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                    ]
                store_data.append(info)
                  
                dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                store[f"{num}"] = {}
                store[f"{num}"] = multiple_classifier_models
                store[f"{num}"]["Feature Info"] = feature_info
                
                dataset2 = dataset_classifiers
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_classifier_models = {}
                    store_data = []
                    
                    multiple_classifier_models[f"{classifier.__class__.__name__}"] = build_classifier_model(classifier = classifier, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Model"].__class__.__name__, 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                        ]
                    store_data.append(info)
                      
                    dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                    store[f"{num}"] = {}
                    store[f"{num}"] = multiple_classifier_models
                    store[f"{num}"]["Feature Info"] = feature_info
                    
                    dataset2 = dataset_classifiers
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
    
    elif not (isinstance(classifier, list) or isinstance(classifier, tuple)) and cv == True:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range((length_col - 1), 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_classifier_models = {}
                store_data = []
                
                multiple_classifier_models[f"{classifier.__class__.__name__}"] = build_classifier_model(classifier = classifier, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold, cross_validation = cv)
                info = [
                    num,
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Model"].__class__.__name__, 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"], 
                    multiple_classifier_models[f"{classifier.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                    ]
                store_data.append(info)
                  
                dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                store[f"{num}"] = {}
                store[f"{num}"] = multiple_classifier_models
                store[f"{num}"]["Feature Info"] = feature_info
                
                dataset2 = dataset_classifiers
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_classifier_models = {}
                    store_data = []
                    
                    multiple_classifier_models[f"{classifier.__class__.__name__}"] = build_classifier_model(classifier = classifier, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Model"].__class__.__name__, 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"], 
                        multiple_classifier_models[f"{classifier.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                        ]
                    store_data.append(info)
                      
                    dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                    store[f"{num}"] = {}
                    store[f"{num}"] = multiple_classifier_models
                    store[f"{num}"]["Feature Info"] = feature_info
                    
                    dataset2 = dataset_classifiers
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
            
    dataset_features = dataset_features.reset_index(drop = True)
    return {"Feature Metrics": dataset_features, "More Info": store}


def build_multiple_regressors_from_features(x, y, regressors: list or tuple, test_size: float, random_state: int,  strategy: str, estimator: str, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
        
    types1 = ["selectkbest", "selectpercentile"]
    types2 = ["rfe", "selectfrommodel"]
    
    if (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == False:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_regressor_models = {}
                store_data = []
                for algorithms in regressors:
                    multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, regressor = algorithms, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test RMSE"]
                        ]
                    store_data.append(info)
                    
                dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                store[f"{num}"] = {}
                store[f"{num}"]["Feature Info"] = feature_info
                store[f"{num}"]["More Info"] = multiple_regressor_models
                
                dataset2 = dataset_regressors
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)


        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_regressor_models = {}
                    store_data = []
                    for algorithms in regressors:
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, regressor = algorithms, kfold = kfold, cross_validation = cv)
                        info = [
                            num,
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test RMSE"]
                            ]
                        store_data.append(info)
                        
                    dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE"])
                    
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                    store[f"{num}"] = {}
                    store[f"{num}"]["Feature Info"] = feature_info
                    store[f"{num}"]["More Info"] = multiple_regressor_models
                    
                    dataset2 = dataset_regressors
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

            
    elif (isinstance(regressors, list) or isinstance(regressors, tuple)) and cv == True:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_regressor_models = {}
                store_data = []
                for algorithms in regressors:
                    multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, regressor = algorithms, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test RMSE"],
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"],
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"],
                        ]
                    store_data.append(info)
                    
                dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                store[f"{num}"] = {}
                store[f"{num}"]["Feature Info"] = feature_info
                store[f"{num}"]["More Info"] = multiple_regressor_models
                
                dataset2 = dataset_regressors
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_regressor_models = {}
                    store_data = []
                    for algorithms in regressors:
                        multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, regressor = algorithms, kfold = kfold, cross_validation = True)
                        info = [
                            num,
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training R2"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training RMSE"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test R2"], 
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test RMSE"],
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"],
                            multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"],
                            ]
                        store_data.append(info)
                        
                    dataset_regressors = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training R2", "Training RMSE", "Test R2", "Test RMSE", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_regressors
                    store[f"{num}"] = {}
                    store[f"{num}"]["Feature Info"] = feature_info
                    store[f"{num}"]["More Info"] = multiple_regressor_models
                    
                    dataset2 = dataset_regressors
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
           
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")

        
        
    dataset_features = dataset_features.reset_index(drop = True)
    return {"Feature Metrics": dataset_features, "More Info": store}
        


def build_multiple_classifiers_from_features(x, y, classifiers: list or tuple, test_size: float, random_state: int, strategy: str, estimator: str, max_num_features: int = None, min_num_features: int = None, kfold: int = None, cv: bool = False, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
    
    types1 = ["selectkbest", "selectpercentile"]
    types2 = ["rfe", "selectfrommodel"]
    
    if (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == False:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range(length_col, 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_classifier_models = {}
                store_data = []
                for algorithms in classifiers:
                    multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, classifier = algorithms, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                        ]
                    store_data.append(info)
                  
                dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                store[f"{num}"] = {}
                store[f"{num}"]["Feature Info"] = feature_info
                store[f"{num}"]["More Info"] = multiple_classifier_models
                
                dataset2 = dataset_classifiers
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_classifier_models = {}
                    store_data = []
                    for algorithms in classifiers:
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, classifier = algorithms, kfold = kfold, cross_validation = cv)
                        info = [
                            num,
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                            ]
                        store_data.append(info)
                      
                    dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score",])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                    store[f"{num}"] = {}
                    store[f"{num}"]["Feature Info"] = feature_info
                    store[f"{num}"]["More Info"] = multiple_classifier_models
                    
                    dataset2 = dataset_classifiers
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
    
    
    elif (isinstance(classifiers, list) or isinstance(classifiers, tuple)) and cv == True:
        data_columns = [col for col in x.columns]
        length_col = len(data_columns)
        store = {}
        dataset_features = pd.DataFrame(columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
        
        if (max_num_features != None) and isinstance(max_num_features, int):
            length_col = max_num_features
        
        if (min_num_features == None):
            for num in range((length_col - 1), 0, -1):
                
                feature_info = {}
                features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                
                strategy = strategy.lower()
                if strategy in types2: 
                    x = features
                    
                elif strategy in types1: 
                    x = features["Dataset ---> Features Selected"]
                   
                    
                x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                
                multiple_classifier_models = {}
                store_data = []
                for algorithms in classifiers:
                    multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, classifier = algorithms, kfold = kfold, cross_validation = cv)
                    info = [
                        num,
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"], 
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                        ]
                    store_data.append(info)
                  
                dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                
                feature_info[f"{num} Feature(s) Selected"] = features
                feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                store[f"{num}"] = {}
                store[f"{num}"]["Feature Info"] = feature_info
                store[f"{num}"]["More Info"] = multiple_classifier_models
                
                dataset2 = dataset_classifiers
                dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                
                
        elif (min_num_features != None) and isinstance(min_num_features, int):
            if (min_num_features <= length_col):
                for num in range(length_col, (min_num_features - 1), -1):
                    
                    feature_info = {}
                    features = select_features(x = x, y = y, strategy = strategy, estimator = estimator, number_of_features = num)
                    
                    strategy = strategy.lower()
                    if strategy in types2: 
                        x = features
                        
                    elif strategy in types1: 
                        x = features["Dataset ---> Features Selected"]
                       
                        
                    x_train, x_test, y_train, y_test = split_data(x = x, y = y, test_size = test_size, random_state = random_state).values()
                    
                    multiple_classifier_models = {}
                    store_data = []
                    for algorithms in classifiers:
                        multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, classifier = algorithms, kfold = kfold, cross_validation = cv)
                        info = [
                            num,
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Built Model"].__class__.__name__, 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"], 
                            multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"], 
                            ]
                        store_data.append(info)
                      
                    dataset_classifiers = pd.DataFrame(store_data, columns = ["No. of features selected", "Algorithm", "Training Accuracy", "Training Precision", "Training Recall", "Training F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Cross Validation Mean", "Cross Validation Standard Deviation"])
                    
                    feature_info[f"{num} Feature(s) Selected"] = features
                    feature_info[f"Model Trained with {num} Feature(s)"] = dataset_classifiers
                    store[f"{num}"] = {}
                    store[f"{num}"]["Feature Info"] = feature_info
                    store[f"{num}"]["More Info"] = multiple_classifier_models
                    
                    dataset2 = dataset_classifiers
                    dataset_features = pd.concat([dataset_features, dataset2], axis = 0)
                    
            else:
                raise ValueError("The parameter 'min_num_features' cannot be more than the number of features in our dataset.")
           
            
    dataset_features = dataset_features.reset_index(drop = True)
    return {"Feature Metrics": dataset_features, "More Info": store}



def classifier_graph(classifier, x_train, y_train, cmap_train = "viridis", cmap_test = "viridis", size_train_marker: float = 10, size_test_marker: float = 10, x_test=None, y_test=None, resolution=100, plot_title="Decision Boundary", warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
        
    feature1 = x_train.iloc[:, 0].name
    feature2 = x_train.iloc[:, 1].name
    
    le = sp.LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    if isinstance(x_train, pd.DataFrame):
        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((x_train.iloc[:, 0].min() - (x_train.iloc[:, 0].min() / 8)), (x_train.iloc[:, 0].max() + (x_train.iloc[:, 0].max() / 8)), resolution),
                                                    np.linspace((x_train.iloc[:, 1].min() - (x_train.iloc[:, 1].min() / 8)), (x_train.iloc[:, 1].max() + (x_train.iloc[:, 1].max() / 8)), resolution))
    elif isinstance(x_train, np.ndarray):
        x1_vals_train, x2_vals_train = np.meshgrid(np.linspace((x_train.iloc[:, 0].min() - (x_train.iloc[:, 0].min() / 8)), (x_train.iloc[:, 0].max() + (x_train.iloc[:, 0].max() / 8)), resolution),
                                                    np.linspace((x_train.iloc[:, 1].min() - (x_train.iloc[:, 1].min() / 8)), (x_train.iloc[:, 1].max() + (x_train.iloc[:, 1].max() / 8)), resolution))
    else:
        raise ValueError("Unsupported input type for x_train. Use either Pandas DataFrame or NumPy array.")

    grid_points_train = np.c_[x1_vals_train.ravel(), x2_vals_train.ravel()]
    predictions_train = classifier.predict(grid_points_train)
    predictions_train = le.inverse_transform(predictions_train)

    plt.figure(figsize = (15, 10))
    
    plt.contourf(x1_vals_train, x2_vals_train, le.transform(predictions_train).reshape(x1_vals_train.shape), alpha=0.3, cmap = cmap_train)
    if isinstance(x_train, pd.DataFrame):
        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
    elif isinstance(x_train, np.ndarray):
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_encoded, cmap=cmap_train, edgecolors='k', s=size_train_marker, marker='o')
    plt.title(f"{classifier.__class__.__name__} Training Classification Graph")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.tight_layout()
    plt.show()

    if x_test is not None and y_test is not None:
        plt.figure(figsize = (15, 10))
        x1_vals_test, x2_vals_test = np.meshgrid(np.linspace((x_test.iloc[:, 0].min() - (x_test.iloc[:, 0].min() / 8)), (x_test.iloc[:, 0].max() + (x_test.iloc[:, 0].max() / 8)), resolution),
                                                  np.linspace((x_test.iloc[:, 1].min() - (x_test.iloc[:, 1].min() / 8)), (x_test.iloc[:, 1].max() + (x_test.iloc[:, 1].max() / 8)), resolution))

        grid_points_test = np.c_[x1_vals_test.ravel(), x2_vals_test.ravel()]
        predictions_test = classifier.predict(grid_points_test)
        predictions_test = le.inverse_transform(predictions_test)

        plt.contourf(x1_vals_test, x2_vals_test, le.transform(predictions_test).reshape(x1_vals_test.shape), alpha=0.3, cmap=cmap_test)

        if isinstance(x_test, pd.DataFrame):
            plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], c=le.transform(y_test), cmap=cmap_test, edgecolors='k', s=size_test_marker, marker='o')
        elif isinstance(x_test, np.ndarray):
            plt.scatter(x_test[:, 0], x_test[:, 1], c=le.transform(y_test), cmap=cmap_test, edgecolors='k', s=size_test_marker, marker='o')

        plt.title(f"{classifier.__class__.__name__} Test Classification Graph")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.tight_layout()
        plt.show()
        
        

def FindK_KNN_Classifier(x_train, y_train, weight = "uniform", algorithm = "auto", metric = "minkowski", max_k_range: int = 31, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
        
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
            model = classifier.fit(x_train, y_train)
            
            # Model Evaluation
            scores_knn.append(model.score(x_train, y_train))
            scores_store[num] = (model.score(x_train, y_train))
        
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


def FindK_KNN_Regressor(x_train, y_train, weight = "uniform", algorithm = "auto", metric = "minkowski", max_k_range: int = 31, warning: bool = False):
    if warning == False:
        warnings.filterwarnings("ignore")
        
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
            model = regressor.fit(x_train, y_train)
            
            # Model Evaluation
            scores_knn.append(model.score(x_train, y_train))
            scores_store[num] = (model.score(x_train, y_train))
        
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
        print(f'\n\nKNN REGRESSOR ------> Finding the besk K value:\nThe best k-value is {b[0]} with a score of {b[1]}.')
        
    else:
        raise ValueError(f"Check that the parameter 'algorithm' is one of the following: {algorithms}. Also, check that the parameter 'weight' is one of the following: {weights}")
    