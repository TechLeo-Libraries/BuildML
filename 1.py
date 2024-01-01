# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:00:41 2023

@author: lEO
"""

# 1) IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from automate import SupervisedLearning
from build_model import classifier_graph, build_single_regressor_from_features
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import datatable as dt
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, f_classif, f_oneway, f_regression, SelectPercentile, chi2

# dataframe = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/3/P14-Part3-Classification/Section 16 - Logistic Regression/Python/Social_Network_Ads.csv")
# dataframe = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/2/P14-Part2-Regression/Section 7 - Multiple Linear Regression/Python/50_Startups.csv")
dataframe = pd.read_csv("C:/Users/lEO/Desktop/Dataset/Kaggle Datasets/diabetes_prediction_dataset.csv")


# # T E S T    1
# # GET THE DATASET
# data = SupervisedLearning(dataframe)
# dataset = data.get_dataset()
# initial_eda = data.eda()
# initial_eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = True)

# # Data Cleaning
# # categories_count = data.count_column_categories()
# group_columns = data.group_data(columns = ["Gender", "EstimatedSalary", "Age"], column_to_groupby = "Gender", aggregate_function = "count", reset_index = True)
# data_transformation = data.categorical_to_numerical()
# drop_column = data.drop_columns("User ID")

# # Further Data Preparation and Segregation
# eda = data.eda()
# eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = False)
# variables = data.select_dependent_and_independent(predict = "Purchased")
# features_selected = data.select_features(strategy = "SelectKBest", estimator = chi2, number_of_features = 2)
# training_test = data.split_data()

# # Model Training
# classifier = RandomForestClassifier(random_state = 0)
# model = data.train_model_classifier(classifier = classifier)

# # Model Prediction
# prediction = data.classifier_predict()

# # Model Evaluation
# evaluate_model = data.classifier_evaluation(cross_validation = True)

# # Graph
# x_train, x_test, y_train, y_test = data.get_training_test_data()
# classifier_graph(classifier, x_train, x_test, y_train, y_test)





# T E S T   2
# GET THE DATASET
data = SupervisedLearning(dataframe)
reduce_memory = data.reduce_data_memory_useage(verbose = True)
dataset = data.get_dataset()
initial_eda = data.eda()
initial_eda_visuals = data.eda_visual(y = "diabetes", before_data_cleaning = True)

# Data Cleaning
data_transformation = data.categorical_to_numerical()


# Further Data Preparation and Segregation
# eda = data.eda()
# eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = False)
variables = data.select_dependent_and_independent(predict = "diabetes")
features = data.select_features(strategy = "rfe", estimator = RandomForestClassifier(random_state = 0), number_of_features = 2)
training_test = data.split_data()

# Model Training
classifiers = [RandomForestClassifier(random_state = 0),
              XGBClassifier(),
              LogisticRegression()
              ]

# classifier = RandomForestClassifier(random_state = 0)


Build_Model = data.build_multiple_classifiers(classifiers = classifiers, cross_validation = True, graph = True, length = 10, width = 20)
# # Build_Model = data.build_multiple_classifiers_from_features(strategy = "SelectKBest", estimator = f_classif, classifiers = classifiers, cv = False, max_num_features = 13, min_num_features = 5)
# Build_Model = data.build_single_classifier_from_features(strategy = "SelectKBest", estimator = f_classif, classifier = classifier, cv = True)
graph = data.classifier_graph(classifier = classifiers[0], resolution = 100)






# # T E S T   3
# # GET THE DATASET
# data = SupervisedLearning(dataframe)
# dataset = data.get_dataset()
# initial_eda = data.eda()
# initial_eda_visuals = data.eda_visual(y = "Profit", before_data_cleaning = True)

# # Data Cleaning
# data_transformation = data.categorical_to_numerical()


# # Further Data Preparation and Segregation
# # eda = data.eda()
# # eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = False)
# variables = data.select_dependent_and_independent(predict = "Profit")
# # training_test = data.split_data()

# # Model Training
# # regressors = [RandomForestRegressor(random_state = 0),
# #               XGBRegressor(),
# #               LinearRegression()
# #               ]

# regressor = LinearRegression()

# a = build_single_regressor_from_features(y = variables["Dependent Variable"], x = variables["Independent Variables"], regressor = regressor, test_size = 0.2, random_state = 0, strategy = "rfe", estimator = regressor)


# # Build_Model = data.build_multiple_regressors_from_features(strategy = "SelectKBest", estimator = f_regression, regressors = regressors, cv = True, min_num_features = 3, max_num_features = 4)
# # Build_Model = data.build_single_regressor_from_features(strategy = "SelectKBest", estimator = f_regression, regressor = regressor, cv = True, min_num_features = 1, max_num_features = 4)






# # T E S T   4
# # GET THE DATASET
# data = SupervisedLearning(dataframe)
# reduce_memory = data.reduce_data_memory_useage(verbose = True)
# dataset = data.get_dataset()
# initial_eda = data.eda()
# initial_eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = True)

# # # Data Cleaning
# drop_column = data.drop_columns("User ID")
# data_transformation = data.categorical_to_numerical()


# # Further Data Preparation and Segregation
# eda = data.eda()
# eda_visuals = data.eda_visual(y = "Purchased", before_data_cleaning = False)
# variables = data.select_dependent_and_independent(predict = "Purchased")
# training_test = data.split_data()

# # Model Training
# classifier = [RandomForestClassifier(random_state = 0),
#               XGBClassifier(),
#               LogisticRegression()
#               ]

# # Build_Model = data.build_multiple_classifiers(classifiers = classifier, cross_validation = True, graph = True, length = 10, width = 20)
# Build_Model = data.build_multiple_classifiers_from_features(strategy = "SelectKBest", estimator = f_regression, classifiers = classifier, cv = False, min_num_features = 2)

