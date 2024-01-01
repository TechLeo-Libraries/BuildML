# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:08:41 2023

@author: lEO
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from __init__ import MultiPartSupervisedLearning, multidata_process
from multiprocess.eda import eda, visualization_eda

data1 = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/2/P14-Part2-Regression/Section 6 - Simple Linear Regression/Python/Salary_Data.csv")
data2 = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/3/P14-Part3-Classification/Section 16 - Logistic Regression/Python/Social_Network_Ads.csv")
data3 = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/4/P14-Part4-Clustering/Section 25 - K-Means Clustering/Python/Mall_Customers.csv")
data4 = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/8/P14-Part8-Deep-Learning/Section 35 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv")


# TESTING THE FUNCTIONS
# EDA
data = [data1, data2, data3, data4]
data_eda = eda(data)
visualization_eda(data)

























# # Data Cleaning
# dataframes = multidata_process([data2, data3, data4])
# dataframes_name = dataframes.name_data(name_datasets = ["Predicting Purchased", "Predicting Genre Male", "Predicting Exited"])
# dropped_columns = dataframes.drop_columns(columns = ["CustomerId", "Surname"])
# categorical_to_numerical = dataframes.categorical_to_numerical()
# remove_outlier = dataframes.remove_outlier(drop_na = False)
# fix_missing_columns = dataframes.fix_missing_values(strategy = "mean")
# eda = dataframes.eda()
# eda_visuals = dataframes.eda_visual(before_data_cleaning = False)
# dependent_independent = dataframes.select_dependent_and_independent(predict = ["Purchased", "Genre_Male", "Exited"])
# scale_dependent = dataframes.scale_independent_variables()
# split_data = dataframes.split_data()
# fix_unbalance = dataframes.fix_unbalanced_dataset(sampler = "SMOTE")
# data = dataframes.get_training_test_data()
# # build_model = dataframes.build_joint_model(model = XGBClassifier(), model_type = "c", cross_validation = True)


# build_model = dataframes.build_multiple_classifiers(models = [XGBClassifier(), RandomForestClassifier()])




# classifier_graph = dataframes.classifier_graph(classifier, xlabel, ylabel, color1, color2)



