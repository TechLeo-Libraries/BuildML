# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:32:38 2024

@author: lEO
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.feature_selection import chi2
from __init__ import SupervisedLearning


dataset = pd.read_csv("C:/Users/lEO/Desktop/Dataset/ML data/2/P14-Part2-Regression/Section 8 - Polynomial Regression/Python/Position_Salaries.csv")
data = SupervisedLearning(dataset)

eda = data.eda()
data.drop_columns("Position")
select_data = data.select_dependent_and_independent("Salary")
# poly_x = data.polyreg_x(degree = 6, include_bias = False)
# training_test = data.split_data()

regressor = [LinearRegression(), RandomForestRegressor(random_state = 0), DecisionTreeRegressor(random_state = 0), SVR()]
# build_model = data.build_single_regressor_from_features(strategy = "selectkbest", estimator = chi2, regressor = regressor)

graph = data.simple_linregres_graph(regressor, whole_dataset = True, title = "I am here", line_marker = None, size_train_marker = 85, size_test_marker = 85)