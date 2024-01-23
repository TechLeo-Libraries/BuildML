# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from automate import SupervisedLearning
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Get the dataset
dataset = pd.read_csv("C:/Users/lEO/Desktop/Github Projects/Data-Science-Tutorial-Programs/Machine_Learning/Social_Network_Ads.csv")
data = pd.read_csv("C:/Users/lEO/Desktop/Github Projects/Data-Science-Tutorial-Programs/Machine_Learning/Social_Network_Ads.csv")

# Using BuildML
automate = SupervisedLearning(data)

# EDA
eda = automate.eda()

# Data Cleaning and Transformation
drop_columns = automate.drop_columns("User ID")
fix_categorical = automate.categorical_to_numerical()

# Further Data Preparation and Segregation
select_variables = automate.select_dependent_and_independent(predict = "Purchased")
split_variables = automate.split_data(test_size = 0.2)

# Model Building
get_neighbors = automate.get_bestK_KNNclassifier()

classifiers = [LogisticRegression(random_state = 0),
                DecisionTreeClassifier(random_state = 0),
                RandomForestClassifier(random_state = 0),
                SVC(),
                KNeighborsClassifier(n_neighbors = 1),
                XGBClassifier(random_state = 0)]

build_model = automate.build_multiple_classifiers(classifiers = classifiers, graph = True, cross_validation = True)
build_model_with_features = automate.build_multiple_classifiers_from_features(strategy = "SelectKBest", estimator = f_classif, classifiers = classifiers, cv = True, max_num_features = 4)