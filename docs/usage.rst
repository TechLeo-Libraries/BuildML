Usage
-----

**Example 1**

.. code:: bash

   # Import Libraries
   import numpy as np
   import pandas as pd
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
   from sklearn.snm import SVC
   from buildml.automate import SupervisedLearning

   # Get Dataset
   dataset = pd.read_csv("Your_file_path")  # Load your dataset(e.g Pandas DataFrame)
   data = SupervisedLearning(dataset)

   # Exploratory Data Analysis
   eda = data.eda()
   eda_visual = data.eda_visual()

   # Build and Evaluate Classifier
   classifiers = ["LogisticRegression(random_state = 0)", "RandomForestClassifier(random_state = 0)", "DecisionTreeClassifier(random_state = 0)", "SVC()"]
   build_model = data.build_multiple_classifiers()
   

**Example 2: Working on a dataset with train and test data given.**

.. code:: bash

    # Import Libraries
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from buildml.automate import SupervisedLearning

    # Get Dataset
    training_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    dataset = pd.concat([training_data, test_data], axis = 0)

    # BuildML on Dataset
    automate_training = SupervisedLearning(training_data)
    automate_test = SupervisedLearning(test_data)

    automate = [automate_training, automate_test]

    # Exploratory Data Analysis
    training_eda = automate_training.eda()
    test_eda = automate_test.eda()

    # Data Cleaning and Transformation 
    training_eda_visual = automate_training.eda_visual(y = "Specify what you are predicting", figsize_barchart = (55, 10), figsize_heatmap = (15, 10), figsize_histogram=(35, 20))

    for data in automate:
        data.reduce_data_memory_useage()
        data.drop_columns("Drop irrelevant columns")
        data.categorical_to_numerical() # If your data has categorical features

    select_variables = automate_training.select_dependent_and_independent(predict = "Loan Status")

    # Further Data Preparation and Segregation
    unbalanced_dataset_check = automate_training.count_column_categories(column = "Specify what you are predicting")
    split_data = automate_training.split_data()
    fix_unbalanced_data = automate_training.fix_unbalanced_dataset(sampler = "RandomOverSampler", random_state = 0)

    # Model Building 
    classifiers = [LogisticRegression(random_state = 0), SVC(), RandomForestClassifier(random_state = 0), DecisionTreeClassifier(random_state = 0), XGBClassifier(random_state = 0)]
    build_model = automate_training.build_multiple_classifiers(classifiers = classifiers, kfold = 10, cross_validation = True, graph = True)

    model = automate_training.train_model_classifier(classifiers[1])
    prediction = automate_training.classifier_predict() 
    evaluation = automate_training.classifier_evaluation(cross_validation = True)