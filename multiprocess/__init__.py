from multipart import MultiPartSupervisedLearning 
    
    
class multidata_process:
    def __init__(self, dataframes: list):
        if isinstance(dataframes, list):
            self.__data_store = []
            for data in dataframes:
                self.__data_store.append(MultiPartSupervisedLearning(data))
        self.__name_activated = False
                
        print(f"Dataframes have been successfully indexed. Please refer to SPECIFIC datasets in line with their index or how they were inserted. You have {len(self.__data_store)} dataframes indexed.")
        
        n = 0
        print("\n\nDATA GUIDE: ")
        for all_data in self.__data_store:
            print(f"\n{n + 1}) Dataset {n + 1} -----> Index {n}")
            n += 1
        
    def name_data(self, name_datasets: list):  
        if isinstance(name_datasets, list):
            self.__name = name_datasets
            if len(self.__data_store) != len(self.__name):
                ValueError(f"Dataframes and names must have the same length. {len(self.__data_store)} specified for dataframes, while {len(self.__name)} specified for names.")
            else:
                self.__index = [index for index in range(len(self.__name))]
                self.__name_activated = True
       
        n = 0
        print("\n\nDATA GUIDE: ")
        for all_data, name, index_count in zip(self.__data_store, self.__name, self.__index):
            print(f"\n{n + 1}) Dataset {n + 1} = {name} -----> Index {n}")
            n += 1
    
    def get_dataset(self, index: int = None):
        data = self.__data_store
        if self.__name_activated == False:
            if index != None:
                if isinstance(index, int):
                    values = {}
                    values[f"Dataset {index + 1}"] = data[index].get_dataset()
                
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        values[f"Dataset {index + 1}"] = data[num].get_dataset()
                
                return values
            
            elif index == None:
                values = {}
                n = 1
                for store in data:
                    values[f"Dataset {n}"] = store.get_dataset()
                    n += 1
                    
                return values
        
        elif self.__name_activated == True:
            if index != None:
                if isinstance(index, int):
                    values = {}
                    values[f"{self.name[index]}"] = data[index].get_dataset()
                
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        values[f"{self.name[index]}"] = data[num].get_dataset()
                
                return values
            
            elif index == None:
                values = {}
                for data_store, name in zip(self.__data_store, self.__name):
                    values[name] = data_store.get_dataset()
                    
                return values
        
        
        
    def get_training_test_data(self, index: int = None):
        data = self.__data_store
        if self.__name_activated == False:
            if index != None:
                if isinstance(index, int):
                    values = {}
                    values[f"Dataset {index + 1}"] = data[index].get_training_test_data()
                
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        values[f"Dataset {index + 1}"] = data[num].get_training_test_data()
                
                return values
            
            elif index == None:
                values = {}
                n = 1
                for store in data:
                    values[f"Dataset {n}"] = store.get_training_test_data()
                    n += 1
                    
                return values
        
        elif self.__name_activated == True:
            if index != None:
                if isinstance(index, int):
                    values = {}
                    values[f"{self.__name[index]}"] = data[index].get_training_test_data()
                
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        values[f"{self.__name[index]}"] = data[num].get_training_test_data()
                
                return values
            
            elif index == None:
                values = {}
                for data_store, name in zip(self.__data_store, self.__name):
                    values[name] = data_store.get_training_test_data()
                    
                return values
    
    
    
    def drop_columns(self, columns: list, index: int = None):
        if self.__name_activated == False:
            if isinstance(columns, str) and index == None:
                values = {}
                n = 1
                for preprocess in self.__data_store:
                    preprocess.drop_columns(columns = columns)
                
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                
                print(f"\n\nNOTE: Any columns in your dataset with the name {str(columns)}, have been successfully removed.")
                return values
            
            elif isinstance(columns, list) and index == None:
                values = {}
                n = 1
                for preprocess in self.__data_store:
                    preprocess.drop_columns(columns = columns)
                
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                
                print(f"\n\nNOTE: All columns in your dataset with the following names: {str(columns)} have all been successfully removed.")
                return values
            
            if index != None:
                if isinstance(columns, str):
                    values = {}
                    n = 1
                    self.__data_store[index].drop_columns(columns = columns)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
                    
                    print(f"\n\nNOTE: Any columns in your specified dataset with the name {str(columns)}, have been successfully removed.")
                    return values
                
                elif isinstance(columns, list):
                    values = {}
                    n = 1
                    for num in index:
                        self.__data_store[num].drop_columns(columns = columns)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
                    
                    print(f"\n\nNOTE: All columns in your specified dataset with the following names: {str(columns)} have all been successfully removed.")
                    return values

        elif self.__name_activated == True:
            if isinstance(columns, str) and index == None:
                values = {}
                for preprocess in self.__data_store:
                    preprocess.drop_columns(columns = columns)
                
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
                
                print(f"\n\nNOTE: Any columns in your dataset with the name {str(columns)}, have been successfully removed.")
                return values
            
            elif isinstance(columns, list) and index == None:
                values = {}
                for preprocess in self.__data_store:
                    preprocess.drop_columns(columns = columns)
                
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
                
                print(f"\n\nNOTE: All columns in your dataset with the following names: {str(columns)} have all been successfully removed.")
                return values
            
            if index != None:
                if isinstance(columns, str):
                    values = {}
                    self.__data_store[index].drop_columns(columns = columns)
                    
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[f"{name}"] = results.get_dataset()
                    
                    print(f"\n\nNOTE: Any columns in your specified dataset with the name {str(columns)}, have been successfully removed.")
                    return values
                
                elif isinstance(columns, list):
                    values = {}
                    for num in index:
                        self.__data_store[num].drop_columns(columns = columns)
                    
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[f"{name}"] = results.get_dataset()
                    
                    print(f"\n\nNOTE: All columns in your specified dataset with the following names: {str(columns)} have all been successfully removed.")
                    return values
            
     
        
     
    def fix_missing_values(self, strategy, index: int = None):
        if self.__name_activated == False:        
            if index == None:
                values = {}
                n = 1
                for preprocess in self.__data_store:
                    preprocess.fix_missing_values(strategy = strategy)
                    
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                    
            elif isinstance(index, int):
                values = {}
                n = 1
                for preprocess in self.__data_store[index]:
                    preprocess.fix_missing_values(strategy = strategy)
                    
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                    
            elif isinstance(index, list):
                values = {}
                n = 1
                for num in index:
                    self.__data_store[num].fix_missing_values(strategy = strategy)
                    
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                        
            return values            
            
        
        elif self.__name_activated == True:
            if index == None:
                values = {}
                for preprocess in self.__data_store:
                    preprocess.fix_missing_values(strategy = strategy)
                    
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
                    
            elif isinstance(index, int):
                values = {}
                for preprocess in self.__data_store[index]:
                    preprocess.fix_missing_values(strategy = strategy)
                    
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
            
            elif isinstance(index, list):
                values = {}
                for num in index:
                    self.__data_store[num].fix_missing_values(strategy = strategy)
                
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
                    
            return values
        
    
    
    
    
    def categorical_to_numerical(self, columns: list = None, index: int = None):
        if self.__name_activated == False:
            if index == None:
                values = {}
                n = 1
                for preprocess in self.__data_store:
                    preprocess.categorical_to_numerical(columns = columns)
                    
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
            
            elif index != None:
                if isinstance(index, list):
                    values = {}
                    n = 1
                    for num in index:
                        self.__data_store[num].categorical_to_numerical(columns = columns)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
                
                elif isinstance(index, int):
                    values = {}
                    n = 1
                    self.__data_store[index].categorical_to_numerical(columns = columns)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
                    
            return values
        
        elif self.__name_activated == True:
            if index == None:
                values = {}
                for preprocess in self.__data_store:
                    preprocess.categorical_to_numerical(columns = columns)
                    
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[name] = results.get_dataset()
            
            elif index != None:
                if isinstance(index, list):
                    values = {}
                    for num in index:
                        self.__data_store[num].categorical_to_numerical(columns = columns)
                    
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[name] = results.get_dataset()
                
                elif isinstance(index, int):
                    values = {}
                    self.__data_store[index].categorical_to_numerical(columns = columns)
                    
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[name] = results.get_dataset()
                    
            return values
    
    
    def remove_outlier(self, drop_na: bool, index: int = None):
        if self.__name_activated == False:
            if index == None:
                values = {}
                n = 1
                for preprocess in self.__data_store:
                    preprocess.remove_outlier(drop_na = drop_na)
                    
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.get_dataset()
                    n += 1
                    
            elif index != None:
                if isinstance(index, int):
                    values = {}
                    n = 1
                    self.__data_store[index].remove_outlier(drop_na = drop_na)
                        
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
                        
                elif isinstance(index, list):
                    values = {}
                    n = 1
                    for num in index:
                        self.__data_store[num].remove_outlier(drop_na = drop_na)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n}"] = results.get_dataset()
                        n += 1
        
            return values
        
        elif self.__name_activated == True:
            if index == None:
                values = {}
                for preprocess in self.__data_store:
                    preprocess.remove_outlier(drop_na = drop_na)
                    
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.get_dataset()
                    
            elif index != None:
                if isinstance(index, int):
                    values = {}
                    self.__data_store[index].remove_outlier(drop_na = drop_na)
                        
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[f"{name}"] = results.get_dataset()
                        
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        self.__data_store[num].remove_outlier(drop_na = drop_na)
                    
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        values[f"{name}"] = results.get_dataset()
        
            return values
    
    
    
    def scale_independent_variables(self):
        if self.__name_activated == False:
            values = {}
            n = 1
            for results in self.__data_store:
                values[f"Dataset {n}"] = results.scale_independent_variables()
                n += 1
                    
            return values
        
        
        elif self.__name_activated == True:
            values = {}
            for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                values[f"{name}"] = results.scale_independent_variables()
                    
            return values
    
    
    
    
    
    def eda(self, index: int = None):
        if self.__name_activated == False:
            if index == None:
                values = {}
                n = 1
                for results in self.__data_store:
                    values[f"Dataset {n}"] = results.eda()
                    n += 1
                    
            elif index != None:        
                if isinstance(index, int):
                    values = {}
                    values[f"Dataset {index + 1}"] = self.__data_store[index].eda()
                    
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        values[f"Dataset {num + 1}"] = self.__data_store[num].eda()
                        
            return values
        
        
        elif self.__name_activated == True:
            if index == None:
                values = {}
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = results.eda()
                    
            elif index != None:        
                if isinstance(index, int):
                    values = {}
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        if index == index_count:
                            values[f"{name}"] = self.__data_store[index].eda()
                    
                elif isinstance(index, list):
                    values = {}
                    for num in index:
                        for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                            if num == index_count:
                                values[f"{name}"] = self.__data_store[num].eda()
                        
            return values
    
    
    
    
    
    def eda_visual(self, before_data_cleaning: bool, index: int = None):
        if self.__name_activated == False:
            if index == None:
                values = {}
                n = 1
                for results in self.__data_store:
                    values[f"Dataset {n}"] = {"Data": results.get_dataset(), "EDA Visual": results.eda_visual(before_data_cleaning = before_data_cleaning, num = n)}
                    n += 1
                    
            elif index != None:        
                if isinstance(index, int):
                    values = {}
                    values[f"Dataset {index + 1}"] = {"Data": self.__data_store[index].get_dataset(), "EDA Visual": self.__data_store[index].eda_visual(before_data_cleaning = before_data_cleaning, num = index + 1)}
                    
                elif isinstance(index, list):
                    values = {}
                    for number in index:
                        values[f"Dataset {number + 1}"] = {"Data": self.__data_store[number].get_dataset(), "EDA Visual": self.__data_store[number].eda_visual(before_data_cleaning = before_data_cleaning, num = number + 1)}
                        
            return values
    
        elif self.__name_activated == True:
            if index == None:
                values = {}
                n = 1
                for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                    values[f"{name}"] = {"Data": results.get_dataset(), "EDA Visual": results.eda_visual(before_data_cleaning = before_data_cleaning, num = n)}
                    n += 1
                    
            elif index != None:        
                if isinstance(index, int):
                    values = {}
                    for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        if index == index_count:
                            values[f"{name}"] = {"Data": self.__data_store[index].get_dataset(), "EDA Visual": self.__data_store[index].eda_visual(before_data_cleaning = before_data_cleaning, num = index + 1)}
                    
                elif isinstance(index, list):
                    values = {}
                    for number in index:
                        for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                            if number == index_count:
                                values[f"{name}"] = {"Data": self.__data_store[number].get_dataset(), "EDA Visual": self.__data_store[number].eda_visual(before_data_cleaning = before_data_cleaning, num = number + 1)}
                        
            return values
    
    
    
    
    def select_dependent_and_independent(self, predict: list):
        if self.__name_activated == False:
            if isinstance(predict, list) and (len(predict) == len(self.__data_store)):
               values = {}
               n = 0
               for items in predict:
                   values[f"Dataset {n + 1}"] = self.__data_store[n].select_dependent_and_independent(predict = items)
                   n += 1
                   
            elif len(predict) != len(self.__data_store):
                raise ValueError("Insert all variables to predict and insert them into list in the right order. Make sure that the number of variables in predict is the same with the number of dataframes you are working with.")
                
            return values

        elif self.__name_activated == True:
            if isinstance(predict, list) and (len(predict) == len(self.__data_store)):
               values = {}
               for results, name, index_count, items in zip(self.__data_store, self.__name, self.__index, predict):
                   values[name] = self.__data_store[index_count].select_dependent_and_independent(predict = items)
                   
            elif len(predict) != len(self.__data_store):
                raise ValueError("Insert all variables to predict and insert them into list in the right order. Make sure that the number of variables in predict is the same with the number of dataframes you are working with.")
                
            return values
    
          
    
    def split_data(self):
        if self.__name_activated == False:
            values = {}
            n = 0
            for items in self.__data_store:
                values[f"Dataset {n + 1}"] = self.__data_store[n].split_data()
                n += 1
        
            return values 
        
        elif self.__name_activated == True:
            values = {}
            for results, name, index_count in zip(self.__data_store, self.__name, self.__index):
                values[name] = self.__data_store[index_count].split_data()
                                
            return values 
    
    
    
    def build_joint_model(self, model: list, model_type: str or list, kfold: int = None, cross_validation: bool = False):
        types_regressor = ["regress", "reg", "regression", "regressor", "r", "regres"]
        types_classifier = ["classifier", "class", "classification", "clas", "c", "classif"]
        
        if self.__name_activated == False:
            if isinstance(model_type, list) and (len(model) == len(model_type)):
                values = {}
                n = 0
                for algorithm, diff_type in zip(model, model_type):
                    if diff_type.lower().strip() in types_regressor:
                        values[f"Dataset {n + 1}"] = self.__data_store[n].build_model_regressor(regressor = algorithm, kfold = kfold, cross_validation = cross_validation)
                        n += 1
                        
                    elif diff_type.lower().strip() in types_classifier:
                        values[f"Dataset {n + 1}"] = self.__data_store[n].build_model_classifier(classifier = algorithm, kfold = kfold, cross_validation = cross_validation)
                        n += 1
                            
            elif isinstance(model_type, list) and (len(model) != len(model_type)):
                raise ValueError("If the parameter model is a list, then model_type should be a list and have the same length as the list of models")
            
            
            elif isinstance(model_type, str) and not isinstance(model, list):
                values = {}
                n = 0
                
                for data in self.__data_store:
                    if model_type.lower().strip() in types_regressor:
                        values[f"Dataset {n + 1}"] = self.__data_store[n].build_model_regressor(regressor = model, kfold = kfold, cross_validation = cross_validation)
                        n += 1
                        
                    elif model_type.lower().strip() in types_classifier:
                        values[f"Dataset {n + 1}"] = self.__data_store[n].build_model_classifier(classifier = model, kfold = kfold, cross_validation = cross_validation)
                        n += 1
                            
            elif isinstance(model_type, str) and isinstance(model, list):
                raise ValueError("If the parameter model is a str, then model_type should be a str")
            
            return values
        
        
        elif self.__name_activated == True:
            if isinstance(model_type, list) and (len(model) == len(model_type)):
                values = {}
                for algorithm, diff_type, results, name, index_count in zip(model, model_type, self.__data_store, self.__name, self.__index):
                    if diff_type.lower().strip() in types_regressor:
                        values[name] = self.__data_store[index_count].build_model_regressor(regressor = algorithm, kfold = kfold, cross_validation = cross_validation)
                           
                    elif diff_type.lower().strip() in types_classifier:
                        values[name] = self.__data_store[index_count].build_model_classifier(classifier = algorithm, kfold = kfold, cross_validation = cross_validation)
                            
            elif isinstance(model_type, list) and (len(model) != len(model_type)):
                raise ValueError("If the parameter model is a list, then model_type should be a list and have the same length as the list of models")
            
            
            elif isinstance(model_type, str) and not isinstance(model, list):
                values = {}
                
                for data, name, index in zip(self.__data_store, self.__name, self.__index):
                    if model_type.lower().strip() in types_regressor:
                        values[name] = self.__data_store[index].build_model_regressor(regressor = model, kfold = kfold, cross_validation = cross_validation)
                        
                    elif model_type.lower().strip() in types_classifier:
                        values[name] = self.__data_store[index].build_model_classifier(classifier = model, kfold = kfold, cross_validation = cross_validation)
                            
            elif isinstance(model_type, str) and isinstance(model, list):
                raise ValueError("If the parameter model is a str, then model_type should be a str")
            
            return values
        
            
    def build_multiple_regressors(self, models: list, kfold: int = None, cross_validation: bool = False):
        if self.__name_activated == False:
            if isinstance(models, list):
                values = {}
                n = 0
                for data in self.__data_store:
                    diff_models = {}
                    for algorithm in models:
                        diff_models[f"{algorithm.__class__.__name__}"] = self.__data_store[n].build_model_regressor(regressor = algorithm, kfold = kfold, cross_validation = cross_validation)
                    
                    values[f"Dataset {n + 1}"] = diff_models
                    n += 1
            
            else:
                raise ValueError("Expected list for model, got None.")
                
            return values
        
        elif self.__name_activated == True:
            if isinstance(models, list):
                values = {}
                for data, name, index in zip(self.__data_store, self.__name, self.__index):
                    diff_models = {}
                    for algorithm in models:
                        diff_models[f"{algorithm.__class__.__name__}"] = self.__data_store[index].build_model_regressor(regressor = algorithm, kfold = kfold, cross_validation = cross_validation)
                    
                    values[name] = diff_models
            
            else:
                raise ValueError("Expected list for model, got None.")
                
            return values
            
    
    def build_multiple_classifiers(self, models: list, kfold: int = None, cross_validation: bool = False):
        if self.__name_activated == False:
            if isinstance(models, list):
                values = {}
                n = 0
                for data in self.__data_store:
                    diff_models = {}
                    for algorithm in models:
                        diff_models[f"{algorithm.__class__.__name__}"] = self.__data_store[n].build_model_classifier(classifier = algorithm, kfold = kfold, cross_validation = cross_validation)
                    
                    values[f"Dataset {n + 1}"] = diff_models
                    n += 1
            
            else:
                raise ValueError("Expected list for model, got None.")
                
            return values
        
        elif self.__name_activated == True:
            if isinstance(models, list):
                values = {}
                for data, name, index in zip(self.__data_store, self.__name, self.__index):
                    diff_models = {}
                    for algorithm in models:
                        diff_models[f"{algorithm.__class__.__name__}"] = self.__data_store[index].build_model_classifier(classifier = algorithm, kfold = kfold, cross_validation = cross_validation)
                    
                    values[name] = diff_models
            
            else:
                raise ValueError("Expected list for model, got None.")
                
            return values
    
    
    
    def test_regressor_model(self, index: int = None):
        pass
    
    def test_classifier_model(self, index: int = None):
        pass

                
    
    def numerical_to_categorical(self, column: str or list or tuple, index: int = None):
        if self.__name_activated == False:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].numerical_to_categorical(column = item)
                            elif item not in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].get_dataset()
                
                elif isinstance(column, str):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        if column in data_columns:
                            values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].numerical_to_categorical(column = column)
                        elif column not in data_columns:
                            values[f"Dataset {index_count + 1}"] = self.__data_store[index_count]
                    
                
                elif isinstance(column, tuple):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].numerical_to_categorical(column = item)
                            elif item not in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count]
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].numerical_to_categorical(column = item)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                    
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].numerical_to_categorical(column = item)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        if column in data_columns:
                            self.__data_store[index].numerical_to_categorical(column = column)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                            
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].numerical_to_categorical(column = item)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].numerical_to_categorical(column = item)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            if column in data_columns:
                                self.__data_store[num].numerical_to_categorical(column = column)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                            
                
                return values

        elif self.__name_activated == True:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[name] = self.__data_store[index_count].numerical_to_categorical(column = item)
                            elif item not in data_columns:
                                values[name] = self.__data_store[index_count].get_dataset()
                
                elif isinstance(column, str):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        if column in data_columns:
                            values[name] = self.__data_store[index_count].numerical_to_categorical(column = column)
                        elif column not in data_columns:
                            values[name] = self.__data_store[index_count]
                    
                
                elif isinstance(column, tuple):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[name] = self.__data_store[index_count].numerical_to_categorical(column = item)
                            elif item not in data_columns:
                                values[name] = self.__data_store[index_count]
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].numerical_to_categorical(column = item)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].numerical_to_categorical(column = item)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        if column in data_columns:
                            self.__data_store[index].numerical_to_categorical(column = column)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].numerical_to_categorical(column = item)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].numerical_to_categorical(column = item)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            if column in data_columns:
                                self.__data_store[num].numerical_to_categorical(column = column)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                
                return values
    
    
    def categorical_to_datetime(self, column: str or list or tuple, index: int = None):
        if self.__name_activated == False:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].categorical_to_datetime(column = item)
                            elif item not in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].get_dataset()
                
                elif isinstance(column, str):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        if column in data_columns:
                            values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].categorical_to_datetime(column = column)
                        elif column not in data_columns:
                            values[f"Dataset {index_count + 1}"] = self.__data_store[index_count]
                    
                
                elif isinstance(column, tuple):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count].categorical_to_datetime(column = item)
                            elif item not in data_columns:
                                values[f"Dataset {index_count + 1}"] = self.__data_store[index_count]
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].categorical_to_datetime(column = item)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                    
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].categorical_to_datetime(column = item)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        if column in data_columns:
                            self.__data_store[index].categorical_to_datetime(column = column)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                            
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].categorical_to_datetime(column = item)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].categorical_to_datetime(column = item)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            if column in data_columns:
                                self.__data_store[num].categorical_to_datetime(column = column)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = self.__data_store[n].get_dataset()
                            n += 1
                            
                            
                
                return values

        elif self.__name_activated == True:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[name] = self.__data_store[index_count].categorical_to_datetime(column = item)
                            elif item not in data_columns:
                                values[name] = self.__data_store[index_count].get_dataset()
                
                elif isinstance(column, str):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        if column in data_columns:
                            values[name] = self.__data_store[index_count].categorical_to_datetime(column = column)
                        elif column not in data_columns:
                            values[name] = self.__data_store[index_count]
                    
                
                elif isinstance(column, tuple):
                    values = {}
                    for data, name, index_count in zip(self.__data_store, self.__name, self.__index):
                        data_columns = self.__data_store[index_count].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                values[name] = self.__data_store[index_count].categorical_to_datetime(column = item)
                            elif item not in data_columns:
                                values[name] = self.__data_store[index_count]
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].categorical_to_datetime(column = item)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        for item in column:
                            if item in data_columns:
                                self.__data_store[index].categorical_to_datetime(column = item)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        data_columns = self.__data_store[index].get_dataset().columns
                        if column in data_columns:
                            self.__data_store[index].categorical_to_datetime(column = column)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].categorical_to_datetime(column = item)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            for item in column:
                                if item in data_columns:
                                    self.__data_store[num].categorical_to_datetime(column = item)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            data_columns = self.__data_store[num].get_dataset().columns
                            if column in data_columns:
                                self.__data_store[num].categorical_to_datetime(column = column)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = self.__data_store[n].get_dataset()
                            n += 1
                
                return values
    
    
    
    
    def extract_date_features(self, datetime_column: str or list, convert_without_extract: bool = False, hrs_mins_sec: bool = False, day_first: bool = False, yearfirst: bool = False, date_format: str = None, index: int = None):
        if self.__name_activated == True:
            if index == None:
                if isinstance(datetime_column, list):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                data.extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                elif isinstance(datetime_column, tuple):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                data.extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                elif isinstance(datetime_column, str):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        if datetime_column in all_columns:
                            data.extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(datetime_column, list):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                self.__data_store[index].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, tuple):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                self.__data_store[index].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, str):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        if datetime_column in all_columns:
                            self.__data_store[index].extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    return values
                
                elif isinstance(index, list):
                    if isinstance(datetime_column, list):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for dates in datetime_column:
                                if dates in all_columns:
                                    self.__data_store[num].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for dates in datetime_column:
                                if dates in all_columns:
                                    self.__data_store[num].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, str):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            if datetime_column in all_columns:
                                self.__data_store[num].extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    return values
        
        elif self.__name_activated == False:
            if index == None:
                if isinstance(datetime_column, list):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                data.extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                elif isinstance(datetime_column, tuple):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                data.extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                elif isinstance(datetime_column, str):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        if datetime_column in all_columns:
                            data.extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                    
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(datetime_column, list):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                self.__data_store[index].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, tuple):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for dates in datetime_column:
                            if dates in all_columns:
                                self.__data_store[index].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, str):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        if datetime_column in all_columns:
                            self.__data_store[index].extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    return values
                
                elif isinstance(index, list):
                    if isinstance(datetime_column, list):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for dates in datetime_column:
                                if dates in all_columns:
                                    self.__data_store[num].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for dates in datetime_column:
                                if dates in all_columns:
                                    self.__data_store[num].extract_date_features(datetime_column = dates, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(datetime_column, str):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            if datetime_column in all_columns:
                                self.__data_store[num].extract_date_features(datetime_column = datetime_column, convert_without_extract = convert_without_extract, hrs_mins_sec = hrs_mins_sec, day_first = day_first, yearfirst = yearfirst, date_format = date_format)
                        
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    return values
                
                
    
    def column_binning(self, column: str or list or tuple, number_of_bins: int = 10, index: int = None):
        if self.__name_activated == True:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                data.column_binning(column = col, number_of_bins = number_of_bins)
                    
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                elif isinstance(column, tuple):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                data.column_binning(column = col, number_of_bins = number_of_bins)
                                
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                elif isinstance(column, str):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        if column in all_columns:
                            data.column_binning(column = column, number_of_bins = number_of_bins)
                            
                    for results, name in zip(self.__data_store, self.__name):
                        values[name] = results.get_dataset()
                        n += 1
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                self.__data_store[index].column_binning(column = col, number_of_bins = number_of_bins)
                                
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                self.__data_store[index].column_binning(column = col, number_of_bins = number_of_bins)
                                
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        if column in all_columns:
                            self.__data_store[index].column_binning(column = column, number_of_bins = number_of_bins)
                            
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    return values
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for col in column:
                                if col in all_columns:
                                    self.__data_store[num].column_binning(column = col, number_of_bins = number_of_bins)
                                    
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for col in column:
                                if col in all_columns:
                                    self.__data_store[num].column_binning(column = col, number_of_bins = number_of_bins)
                                    
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            if column in all_columns:
                                self.__data_store[num].column_binning(column = column, number_of_bins = number_of_bins)
                                
                        for results, name in zip(self.__data_store, self.__name):
                            values[name] = results.get_dataset()
                            n += 1
                            
                    return values
                
        elif self.__name_activated == False:
            if index == None:
                if isinstance(column, list):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                data.column_binning(column = col, number_of_bins = number_of_bins)
                                
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                elif isinstance(column, tuple):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                data.column_binning(column = col, number_of_bins = number_of_bins)
                                
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                elif isinstance(column, str):
                    values = {}
                    n = 0
                    for data, name, index in zip(self.__data_store, self.__name, self.__index):
                        all_columns = data.get_dataset().columns
                        if column in all_columns:
                            data.column_binning(column = column, number_of_bins = number_of_bins)
                            
                    for results in self.__data_store:
                        values[f"Dataset {n + 1}"] = results.get_dataset()
                        n += 1
                        
                return values
            
            elif index != None:
                if isinstance(index, int):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                self.__data_store[index].column_binning(column = col, number_of_bins = number_of_bins)
                                
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        for col in column:
                            if col in all_columns:
                                self.__data_store[index].column_binning(column = col, number_of_bins = number_of_bins)
                                
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        all_columns = self.__data_store[index].get_dataset().columns
                        if column in all_columns:
                            self.__data_store[index].column_binning(column = column, number_of_bins = number_of_bins)
                            
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    return values
                
                elif isinstance(index, list):
                    if isinstance(column, list):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for col in column:
                                if col in all_columns:
                                    self.__data_store[num].column_binning(column = col, number_of_bins = number_of_bins)
                                    
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, tuple):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            for col in column:
                                if col in all_columns:
                                    self.__data_store[num].column_binning(column = col, number_of_bins = number_of_bins)
                                    
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    elif isinstance(column, str):
                        values = {}
                        n = 0
                        for num in index:
                            all_columns = self.__data_store[num].get_dataset().columns
                            if column in all_columns:
                                self.__data_store[num].column_binning(column = column, number_of_bins = number_of_bins)
                                
                        for results in self.__data_store:
                            values[f"Dataset {n + 1}"] = results.get_dataset()
                            n += 1
                            
                    return values

        
    
    def fix_unbalanced_dataset(self, sampler: str, k_neighbors: int = None, index: int = None):
        if self.__name_activated == False:
            if index == None:
                if isinstance(sampler, str):
                    values = {}
                    n = 1
                    for data in self.__data_store:
                        values[f"Dataset {n}"] = data.fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        n += 1
                    
                return values
            
            if index != None:
                if isinstance(index, int):
                    if isinstance(sampler, str):
                        values = {}
                        values[f"Dataset {index + 1}"] = self.__data_store[index].fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        
                    return values
               
                elif isinstance(index, list):
                    if isinstance(sampler, str):
                        values = {}
                        for num in index:
                            values[f"Dataset {num + 1}"] = self.__data_store[num].fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        
                    return values
                
        elif self.__name_activated == True:
            if index == None:
                if isinstance(sampler, str):
                    values = {}
                    n = 1
                    for data, name in zip(self.__data_store, self.__name):
                        values[name] = data.fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        n += 1
                    
                return values
            
            if index != None:
                if isinstance(index, int):
                    if isinstance(sampler, str):
                        values = {}
                        values[f"{self.__name[index]}"] = self.__data_store[index].fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        
                    return values
               
                elif isinstance(index, list):
                    if isinstance(sampler, str):
                        values = {}
                        for num, name in zip(index, self.__name):
                            values[name] = self.__data_store[num].fix_unbalanced_dataset(sampler = sampler, k_neighbors = k_neighbors)
                        
                    return values
        
       
                    
                    
      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        