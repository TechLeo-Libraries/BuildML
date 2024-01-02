import numpy as np
import pandas as pd
import sklearn.impute as si
import sklearn.preprocessing as sp
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
import datatable as dt

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"
__version__ = "0.0.1"


__all__ = [
    "column_binning",
    "categorical_to_numerical",
    "count_column_categories",
    "drop_columns",
    "filter_data",
    "fix_missing_values",
    "fix_unbalanced_dataset",
    "group_data",
    "load_large_dataset",
    "numerical_to_categorical",
    "remove_duplicates",
    "remove_outlier",
    "rename_columns",
    "replace_values",
    "reset_index",
    "scale_independent_variables",
    "select_datatype",
    "set_index",
    "sort_index",
    "sort_values",
    "replace_values",
]



def group_data(dataset, columns: list or tuple, column_to_groupby: str or list or tuple, aggregate_function: str, reset_index: bool = False):
    agg = ["mean", "count", "min", "max", "std", "var", "median"]
    
    if reset_index == False:
        if isinstance(columns, list) or isinstance(columns, tuple):
            if isinstance(column_to_groupby, str) or isinstance(column_to_groupby, list) or isinstance(column_to_groupby, tuple):
                aggregate_function = aggregate_function.lower().strip()
                if aggregate_function in agg and isinstance(aggregate_function, str):
                    if aggregate_function == "mean":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).mean()
                    
                    elif aggregate_function == "count":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).count()
                        
                    elif aggregate_function == "min":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).min()
                        
                    elif aggregate_function == "max":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).max()
                        
                    elif aggregate_function == "std":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).std()
                        
                    elif aggregate_function == "var":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).var()
                        
                    elif aggregate_function == "median":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).median()
                        
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
                        grouped_columns = dataset[columns].groupby(column_to_groupby).mean()
                        grouped_columns = grouped_columns.reset_index()
                    
                    elif aggregate_function == "count":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).count()
                        grouped_columns = grouped_columns.reset_index()
                        
                    elif aggregate_function == "min":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).min()
                        grouped_columns = grouped_columns.reset_index()
                        
                    elif aggregate_function == "max":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).max()
                        grouped_columns = grouped_columns.reset_index()
                        
                    elif aggregate_function == "std":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).std()
                        grouped_columns = grouped_columns.reset_index()
                        
                    elif aggregate_function == "var":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).var()
                        grouped_columns = grouped_columns.reset_index()
                        
                    elif aggregate_function == "median":
                        grouped_columns = dataset[columns].groupby(column_to_groupby).median()
                        grouped_columns = grouped_columns.reset_index()
                        
                else:
                    raise ValueError(f"Specify the right aggregate function from the following: {agg}")
                        
        else:
            raise ValueError("You need to select more than one column as a list or tuple to perform a groupby operation.")
    
    
    else:
        raise ValueError("The arguments for 'reset_index' must be boolean of TRUE or FALSE.")
    
    return grouped_columns               
         

def count_column_categories(dataset, column: str or list or tuple, reset_index: bool = False):
    if reset_index == False:
        if isinstance(column, str) or isinstance(column, list) or isinstance(column, tuple):
            categories_count = dataset[column].value_counts()
            
        else:
            raise ValueError("Column inserted must be a string, list, or tuple.")
            
    elif reset_index == True:
        if isinstance(column, str) or isinstance(column, list) or isinstance(column, tuple):
            categories_count = dataset[column].value_counts()
            categories_count = categories_count.reset_index()
            
        else:
            raise ValueError("Column inserted must be a string, list, or tuple.")
            
    else:
        raise ValueError("The arguments for 'reset_index' must be boolean of TRUE or FALSE.")
        
    return categories_count


def replace_values(dataset, replace: int or float or str or list or tuple or dict, new_value: int or float or str or list or tuple):
    if isinstance(replace, str) or isinstance(new_value, int) or isinstance(new_value, float):
        if isinstance(new_value, str) or isinstance(new_value, int) or isinstance(new_value, float):
            dataset.replace(to_replace = replace, value = new_value, inplace = True)
        else:
            raise ValueError("If replace is a string, integer, or float, then new value must be either a string, integer, or float.")
    
    elif isinstance(replace, list) or isinstance(replace, tuple):
        if isinstance(new_value, str) or isinstance(new_value, int) or isinstance(new_value, float):
            dataset.replace(to_replace = replace, value = new_value, inplace = True)
        elif isinstance(new_value, list) or isinstance(new_value, tuple):
            for word, new in zip(replace, new_value):
                dataset.replace(to_replace = word, value = new, inplace = True)
        else:
            raise ValueError("If replace is a list or tuple, then value can be any of int, str, float, list, or tuple.")
                
    elif isinstance(replace, dict):
        dataset.replace(to_replace = replace, value = new_value, inplace = True)
        
    else:
        raise ValueError("Check your input arguments for the parameters: replace and new_value")
    
    return {"Dataset ---> Dataset with Replaced Values": dataset}



def sort_values(dataset, column: str or list, ascending: bool = True, reset_index: bool = False):
    if isinstance(column, str) or isinstance(column, list):
        dataset.sort_values(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
    
    return {"Dataset ---> Sorted Dataset": dataset}



def set_index(dataset, column: str or list):
    if isinstance(column, str) or isinstance(column, list):
        dataset = dataset.set_index(column)
    
    return {"Dataset ---> Index Set": dataset}



def sort_index(dataset, column: str or list, ascending: bool = True, reset_index: bool = False):
    if isinstance(column, str) or isinstance(column, list):
        dataset.sort_index(by = column, ascending = ascending, ignore_index = reset_index, inplace = True)
    
    return {"Dataset ---> Sorted Dataset": dataset}



def rename_columns(dataset, old_column: str or list, new_column: str or list):
    if isinstance(old_column, str) and isinstance(new_column, str):
        dataset.rename({old_column: new_column}, axis = 1, inplace = True)
        
    elif isinstance(old_column, list) and isinstance(new_column, list):
        dataset.rename({key:value for key, value in zip(old_column, new_column)}, axis = 1, inplace = True)
    
    return {"Dataset ---> Column Changed": dataset}



def reset_index(dataset, drop_index_after_reset: bool = False):
    dataset.reset_index(drop = drop_index_after_reset, inplace = True)
    
    return {"Dataset ---> Column Changed": dataset}



def filter_data(dataset, column: str or list or tuple, operation: str or list or tuple = None, value: int or float or str or list or tuple = None):
    possible_operations = ['greater than', 'less than', 'equal to', 'greater than or equal to', 'less than or equal to', 'not equal to', '>', '<', '==', '>=', '<=', '!=']
    if column != None:
        if isinstance(column, str):
            if isinstance(value, int) or isinstance(value, float):
                if isinstance(operation, str):
                    if operation.lower() not in possible_operations:
                        raise ValueError(f"This operation is not supported. Please use the following: {possible_operations}")
                        
                    elif (operation.lower() == 'greater than' or operation == '>'):
                        condition = dataset[column] > value
                        dataset = dataset[condition]
                        
                    elif (operation.lower() == 'less than' or operation == '<'):
                        condition = dataset[column] < value
                        dataset = dataset[condition]
                        
                    elif (operation.lower() == 'equal to' or operation == '=='):
                        condition = dataset[column] == value
                        dataset = dataset[condition]
                        
                    elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                        condition = dataset[column] >= value
                        dataset = dataset[condition]
                        
                    elif (operation.lower() == 'less than or equal to' or operation == '<='):
                        condition = dataset[column] <= value
                        dataset = dataset[condition]
                        
                    elif (operation.lower() == 'not equal to' or operation == '!='):
                        condition = dataset[column] != value
                        dataset = dataset[condition]
                        
                elif isinstance(operation, list) or isinstance(operation, int) or isinstance(operation, float) or isinstance(operation, tuple):
                    raise ValueError("When column is set to string and value is set to either float, int, or string. Operation can not be a list or tuple. Must be set to string")
                   
                    
                   
            elif isinstance(value, str):
                if (operation.lower() == 'equal to' or operation == '=='):
                    condition = dataset[column] == value
                    dataset = dataset[condition]
                    
                elif (operation.lower() == 'not equal to' or operation == '!='):
                    condition = dataset[column] != value
                    dataset = dataset[condition]  
                    
                else:
                    raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                    
                    
                    
            elif isinstance(value, list) or isinstance(value, tuple):
                if isinstance(operation, str):
                    raise ValueError("Length of values should be same as length of available operations to perform")
                            
                elif isinstance(operation, list) or isinstance(operation, tuple):
                    for item, symbol in zip(value, operation):
                        if isinstance(item, str):
                            if (symbol.lower() == 'equal to' or symbol == '=='):
                                condition = dataset[column] == item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                condition = dataset[column] != item
                                dataset = dataset[condition]  
                                
                            else:
                                raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                                
                        
                        elif isinstance(item, int) or isinstance(item, float):
                            if (symbol.lower() == 'greater than' or symbol == '>'):
                                condition = dataset[column] > item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'less than' or symbol == '<'):
                                condition = dataset[column] < item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'equal to' or symbol == '=='):
                                condition = dataset[column] == item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                                condition = dataset[column] >= item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                                condition = dataset[column] <= item
                                dataset = dataset[condition]
                                
                            elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                condition = dataset[column] != item
                                dataset = dataset[condition]
        
        
        
        elif isinstance(column, list) or isinstance(column, tuple):
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                raise ValueError("If column is a list or tuple, then value must assume form of a list or tuple with same length.")
                
            elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) == len(column)):
                if isinstance(operation, str):
                    for col, item in zip(column, value):
                        if isinstance(item, str):
                            if (operation.lower() == 'equal to' or operation == '=='):
                                condition = dataset[col] == item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'not equal to' or operation == '!='):
                                condition = dataset[col] != item
                                dataset = dataset[condition]  
                                
                            else:
                                raise ValueError("When value is a string, comparison of greater than or less than cannot be made. Consider switching operation to a list or tuple for more control.")
                                
                        
                        elif isinstance(item, int) or isinstance(item, float):
                            if (operation.lower() == 'greater than' or operation == '>'):
                                condition = dataset[col] > item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'less than' or operation == '<'):
                                condition = dataset[col] < item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'equal to' or operation == '=='):
                                condition = dataset[col] == item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                                condition = dataset[col] >= item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'less than or equal to' or operation == '<='):
                                condition = dataset[col] <= item
                                dataset = dataset[condition]
                                
                            elif (operation.lower() == 'not equal to' or operation == '!='):
                                condition = dataset[col] != item
                                dataset = dataset[condition]
                                
                            
                elif isinstance(operation, list) or isinstance(operation, tuple):
                    if len(operation) == len(value) == len(column):
                        for col, item, symbol in zip(column, value, operation):
                            if isinstance(item, str):
                                if (symbol.lower() == 'equal to' or symbol == '=='):
                                    condition = dataset[col] == item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                    condition = dataset[col] != item
                                    dataset = dataset[condition]  
                                    
                                else:
                                    raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                                    
                            
                            elif isinstance(item, int) or isinstance(item, float):
                                if (symbol.lower() == 'greater than' or symbol == '>'):
                                    condition = dataset[col] > item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'less than' or symbol == '<'):
                                    condition = dataset[col] < item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'equal to' or symbol == '=='):
                                    condition = dataset[col] == item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                                    condition = dataset[col] >= item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                                    condition = dataset[col] <= item
                                    dataset = dataset[condition]
                                    
                                elif (symbol.lower() == 'not equal to' or symbol == '!='):
                                    condition = dataset[col] != item
                                    dataset = dataset[condition]
                    
                    
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
                   condition = dataset > value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'less than' or operation == '<'):
                   condition = dataset < value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'equal to' or operation == '=='):
                   condition = dataset == value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'greater than or equal to' or operation == '>='):
                   condition = dataset >= value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'less than or equal to' or operation == '<='):
                   condition = dataset <= value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'not equal to' or operation == '!='):
                   condition = dataset != value
                   dataset = dataset[condition]
                   
            
            elif isinstance(value, str):
               if (operation.lower() == 'equal to' or operation == '=='):
                   condition = dataset == value
                   dataset = dataset[condition]
                   
               elif (operation.lower() == 'not equal to' or operation == '!='):
                   condition = dataset != value
                   dataset = dataset[condition]  
                   
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
                            condition = dataset == item
                            dataset = dataset[condition]
                           
                        elif (symbol.lower() == 'not equal to' or symbol == '!='):
                            condition = dataset != item
                            dataset = dataset[condition]  
                           
                        else:
                            raise ValueError("When value is a string, comparison of greater than or less than cannot be made.")
                           
                   
                    elif isinstance(item, int) or isinstance(item, float):
                        if (symbol.lower() == 'greater than' or symbol == '>'):
                            condition = dataset > item
                            dataset = dataset[condition]
                           
                        elif (symbol.lower() == 'less than' or symbol == '<'):
                            condition = dataset < item
                            dataset = dataset[condition]
                            
                        elif (symbol.lower() == 'equal to' or symbol == '=='):
                            condition = dataset == item
                            dataset = dataset[condition]
                            
                        elif (symbol.lower() == 'greater than or equal to' or symbol == '>='):
                            condition = dataset >= item
                            dataset = dataset[condition]
                            
                        elif (symbol.lower() == 'less than or equal to' or symbol == '<='):
                            condition = dataset <= item
                            dataset = dataset[condition]
                            
                        elif (symbol.lower() == 'not equal to' or symbol == '!='):
                            condition = dataset != item
                            dataset = dataset[condition]
                            
            elif (isinstance(value, list) or isinstance(value, tuple)) and (len(value) != len(operation)):
                raise ValueError("If operation is list or tuple, then value must be list or tuple of same size.")
        
    return dataset



def remove_duplicates(dataset, which_columns: str or list or tuple = None):
    if isinstance(which_columns, str) or isinstance(which_columns, list) or isinstance(which_columns, tuple):
        dataset.drop_duplicates(inplace = True, subset = which_columns)
        
    else:
        raise ValueError("Removing duplicates from your dataset must be done by indicating the column as either a string, list, or tuple.")
    
    return {"Dataset ---> Removed Duplicates": dataset}


def scale_independent_variables(x):
    scaler = sp.StandardScaler()
    x = scaler.fit_transform(x)
    x = pd.DataFrame(x, columns = scaler.feature_names_in_)
    return x

def load_large_dataset(dataset: str):
    data = dt.fread(dataset).to_pandas()
    return data

def reduce_data_memory_useage(dataset, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = dataset.memory_usage().sum() / 1024 ** 2
    for col in dataset.columns:
        col_type = dataset[col].dtypes
        if col_type in numerics:
            c_min = dataset[col].min()
            c_max = dataset[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataset[col] = dataset[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataset[col] = dataset[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataset[col] = dataset[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataset[col] = dataset[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    dataset[col] = dataset[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    dataset[col] = dataset[col].astype(np.float32)
                else:
                    dataset[col] = dataset[col].astype(np.float64)
    end_mem = dataset.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return dataset

def drop_columns(dataset, columns: list, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
    dataset = dataset.drop(columns, axis = 1)
    return dataset
    
def fix_missing_values(dataset, strategy: str = None, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
        
    if strategy == None:
        imputer = si.SimpleImputer(strategy = "mean")
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns = imputer.feature_names_in_)
        return dataset
        
    elif strategy.lower().strip() == "mean":
        imputer = si.SimpleImputer(strategy = "mean")
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns = imputer.feature_names_in_)
        return dataset
        
    elif strategy.lower().strip() == "median":
        imputer = si.SimpleImputer(strategy = "median")
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns = imputer.feature_names_in_)
        return dataset
        
    elif strategy.lower().strip() == "mode":
        imputer = si.SimpleImputer(strategy = "most_frequent")
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns = imputer.feature_names_in_)
        return dataset
    
def categorical_to_numerical(dataset, columns: list = None, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
    
    if columns == None:
        dataset = pd.get_dummies(dataset, drop_first = True, dtype = int)
        return dataset
        
    else:
        dataset = pd.get_dummies(dataset, columns = columns, drop_first = True, dtype = int)
        return dataset
       
def remove_outlier(dataset, warning: bool = False):   
    if warning == True:
        warnings.filterwarnings("ignore")
        
    scaler = sp.StandardScaler()
    dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns = scaler.feature_names_in_)
    dataset = dataset[(dataset >= -3) & (dataset <= 3)]
    dataset = pd.DataFrame(scaler.inverse_transform(dataset), columns = scaler.feature_names_in_)
    return dataset  

def select_datatype(dataset, datatype_to_select: str = None, datatype_to_exclude: str = None, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
        
    selected_data = dataset.select_dtypes(include = datatype_to_select, exclude = datatype_to_exclude)
    return selected_data

def numerical_to_categorical(dataset, column, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
        
    if isinstance(column, list):
        for items in column:
            dataset[items] = dataset[items].astype("object")
    
    elif isinstance(column, str):
        dataset[column] = dataset[column].astype("object")
        
    elif isinstance(column, tuple):
        for items in column:
            dataset[items] = dataset[items].astype("object")
    return dataset

def column_binning(data, column, number_of_bins: int = 10, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
        
    if isinstance(column, list):
        for items in column:
            data[items] = pd.cut(data[items], bins = number_of_bins, labels = False)
    
    elif isinstance(column, str):
        data[column] = pd.cut(data[column], bins = number_of_bins, labels = False)
        
    elif isinstance(column, tuple):
        for items in column:
            data[items] = pd.cut(data[items], bins = number_of_bins, labels = False)
    return data

def fix_unbalanced_dataset(x_train, y_train, sampler: str, k_neighbors: int = None, warning: bool = False):
    if warning == True:
        warnings.filterwarnings("ignore")
        
    if sampler == "SMOTE" and k_neighbors != None:
        technique = SMOTE(random_state = 0, k_neighbors = k_neighbors)
        x_train, y_train = technique.fit_resample(x_train, y_train)
    
    elif sampler == "SMOTE" and k_neighbors == None:
        technique = SMOTE(random_state = 0)
        x_train, y_train = technique.fit_resample(x_train, y_train)
        
    elif sampler == "Random over sampler" and k_neighbors == None:
        technique = RandomOverSampler(random_state = 0)
        x_train, y_train = technique.fit_resample(x_train, y_train)
        
    elif sampler == "Random under sampler" and k_neighbors == None:
        technique = RandomUnderSampler(random_state = 0)
        x_train, y_train = technique.fit_resample(x_train, y_train)
        
    else:
        ValueError("k_neighbors works with only the SMOTE algorithm.")
    
    return {"Training X": x_train, "Training Y": y_train}