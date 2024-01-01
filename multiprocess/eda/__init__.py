import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(dataset: list, data_name: list = None):
    values = {}
    index = [a for a in range(len(dataset))]
    if data_name == None:
        if isinstance(dataset, list):
            for data, index_count in zip(dataset, index):
                data.info()
                print("\n\n")
                data_head = data.head()
                data_tail = data.tail()
                data_descriptive_statistic = data.describe()
                data_more_descriptive_statistics = data.describe(include = "all")
                data_mode = data.mode()
                data_distinct_count = data.nunique()
                data_null_count = data.isnull().sum()
                data_total_null_count = data.isnull().sum().sum()
                data_correlation_matrix = data.corr()
    
                values[f"Dataset {index_count + 1}"] = {"Dataset": data, "Data_Head": data_head, "Data_Tail": data_tail, "Data_Descriptive_Statistic": data_descriptive_statistic, "Data_More_Descriptive_Statistic": data_more_descriptive_statistics, "Data_Mode": data_mode, "Data_Distinct_Count": data_distinct_count, "Data_Null_Count": data_null_count, "Data_Total_Null_Count": data_total_null_count, "Data_Correlation_Matrix": data_correlation_matrix}
        
        else:
            raise ValueError("Data must be a list as this function works for multi-dataset processing.")
            
    elif data_name != None:
        if isinstance(dataset, list) and isinstance(data_name, list) and (len(dataset) == len(data_name)):
            for data, name, index_count in zip(dataset, data_name, index):
                data.info()
                print("\n\n")
                data_head = data.head()
                data_tail = data.tail()
                data_descriptive_statistic = data.describe()
                data_more_descriptive_statistics = data.describe(include = "all")
                data_mode = data.mode()
                data_distinct_count = data.nunique()
                data_null_count = data.isnull().sum()
                data_total_null_count = data.isnull().sum().sum()
                data_correlation_matrix = data.corr()
    
                values[f"{name}"] = {"Dataset": data, "Data_Head": data_head, "Data_Tail": data_tail, "Data_Descriptive_Statistic": data_descriptive_statistic, "Data_More_Descriptive_Statistic": data_more_descriptive_statistics, "Data_Mode": data_mode, "Data_Distinct_Count": data_distinct_count, "Data_Null_Count": data_null_count, "Data_Total_Null_Count": data_total_null_count, "Data_Correlation_Matrix": data_correlation_matrix}
        
        else:
            raise ValueError("Data must be a list as this function works for multi-dataset processing.")
    
    return values


def visualization_eda(dataset: list, data_name: list = None, before_data_cleaning: bool = True):   
    index = [a for a in range(len(dataset))]
    if data_name == None:
        if isinstance(dataset, list):
            for data, index_count in zip(dataset, index):
                if before_data_cleaning == False:
                    data_histogram = data.hist(figsize = (15, 10), bins = 10)
                    plt.show()
                    
                    plt.figure(figsize = (15, 10))
                    data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
                    plt.title(f'Correlation Matrix for Dataset {index_count + 1}')
                    plt.show()
                
                elif before_data_cleaning == True:
                    # Visualize the distribution of categorical features
                    categorical_features = data.select_dtypes(include = "object").columns
                    for feature in categorical_features:
                        plt.figure(figsize=(8, 5))
                        sns.countplot(x=feature, data = data)
                        plt.title(f'Distribution of {feature} for Dataset {index_count + 1}')
                        plt.show()
                      
                    data_histogram = data.hist(figsize = (15, 10), bins = 10)
                    plt.show()
                    
                    plt.figure(figsize = (15, 10))
                    data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
                    plt.title(f'Correlation Matrix for Dataset {index_count + 1}')
                    plt.show()
                    
        else:
            raise ValueError("Data must be a list as this function works for multi-dataset processing.")
            
    elif data_name != None:
        if isinstance(dataset, list) and isinstance(data_name, list) and (len(dataset) == len(data_name)):
            for data, name, index_count in zip(dataset, data_name, index):
                for data, index_count in zip(dataset, index):
                    if before_data_cleaning == False:
                        data_histogram = data.hist(figsize = (15, 10), bins = 10)
                        plt.show()
                        
                        plt.figure(figsize = (15, 10))
                        data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
                        plt.title(f'Correlation Matrix for {name}')
                        plt.show()
                    
                    elif before_data_cleaning == True:
                        # Visualize the distribution of categorical features
                        categorical_features = data.select_dtypes(include = "object").columns
                        for feature in categorical_features:
                            plt.figure(figsize=(8, 5))
                            sns.countplot(x=feature, data = data)
                            plt.title(f'Distribution of {feature} for {name}')
                            plt.show()
                          
                        data_histogram = data.hist(figsize = (15, 10), bins = 10)
                        plt.show()
                        
                        plt.figure(figsize = (15, 10))
                        data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
                        plt.title(f'Correlation Matrix for Dataset {name}')
                        plt.show()
                        
        else:
            raise ValueError("Data must be a list as this function works for multi-dataset processing.")
    