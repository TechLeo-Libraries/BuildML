import pandas as pd

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"


__all__ = [
    "categorical_to_datetime",
    "extract_date_features",
    ]

def categorical_to_datetime(data, column):
    if isinstance(column, list):
        for items in column:
            data[items] = pd.to_datetime(data[items])
    
    elif isinstance(column, str):
        data[column] = pd.to_datetime(data[column])
        
    elif isinstance(column, tuple):
        for items in column:
            data[items] = pd.to_datetime(data[items])
    
    return data

def extract_date_features(data, datetime_column, hrs_mins_sec: bool = False):
    if hrs_mins_sec == False:
        if isinstance(datetime_column, list):
            for items in datetime_column:
                data[f"Day_{items}"] = data[items].day
                data[f"Month_{items}"] = data[items].month
                data[f"Year_{items}"] = data[items].year
                data[f"Quarter_{items}"] = data[items].quarter
                data[f"Day_of_Week_{items}"] = data[items].day_of_week
        
        elif isinstance(datetime_column, str):
            data["Day"] = data[datetime_column].day
            data["Month"] = data[datetime_column].month
            data["Year"] = data[datetime_column].year
            data["Quarter"] = data[datetime_column].quarter
            data["Day_of_Week"] = data[datetime_column].day_of_week
            
        elif isinstance(datetime_column, tuple):
            for items in datetime_column:
                data[f"Day_{items}"] = data[items].day
                data[f"Month_{items}"] = data[items].month
                data[f"Year_{items}"] = data[items].year
                data[f"Quarter_{items}"] = data[items].quarter
                data[f"Day_of_Week_{items}"] = data[items].day_of_week
        
        return data
    
    elif hrs_mins_sec == False:
        if isinstance(datetime_column, list):
            for items in datetime_column:
                data[f"Day_{items}"] = data[items].day
                data[f"Month_{items}"] = data[items].month
                data[f"Year_{items}"] = data[items].year
                data[f"Quarter_{items}"] = data[items].quarter
                data[f"Day_of_Week_{items}"] = data[items].day_of_week
                data[f"Hour_{items}"] = data[items].hour
                data[f"Minutes_{items}"] = data[items].minute
                data[f"Seconds_{items}"] = data[items].second                   
                
        elif isinstance(datetime_column, str):
            data["Day"] = data[datetime_column].day
            data["Month"] = data[datetime_column].month
            data["Year"] = data[datetime_column].year
            data["Quarter"] = data[datetime_column].quarter
            data["Day_of_Week"] = data[datetime_column].day_of_week
            data[f"Hour_{items}"] = data[items].hour
            data[f"Minutes_{items}"] = data[items].minute
            data[f"Seconds_{items}"] = data[items].second  
            
        elif isinstance(datetime_column, tuple):
            for items in datetime_column:
                data[f"Day_{items}"] = data[items].day
                data[f"Month_{items}"] = data[items].month
                data[f"Year_{items}"] = data[items].year
                data[f"Quarter_{items}"] = data[items].quarter
                data[f"Day_of_Week_{items}"] = data[items].day_of_week
                data[f"Hour_{items}"] = data[items].hour
                data[f"Minutes_{items}"] = data[items].minute
                data[f"Seconds_{items}"] = data[items].second  
        
        return data