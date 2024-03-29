import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pp
import sweetviz as sv

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"

def eda(data):
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
    return {"Dataset": data, "Data_Head": data_head, "Data_Tail": data_tail, "Data_Descriptive_Statistic": data_descriptive_statistic, "Data_More_Descriptive_Statistic": data_more_descriptive_statistics, "Data_Mode": data_mode, "Data_Distinct_Count": data_distinct_count, "Data_Null_Count": data_null_count, "Data_Total_Null_Count": data_total_null_count, "Data_Correlation_Matrix": data_correlation_matrix}
    
def eda_visual(data, y: str, histogram_bins: int = 10, figsize_heatmap: tuple = (15, 10), figsize_histogram: tuple = (15, 10), figsize_barchart: tuple = (15, 10), before_data_cleaning: bool = True):
    if before_data_cleaning == False:
        data_histogram = data.hist(figsize = figsize_histogram, bins = histogram_bins)
        plt.figure(figsize = figsize_heatmap)
        data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
    
    elif before_data_cleaning == True:
        # Visualize the distribution of categorical features
        categorical_features = data.select_dtypes(include = "object").columns
        for feature in categorical_features:
            plt.figure(figsize=figsize_barchart)
            sns.countplot(x=feature, data = data)
            plt.title(f'Distribution of {feature}')
            plt.show()
        
        data_histogram = data.hist(figsize = figsize_histogram, bins = histogram_bins)
        plt.figure(figsize = figsize_heatmap)
        data_heatmap = sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        
def sweet_viz(dataset, filename: str, auto_open: bool = False):
    report1 = sv.analyze(dataset)
    report1.show_html(filepath = filename, open_browser = auto_open)
    
def pandas_profiling(dataset, output_file: str = "Pandas Profile Report.html", dark_mode: bool = False, title: str = "Report"):
    report = pp.ProfileReport(df = dataset, dark_mode = dark_mode, explorative = True, title = title)
    report.to_widgets()
    report.to_file(output_file = output_file)