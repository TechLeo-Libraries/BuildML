__author__ = "Leonard Onyiriuba"
__email__ = "leonard.c.onyiriuba@gmail.com"
__copyright__ = "Copyright (c) 2023-2026 Leonard Onyiriuba"
__license__ = "MIT"

def output_dataset_as_csv(dataset, file_name: str, file_path: str = None):
    if file_path == None:
        dataset.to_csv(rf"{file_name}.csv", index = True)
    elif file_path != None:
        dataset.to_csv(rf"{file_path}/{file_name}.csv", index = True)
        
def output_dataset_as_excel(dataset, file_name: str, file_path: str = None):
    if file_path == None:
        dataset.to_excel(rf"{file_name}.xlsx", index = True)
    elif file_path != None:
        dataset.to_excel(rf"{file_path}/{file_name}.xlsx", index = True)