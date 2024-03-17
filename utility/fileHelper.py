import os, shutil
import pandas as pd

script_dir = os.path.dirname(__file__)
rel_path = "\\files\\"
archive_path="\\archive\\"
FILE_PATH_LOCATION = script_dir + "\\..\\"+ rel_path

def save_file(file_name, content):
    file_path = FILE_PATH_LOCATION + file_name
    file = open(file_path, 'a')
    file.write(content)
    file.close()

def move_file(file_name):
    file_path = FILE_PATH_LOCATION + file_name
    new_file_path = FILE_PATH_LOCATION + archive_path + file_name
    shutil.move(file_path, new_file_path)

def get_all_files_from_directory():
    merged_data = pd.DataFrame()
    files =  os.listdir(FILE_PATH_LOCATION)
    if(not files):
        return False
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(FILE_PATH_LOCATION, filename)
            df = pd.read_csv(file_path)            
            merged_data = pd.concat([merged_data, df])
            move_file(filename)

    merged_data.to_csv(FILE_PATH_LOCATION+'dataset.csv', index=False)
    return True
