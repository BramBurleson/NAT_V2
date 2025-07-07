from pathlib import Path
import os
from datetime import datetime
import shutil

first_pos_file_id = "0_POSITION" #define wild cards for first and last position files
last_pos_file_id = "13_POSITION"

ROOT_DATASET = Path(__file__).resolve().parent #get main folder path
print(ROOT_DATASET)§

raw_folder = Path(ROOT_DATASET, "raw_behav_data")
sorted_folder = Path(ROOT_DATASET, "sorted_behav_data") 
if not sorted_folder.exists(): #if it doesn't already exist create sorted_raw_behav_data_folder
    os.mkdir(sorted_folder)

raw_run_folders = [Path(f"{run}/1") for run in raw_folder.iterdir() if run.is_dir()]
for raw_run_folder in raw_run_folders: # iterate through raw_behav_data subfolders if these contain at least 13 POSITION Files (either look for filename or count them)
    position_files = list(raw_run_folder.glob("*POSITION*"))
    last_pos_file = [f for f in position_files if last_pos_file_id in f.name] #select the last_pos_file
    if last_pos_file : #if it exists do the following
        first_pos_file = [f for f in position_files if first_pos_file_id in f.name][0] #selet the first_pos_file
        print(first_pos_file)
        #get file creation datetime and reformat it
        first_pos_file_time = os.path.getmtime(first_pos_file)
        first_pos_file_time = datetime.fromtimestamp(first_pos_file_time)
        first_pos_file_time = first_pos_file_time.strftime('%Y%m%d_%H%M')
        print(first_pos_file_time)  # e.g., '13h45'
        
        sorted_run_folder = Path(sorted_folder, first_pos_file_time) #if it doesn't already exist created sorted_run_folder
        if not sorted_run_folder.exists():
             os.mkdir(sorted_run_folder)

        for position_file in position_files: #save each of the position files to the sorted_run_folder
            shutil.copy2(position_file, sorted_run_folder)