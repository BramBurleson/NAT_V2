from pathlib import Path
import os
import shutil
import pandas as pd



ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')

subject_w_files_to_merge = 

mygreen = (0.3, 0.7, 0.5)  # bright green

subject = eyetrack_foé

#load eye data 
raw_eyetrack_folder = Path(subject, task, 'raw')
file = list(raw_eyetrack_folder.glob('2025*.txt'))

tmpload     = pd.read_csv(file[0], usecols=[0], delimiter='\t',header=None) #extract first column of eye data file contains tags -- indices for differents sections of file (by rows)
not_header  = tmpload.index[tmpload.iloc[:,0] != 5].tolist()
not_data = tmpload.index[~tmpload.iloc[:, 0].isin([10, 12])].tolist()

header          = pd.read_csv(file[0], skiprows=not_header, nrows=1, delimiter='\t').columns.tolist() #extract header by skipping all rows except header.       
data            = pd.read_csv(file[0], skiprows=not_data, delimiter='\t', header=None) #extract data by skipping all rows except data
data.columns    = header  # Add header to data

#identify run_starts and run_ends
run_starts = list(data.loc[data['DeltaTime'] == '5'].index)
print(run_starts)
run_ends = np.array(list(data.loc[data['DeltaTime'] == 'Experiment'].index))-1
run_ends = run_ends.tolist()
print(run_ends)

#match starts to ends

for run_number, (run_start, run_end) in enumerate(list(zip(run_starts, run_ends))):



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