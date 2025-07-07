from pathlib import Path
import numpy as np
import pandas as pd
import datetime

def parameters():
    # SET UP 
    print('hi yall')
    run_for_subset = True
    skip_existing_files = False
    remove_first_flashes_from_data = True #these are all bullshit in "NAT_V1_fmri_pilots" dataset

    # get data folder
    # datafolder = Path(fr'K:\BramBurleson\01_Data\NAT_fmripilots12345')
    datafolder = Path(fr'K:\BramBurleson\01_Data\NAT_V1_fMRI_pilots\behav')
    results_dir = Path(fr'K:\BramBurleson\01_Data\NAT_V1_fMRI_pilots\behav\results\{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
    results_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory and any necessary parent directories
    
    define_subset = [
                # 'p01_20240326_mri_pilotset1',
                'p00_20241007_mri_pilotset2_eyetracker',
                # 'p02_20240826_mri_pilotset2_badbehav',
                'p02_20241129_mri_pilotset2_eyetracker',
                # 'p03_20240829_mri_pilotset2_badbehav',
                'p12_20241111_mri_pilotset2_eyetracker',
            ]
    
    subjectnames = {
    'p00_20241007_mri_pilotset2_eyetracker' : 'sub-04',
    # 'p01_20240326_mri_pilotset1'            : 'sub-01',
    'p02_20240826_mri_pilotset2_badbehav'   : 'sub-02',
    'p02_20241129_mri_pilotset2_eyetracker' : 'sub-22',
    'p03_20240829_mri_pilotset2_badbehav'   : 'sub-03',
    'p12_20241111_mri_pilotset2_eyetracker' : 'sub-05',
    }
    print(define_subset)

    # filter subjectnames by subset
    filtered_subjectnames = {}
    for key in define_subset:
        if key in subjectnames:
            filtered_subjectnames[key] = subjectnames[key]
    subjectnames = filtered_subjectnames

    # get subject folder paths
    if datafolder and datafolder.is_dir():
        if run_for_subset:
            subject_folders = define_subset
            subjects = [datafolder / folder for folder in subject_folders]
        else:
            subjects = [sub for sub in datafolder.iterdir() if sub.is_dir()]  # Get subject folders from inside datafolder
        
        subjects.sort()
        

        # names = []
        # for subject in subjects:
        #     names.append(Path(subject).name)
        # # names = ['sub-04']
        # sortednames = names.copy()
        # sortednames.sort()
        # print(f'{sortednames}')
        # name

    return{
            'run_for_subset'                : run_for_subset, 
            'skip_existing_files'           : skip_existing_files, 
            'datafolder'                    : datafolder, 
            'subjects'                      : subjects,
            'subjectnames'                  : subjectnames,
            'remove_first_flashes_from_data'  : remove_first_flashes_from_data,
            'results_dir'                     : results_dir,
        }
    
    
  
def convert_to_array(row):
    if pd.notnull(row):
        #print(row)
        try:
            return np.array([float(x) for x in row.split()])  # Convert to NumPy array np.array(ast.literal_eval(row))
        except ValueError:
            return np.nan  # Return NaN if conversion fails
    else:
        return np.nan  # Return NaN for null values
  