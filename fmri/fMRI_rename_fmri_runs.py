import numpy as np
import os
import glob
# from pathlib import Path

# subject_labels = ['sub-01', 'sub-02','sub-03','sub-04','sub-05'] 
subject_labels = ['sub-02','sub-03','sub-04','sub-05'] 

bad_runs = {
    'sub-01' : [],
    'sub-02' : '04', #13 is fhof?
    'sub-03' : '01', #02 is fhof?
    'sub-04' : [],      #13 is fhof?
    'sub-05' : '04' # 13 is fhof and 04 is bad?
}

all_runs = [f"{i:02}" for i in range(1, 13)]

fhof_runs = {
    'sub-01' : [],
    'sub-02' : '13', #13 is fhof?
    'sub-03' : '02', #02 is fhof?
    'sub-04' : '13',      #13 is fhof?
    'sub-05' : '13' # 13 is fhof and 04 is bad?
}

#step 1 list all files and folder that include 'nat'

main = r'K:\BramBurleson\01_Data\NAT_fMRI_pilots\derivatives'
# results_dir = Path(r'K:\BramBurleson\03_Results\fmri_pilots_234_fmri')

for subject_label in subject_labels: #for each subject
    # #change name to bad run:
    # pattern = rf'{main}/{subject_label}/func/*nat_run-{bad_runs[subject_label]}*'
    # #change name to fhof
 
    # run_files = glob.glob(pattern)

    # for run_file in run_files:
    #     print(run_file)
    #     new_name = str(run_file).replace(f"run-{bad_runs[subject_label]}", f"run-bad-{bad_runs[subject_label]}")
    #     # print(new_name)
    #     os.replace(run_file, new_name)
    #     print(run_file)


    # pattern2 = rf'{main}/{subject_label}/func/*nat_run-{fhof_runs[subject_label]}*'
    # #change name to fhof
 
    # run_files2 = glob.glob(pattern2)

    # for run_file2 in run_files2:
    #     print(run_file2)
    #     new_name2 = str(run_file2).replace(f"task-nat", f"task-fhof")
    #     new_name2 = str(new_name2).replace(f"run-{fhof_runs[subject_label]}", f"run-01")
    #     # print(new_name)
    #     os.replace(run_file2, new_name2)
    #     print(new_name2)


    # #%%
    # pattern3 = rf'{main}/{subject_label}/events/*events.tsv'
    # event_files = glob.glob(pattern3)

    # all_runs_subj = []
    # for run in all_runs:
    #     if run != bad_runs[subject_label]:
    #         all_runs_subj.append(run)
    # for e, event_file in enumerate(event_files):
    #     new_name = event_file.replace(f'run-{all_runs[e]}', f'run-{all_runs_subj[e]}')
    #     os.replace(event_file, new_name)

# %%
