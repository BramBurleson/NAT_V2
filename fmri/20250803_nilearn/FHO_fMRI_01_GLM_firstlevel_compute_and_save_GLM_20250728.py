#REQUIRES nilearn v 0.10.4: use pip install nilearn==0.10.4
#   model: hrf => spm
#   regressors from fmri-prep confounds.tsv 
#       motion trans xyz, rot xyz
#       csf/white matter : a_comp_cor 00... 09
#       global signal
#   conditions : 'Explore', 'Key Ego', 'Key Allo', 'Key Color', 'Eagle Ego', 'Eagle Allo', 'Eagle Color'}



import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime as dt
from scipy.stats import norm

from nilearn import image, plotting, datasets
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix,  FirstLevelModel
from nilearn.glm import threshold_stats_img

import joblib

#SETUP
smoothing_parameter = 6
subject_labels = ['sub-01']
glm_folder = "glm"
task_id =  "FHOegoallo"
subtask = "Ota"

#load FMRI_master parameters
ROOT_DATASET    = Path(__file__).resolve().parents[3]
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')

#set model Parameters:
t_r                     = 1.5 #repetition time, see e.g. derivatives/sub-01/func/sub-01_task-nat_run-01_space-T1w_desc-preproc_bold.json
# slice_time_ref          = 0.0 #slice_time_ref ?
hrf_model               = "spm" #if you run with derivative you also need to account for beta maps from derivative
mni_template            = datasets.load_mni152_template()
data_dict = {} #create dictionary to map subjects to their designmatrices from each run.

#%% assemble volumes and create design matrices
print("creating design matrices")
for subject_label in subject_labels:
    subj_img                         = [] #pre-allocate BOLD
    subj_design_matrices_confounds   = [] #pre-allocate first-level design matrices with confounds
    subj_confounds                   = [] #pre-allocate confounds
    subj_contrasts                   = [] #pre-allocate contrasts
    subj_run_ids                     = [] #pre-allocate run ids
    print(subject_label)

    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders

    subject_folder_glm = Path(subjectfolder, glm_folder)
    if not subject_folder_glm.exists():
        os.makedirs(subject_folder_glm), print(f"made {subject_folder_glm}")

    run_BOLD_files_raw = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz*') #get bold files
    run_BOLD_files_smooth = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold*{smoothing_parameter}*.nii.gz')     
    run_BOLD_files_mask = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz') #get brain mask files
    print(run_BOLD_files_raw)
    print(run_BOLD_files_smooth)

    for r, run_BOLD_file in enumerate(run_BOLD_files_raw): 
        pattern = r"run-(\d+)"
        run_id  = f"{re.search(pattern, run_BOLD_file).group()}"

        subj_run_ids.append(run_id) #if event file exists add to subject run list
        run_events_file = Path(subjectfolder, 'events', f'FHOF_events_{subject_label}_{run_id}.tsv') # 'sub-01_task-FHOegoallo_run-01_events) #Get events files

        events = pd.read_table(run_events_file) #load events
        #Load BOLD data
        print(f'BOLD FILE : {run_BOLD_file} and {run_BOLD_files_smooth[r]}')
        img = image.load_img(run_BOLD_files_smooth[r])  #Load the BOLD image
        mask_img = image.load_img(run_BOLD_files_mask[r])  #Load the mask image
        subj_img.append(img)  #Append the smoothed image to sub_img
          
        # prep for first-level design matrix
        n_scans = img.shape[-1] 
        frame_times = np.arange(n_scans) * t_r #print(frame_times)
      
        # Get confounds(noise regressors)
        confounds_data = load_confounds(
            run_BOLD_file, 
            strategy = ('high_pass','motion','compcor','global_signal'),  # high pass filter must be included for compcor
            # high_pass applies discrete cosine transfer function (DCT) https://en.wikipedia.org/wiki/Discrete_cosine_transform#/media/File:DCT_filter_comparison.png
            # and will create additional cosine columns in design matrix for longer runs!!
            motion          = 'basic',           #basic: translation/rotation (6 parameters)
            compcor         = 'anat_combined',   #anat_combined: white matter and CSF combined anatomical mask
            n_compcor       = 10,                #10: number of noise components of mask
            global_signal   = 'basic',           #basic: just the global signal (1 parameter)
            )
   
        confounds = confounds_data[0]  #load_confounds outputs a tuple, with a df and ?, we just want the df so get df with [0]
        subj_confounds.append(pd.DataFrame(confounds))
        design_matrix_confounds = make_first_level_design_matrix(      
            frame_times,   #create first-level design matrix with confounds
            events,
            hrf_model   = hrf_model,
            drift_model = None, #default : "Drift" discrete cosine transform to model signal drifts, despite this because of using highpass in confounds there will be drift model.
            high_pass   = None, #default: 0.01Hz cutoff for the drift model, see above
            add_regs    = confounds,
        )

        # Plot and save subject design matrices
        print("Plotting design_matrices and saving plots")
        output_file = f'{subjectfolder}/{glm_folder}/{subject_label}_{run_id}_design_matrix.png'
        plotting.plot_design_matrix(
            design_matrix_confounds,
            output_file = output_file,     
        ) 
        design_matrix_confounds.to_csv(f'{subjectfolder}/{glm_folder}/{subject_label}_{run_id}_design_matrix.csv', index=False)
        subj_design_matrices_confounds.append(design_matrix_confounds) #append design matrix to subject design matrices

    subj_data = {
        'run_ids':subj_run_ids,
        'imgs':subj_img,
        'design_matrices':subj_design_matrices_confounds,
        'confounds': subj_confounds,
    }

    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') 
    subjectfolder_glm = Path(subjectfolder, glm_folder)

    #Fit GLM  with noise regressors "confounds from fMRIPrep."
    print("Fitting a GLM with noise regressors")
    # glm = FirstLevelModel(mask_img=mask_img)
    glm = FirstLevelModel()
    glm = glm.fit(subj_data['imgs'], design_matrices = subj_data['design_matrices'])

    #save GLM
    glm_file = subjectfolder_glm / f'{subtask}_{subject_label}_firstlevelmodel.pkl'
    joblib.dump(glm, glm_file)