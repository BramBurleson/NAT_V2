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
import joblib

from nilearn import image, plotting, datasets
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix,  FirstLevelModel
from nilearn.glm import threshold_stats_img

# import joblib

#SETUP
smoothing_parameter = 6
subject_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
mvpa_folder = "run_level_glm"
task_id =  "nat"

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
    print(subject_label)

    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders

    single_run_glm = Path(subjectfolder, mvpa_folder)
    if not single_run_glm.exists():
        os.makedirs(single_run_glm), print(f"made {single_run_glm}")

    run_BOLD_files_raw = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz*') #get bold files
    run_BOLD_files_smooth = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold*{smoothing_parameter}*.nii')     
    run_BOLD_files_mask = glob.glob(f'{subjectfolder}/func/sub*task-{task_id}*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz') #get brain mask files
    print(run_BOLD_files_raw)
    print(run_BOLD_files_smooth)

    for r, run_BOLD_file in enumerate(run_BOLD_files_raw): 
        pattern = r"run-(\d+)"
        run_id  = f"{re.search(pattern, run_BOLD_file).group()}"

        run_events_file = Path(subjectfolder, 'events', f'{subject_label}_{run_id}_events.tsv') # 'sub-01_task-FHOegoallo_run-01_events) #Get events files

        events = pd.read_table(run_events_file) #load events
        #Load BOLD data
        print(f'BOLD FILE : {run_BOLD_file} and {run_BOLD_files_smooth[r]}')
        run_img = image.load_img(run_BOLD_files_smooth[r])  #Load the BOLD image
        mask_img = image.load_img(run_BOLD_files_mask[r])  #Load the mask image

          
        # prep for first-level design matrix
        n_scans = run_img.shape[-1] 
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
        output_file = f'{single_run_glm}/{subject_label}_{run_id}_design_matrix.png'
        plotting.plot_design_matrix(
            design_matrix_confounds,
            output_file = output_file,     
        ) 
        design_matrix_confounds.to_csv(f'{single_run_glm}/{subject_label}_{run_id}_design_matrix.csv', index=False)
               
        #Fit GLM  with noise regressors "confounds from fMRIPrep."
        print("Fitting a GLM with noise regressors")
        # glm = FirstLevelModel(mask_img=mask_img)
        glm = FirstLevelModel()
        glm = glm.fit(run_img, design_matrices = design_matrix_confounds)

        #save GLM
        glm_file = single_run_glm/ f'{subject_label}_{run_id}_glm.pkl'
        joblib.dump(glm, glm_file)