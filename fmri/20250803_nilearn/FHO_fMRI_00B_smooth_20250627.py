#REQUIRES nilearn v 0.10.4: use pip install nilearn==0.10.4
#First-level GLM concatenating all subject runs together to observe effect for small sample.
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


# SET UP
#1. select subjects to process by adding them to subject_labels list:
# subject_labels = ['sub-01']
subject_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] 
# ff_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07', 'sub-08', 'sub-09'] 

#2. select behavioral task used either 'nat' (derivatives_folderexperiment) or 'fhof' (localizer) 
task = "nat"

#load FMRI_master parameters
ROOT_DATASET    = Path(__file__).resolve().parent.parent.parent.parent
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')


smoothing_parameters = [6]


#%% assemble volumes and create design matrices
print("smoothing and saving")
for subject_label in subject_labels:
    print(subject_label)
    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders
    # sub-01_task-FHOegoallo_run-6_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii
    run_BOLD_files = list(Path(subjectfolder, 'func').glob(f'sub*task-{task}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii*'))     
    for run_BOLD_file in run_BOLD_files: #For each BOLD file (run) get run_id (number)
        print(run_BOLD_file.name)
        for smoothing_parameter in smoothing_parameters:
            print(f"smoothing with {smoothing_parameter} mm kernel")
            smoothed_filename = run_BOLD_file.with_name(run_BOLD_file.name.replace(".nii", f"_smooth_{smoothing_parameter}_mm.nii"))
            print(smoothed_filename)
            img = image.load_img(run_BOLD_file)  #Load the BOLD image
            img = image.smooth_img(img, fwhm=smoothing_parameter)  #smooth BOLD data
      
            img.to_filename(smoothed_filename)
          
    
# %%
