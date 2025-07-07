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


ROOT_DATASET    = Path(__file__).resolve().parent.parent.parent
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
# motioncorrection_dir = Path(ROOT_DATASET, 'data', 'motioncorrection')

task = "nat" #"fhof"
#'sub-02',
subject_labels = ['sub-01',  'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] 
#%% assemble volumes and create design matrices
for subject_label in subject_labels:
    print(subject_label)
    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders
    subject_motioncorrection_dir = Path(subjectfolder, 'motioncorrection')
    if not subject_motioncorrection_dir.exists():
        os.makedirs(subject_motioncorrection_dir)

    run_BOLD_files_raw = glob.glob(f'{subjectfolder}/func/*task-{task}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz') #get bold files
    for r, run_BOLD_file in enumerate(run_BOLD_files_raw): #For each BOLD file (run) get run_id (number)
            
        run_id = re.search(r'run-(\d+)', run_BOLD_file).group(0) # identify run numbers and add to run numbers  => entire pattern e.g. run-12 match.group(1) = 12
        print(run_id)

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

        # motion correction parameters: load, plot and save
        motion_corrs = confounds_data[0]
        motion_dict = {
            'rotations': ['rot_x', 'rot_y', 'rot_z'],
            'translations': ['trans_x', 'trans_y', 'trans_z']
        }

        fig, axs = plt.subplots(2,1,figsize=(100,20))
        for r, (label, motions) in enumerate(motion_dict.items()):
            ax = axs[r]
            for motion in motions:
                if label == 'rotations':
                    moco = np.rad2deg(motion_corrs[motion])
                else:
                    moco = motion_corrs[motion]
                ax.plot(motion_corrs.index, moco, label=motion.replace('_', ' '),linewidth=2, alpha=0.8 )
            ax.legend(title=label.capitalize())
            ax.set_title(label)
            # ax.set_xticks(motion_corrs.index[motion_corrs['run_start_index']])
            # ax.set_xticklabels(motion_corrs['run_id'].loc[motion_corrs['run_start_index']])
            ax.set_xlabel('images')
            ax.set_ylabel(['degrees °', 'mm'][r])
            ax.set_ylim([[-0.5, 1],[-0.5, 0.5]][r])
        
        fig.suptitle(f'{subject_label}_{run_id} - Motion Correction parameters')
        fig.savefig(f'{subject_motioncorrection_dir}/{subject_label}_{run_id}_motion_correction_parameters.png')
        # print(motion_corrs.columns)
        motion_corrs.to_csv(f'{subject_motioncorrection_dir}/{subject_label}_{run_id}_motion_correction_parameters.csv')
