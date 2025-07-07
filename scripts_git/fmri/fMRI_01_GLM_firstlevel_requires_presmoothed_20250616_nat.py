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

#SAVING IS WRONG

# SET UP
#1. select subjects to process by adding them to subject_labels list:
subject_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] 
#2. select behavioral task used either 'nat' (derivatives_folderexperiment) or 'fhof' (localizer) 

task = "nat" #"fhof"
smoothing_parameter = 6 #mm

main_effects            = ["Allo_Main", "Ego_Main", "Color_Main", "Instructions"]
main_effects_pval       = 0.01
main_effects_zscore = norm.isf(main_effects_pval)
print(f'main_effects_pval = {main_effects_pval}, threshold: main_effects_zscore = {main_effects_zscore}')

contrast_effects        = ["Ego-Color", "Allo-Color", "Allo-Ego"]
contrast_effects_pval   = 0.1
contrast_effects_zscore = norm.isf(contrast_effects_pval)
print(f'contrast_effects_pval = {contrast_effects_pval}, threshold: contrast_effects_zscore = {contrast_effects_zscore}')

#load FMRI_master parameters
ROOT_DATASET    = Path(__file__).resolve().parent.parent.parent
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
results_dir     = Path(ROOT_DATASET, 'results', 'fmri')
motioncorrection_dir = Path(ROOT_DATASET, 'data', 'motioncorrection')

#set model Parameters:
t_r                     = 1.5 #repetition time, see e.g. derivatives/sub-01/func/sub-01_task-nat_run-01_space-T1w_desc-preproc_bold.json
slice_time_ref          = 0.0 #slice_time_ref ?
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
    run_BOLD_files_raw = glob.glob(f'{subjectfolder}/func/*task-{task}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz') #get bold files
    run_BOLD_files_smooth = glob.glob(f'{subjectfolder}/func/*task-{task}*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold*{smoothing_parameter}*.nii')     

    for r, run_BOLD_file in enumerate(run_BOLD_files_raw): #For each BOLD file (run) get run_id (number)
        if task == 'nat':
            run_id = re.search(r'run-(\d+)', run_BOLD_file).group(0) # identify run numbers and add to run numbers  => entire pattern e.g. run-12 match.group(1) = 12
        elif task == 'fhof':
            run_id = '00'

        print(run_id)
        subj_run_ids.append(run_id) #if event file exists add to subject run list

        #Get events files
        if task == 'nat':
            run_events_file = Path(subjectfolder,'events', f'{subject_label}_{run_id}_events.tsv')
        elif task == 'fhof':
            run_events_file = Path(subjectfolder,'events', f'*BLOCK*.tsv')
        events = pd.read_table(run_events_file) #load events

        #Load BOLD data
        print(f'BOLD FILE : {run_BOLD_file} and {run_BOLD_files_smooth[r]}')
        img = image.load_img(run_BOLD_files_smooth[r])  #Load the BOLD image
        subj_img.append(img)  #Append the smoothed image to sub_img
          
        # prep for first-level design matrix
        n_scans = img.shape[-1] 
        frame_times = np.arange(n_scans) * t_r #print(frame_times)
      
        # Get confounds(noise regressors)
        # more info https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds
        # all options:(motion='basic', scrub=5, fd_threshold=0.2, std_dvars_threshold=3, wm_csf='basic', global_signal='basic', compcor='anat_combined', n_compcor=10)
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
        #create first-level design matrix with confounds
        design_matrix_confounds = make_first_level_design_matrix(
            frame_times,
            events,
            hrf_model   = hrf_model,
            drift_model = None, #default : "Drift" discrete cosine transform to model signal drifts, despite this because of using highpass in confounds there will be drift model.
            high_pass   = None, #default: 0.01Hz cutoff for the drift model, see above
            add_regs    = confounds,
        )
        
        # Plot and save subject design matrices
        print("Plotting design_matrices and saving plots")
        output_file = f'{subjectfolder}/glm/{subject_label}_{run_id}_design_matrix.png'
        plotting.plot_design_matrix(
            design_matrix_confounds,
            output_file     = output_file,     
        ) 
        design_matrix_confounds.to_csv(f'{subjectfolder}/glm/{subject_label}_{run_id}_design_matrix.csv', index=False)
        subj_design_matrices_confounds.append(design_matrix_confounds) #append design matrix to subject design matrices

    subj_data = {
        'run_ids':subj_run_ids,
        'imgs':subj_img,
        'design_matrices':subj_design_matrices_confounds,
        'confounds': subj_confounds,
    }
    data_dict[subject_label] = subj_data

#prep data for GLM: different for super subject (fixed effects only) or individual subjects (fixed and random effects)
data_for_glm            = {}
keys_for_glm            = []
names                   = []       

#individual subject
data_for_glm            = data_dict
keys_for_glm            = subject_labels
names                   = subject_labels

#%% fit first level glms and plot results for each subject (or supersubject)
for subject_label in subject_labels:
    print(subject_label) 
    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') 
    subjectfolder_glm = Path(subjectfolder, 'glm')

    #Fit GLM  with noise regressors "confounds from fMRIPrep."
    print("Fitting a GLM with noise regressors")
    glm = FirstLevelModel()
    glm = glm.fit(data_for_glm[subject_label]['imgs'], design_matrices=data_for_glm[subject_label]['design_matrices'])

    columns = glm.design_matrices_[0].columns #assumes that design matrices for all runs are identical to first run of each sub
    effect_matrix = np.eye(len(columns))
    effect_contrasts = {
        column: effect_matrix[i] for i, column in enumerate(columns)
    }

    if task == "nat":
        effect_dict = {
            "Allo_Main":   effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"],
            "Ego_Main":    effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"],
            "Color_Main":  effect_contrasts["color_key"] + effect_contrasts["color_eagle"],
            # "Allo-Explore":   effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["explore"],
            # "Ego-Explore":    effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"] - effect_contrasts["explore"],
            # "Color-Explore":  effect_contrasts["color_key"] + effect_contrasts["color_eagle"] - effect_contrasts["explore"],
            "Allo-Color":  effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["color_key"] - effect_contrasts["color_eagle"],
            "Ego-Color":   effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"] - effect_contrasts["color_key"] - effect_contrasts["color_eagle"],
            "Allo-Ego":    effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["ego_key"] - effect_contrasts["ego_eagle"],
        }
    elif task == "ff":
          effect_dict = {
              "Color Flags",                  effect_contrasts["Color Flags"],
              "Ego Flags",                    effect_contrasts["Ego Flags"],
              "Allo Flags",                   effect_contrasts["Allo Flags"],
              "Color Faces	",                effect_contrasts["Color Faces	"],
              "Ego Faces",                    effect_contrasts["Ego Faces"],
              "Allo Faces",                   effect_contrasts["Allo Faces"],
              
            # "Allo_Main":   effect_contrasts["allo_face"] + effect_contrasts["allo_fglag"],
            # "Ego_Main":    effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"],
            # "Color_Main":  effect_contrasts["color_key"] + effect_contrasts["color_eagle"],
            # # "Allo-Explore":   effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["explore"],
            # # "Ego-Explore":    effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"] - effect_contrasts["explore"],
            # # "Color-Explore":  effect_contrasts["color_key"] + effect_contrasts["color_eagle"] - effect_contrasts["explore"],
            # "Allo-Color":  effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["color_key"] - effect_contrasts["color_eagle"],
            # "Ego-Color":   effect_contrasts["ego_key"] + effect_contrasts["ego_eagle"] - effect_contrasts["color_key"] - effect_contrasts["color_eagle"],
            # "Allo-Ego":    effect_contrasts["allo_key"] + effect_contrasts["allo_eagle"] - effect_contrasts["ego_key"] - effect_contrasts["ego_eagle"],
        }
          
    # for dm in glm.design_matrices_:
    #     print(len(dm.columns))


    #Compute effects and save to file and save to dictionary
    print("Computing and saving effects")
    z_maps_to_plot = {}

    for effect_id, effect_val in effect_dict.items():
        print(f"\teffect id: {effect_id}")

        z_map = glm.compute_contrast(effect_val, output_type="z_score") # compute the effects
        
        output_file = f'{subjectfolder_glm}/{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_z_map.nii.gz'
        z_map.to_filename(output_file) #save niimg to file ULTIMATELEY SAVE SOMEWHERE USEFUL MAYBE A effectS FOLDER?
        z_maps_to_plot[effect_id] = z_map #save to dictionary

    #Thresholds 
    for i, effect_id in enumerate(z_maps_to_plot): #for each key in z_maps_to_plot
        print(i)
        print(effect_id)
        if effect_id in main_effects:
            pval = main_effects_pval   
            zscore_threshold = main_effects_zscore           
          
        elif effect_id in contrast_effects:   
            pval = contrast_effects_pval 
            zscore_threshold = contrast_effects_zscore
         
        # PLOT 3D SURFACES PIAL STATIC .PNG
        output_file     =  rf'{subjectfolder_glm}/{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_uncorrected_p={pval}_surface.png'
        plotting.plot_img_on_surf(
            z_maps_to_plot[effect_id], 
            surf_mesh       = 'fsaverage',
            views           = ['lateral'], #'medial', 'ventral'],
            hemispheres     = ['left', 'right'],
            colorbar        = True,
            threshold       = zscore_threshold,
            darkness        = 1.0,
            title           = f'{subject_label}_glm_{effect_id}_uncorrected_p={pval}',
            output_file     = output_file,
        )