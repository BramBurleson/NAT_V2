#REQUIRES nilearn v 0.10.4: use pip install nilearn==0.10.4
#   model: hrf => spm
#   regressors from fmri-prep confounds.tsv 
#       motion trans xyz, rot xyz
#       csf/white matter : a_comp_cor 00... 09
#       global signal
#   conditions : 'Explore', 'Key Ego', 'Key Allo', 'Key Color', 'Eagle Ego', 'Eagle Allo', 'Eagle Color'}


from pathlib import Path
import numpy as np
from nilearn import plotting, datasets

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
for subject_label in subject_labels:
    print(subject_label)
    
    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders

    subjectfolder_glm = Path(subjectfolder, glm_folder)
    glm_file = subjectfolder_glm/ f'{subtask}_{subject_label}_firstlevelmodel.pkl'
    glm = joblib.load(glm_file)

    design_matrix = glm.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])

    basic_effects = {
    column: contrast_matrix[i]
    for i, column in enumerate(design_matrix.columns)
    }

    effects = {
               "Allo-Ego_OTA": basic_effects['Allo_OTA'] - basic_effects['Ego_OTA'],
                "Ego-Color_Face": basic_effects['Ego_Face'] - basic_effects['Color_Face'],
    }

    z_maps_to_plot = {}
    for effect_id, effect_val in effects.items():
        print(f"\teffect id: {effect_id}")
        effect_val = np.array(effect_val)  # don't normalize!
        z_map = glm.compute_contrast(effect_val, output_type="z_score") # compute the effects   
        z_output_file = f'{subjectfolder_glm}/{subtask}_{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_z_map_20250728.nii.gz'
        z_map.to_filename(z_output_file)

        t_map = glm.compute_contrast(effect_val, stat_type='t', output_type='stat')
        t_output_file = f'{subjectfolder_glm}/{subtask}_{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_T_map_20250728.nii.gz'
        t_map.to_filename(t_output_file)

        beta_map = glm.compute_contrast(effect_val, output_type='effect_size')
        beta_output_file = f'{subjectfolder_glm}/{subtask}_{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_beta_map_20250728.nii.gz'
        beta_map.to_filename(beta_output_file)

        # PLOT 3D SURFACES PIAL STATIC .PNG
        output_file     =  rf'{subjectfolder_glm}/{subtask}_{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_UNTHRESHOLDED_surface_zmap_20250724.png'
        plotting.plot_img_on_surf(
            z_map, 
            surf_mesh       = 'fsaverage',
            views           = ['lateral'], #'medial', 'ventral',
            hemispheres     = ['left', 'right'],
            colorbar        = True,
            threshold       = None,
            darkness        = 1.0,
            title           = f'{subtask}_{subject_label}_glm_{effect_id}_unthresholded',
            output_file     = output_file,
        )

        
        # PLOT 3D SURFACES PIAL STATIC .PNG
        output_file     =  rf'{subjectfolder_glm}/{subtask}_{subject_label}_glm_{effect_id}_smooth_{smoothing_parameter}_UNTHRESHOLDED_surface_tmap_20250724.png'
        plotting.plot_img_on_surf(
            z_map, 
            surf_mesh       = 'fsaverage',
            views           = ['lateral'], #'medial', 'ventral',
            hemispheres     = ['left', 'right'],
            colorbar        = True,
            threshold       = None,
            darkness        = 1.0,
            title           = f'{subtask}_{subject_label}_glm_{effect_id}_unthresholded',
            output_file     = output_file,
        )

# %%

from nilearn.datasets import load_mni152_template

report = glm.generate_report(
    effects,
    threshold=3.1, 
    bg_img=load_mni152_template(),
    height_control=None,
)
report.save_as_html('report.html')
# %%
