#REQUIRES nilearn v 0.10.4: use pip install nilearn==0.10.4
#   model: hrf => spm
#   regressors from fmri-prep confounds.tsv 
#       motion trans xyz, rot xyz
#       csf/white matter : a_comp_cor 00... 09
#       global signal
#   conditions : 'Explore', 'Key Ego', 'Key Allo', 'Key Color', 'Eagle Ego', 'Eagle Allo', 'Eagle Color'}


from pathlib import Path
import numpy as np
import re
from nilearn import plotting, datasets
import joblib

#SETUP
smoothing_parameter = 6
subject_labels = ['sub-01']
glm_folder = "run_level_glm"
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
for subject_label in subject_labels:
    print(subject_label)
    
    subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') #Get subject folders

    subjectfolder_glm = Path(subjectfolder, glm_folder)
    glm_files = subjectfolder_glm.glob(f'{subject_label}_*glm.pkl')
    for g, glm_file in enumerate(glm_files):
        pattern = r"run-(\d+)"
        run_id  = f"{re.search(pattern, glm_file.name).group()}"
        print(f"\t{run_id}")
        glm = joblib.load(glm_file)

        design_matrix = glm.design_matrices_[0]
        contrast_matrix = np.eye(design_matrix.shape[1])

        basic_effects = {
            column: contrast_matrix[i]
            for i, column in enumerate(design_matrix.columns)
            }

        effects = {
                "allo": basic_effects['allo_key'] + basic_effects['allo_eagle'],
                "ego": basic_effects['ego_key'] + basic_effects['ego_eagle'],
        }
    
        for effect_id, effect_val in effects.items():
            print(f"\t\t {effect_id}")

            beta_map = glm.compute_contrast(effect_val, output_type='effect_size')
            beta_output_file = f'{subjectfolder_glm}/{task_id}_{subject_label}_{run_id}_glm_{effect_id}_smooth_{smoothing_parameter}_beta_map.nii.gz'
            beta_map.to_filename(beta_output_file)

# %%
