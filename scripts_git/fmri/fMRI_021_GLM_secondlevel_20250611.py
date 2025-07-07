#one model for each contrast:
import os
from pathlib import Path
import pandas as pd
from scipy.stats import norm


from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import load_img
from nilearn.glm import threshold_stats_img
from nilearn import plotting



# SET UP
#1. select subjects to process by adding them to subject_labels list:
subject_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] 
#2. select behavioral task used either 'nat' (derivatives_folderexperiment) or 'fhof' (localizer) 
task = "nat" #"fhof"

#load FMRI_master parameters
ROOT_DATASET    = Path(__file__).resolve().parent.parent.parent
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
results_dir     = Path(ROOT_DATASET, 'results', 'fmri')
motioncorrection_dir = Path(ROOT_DATASET, 'data', 'motioncorrection')
 

#to include subject level confounds:
# subject_info = pd.DataFrame({"subject_label": subject_label,"age": range(15, 15 + n_subjects),"sex":([0, 1] * (n_subjects // 2)) + [0] * (n_subjects % 2),})
# print(subject_info)
# design_matrix = make_second_level_design_matrix(subject_label, subject_info)

#create design matrix 
design_matrix = pd.DataFrame([1] * len(subject_labels), columns=["intercept"])

# plot design matrix
plot_design_matrix(design_matrix)

#Run secondlevel GLM and plot for each contrast 1 ego-color
contrasts_to_model = ["Allo_Main", "Ego_Main", "Color_Main", "Ego-Color", "Allo-Color", "Allo-Ego"] 

main_effects     = ["Allo_Main", "Ego_Main", "Color_Main"]
contrast_effects = ["Ego-Color", "Allo-Color", "Allo-Ego"]
main_effects_pval     = 0.01
contrast_effects_pval = 0.05
print(f'Running Second-Level GLMs for the following contrasts: {contrasts_to_model} and subjects: {subject_labels}')
      
#change contrast vector to decide which contrast to run, contrast name must match filename contrast
#'/home/bramb/Bram/NAT_fMRI_Analysis',  f'{subject}_glmconfounds_contrast*****Ego-Color*****_z_map.nii.gz'

for contrast in contrasts_to_model:
    print(contrast)
    contrast_img = []
    for subject_label in subject_labels: #load images
        print(subject_label)
        subjectfolder = Path(derivatives_folder, subject_label)
        #Get contrast file
        contrast_file = list(Path(subjectfolder, 'glm').glob(f'*glm*{contrast}*z_map.nii*'))[0]


        print(contrast_file)
        contrast_img.append(load_img(contrast_file))  #Append the loaded image to sub_img

    glm2 = SecondLevelModel(n_jobs=2).fit(contrast_img, design_matrix=design_matrix)

    z_map = glm2.compute_contrast("intercept", output_type="z_score")
    # if the second-level model glm2_ec includes design_matrix with multiple columns 
    # use second_level_input_ i.e., the first argument of compute_contrast to specify the contrast(s) 

    #variations on thresholding;:
    #default FWER => very stringent because corrcts for all voxels
    #but if i threshold first then there are fewer voxels => FWER on remaining voxels
    #FDR is more or less division by number of identified clusters instead of number of voxels.
    
    #Thresholds 
    if contrast in main_effects:
        p_val = main_effects_pval
    elif contrast in contrast_effects:
        p_val = contrast_effects_pval

    p_uncorrected = norm.isf(p_val)
    print(f'threshold = {p_uncorrected}')

        
    # PLOT 3D SURFACES PIAL STATIC .PNG
    output_file     =  rf'{derivatives_folder}/secondlevel_glm_{contrast}_uncorrected_p={p_val}_surface.png'
    plotting.plot_img_on_surf(
        z_map, 
        surf_mesh       = 'fsaverage',
        views           = ['lateral', 'medial', 'ventral'],
        hemispheres     = ['left', 'right'],
        colorbar        = True,
        threshold       = p_uncorrected,
        darkness        = 1.0,
        title           = f'secondlevel_glm_{contrast}_uncorrected_p={p_val}',
        output_file     = output_file,
        )

        #threshold 1: restrict to pval
    thresholded_map1, threshold1 = threshold_stats_img(
    z_map, alpha=p_val height_control="fdr"
    )
    #threshold 2: run FWER
    thresholded_map2, threshold2 = threshold_stats_img(
    z_map, alpha=0.05, height_control="bonferroni"
    )
    print(f"The p<.05 Bonferroni-corrected threshold is {threshold2:.3g}")  

    # #Plot raw group-level contrast 
    # display = plotting.plot_stat_map(z_map, title="Raw z map")
    # # Corrections: 
    # # multiple comparisons correction 
    # thresholded_map1, threshold1 = threshold_stats_img(
    #     z_map,
    #     alpha=0.1,
    #     height_control="fpr",
    #     cluster_threshold=10,
    #     two_sided=True,
    # )
    # # More stringent thresholds : FDR and FWER (Bonferroni) corrections availabe here: https://nilearn.github.io/stable/auto_examples/05_glm_second_level/plot_thresholding.html  

    # # Second, the p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    # display = plotting.plot_stat_map(
    #     thresholded_map1,
    #     cut_coords=display.cut_coords,
    #     threshold=threshold1,
    #     title="Thresholded z map, fpr <.001, clusters > 10 voxels",
    # )