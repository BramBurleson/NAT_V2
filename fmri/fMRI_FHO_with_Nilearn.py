import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from nilearn import image, plotting, datasets
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import make_first_level_design_matrix,  FirstLevelModel
from nilearn.glm import threshold_stats_img  



contrast_paths = pd.read_csv(r'k:\BramBurleson\02_Analysis\NAT_fMRI_Analysis_Comparisons\FHO_contrast_info.txt')
t_maps_to_plot = {}

subject_names = {
    "Sub_01_glm3_FHOegoallo_Cofnds17": "sub-03",
    "Sub_05_glm3_FHOegoallo_Cofnds17": "sub-04",
}

contrast_names = {
    "EgoFH_vs_ColorFH - All Sessions"   : "Ego-Color FH",
    "AlloFH_vs_ColorFH - All Sessions"  : "Allo-Color FH",
    "Allo_vs_Ego_FH - All Sessions"     : "Allo-Ego FH",
    "Ego_vs_Color_OTA - All Sessions"   : "Ego-Color OTA",
    "Allo_vs_Color_OTA - All Sessions"  : "Allo-Color OTA",
    "Allo_vs_Ego_OTA - All Sessions"    : "Allo-Ego OTA",
}

# for contrast in ucontrast_paths['t_map_files']:
for contrast_path in contrast_paths.iterrows():
    print(contrast_path)

    FHO_sub_name = contrast_path[1]['subject_names']
    NAT_sub_name = subject_names[FHO_sub_name]
    con_name = contrast_names[contrast_path[1]['con_int']]
        
    img = image.load_img(contrast_path[1]['t_map_files'])
    t_maps_to_plot[f'{NAT_sub_name} {con_name}'] = img

tvalue = 2.3291
results_dir = Path(r'K:\BramBurleson\03_Results\20241107_fMRI_FH_OTA_tval2.3291')
for i, contrast_id in enumerate(t_maps_to_plot):
    output_file     = results_dir / f'{contrast_id}_glm_t_value_{tvalue}_surface.png'
    plotting.plot_img_on_surf(
    t_maps_to_plot[contrast_id], 
    surf_mesh       = 'fsaverage',
    views           = ['lateral', 'medial', 'ventral'],
    hemispheres     = ['left', 'right'],
    colorbar        = True,
    threshold       = tvalue,
    darkness        = 1.0,
    title           = f'{contrast_id}_glm_t_value_{tvalue}',
    output_file     = output_file)

# z_map.to_filename(f'{names[s]}_glmconfounds_contrast{contrast_id}_z_map.nii.gz') #save niimg to file
# z_maps_to_plot[contrast_id] = z_map #save to dictionary
# img = image.load_img(run)  #Load the BOLD image
# img = image.smooth_img(img, fwhm=6)   #print(img.shape)
# z_maps_to_plot[contrast_id], 

