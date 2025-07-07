# %%
from nilearn import image, plotting

import numpy as np

from nilearn.datasets import load_sample_motor_activation_image

stat_img = load_sample_motor_activation_image()

# Load a template (e.g., fsaverage) image to match the surface space
# template_img = r'K:\BramBurleson\02_Analysis\NAT_fMRI_Analysis\sub-05_glmconfounds_contrastEgo_Main_z_map.nii.gz'  # Replace with a relevant template image
# template = image.load_img(template_img)

# Create a blank image with the same shape as the template
# blank_data = np.ones_like(template.shape)
# blank_img = image.new_img_like(template, blank_data)

#%% Now plot the surface with the blank image
plotting.plot_img_on_surf(
    stat_img,  # Use the blank image here
    surf_mesh='fsaverage',
    views=['lateral', 'medial', 'ventral'],
    hemispheres=['left', 'right'],
    colorbar=True,
    threshold=10.0,  # Adjust the threshold if needed
    darkness=1.0,
    title='Blank Pial Brain Surface',
    output_file='blank_pial_surface.png'
)
plotting.show()
# %%
