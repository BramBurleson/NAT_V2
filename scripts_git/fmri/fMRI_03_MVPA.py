

from nilearn import image
from nilearn.image import index_img
from nilearn.decoding import Decoder
import numpy as np

from pathlib import Path
import os 

# from nilearn import plotting
# from nilearn.image import mean_img
# # import numpy as np

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
run_for_subset = True
skip_existing_files = False

datafolder = Path(ROOT_DATASET, 'data', 'fmri_MVPA')

# for subject in subjects.....
    #plot raw bold, later you will need to transform it to matrix
subject = Path(datafolder, os.listdir(datafolder)[0])

condition_regexes = ['*Allo_Main*.nii.gz', '*Color_Main*.nii.gz*', '*Ego_Main*.nii.gz*']
condition_names = ['Allo', 'Color', 'Ego']
conditions_dict = dict(zip(condition_names, condition_regexes))

BOLD_list = []
conditions = []
for name, regex in conditions_dict.items():
    BOLD_files = list(subject.glob(regex))

    for BOLD_file in BOLD_files:

        #imgs = read () #read BOLD_files
        # BOLD_list.append(BOLD_file)
        img = image.load_img(BOLD_file)  #Load the BOLD image
        BOLD_list.append(img)
        conditions.append(name)

import random
imgs_conditions = list(zip(BOLD_list, conditions))
random.shuffle(imgs_conditions)
BOLD_imgs, conditions = zip(*imgs_conditions)

BOLD_imgs = image.concat_imgs(BOLD_imgs) 
conditions = np.array(conditions)

# #create Support Vector Machine Classifier ("svc") decoder object
decoder = Decoder( #presumably create new decoder for each subj
    estimator="svc", standardize="zscore_sample" 
)
# BOLD_imgs = image.concat_imgs(BOLD_list)
decoder.fit(BOLD_imgs, conditions)

prediction = decoder.predict(BOLD_imgs)
print(prediction) 
print((prediction == conditions).sum() / float(len(conditions)))


#%%
from sklearn.model_selection import KFold

cv = KFold(n_splits=4)

for fold, (train, test) in enumerate(cv.split(conditions), start=1):
    decoder = Decoder(
        estimator="svc", standardize="zscore_sample" # consider using mask later!!! estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    decoder.fit(index_img(BOLD_imgs, train), conditions[train])
    prediction = decoder.predict(index_img(BOLD_imgs, test))
    predicton_accuracy = (prediction == conditions[test]).sum() / float(
        len(conditions[test])
    )
    print(
        f"CV Fold {fold:01d} | "
        f"Prediction Accuracy: {predicton_accuracy:.3f}"
    )



# decoder.fit(BOLD_list, conditions)

# prediction = decoder.predict(BOLD_list)
# print(prediction)
    
# print((prediction == conditions).sum() / float(len(conditions)))
# %%
# This prediction accuracy score is meaningless. Why?
# Because it is predicting on exactly the same data it was fit on.

# fmri_filenames = [
# r"K:\BramBurleson\000_datasets_and_scripts\NAT_V1_fMRI_pilots\data\fmri\derivatives\sub-04\func\sub-04_task-nat_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
# # r"K:\BramBurleson\000_datasets_and_scripts\NAT_V1_fMRI_pilots\data\fmri\derivatives\sub-04\func\sub-04_task-nat_run-01_desc-coreg_boldref.nii.gz"
# ]


# #load data start with a single subject
# # skip plotting step 
# for fmri_filename in fmri_filenames:
#     # plotting.view_img(mean_img(fmri_filename), threshold=None)
