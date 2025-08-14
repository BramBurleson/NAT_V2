

from nilearn.masking import intersect_masks
from nilearn.image import load_img, index_img, concat_imgs
from nilearn.decoding import Decoder
from nilearn.plotting import plot_epi
import joblib
from sklearn.model_selection import KFold



import numpy as np
from pathlib import Path
import pandas as pd
import glob
import os


ROOT_DATASET =  Path(__file__).resolve().parents[3]
print(ROOT_DATASET)

derivatives_folder = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
task = "nat"
# run_level_glm_folder = "run_level_glm"
mvpa_folder = "mvpa"



tr = 1.5
condition_map = { "allo_key": "allo",  "allo_eagle": "allo",  "ego_key": "ego",  "ego_eagle": "ego"}
subject_labels = ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
run_labels = ['run-01', 'run-02', 'run-03', 'run-04', 'run-05', 'run-06', 'run-07']

train_runs = ['run-01', 'run-02', 'run-03', 'run-04', 'run-05', 'run-06']
test_runs = ['run-07']

for subject_label in subject_labels:
    print(subject_label)
    subject_folder = Path(derivatives_folder, subject_label)

    subjectfolder_func = Path(subject_folder, 'func')
    subjectfolder_events = Path(subject_folder, 'events')
    subjectfolder_mvpa = Path(subject_folder, 'mvpa')
    os.makedirs(subjectfolder_mvpa, exist_ok=True)

    train_niimgs = []
    train_labels = []
    test_niimgs = []
    test_labels = []
    
    mask_imgs = []
    for run_label in run_labels:
        print(f"\t{run_label}")
        BOLD_paths = list(subjectfolder_func.glob(f"{subject_label}*{task}_{run_label}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth_6_mm.nii"))
        BOLD = load_img(BOLD_paths[0])

        mask_paths = list(subjectfolder_func.glob(f"{subject_label}*{task}_{run_label}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii*"))
        mask_img = load_img(mask_paths[0])
        mask_imgs.append(mask_img)
        # mean_smoothed_BOLD = image.mean_img(smoothed_BOLD_path)
        # plot_epi(mean_smoothed_BOLD)
        event_paths = list(subjectfolder_events.glob(f"{subject_label}*{run_label}*.tsv"))
        events = pd.read_csv(event_paths[0], sep="\t")      
   
        start = 0
        counts = (events['duration'] / tr).round().astype(int).to_numpy()

        volumes_df = pd.DataFrame({
            'volume': start + np.arange(counts.sum()),
            'trial_type': np.repeat(events['trial_type'].to_numpy(), counts)
        })
        volumes_df = volumes_df[:-1] #remove last volume from volumes_df (because last event finish block has a duration of 0)
       
        min_length = np.min([BOLD.shape[3], len(volumes_df)])
        
        BOLD = index_img(BOLD, range(min_length))
        volumes_df = volumes_df[0:min_length]


        volumes_df["condition"] = volumes_df["trial_type"].map(condition_map).fillna("")  # select allo and ego only and collapse key and eagle
        
        volumes_mask = volumes_df["condition"].isin(["allo", "ego"]).to_numpy()

        labels = volumes_df.loc[volumes_mask, "condition"].to_numpy()
        niimgs = index_img(BOLD, volumes_mask)

        if run_label in train_runs:
            train_niimgs.append(niimgs)
            train_labels.append(labels)
        elif run_label in test_runs:
            test_niimgs.append(niimgs)
            test_labels.append(labels)


    subject_mask = intersect_masks(mask_imgs, threshold=1.0)


    decoder = Decoder(
        estimator="svc", mask=subject_mask, standardize="zscore_sample",
    #     screening_percentile=5,
    #     cv=LeaveOneGroupOut(),  # from sklearn.model_selection import LeaveOneGroupOut
    )

    X_train = concat_imgs(train_niimgs)
    y_train = np.concat(train_labels)

    X_test = concat_imgs(test_niimgs)
    y_test = np.concat(test_labels)
    decoder.fit(X_train, y_train) #, groups=groups_train)

    # Cross-validated accuracy on the training folds (leave-one-run-out)
    cv_acc = np.mean(list(decoder.cv_scores_.values()))
    chance = 1.0 / len(np.unique(np.concatenate(train_labels)))

    # Held-out test accuracy on run-07
    y_pred = decoder.predict(X_test)
    test_acc = (np.array(y_pred) == np.array(y_test)).mean()

    print(
        f"CV accuracy (LOGO over train runs): {cv_acc:.4f} — Chance: {chance:.2f}\n"
        f"Held-out test accuracy (run-07):     {test_acc:.4f}"
    )



    # save
    joblib.dump(decoder, f"{subjectfolder_mvpa}/20250813_decoder_{subject_label}_allo_v_ego_intersect_mask.joblib")

