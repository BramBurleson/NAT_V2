

from nilearn import image
from nilearn.image import index_img
from nilearn.decoding import Decoder
import numpy as np
from pathlib import Path
import pandas as pd
import glob


ROOT_DATASET =  Path(__file__).resolve().parents[3]
print(ROOT_DATASET)

derivatives_folder = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
task = "nat"
# run_level_glm_folder = "run_level_glm"
mvpa_folder = "mvpa"

subject_labels = ['sub-01'] #, 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
run_labels = ['run-01', 'run-02', 'run-03', 'run-04', 'run-05', 'run-06', 'run-07']

train_runs = ['run-01', 'run-02', 'run-03', 'run-04', 'run-05', 'run-06']
test_runs = ['run-07']
effect_labels = ['allo', 'ego']
for subject_label in subject_labels:
    subject_folder = Path(derivatives_folder, subject_label)

    subjectfolder_func = Path(subject_folder, 'func')
    subjectfolder_events = Path(subject_folder, 'events')
    effects = []
    beta_maps = []
    runs = []
    for run_label in run_labels:
        smoothed_BOLD = subjectfolder_func.glob(f"{subject_label}*{task}_{run_label}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_smooth_6_mm.nii")[0]
        events = pd.read_csv(subjectfolder_events.glob(f"{subject_label}*{run_label}*")[0])
        
    df = pd.DataFrame({
        'run': runs,
        'effect': effects,
        'beta_map': beta_maps
    })

    df_train = df[df['run'].isin(train_runs)]
    df_test  = df[df['run'].isin(test_runs)]

    X_train = df_train['beta_map'].tolist()
    y_train = df_train['effect'].tolist()
    groups_train = df_train['run'].tolist()

    X_test = df_test['beta_map'].tolist()
    y_test = df_test['effect'].tolist()

    from sklearn.model_selection import LeaveOneGroupOut


    decoder = Decoder(
        estimator="svc",
        standardize=False,
        screening_percentile=5,
        cv=LeaveOneGroupOut(),
    )
    decoder.fit(X_train, y_train, groups=groups_train)

    # Cross-validated accuracy on the training folds (leave-one-run-out)
    cv_acc = np.mean(list(decoder.cv_scores_.values()))
    chance = 1.0 / df_train['effect'].nunique()

    # Held-out test accuracy on run-07
    y_pred = decoder.predict(X_test)
    test_acc = (np.array(y_pred) == np.array(y_test)).mean()

    print(
        f"CV accuracy (LOGO over train runs): {cv_acc:.4f} — Chance: {chance:.2f}\n"
        f"Held-out test accuracy (run-07):     {test_acc:.4f}"
    )