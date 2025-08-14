

# from nilearn.masking import intersect_masks
# from nilearn.image import load_img, index_img, concat_imgs
# from nilearn.decoding import Decoder

from pathlib import Path
import joblib
from nilearn.plotting import view_img #plot_epi, plot_stat_map, 
from nilearn import datasets

mni = datasets.load_mni152_template()


ROOT_DATASET =  Path(__file__).resolve().parents[3]
print(ROOT_DATASET)

derivatives_folder = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
task = "nat"
mvpa_folder = "mvpa"

subject_labels = ['sub-01'] #, 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
condition_labels = ['allo', 'ego']

for subject_label in subject_labels:
    print(subject_label)
    subject_folder = Path(derivatives_folder, subject_label)
    subjectfolder_mvpa = Path(subject_folder, 'mvpa')

    decoder = joblib.load(f"{subjectfolder_mvpa}/20250813_decoder_{subject_label}_allo_v_ego_intersect_mask.joblib")

    for key in condition_labels:
        coef_img = decoder.coef_img_[key]

        v = view_img(
            coef_img,
            bg_img=mni,
            title=f"{key} svc weights",
            dim=-1,
        )

        v.open_in_browser()  # opens your default browser