import pandas as pd
from pathlib import Path

ROOT_DATASET = Path(__file__).resolve().parents[3]

# confounds_tsv = \data\fmri\derivatives\sub-01\func

derivatives_folder = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
confounds_of_interest = ['global_signal', 'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05', 'a_comp_cor_06', 'a_comp_cor_07', 'a_comp_cor_08', 'a_comp_cor_09', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']



subjects = ["sub-01"]

for subject in subjects:
    print(subject)
    subject = Path(derivatives_folder, subject)


    confounds_tsvs = list(Path(subject,'func').glob(f'{subject.name}_*_timeseries.tsv'))

    for confounds_tsv in confounds_tsvs:
        print(confounds_tsv)
        all_confounds = pd.read_csv(confounds_tsv, sep='\t')
        selected_confounds = all_confounds[confounds_of_interest]
        print("saving selected confounds to CSV")
        selected_confounds.to_csv(confounds_tsv.with_name(f'{confounds_tsv.stem}_SELECTED.txt'), sep='\t', index=False, header=False)

