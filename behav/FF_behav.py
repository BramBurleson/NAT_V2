from pathlib import Path
import os
import numpy as np 
import pandas as pd


ROOT_DATASET = Path(__file__).resolve().parent.parent.parent
behav_folder = Path(ROOT_DATASET, 'data', 'behav')

subjects = list(behav_folder.glob(f"*sub*"))

all_dfs = []
for subject in subjects:
    print(subject.name)
    ff_behav = list(Path(subject, 'ff').glob("*sub*_ff*"))
    if len(ff_behav)<=0: #if no ff skip
        print(f"no ff file skipping {subject.name}")
        continue
    df = pd.read_csv(ff_behav[0])
    df.insert(3, 'performance_calculated', np.nan)
    df.insert(0, 'subject', subject.name)
    df['trial_expected_resp'] = df['trial_expected_resp'].astype(int)
    # df['trial_resp_type'].fillna(0, inplace=True)
    for block in df['trial_block'].unique():

        tmp_df = df[df['trial_block'] == block]
        
        
        performance_calculated = (sum(tmp_df['trial_resp_type'] == tmp_df['trial_expected_resp']) / len(tmp_df)) *100

        df.loc[df['trial_block'] == block, 'performance_calculated'] = performance_calculated

        print(block, performance_calculated, tmp_df['trial_performance'].iloc[0])

        if  performance_calculated != tmp_df['trial_performance'].iloc[0]:
            print(tmp_df)
    all_dfs.append(df)


all_df = pd.concat(all_dfs)
















