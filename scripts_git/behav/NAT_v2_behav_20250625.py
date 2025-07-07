from pathlib import Path
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats



def ci95(series):
    n = series.count()
    if n < 2:
        return np.nan
    sem = series.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n-1)  # two-tailed 95%
    return sem * t


def plot_condition_bars_with_subjects(ax, data, value_col='trial_performance', condition_col='trial_condition', subject_col='subject', title='', colors=None, ylimit = 100):

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    # Calculate means and SEMs
    means = data.groupby(condition_col)[value_col].mean()
    # sems = data.groupby(condition_col)[value_col].sem()
    cis = data.groupby(condition_col)[value_col].apply(ci95)
    # conditions = means.index.tolist()

    wildcard_order = ['allo', 'ego', 'color']
    conditions = [c for pattern in wildcard_order for c in means.index if pattern in c]

    # Bar locations
    bar_locs = np.arange(len(conditions))
    width = 0.6

    # Plot bars (without error bars)
    for i, cond in enumerate(conditions):
        ax.bar(
            bar_locs[i], means[cond],
            width=width, color=colors[i],
            edgecolor=None, zorder=1
        )

    # Plot error bars separately
    for i, cond in enumerate(conditions):
        ax.errorbar(
            bar_locs[i], means[cond], yerr=cis[cond],
            fmt='none', ecolor='black', capsize=5,
            zorder=2
        )

    # Plot subject lines
    for subject in data[subject_col].unique():
        subject_points = {'x': bar_locs, 'y': []}
        for cond in conditions:
            y_val = data[value_col][
                (data[condition_col] == cond) & (data[subject_col] == subject)
            ].drop_duplicates()
            subject_points['y'].append(y_val.values[0] if len(y_val) else np.nan)

        ax.plot(
            subject_points['x'], subject_points['y'],
            'o-', zorder=3
        )
        # # Scatter subject-level points
        # subj_vals = data[data[condition_col] == cond][[subject_col, value_col]].drop_duplicates()
        # jitter = 0.1 * (np.random.rand(len(subj_vals)) - 0.5)
        # ax.scatter(bar_locs[i] + jitter, subj_vals[value_col], color='black', s=10, zorder=10)

    # Tukey HSD
    tukey = pairwise_tukeyhsd(data[value_col], data[condition_col])
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # Add significance asterisks
    sig_thresholds = [(0.0001, '***'), (0.001, '**'), (0.01, '*')]
    spacing = 5
    sig_y = (means.max()) + spacing
    for _, row in tukey_df.iterrows():
        if row['reject']:
            cond1, cond2 = row['group1'], row['group2']
            i1, i2 = conditions.index(cond1), conditions.index(cond2)
            x1, x2 = bar_locs[i1], bar_locs[i2]
            y = sig_y
            ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y], lw=1.5, color='black')
            for pval, star in sig_thresholds:
                if row['p-adj'] < pval:
                    ax.text((x1 + x2)/2, y + 1.5, star, ha='center', va='bottom', fontsize=12)
                    break
            sig_y += spacing  # increment for stacked annotations

    ax.set_ylabel('%')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylim(0, ylimit)
    ax.text(0.5, -0.25, "* p < 0.01, ** p < 0.001, *** p < 0.0001 \n 95% CIs", ha='center', va='center', fontsize=8, transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
# SET UP

for task in ['single_stream', 'ff']:

    ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
    print(ROOT_DATASET)
    behav_folder = Path(ROOT_DATASET, 'data', 'behav')
    aggregate_level = ['subject']
    subjects = list(behav_folder.glob("sub*"))

    if task == 'single_stream':

        #load and concatenate all data into single df
        all_dfs = []
        for subject in subjects:
            print(subject.name)
            runs = list(Path(subject, 'single_stream').glob("run*"))
            for r, run in enumerate(runs):
                if r not in [5,6]:
                    print(run.name)
                    response_file = list(run.glob("flag_file_*.csv"))
                    all_dfs.append(pd.read_csv(response_file[0]))
        all_df = pd.concat(all_dfs)
        all_df['correct'] *= 100

        ylimits = [100, 1500]
        for b, behavior_metric in enumerate(['correct']):   
            aggregate = ['condition_name2'] + aggregate_level
            all_data = all_df.groupby(aggregate, as_index=False).agg({behavior_metric: 'mean'})

            data = all_data
            data = data[data['condition_name2']!='explore'].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_condition_bars_with_subjects(ax, data, value_col=behavior_metric, condition_col='condition_name2', title=f"{task} \n {behavior_metric}", colors = [(1, 0.7, 0.7), (0.7, 0.7, 0.7), (0.7, 0.7, 1)], ylimit = ylimits[b])
            fig.savefig(rf"{behav_folder}/condition_comparison_{task}_{behavior_metric}.png")

    
    elif task == 'ff':

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
            df['trial_RT_ms'] = df['trial_RT']*1000
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



        ylimits = [100, 1500]

        #split into flag and faces tasks:
        for b, behavior_metric in enumerate(['trial_performance', 'trial_RT_ms']):    #aggregate data by condition and aggregate level
            aggregate = ['trial_condition'] + aggregate_level 
            all_data = all_df.groupby(aggregate, as_index=False).agg({behavior_metric: 'mean'})

            for stimulus_type in ['flags', 'faces']:
                data = all_data[all_data['trial_condition'].str.contains(stimulus_type)]

                # data['col_id'] = df['column'].astype('category').cat.codes

                fig, ax = plt.subplots(figsize=(6, 6))
                plot_condition_bars_with_subjects(ax, data, value_col=behavior_metric, title=f"{task} \n {behavior_metric} {stimulus_type}", colors = [(1, 0.7, 0.7), (0.7, 0.7, 1), (0.7, 0.7, 0.7), ], ylimit = ylimits[b])
                fig.savefig(rf"{behav_folder}/condition_comparison_{task}_{stimulus_type}_{behavior_metric}.png")

