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
    conditions = means.index.tolist()
    n_conditions = len(conditions)

    # Bar locations
    bar_locs = np.arange(n_conditions)
    width = 0.6
    
    # Plot bars
    for i, cond in enumerate(conditions):
        ax.bar(bar_locs[i], means[cond], yerr=cis[cond], capsize=5, width=width, color=colors[i], edgecolor='black')

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


ROOT_DATASET = Path(__file__).resolve().parent.parent.parent
behav_folder = Path(ROOT_DATASET, 'data', 'behav')
subjects = list(behav_folder.glob(f"*sub*"))
aggregate_level = ['subject']

all_dfs = []
for subject in subjects:
    print(subject.name)
    ff_behav = list(Path(subject, 'ff').glob("*sub*corr*_ff*"))
    if len(ff_behav)<=0: #if no ff skip
        print(f"no ff file skipping {subject.name}")
        continue
    df = pd.read_csv(ff_behav[0])
    df.insert(3, 'correct', np.nan)
    df.insert(4, 'performance_calculated', np.nan)
    df.insert(0, 'subject', subject.name)
    df['trial_expected_resp'] = df['trial_expected_resp'].astype(int)
    df['trial_RT_ms'] = df['trial_RT']*1000
    df['correct'] = df['trial_resp_type'] == df['trial_expected_resp']
    # df['trial_resp_type'].fillna(0, inplace=True)
    for block in df['trial_block'].unique():
        tmp_df = df[df['trial_block'] == block]     
        performance_calculated = (sum(tmp_df['trial_resp_type'] == tmp_df['trial_expected_resp']) / len(tmp_df))*100
        df.loc[df['trial_block'] == block, 'performance_calculated'] = performance_calculated
        if  performance_calculated != tmp_df['trial_performance'].iloc[0]:
            print(f"{subject.name}_block_{block} performance calculated {performance_calculated} does not match psychtoolbox performance {tmp_df['trial_performance'].iloc[0]}")
            # print(tmp_df)          
    all_dfs.append(df)
all_df = pd.concat(all_dfs)

ylimits = [100, 1500]

#split into flag and faces tasks:
for b, behavior_metric in enumerate(['trial_performance', 'trial_RT_ms']):    #aggregate data by condition and aggregate level

    if behavior_metric == 'trial_RT_ms':
        all_df_copy = all_df[all_df['correct']]
    else:
        all_df_copy = all_df
    aggregate = ['trial_condition'] + aggregate_level 
    
    all_data = all_df_copy.groupby(aggregate, as_index=False).agg({behavior_metric: 'mean'})

    for stimulus_type in ['flags', 'faces']:
        data = all_data[all_data['trial_condition'].str.contains(stimulus_type)]

        # data['col_id'] = df['column'].astype('category').cat.codes

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_condition_bars_with_subjects(ax, data, value_col=behavior_metric, title=f" Condition Comparison \n {behavior_metric} {stimulus_type} ", colors = [(1, 0.7, 0.7), (0.7, 0.7, 0.7), (0.7, 0.7, 1)], ylimit = ylimits[b])
        fig.savefig(rf"{behav_folder}/condition_comparison_{behavior_metric} {stimulus_type}.png")