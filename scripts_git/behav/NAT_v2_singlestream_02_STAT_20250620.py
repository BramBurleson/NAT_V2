#prioritize and execute. simple models then more complicated.
#get a nice dataset and use that...
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

import matplotlib
print(matplotlib.get_backend())

#FOR CORRECT RESPONSES:
# TRIAL-LEVEL => BERNOUILLI DISTRIBUTION
# Proportion subject-level => BINOMIAL DISTRIBUTION => USE BINOMIAL LINK FUNCTION IN LMM

DoStat = True
RestrictToHits = False
# WHEN SPECIFYING AGGREGATE LEVEL SPECIFY HIGHER LEVELS AS WELL (I.E., IF 'block' also include 'subject', 'run')
# otherwise if aggregate only by block then we end up with only len(block.unique()) data points even if many runs/subjects
# aggregate_level = ['subject', 'run', 'block']
aggregate_level = ['subject']

# SET UP
ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
behav_folder = Path(ROOT_DATASET, 'data', 'behav')
subjects = list(behav_folder.glob("sub*"))

#load and concatenate all data into single df
all_data = []
for subject in subjects:
    print(subject.name)
    runs = list(Path(subject, 'single_stream').glob("run*"))
    for r, run in enumerate(runs):
        if r not in [5,6]:
            print(run.name)
            response_file = list(run.glob("flag_file_*.csv"))
            all_data.append(pd.read_csv(response_file[0]))
all_data = pd.concat(all_data)
response_types  = ['correct'] #, 'error'] error is superfluous because simply inverse of response type..

#aggregate data by condition and aggregate level
aggregate = ['condition_name2'] + aggregate_level

#separate for proportion and rts 
agg_proportions = {f"{col}": 'mean' for col in response_types}

#agg
data = all_data.groupby(aggregate, as_index=False).agg(agg_proportions)

data = data[data['condition_name2']!='explore'].reset_index(drop=True)

#%%Do stat if option ticked and data adequate
if len(data[aggregate_level])<3:
    print(f"too few data points!: {len(data[aggregate_level[0]].unique())} < 3 data point names = {data[aggregate_level[0]].unique()[0]}")
    DoStat=False
if(DoStat): #requires at least 3 unique subjects  !!!  
    #Alternative to repeated measures due to violation of normality assumption
    from scipy import stats as scipy_stats
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import statsmodels.api as sm
    from scipy.special import logit



    prop_results = []
    rt_results = []
    comp = ['allo_color', 'allo_ego', 'color_ego'] #this is entirely reliant on alphabetical order !!!

    #stat for proportions
    for prop_key in response_types:
        mydata = data[aggregate + [prop_key]].copy()

        # do normality 
        fig, axs = plt.subplots(1,3, figsize = (5,5))
        fig2, axs2 = plt.subplots(1,3, figsize=(18, 6))
        condition_datas= []
        for c, condition in enumerate(data['condition_name2'].unique()):
            condition_data = data[prop_key][data['condition_name2']==condition]
            
            condition_datas.append(condition_data)
            ax = axs[c]
    
            condition_data_percent = condition_data*100
            ax.scatter(np.arange(len(condition_data_percent))+1, condition_data_percent)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 110)
            # if c>0:
            #     ax.set_yticks([])
            #     ax.set_xticks([])

            cond_mean = np.mean(condition_data_percent)
            cond_sem = np.std(condition_data_percent)/len(condition_data_percent)

            # ax.scatter(5, cond_mean)
            ax.errorbar(5, cond_mean, yerr=cond_sem, fmt='none', ecolor='black', capsize=5, elinewidth=2, capthick=2, zorder=0)         
            ax.set_title(condition)

            condition_data_logit = logit(condition_data)  # Transform to (-inf, inf)
            ax2 = axs2[c]
            sm.qqplot(condition_data_logit, line='45', ax=ax2)
            ax2.set_title("Q-Q Plot of Proportions")

            shap_statistic, shap_pvalue = scipy_stats.shapiro(condition_data)

        #do homoscedasticity (homogeneity of variances)
      

        homo_statistic, homo_pvalue = scipy_stats.levene(condition_datas[0], condition_datas[1], condition_datas[2], center='median', proportiontocut=0.05, axis=0, nan_policy='propagate', keepdims=False)


       
        mydata[prop_key]
        if len(mydata[prop_key]) > 0:
            model_acc = smf.mixedlm(f"{prop_key} ~ condition_name2", mydata, groups=mydata[aggregate_level[0]])
            model_acc_fitted = model_acc.fit()
            print(model_acc_fitted.summary())
            tukey_result = pairwise_tukeyhsd(mydata[prop_key], mydata['condition_name2'], alpha=0.05)
            print(tukey_result)
            pvals = tukey_result.pvalues.tolist()
        else:
            pvals = np.full(len(comp),np.nan) #careful comp = ['allo_color', 'allo_ego', 'color_ego'] #this is entirely reliant on alphabetical order !!!
        prop_results.append(dict(zip(comp, pvals)))
    prop_results_df = pd.DataFrame(prop_results, index=response_types)

    #extract significant comparisons for plotting
    arr = prop_results_df
    significance_prop_results_df= pd.DataFrame()
    sig_conditions = [
        (prop_results_df< 0.0001),
        (prop_results_df< 0.001) & (prop_results_df>= 0.0001),
        (prop_results_df< 0.01) & (prop_results_df>= 0.001),
        ]
        # keep for testing purposes:
        # sig_conditions = [
        # (prop_results_df< 0.9),
        # (prop_results_df< 0.7) & (prop_results_df>= 0.6),
        # (prop_results_df< 0.6) & (prop_results_df>= 0.2),
        # ]
    categories = ['<0.0001', '<0.001', '<0.01']
    categories = ['***', '**', '*']
    categorized_arr = np.select(sig_conditions, categories, default='')
    significance_prop_results_df = pd.DataFrame(categorized_arr, index=arr.index, columns=arr.columns)
else:
      significance_prop_results_df = []

#PLOT_BAR
def plot_bars(ax, pointdata, response_types, conditions, significance_array, unit, colors):
    width = 0.2
    x = np.arange(len(response_types))
    tmp = np.arange(len(conditions))
    offsets = (tmp-len(tmp)//2) * width
    comparisons = list(combinations(offsets, 2))

    #sig array has following order : 'allo_color', 'allo_ego', 'color_ego' 
    #bars are allo, ego, color
    #if we want the bar heights to be: 0 : allo-color, 1: color_ego, and 2: allo_ego we need to rearrange tmp as follows:
    tmp = np.array([0, 2, 1])
    sig_heights = unit*1.15 - unit/20 + unit/20 * tmp

    for i, condition in enumerate(conditions):
        bar_locs = x + offsets[i]

        #bars 
        ax.bar(bar_locs, pointdata.loc[[condition]].mean(), width, label=condition, color=colors[i])   

        #errorbars #USE CI 95% INSTEAD
        ax.errorbar(bar_locs, pointdata.loc[[condition]].mean(), yerr=pointdata.loc[[condition]].sem(), fmt='none', ecolor='black', capsize=5, elinewidth=2, capthick=2, zorder=0)         

        #subjects
        points = pointdata.loc[[condition]].values    
        jitter = np.random.rand(len(points)) #jitter points: 
        for p, point in enumerate(points):
            ax.scatter(bar_locs -0.05 + 0.1*jitter[p], point, color='black', s=5, zorder=10)

        #asterixes  
        if len(significance_array)>0:
            sig_locs = np.mean(np.vstack([x,bar_locs]), axis=0)   #sig locs are central between bars
            sig_height = sig_heights[i]
    
            for r, _ in enumerate(response_types):
                if significance_array.iloc[r,i] != '':
                    ax.text(sig_locs[r], sig_height+unit/100, significance_array.iloc[r,i], ha='center', va='center', fontsize=12, color='black')     
                    ax.plot([x[r]+comparisons[i][0], x[r]+comparisons[i][1]], [sig_height, sig_height], marker='|', color='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(response_types)
        ax.set_yticks(np.arange(0,unit+unit/10,unit/10))
        ax.set_xlim(-0.5, len(response_types) - 0.5)
        ax.set_ylim(0, unit*1.25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncols=1, fontsize=10)

#%% Prep Plots:
fig_width = 6
fig_height = 6
conditions = data['condition_name2'].unique()
colors =   [(1, 0.7, 0.7), (0.7, 0.7, 0.7), (0.7, 0.7, 1)]

# Prep prop data
props = data[['condition_name2'] + response_types].set_index('condition_name2')*100

# Plot props
# fig, ax = plt.subplots(layout='constrained', figsize = (fig_width, fig_height))
fig, ax = plt.subplots(layout='constrained', figsize = (fig_width, fig_height))
plot_bars(ax, props, response_types, conditions, significance_prop_results_df, 100, colors)
ax.set_ylabel('%')
ax.set_title(f'Proportion \n level : {aggregate_level[-1]} \n restricted to hits = {RestrictToHits}')
ax.text(0.5, -0.1, "pairwise tukeys hsd (subjects pooled) * p < 0.01, ** p < 0.001, *** p < 0.0001", 
        ha='center', va='center', fontsize=8, transform=ax.transAxes)
plt.show()
fig.savefig(rf"{behav_folder}/single_stream_results/Accuracy_{aggregate_level[-1]}_level_stat_{DoStat}_hitsonly_{RestrictToHits}.png")