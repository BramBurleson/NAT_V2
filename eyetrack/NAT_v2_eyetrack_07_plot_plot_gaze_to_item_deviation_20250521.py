import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

#important variables:
# flash_side = {'left':0, 'right':1}

eyes = {'left':['X_Gaze', 'Y_Gaze'], 'right':['X_Gaze.1', 'Y_Gaze.1']}
corrected_eyes = {'left':['X_CorrectedGaze', 'Y_CorrectedGaze'], 'right':['X_CorrectedGaze.1', 'Y_CorrectedGaze.1']}
#to toggle if using raw or Viewpoint Eyetracker's corrected gaze points comment/uncomment below:
# eyes = corrected_eyes

#plot parameters
#font sizes:
small, medium, large = 14, 16, 32
plt.rc('font', size=medium)
plt.rcParams.update({
    'axes.titlesize': medium, #'axes.titleweight': 'bold',
    'axes.labelsize': medium, 'xtick.labelsize': medium,
    'ytick.labelsize': medium, 'legend.fontsize': medium,
    'figure.titlesize': large, #'figure.titleweight': 'bold'
})

#colors:
magenta = (1, 0, 1)  # RGB for magenta
teal = (0, 1, 1)     # RGB for teal
darkgray = (0.9,0.9,0.9)
darkergray = (0.7,0.7,0.7)
darkmagenta = (0.5, 0, 0.5)  # RGB for dark magenta
darkteal = (0, 0.5, 0.5)     # RGB for teal
gaze_color = (1,0,0)
colors = [darkmagenta, darkteal]
colors2 = [magenta, teal]
dotsize = 10 

#time window parameters
min_time = -300 #ms before/after press
max_time = 100    #ms before/after press

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
data_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = data_folder.glob("sub*")

def get_value(df, row, pattern):
    val = df.loc[row, df.columns.str.contains(pattern)].values
    return val[0] if len(val) else np.nan  # or raise an error

all_presses_data = []
# Main loop
for s, subject in enumerate(subjects):
    # if s==0:
        print(subject)
        # runs = [run for run in subject.iterdir() if run.is_dir()]
        runs = list(Path(subject, 'single_stream').glob("run_*"))
        for r, run in enumerate(runs):  
            # if r==0:
                print(run)
                flag_gaze_press_plots_folder = Path(rf"{run}/flag_gaze_press_plots")
                if not os.path.exists(flag_gaze_press_plots_folder):
                    os.mkdir(flag_gaze_press_plots_folder)

                data_file = list(run.glob("*eye_and_pos*"))[0]
                data = pd.read_csv(data_file)

                # #select data after flashes
                # press_indices = data['response']#########extract indices of responses      
                press_indices   = data.index[data['Events'] == 'KeyPress']
                mean_deltatime  = np.mean(data['DeltaTime'][data['5']!=12].astype(float))  # was 16.68624364606602

                timewindow_start_index  = np.round(min_time / mean_deltatime).astype(int)
                timewindow_end_index    = np.round(max_time / mean_deltatime).astype(int)     

                data_timewindow_start_indices = press_indices + timewindow_start_index    
                data_timewindow_end_indices   = press_indices + timewindow_end_index

                conditions = data['condition_name1'].iloc[press_indices].values
                conditions2 = data['condition_name2'].iloc[press_indices].values

                flag_names = data['flag_name'].iloc[press_indices].values
                flag_configs = data['flag_config'].iloc[press_indices].values    

                blocks = data['block'].iloc[press_indices].values

                target_item_position_x = []
                target_item_position_y = []         

                for p, (press_index, condition, condition2, flag_name, flag_config, timewindow_start_ind, timewindow_end_ind, block) in enumerate(list(zip(press_indices, conditions, conditions2, flag_names, flag_configs, data_timewindow_start_indices, data_timewindow_end_indices, blocks))):  
                    # times = np.arange(timewindow_start_ind, timewindow_end_ind) - timewindow_start_ind) * mean_deltatime 

                    press_df = data.iloc[timewindow_start_ind:timewindow_end_ind]
                    press_df = press_df.reset_index(drop=True)

                    if not flag_name.startswith('no_match'): 
                        if "key" in condition: # if key
                            if "OKE" in flag_config:
                                x = get_value(press_df, r, f'{flag_name}_LeftPosX$')
                                y = get_value(press_df, r, f'{flag_name}_LeftPosY$')
                            elif "OEK" in flag_config:
                                x = get_value(press_df, r, f'{flag_name}_RightPosX$')
                                y = get_value(press_df, r, f'{flag_name}_RightPosY$')

                        if "eagle" in condition: # if eagle
                            if "OKE" in flag_config:
                                x = get_value(press_df, r, f'{flag_name}_LeftPosX$')
                                y = get_value(press_df, r, f'{flag_name}_LeftPosY$')
                            elif "OEK" in flag_config:
                                x = get_value(press_df, r, f'{flag_name}_RightPosX$')
                                y = get_value(press_df, r, f'{flag_name}_RightPosY$')

                        # press_df[eyes[eye][0]] *= 1920 #transform to pixel coordinates
                        # press_df[eyes[eye][1]] *= 1080 #transform to pixel coordinates
                        # target_item_position_x.append(x)
                        # target_item_position_y.append(y)
                        press_df['target_item_position_x'] = x
                        press_df['target_item_position_y'] = y
                        press_df['press_id'] = f"{s}_{r}_{p}"
                        press_df['press_times'] = (press_df['Time'] - press_df['Time'].iloc[0]) * 1000 + min_time

                        press_df['unique_block'] = f"{run.name}_{block}"
                        all_presses_data.append(press_df)
data = pd.concat(all_presses_data).reset_index(drop=True)
tata = pd.concat(all_presses_data).reset_index(drop=True)

#%% plot for each eye all gaze points (grey) and mean/std 
press_df_eyes = []
for e, eye in enumerate(eyes):    
    press_dfs =  []
    for press_id in data['press_id'].unique():
        press_df = data[data['press_id']==press_id]
        x_target = press_df['target_item_position_x']
        y_target = press_df['target_item_position_y']
        
        press_df['x_dist_target'] = press_df[eyes[eye][0]]*1920 - x_target   
        press_df['y_dist_target'] = press_df[eyes[eye][1]]*1080 - y_target 
        ed_gaze_target = np.sqrt(press_df['x_dist_target']**2 + press_df['y_dist_target']**2)
        press_df['ed_gaze_target'] = ed_gaze_target

        press_df['x_dist_center'] = press_df[eyes[eye][0]]*1920 - 1920/2
        press_df['y_dist_center'] = press_df[eyes[eye][1]]*1080 - 1080/2
        press_df['ed_gaze_center']  = np.sqrt(press_df['x_dist_center']**2 + press_df['y_dist_center']**2)

        press_dfs.append(press_df)
    press_df_eyes.append(pd.concat(press_dfs))


#%% Create model that includes conditions and blocks as factors.
# use statsmodels.glm ? or something maybe you need to first define glm and then fit it to your data
data = press_df_eyes[0]
mdl_data = data[['ed_gaze_target', 'condition', 'unique_block', 'subject_eye']].dropna().reset_index(drop = True)
mdl_data['collapsed_condition'] = ""
mdl_data['collapsed_condition'].loc[mdl_data['condition'].isin([1,4])] = 'ego'
mdl_data['collapsed_condition'].loc[mdl_data['condition'].isin([2,5])] = 'allo'
mdl_data['collapsed_condition'].loc[mdl_data['condition'].isin([3,6])] = 'color'
mdl_data['collapsed_condition'].loc[mdl_data['condition'].isin([0])]   = 'aa_explore'

for col in ['condition', 'collapsed_condition', 'unique_block', 'subject_eye']:
    mdl_data[col] = mdl_data[col].astype('category')
from statsmodels.regression.mixed_linear_model import MixedLM

model = MixedLM.from_formula(
     'ed_gaze_target ~ collapsed_condition',
      groups = 'subject_eye', 
      vc_formula={'unique_block': '0 + C(unique_block)'},
      data=mdl_data)


result = model.fit()
result.summary()
# #%%
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# tukey_result = pairwise_tukeyhsd(mdl_data['ed_gaze_target'], mdl_data['condition'], alpha=0.05)
# from itertools import combinations
#%%
from itertools import combinations
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

levels   = mdl_data['collapsed_condition'].cat.categories
rows     = []

for a, b in combinations(levels, 2):
    # build constraint string once
    # (reference level is implicit, so if a==baseline its β is 0)
    if a == levels[0]:
        constr_str = f'C(collapsed_condition)[T.{b}] = 0'
    elif b == levels[0]:
        constr_str = f'C(collapsed_condition)[T.{a}] = 0'
    else:
        constr_str = f'C(collapsed_condition)[T.{b}] = C(collapsed_condition)[T.{a}]'

    # Patsy turns the string into (L, q)
    LC       = result.model.data.design_info.linear_constraint(constr_str)
    test     = result.wald_test(LC)        # identical to t_test for 1-df
    est, se  = test.effect.item(), np.sqrt(test.cov.item())
    rows.append([a, b, est, se,
                 test.statistic.item(), test.pvalue.item()])

pairwise = (pd.DataFrame(rows,
                         columns=['A','B','diff','SE','z','p_raw'])
            .assign(p_adj=lambda d:
                    multipletests(d.p_raw, method='holm')[1])
            .sort_values('p_adj'))
print(pairwise)





# import statsmodels.api as sm

# import statsmodels.formula.api as smf

#in statsmodels GLM refers to generalized linear model and not general linear model
#to fit a general linear model (the linear kind, use .ols)

# formula = "SUCCESS ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT + \
#            PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF"


# model = smf.ols(formula = 'ed_gaze_target ~ condition + unique_block + subject')
# print(mod1.summary
# model1 = sm.GLM(data[])

# from statsmodels.regression.mixed_linear_model import MixedLM

# model = MixedLM.from_formula('y ~ x1 + x2', groups='group', data=data)
# result = model.fit()


#
#
#
#% Do stat (ttest) on gaze point distance in allo vs. ego
    # import statsmodels.formula.api as smf
    # from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import pingouin as pg
#   tukey_result = pairwise_tukeyhsd(mydata[prop_key], mydata['condition_name2'], alpha=0.05)
# output = pg.ttest(np.array(gaze_deviations_from_center[eye]['ego']), gaze_deviations_from_center[eye]['allo'])
#%%

# ego_gaze_dev = np.array(gaze_deviations_from_center[eye]['ego'])
# allo_gaze_dev = np.array(gaze_deviations_from_center[eye]['allo'])

# ego_gaze_dev = ego_gaze_dev[~np.isnan(ego_gaze_dev)]
# allo_gaze_dev = allo_gaze_dev[~np.isnan(allo_gaze_dev)]
# ego_gaze_dev = ego_gaze_dev[0:7000]
# allo_gaze_dev = allo_gaze_dev[0:7000]


# from scipy.stats import ttest_ind
# tmp = np.round(ttest_ind(ego_gaze_dev, allo_gaze_dev, equal_var=True), 100)

# #try to run 2 factors GLM, blocks and conditions and see if conditions is still significant if including block factor
# #
# import statsmodels.api as sm
# model1 = sm.GLM()

#hugely significant difference between ego and allo should try cohen's d
#then shift to allo color
#also really really hard to see if differences are due to differences
#within or between conditions
#probably better to compare differences within ego and within allo
#then see if the difference between ego and allo is larger than difference within
#ego and within allo (im not sure if ttest already does something like this)
#the other thing  would be to train some kind of ML model to see if and how it can 
#discriminate between the two conditions.

#the other thing you can try is to bin and do like histogram of time on target or bin a few distances
# a really hot plot might have some like topo lines to show standard deviation in polar coordinates
# and like of different colors with shaded areas for the different conditions
#and maybe very small gaze points.

#%%
#%%
# 
# # fig.suptitle(f"gaze_from_target_center by condition")
# for ego_allo_color in np.unique(conditions2):
#     for press_id in data['press_id'][data['condition_name1'] == ego_allo_color]:
#         for eye in eyes:
#             x_dist = data[eyes[eye][0]]*1920 - x   
#             y_dist = data[eyes[eye][1]]*1080 - y 
#             ax.set_xlim(0, 1920)
#             ax.set_ylim(0, 1080)
#             ax.scatter((x_dist + 1920/2), (y_dist + 1080/2), color =  colors[], s=dotsize)


        


 
                   
  
# fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
# fig.suptitle(f"gaze_from_target_center {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center

# for e, eye in enumerate(eyes):
#     ax = axs[e]
#     ax.set_xlim(0, 1920)
#     ax.set_ylim(0, 1080)
#     ax.scatter(gazes[eye][0], gazes[eye][1], color = darkteal, s=dotsize)

# ##%%
# # plot for each eye, all gaze proximity trajectories       
# fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
# fig.suptitle(f"gaze_proximity_trajectories (euclidean_distance) {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
# for e, eye in enumerate(eyes):
#     ax = axs[e]
#     for gaze_time, ed_standard_gaze in list(zip(gaze_times[eye], ed_standard_gazes[eye])):
#         ax.plot(gaze_time, ed_standard_gaze)


# # #%% plot mean gaze traj +- sem
# # plot for each eye, mean gaze trajectory and        
# fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
# fig.suptitle(f"mean_gaze_proximity_trajectories (euclidean_distance) {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center


# for e, eye in enumerate(eyes):                
#     gaze_array = np.array(ed_standard_gazes[eye])
#     mean_gaze_traj = np.mean(gaze_array,0)
#     sem_gaze_traj = np.std(gaze_array, 0) / np.sqrt(np.shape(gaze_array)[0])
#     upperbound = mean_gaze_traj + sem_gaze_traj
#     lowerbound = mean_gaze_traj - sem_gaze_traj
#     mean_gaze_time = np.mean(np.array(gaze_times[eye]), 0)
    
#     # mask any index where either bound is not finite
#     mask = np.isfinite(lowerbound) & np.isfinite(upperbound)

#     # SE = σ / √n
#     ax = axs[e]
#     # plot the central trajectory (matplotlib will leave gaps at NaNs by itself)
#     ax.plot(mean_gaze_time[mask], mean_gaze_traj[mask], color='C0', label='mean gaze')

#     # shaded ±1 SEM envelope
#     ax.fill_between(
#         mean_gaze_time[mask],
#         lowerbound[mask],
#         upperbound[mask],
#         color='C0',
#         alpha=0.25,
#         linewidth=0,
#         label='±1 SEM'
#     )

#     # for gaze_time, ed_standard_gaze in list(zip(gaze_times[eye], ed_standard_gazes[eye])):
#     #     ax.plot(gaze_time, ed_standard_gaze)



# # # plot for each eye mean +/- SEM gaze trajectories
# # fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
# # fig.suptitle(f"euclidean_distance_gaze_from_target_center {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
    
# # %%

a# %%
