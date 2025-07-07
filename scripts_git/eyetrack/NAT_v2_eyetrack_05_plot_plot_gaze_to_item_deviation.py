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
min_time = -1000 #ms before/after press
max_time = 1000    #ms before/after press

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
data_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = data_folder.glob("sub*")

def get_value(df, row, pattern):
    val = df.loc[row, df.columns.str.contains(pattern)].values
    return val[0] if len(val) else np.nan  # or raise an error

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

                data_timewindow_start_indices   = press_indices + timewindow_start_index    
                data_timewindow_end_indices     = press_indices + timewindow_end_index

                conditions = data['condition_name1'].iloc[press_indices].values
                conditions2 = data['condition_name2'].iloc[press_indices].values

                flag_names = data['flag_name'].iloc[press_indices].values
                flag_configs = data['flag_config'].iloc[press_indices].values

            

                for ego_allo_color in np.unique(conditions2):

            
                    gazes               = {'left': [[],[]], 'right': [[],[]]}
                    ed_gazes            = {'left': [], 'right': []}
                    ed_standard_gazes   = {'left': [], 'right': []}
                    gaze_times          = {'left': [], 'right': []}


                    for p, (press_index, condition, condition2, flag_name, flag_config, timewindow_start_ind, timewindow_end_ind) in enumerate(list(zip(press_indices, conditions, conditions2, flag_names, flag_configs, data_timewindow_start_indices, data_timewindow_end_indices))):
                        
                        # times = np.arange(timewindow_start_ind, timewindow_end_ind) - timewindow_start_ind) * mean_deltatime 
                       
                        press_df = data.iloc[timewindow_start_ind:timewindow_end_ind]
                        press_df = press_df.reset_index(drop=True)

        
                        for e, eye in enumerate(eyes):
                            if ego_allo_color in condition2:
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

                                    x_dist = press_df[eyes[eye][0]]*1920 - x   
                                    y_dist = press_df[eyes[eye][1]]*1080 - y 

                                    gazes[eye][0].append(x_dist + 1920/2)
                                    gazes[eye][1].append(y_dist + 1080/2)


                                    times = (press_df['Time'] - press_df['Time'].iloc[0]) * 1000 + min_time
                                    gaze_times[eye].append(times)
                                    # e

                                    ed_gaze = np.sqrt(x_dist**2 + y_dist**2) #euclidean distance between gaze and target center
                                    ed_gazes[eye].append(ed_gaze)
      
                                    ed_standard_gaze = ed_gaze #- ed_gaze[0]
                                    ed_standard_gazes[eye].append(ed_standard_gaze)
                    
                                    # ed_gazes[eye].append(e)
                  
                   
                   
                    #plot for each eye all gaze points (grey) and mean/std       
                    fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                    fig.suptitle(f"gaze_from_target_center {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
                    
                    for e, eye in enumerate(eyes):
                        ax = axs[e]
                        ax.set_xlim(0, 1920)
                        ax.set_ylim(0, 1080)
                        ax.scatter(gazes[eye][0], gazes[eye][1], color = darkteal, s=dotsize)

                    ##%%
                    # plot for each eye, all gaze proximity trajectories       
                    fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                    fig.suptitle(f"gaze_proximity_trajectories (euclidean_distance) {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
                    for e, eye in enumerate(eyes):
                        ax = axs[e]
                        for gaze_time, ed_standard_gaze in list(zip(gaze_times[eye], ed_standard_gazes[eye])):
                            ax.plot(gaze_time, ed_standard_gaze)


                    # #%% plot mean gaze traj +- sem
                    # plot for each eye, mean gaze trajectory and        
                    fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                    fig.suptitle(f"mean_gaze_proximity_trajectories (euclidean_distance) {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
               

                    for e, eye in enumerate(eyes):                
                        gaze_array = np.array(ed_standard_gazes[eye])
                        mean_gaze_traj = np.mean(gaze_array,0)
                        sem_gaze_traj = np.std(gaze_array, 0) / np.sqrt(np.shape(gaze_array)[0])
                        upperbound = mean_gaze_traj + sem_gaze_traj
                        lowerbound = mean_gaze_traj - sem_gaze_traj
                        mean_gaze_time = np.mean(np.array(gaze_times[eye]), 0)
                        
                        # mask any index where either bound is not finite
                        mask = np.isfinite(lowerbound) & np.isfinite(upperbound)

                        # SE = σ / √n
                        ax = axs[e]
                        # plot the central trajectory (matplotlib will leave gaps at NaNs by itself)
                        ax.plot(mean_gaze_time[mask], mean_gaze_traj[mask], color='C0', label='mean gaze')

                        # shaded ±1 SEM envelope
                        ax.fill_between(
                            mean_gaze_time[mask],
                            lowerbound[mask],
                            upperbound[mask],
                            color='C0',
                            alpha=0.25,
                            linewidth=0,
                            label='±1 SEM'
                        )

                        # for gaze_time, ed_standard_gaze in list(zip(gaze_times[eye], ed_standard_gazes[eye])):
                        #     ax.plot(gaze_time, ed_standard_gaze)
                    


                    # # plot for each eye mean +/- SEM gaze trajectories
                    # fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                    # fig.suptitle(f"euclidean_distance_gaze_from_target_center {ego_allo_color}") #place target in center of plot and then plot mean gaze from item center
                        
# %%
