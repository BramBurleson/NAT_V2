# Acquire and plot eyetracking data for every subject/sub
import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fixation_id_method_for_correction = 'kde_scott'

use_corrected = False #to toggle if using raw or Viewpoint Eyetracker's corrected gaze points comment/uncomment below:
eyes = {'left':{'gaze':['X_Gaze', 'Y_Gaze'], 'pupil':['PupilWidth', 'PupilHeight'], 'fixation':'Fixation'}, 
        'right':{'gaze':['X_Gaze.1', 'Y_Gaze.1'], 'pupil':['PupilWidth.1', 'PupilHeight.1'], 'fixation':'Fixation.1'}}

corrected_eyes = {'left':{'gaze':['X_CorrectedGaze', 'Y_CorrectedGaze'], 'pupil':['PupilWidth', 'PupilHeight'], 'fixation':'Fixation'},  
        'right':{'gaze':['X_CorrectedGaze.1', 'Y_CorrectedGaze.1'], 'pupil':['PupilWidth.1', 'PupilHeight.1'], 'fixation':'Fixation.1'}}
if use_corrected: eyes = corrected_eyes

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = list(eyetrack_folder.glob("sub*"))
tasks = ['single_stream'] #, 'dual_stream']
mygreen = (0.3, 0.7, 0.5)  # bright green

#Main loop
for subject in subjects[2:3]:
    # if subject.name in ['sub-07']:
    #     print(subject.name)
        for task in tasks:
            
            #load eye data 
            raw_eyetrack_folder = Path(subject, task, 'raw')
            file = list(raw_eyetrack_folder.glob('2025*.txt'))

            tmpload     = pd.read_csv(file[0], usecols=[0], delimiter='\t',header=None) #extract first column of eye data file contains tags -- indices for differents sections of file (by rows)
            not_header  = tmpload.index[tmpload.iloc[:,0] != 5].tolist()
            # not_data = tmpload.index[~tmpload.iloc[:, 0].isin([10, 12])].tolist()
            not_data= tmpload.index[~tmpload.iloc[:, 0].isin([10, 12])].tolist()

            header          = pd.read_csv(file[0], skiprows=not_header, nrows=1, delimiter='\t').columns.tolist() #extract header by skipping all rows except header.        
            data            = pd.read_csv(file[0], skiprows=not_data, delimiter='\t', header=None) #extract data by skipping all rows except data
            data.columns    = header  # Add header to data

            #identify run_starts and run_ends
            run_starts = list(data.loc[data['DeltaTime'] == '5'].index)
            print(run_starts)
            run_ends = np.array(list(data.loc[data['DeltaTime'] == 'Experiment'].index))-1
            run_ends = run_ends.tolist()
            print(run_ends)

            print(sum(data['DeltaTime'] == 'startBlock'))

            # #match starts to ends

            for run_number, (run_start, run_end) in enumerate(list(zip(run_starts, run_ends))):
                    new_run_folder = f"{subject}/{task}/run_{run_number+1:02d}"
                    if not os.path.exists(new_run_folder): #read individual eyetrack file and split into runs and create new run folders e.g., run01 run02 etc..
                        os.mkdir(new_run_folder)           #that way these can be reused in next script

                    run = Path(new_run_folder)

                    preprocess_steps_folder = f"{run}/preproc_steps"
                    if not os.path.exists(preprocess_steps_folder):
                        os.mkdir(preprocess_steps_folder)
                    run_preproc = Path(preprocess_steps_folder)
            
                    print(run_start, run_end)

                    run_data = data.loc[run_start:run_end].reset_index(drop=True) #cut run_data to run.
                    run_data['subject_eye'] = subject.name
                    run_data['run_eye'] = f"run_{run_number:02d}"
                    run_data['TotalTime'] -= run_data['TotalTime'].loc[0] #realign time so first start of each run == 0

                    run_start = 0  #reset run_start and run_end indices
                    run_end = len(run_data)

                    origin_run_data = run_data.copy()
                    start_block_indices =  run_data.loc[run_data['DeltaTime'] == 'startBlock'].index
                    finish_block_indices = run_data.loc[run_data['DeltaTime'] == 'finishBlock'].index

                    pause_starts = [run_start] + list(finish_block_indices)[:-1]
                    pause_ends = list(start_block_indices)

                    #extract fixation time windows (start-end indices)
                    fixation_windows = []      
                    for pause_start_end in list(zip(pause_starts, pause_ends)):
                        fixation_start = pause_start_end[0]
                        fixation_end = pause_start_end[0] + (pause_start_end[1] - pause_start_end[0]) // 2
                        fixation_windows.append([fixation_start, fixation_end])

                    #extract block (run_data) time windows (start-end indices)
                    block_windows = []
                    for block_start_end in list(zip(pause_starts, finish_block_indices)):
                        block_start = block_start_end[0]
                        block_end   = block_start_end[1]
                        block_windows.append([block_start, block_end])
                    
                    #create a copy of original run data on which to apply the selected correction:
                    corrected_run_data = origin_run_data.copy()

                    # Plot gaze and do recentering
                    fig, axs = plt.subplots(4,1, layout='constrained',figsize=(20,20))
                    #plot screen centers
                    for ax in axs:
                        ax.plot([run_data['TotalTime'].iloc[0], run_data['TotalTime'].iloc[-1]], [0.5, 0.5], linewidth = 1, color='k')
                    
                    plot_indices = [[0,1], [2,3]]
                    for e, eye in enumerate(eyes):
                        # if e == 0:
                        x_gaze = eyes[eye]['gaze'][0]
                        y_gaze = eyes[eye]['gaze'][1]
                        pupil_width  = eyes[eye]['pupil'][0]
                        pupil_height = eyes[eye]['pupil'][1]
                        x_plot_index = plot_indices[e][0]
                        y_plot_index = plot_indices[e][1]

                    #     #recenter block run_data based on some metric of central tendency of eye position in fixation time windows.
                        lines = []
                    #     # fixation_id_methods = ["none_raw_run_data", "correct_mean", "correct_median", 'kde_scott', 'kde_silverman', 'kde_0.1', 'kde_0.7']
                    #     fixation_id_methods = ["none_raw_run_data", "correct_mean", "correct_median", 'kde_scott']
                        fixation_id_methods = ['none_raw_run_data', 'kde_scott','kde_silverman']
                        for fixation_id_method in fixation_id_methods:
                            run_data = origin_run_data.copy() #start with a fresh copy of run_data for each fixation-based gaze recentering method   
                            for fixation_window, block_window in list(zip(fixation_windows, block_windows)):
                                #get indices
                                fixation_indices = np.arange(fixation_window[0], fixation_window[1])
                                block_indices = np.arange(block_window[0], block_window[1])

                                if fixation_id_method == "none_raw_run_data":
                                    center_x = 0.5
                                    center_y = 0.5
                            
                                elif fixation_id_method == "correct_mean":    #just mean
                                    fixation_mean_x = np.nanmean(run_data[y_gaze].iloc[fixation_indices])
                                    fixation_mean_y = np.nanmean(run_data[x_gaze].iloc[fixation_indices])
                                    center_x = fixation_mean_x
                                    center_y = fixation_mean_y

                                elif fixation_id_method == "correct_median":
                                    fixation_median_x = np.nanmedian(run_data[y_gaze].iloc[fixation_indices])
                                    fixation_median_y = np.nanmedian(run_data[x_gaze].iloc[fixation_indices])
                                    center_x = fixation_median_x
                                    center_y = fixation_median_y

                                else:
                                    #would be very good if using optimal kernel size. method used may depend on number of samples I only have 120-150 samples.
                                    tmp = run_data[[x_gaze, y_gaze]].iloc[fixation_indices].dropna().copy()
                                    x = tmp[y_gaze]
                                    y = tmp[x_gaze]

                                    X = np.vstack([x, y])
                                    if fixation_id_method   == 'kde_scott':
                                        kde_method = 'scott'
                                    elif fixation_id_method == 'kde_silverman':
                                        kde_method = 'silverman'
                                    elif fixation_id_method == 'kde_0.1':
                                        kde_method = 0.1
                                    elif fixation_id_method == 'kde_0.7':
                                        kde_method = 0.7

                                    kde = gaussian_kde(X, bw_method=kde_method)  # or 'silverman', or a float
                                    
                                    # To find the max of the KDE, you can do a grid search:
                                    x_grid = np.linspace(x.min(), x.max(), 200)
                                    y_grid = np.linspace(y.min(), y.max(), 200)
                                    xx, yy = np.meshgrid(x_grid, y_grid)
                                    coords = np.vstack([xx.ravel(), yy.ravel()])
                                    densities = kde(coords)
                                    idx_max = np.argmax(densities)
                                    best_x = coords[0, idx_max]
                                    best_y = coords[1, idx_max]

                                    #define output
                                    center_x = best_x
                                    center_y = best_y

                                    #first assign corrected data to corrected_run data columns #this ensures that the correction is saved for both eyes and written to.csv
                                    if fixation_id_method == fixation_id_method_for_correction:
                                        corrected_run_data.loc[block_indices, y_gaze] = run_data.loc[block_indices, y_gaze] - center_x + 0.5
                                        corrected_run_data.loc[block_indices, x_gaze] = run_data.loc[block_indices, x_gaze] - center_y + 0.5
                                        if e == 1: #if second eye save it
                                            corrected_run_data.to_csv(f'{run_preproc}/eyetrack_run_data_{fixation_id_method_for_correction}_{subject.name}_{run.name}.csv')

                                    #apply changes to current run/eye data columns
                                    run_data.loc[block_indices, y_gaze] = run_data.loc[block_indices, y_gaze] - center_x + 0.5
                                    run_data.loc[block_indices, x_gaze] = run_data.loc[block_indices, x_gaze] - center_y + 0.5

                            #plot current run, eye, and fixation method data
                            line, = axs[x_plot_index].plot(run_data['TotalTime'], run_data[x_gaze], linewidth = 0.4)
                            lines.append(line)
                            axs[x_plot_index].set_ylim(0,1)
                            axs[x_plot_index].set_title("raw_run_data and recentered_run_data X")
                            axs[x_plot_index].vlines(run_data['TotalTime'].iloc[start_block_indices], 0, 1, 'g')
                            axs[x_plot_index].vlines(run_data['TotalTime'].iloc[finish_block_indices], 0, 1, 'r')

                            axs[y_plot_index].plot(run_data['TotalTime'], run_data[y_gaze], linewidth = 0.4)
                            axs[y_plot_index].set_ylim(0,1)
                            axs[y_plot_index].set_title("raw_run_data and recentered_run_data Y") 
                            axs[y_plot_index].vlines(run_data['TotalTime'].iloc[start_block_indices], 0, 1, 'g')
                            axs[y_plot_index].vlines(run_data['TotalTime'].iloc[finish_block_indices], 0, 1, 'r')
            
                        axs[x_plot_index].legend(lines, fixation_id_methods, loc='lower right')
                        axs[y_plot_index].legend(lines, fixation_id_methods, loc='lower right')

                    fig.suptitle(f"recentering_eye_run_data_based_on_central_fixation_{subject.name}_{run.name}_both_eyes")
                    fig.savefig(f"{run}/recentering_eye_run_data_based_on_central_fixation_{subject.name}_{run.name}_both_eyes")
            


                    # 20250411_ extract fixations from fixation columns
                    #%%
                    fixation_col = [eyes[eye]['fixation']]
                    fixations = run_data[fixation_col][~np.isnan(run_data[fixation_col].to_numpy())]

                    fig, axs = plt.subplots(1,1)
                    counts, bins = np.histogram(fixations.to_numpy(), 10000)
                    axs.stairs(counts, bins)
                    axs.set_xlim(0,0.3)
                    axs.set_title(f"{eye}_eye fixations")

                    # standard minimum for fixation duration is 0.05
                    #so extract values >0.05 work backwards until previous value that is not larger than subsequent and go from there
                    #or I mean you also just kind of know your sampling rate and shit so you can just divide those values by your sampling rate
                    #and then plot - n frames

                    fixation_true = run_data[fixation_col]!=0 & run_data[fixation_col].notna()
                    # print(fixation_true)
                    fixation_true*=1
                    # print(fixation_true)
                # threshold fixation appears to be much too low
                # extract fixation longer than x and then work backwards until first value of that fixation?

                # #
                # #so at the  moment you are using the run_data defined above
                # #Pupil diameter based blink detection
                # #reload run_data created using KDE_Scott method
                # del run_data
                # run_data = pd.read_csv( f'{run_preproc}/eyetrack_run_data_kde_scott_{subject.name}_{run.name}.csv')
            
                # #Plot pupil diameter histograms
                # fig, axs = plt.subplots(2,2, layout='constrained',figsize=(20,20))
                # axs = axs.ravel()
                # # colors = ['r', 'g', 'b', 'k']
                # # for ax, c, in list(zip(axs, colors)):
                # #     ax.scatter(1,1, 50, color=c)

                # plot_indices = [[0,2], [1,3]]
                # for e, eye in enumerate(eyes):
                #     x_gaze = eyes[eye]['gaze'][0]
                #     y_gaze = eyes[eye]['gaze'][1]
                #     pupil_width  = eyes[eye]['pupil'][0]
                #     pupil_height = eyes[eye]['pupil'][1]
                #     width_plot_index = plot_indices[e][0]
                #     height_plot_index = plot_indices[e][1]

                #     for pupil_metric, plot_index in list(zip([pupil_width, pupil_height], [width_plot_index, height_plot_index])):
                #         clean_pupil_metric = run_data[pupil_metric][run_data['5']==10].to_numpy()
                #         clean_pupil_metric[np.isnan(clean_pupil_metric)] = 0
                #         counts, bins = np.histogram(clean_pupil_metric,15)
                #         axs[plot_index].stairs(counts, bins)
                #         axs[plot_index].set_xlim(0,0.3)
                #         axs[plot_index].set_title(f"{eye}_eye {pupil_metric}")
                    
                # figname= f"pupil_metrics_histograms_{subject.name}_{run.name}" 
                # fig.suptitle(figname)
                # fig.savefig(f"{run_preproc}/{figname}")
        
                # # #based on histograms lets define blinks as anything below 0.05 mm (could be more stringent with 0.7)
                # # #then from there based on literature we can define maybe previous and subsequent 50 ms 
                # # #from data. presumably we remove and then we can try to smooth or interpolate or simply leave out.

                
                # # Plot Pupil Diameter
                # plot_indices = [[0,1], [2,3]]
                # fig, axs = plt.subplots(4,1, layout='constrained',figsize=(20,20))
                # axs = axs.ravel()
                # for e, eye in enumerate(eyes):
                #     x_gaze = eyes[eye]['gaze'][0]
                #     y_gaze = eyes[eye]['gaze'][1]
                #     pupil_width  = eyes[eye]['pupil'][0]
                #     pupil_height = eyes[eye]['pupil'][1]
                #     width_plot_index = plot_indices[e][0]
                #     height_plot_index = plot_indices[e][1]

                #     for pupil_metric, plot_index in list(zip([pupil_width, pupil_height], [width_plot_index, height_plot_index])):
                #         #plot raw pupil metrics
                #         axs[plot_index].plot(run_data['TotalTime'], run_data[pupil_metric], linewidth = 0.4)
                #         axs[plot_index].set_ylim(0,0.5)
                #         axs[plot_index].set_title(f"{eye}_{pupil_metric}")
                    

                #         #compute pupil metric based blinks
                #         mask = run_data[pupil_metric] <= 0.05
                #         mask_previous = mask.shift(1, fill_value=False)    
                #         mask_next     = mask.shift(-1, fill_value=False)
                #         blinks = mask|mask_previous|mask_next


                #         # tmp1 = run_data.index[run_data[pupil_metric].isna()]
                #         # tmp2 = run_data.index[run_data[pupil_metric]<=0.05]
                #         # fig, axs = plt.subplots(1,1)
                #         # axs.plot(np.arange(len(tmp1)), tmp1)
                #         # axs.plot(np.arange(len(tmp2)), tmp2)

                #         #plot the blinks
                #         pupil_metric_based_blink_times = run_data['TotalTime'][blinks]  
                #         axs[plot_index].vlines(pupil_metric_based_blink_times, 0.3, 0.4, colors='r', linestyles='--', linewidth=0.4)
                    

                #         #save the blinks to the df
                #         run_data[f"blinks_{pupil_metric}"] = blinks

                #     figname= f"pupil_metrics_timeseries_{subject.name}_{run.name}" 
                #     fig.suptitle(figname)
                #     fig.savefig(f"{run_preproc}/{figname}")

                # # Plot blinks on gaze
                # fig, axs = plt.subplots(4,1, layout='constrained',figsize=(20,20))
                # axs = axs.ravel()
                # plot_indices = [[0,1], [2,3]]
                # for e, eye in enumerate(eyes):
                #     x_gaze = eyes[eye]['gaze'][0]
                #     y_gaze = eyes[eye]['gaze'][1]
                #     pupil_width  = eyes[eye]['pupil'][0]
                #     pupil_height = eyes[eye]['pupil'][1]
                #     x_plot_index = plot_indices[e][0]
                #     y_plot_index = plot_indices[e][1]

                #     eye_blinks = run_data[f"blinks_{pupil_width}"]|run_data[f"blinks_{pupil_height}"]

                #     for gaze, plot_index in list(zip([x_gaze, y_gaze], [x_plot_index, y_plot_index])):
                #         #plot gaze like before (from kdescott method)
                #         axs[x_plot_index].plot(run_data['TotalTime'], run_data[x_gaze], linewidth = 0.4)
                #         axs[x_plot_index].set_ylim(0,1)
                #         axs[x_plot_index].set_title(f"eye_{eye}: X Gaze")
                    
                #         axs[y_plot_index].plot(run_data['TotalTime'], run_data[y_gaze], linewidth = 0.4)
                #         axs[y_plot_index].set_ylim(0,1)
                #         axs[y_plot_index].set_title(f"eye_{eye}: Y Gaze") 

                #         #plot blinks on gaze
                #         axs[x_plot_index].vlines(run_data['TotalTime'][eye_blinks], 0.3, 0.4, colors='r', linestyles='--', linewidth=0.4)
                #         axs[x_plot_index].vlines(run_data['TotalTime'][eye_blinks], 0.3, 0.4, colors='r', linestyles='--', linewidth=0.4)

                # figname = f"gaze_timeseries_with_blinks_{subject.name}_{run.name}"
                # fig.suptitle(figname)
                # fig.savefig(f"{run_preproc}/{figname}")

                # #plot blinks sans gaze
                # fig, axs = plt.subplots(4,1, layout='constrained',figsize=(20,20))
                # axs = axs.ravel()
                # plot_indices = [[0,1], [2,3]]
                # for e, eye in enumerate(eyes):
                #     x_gaze = eyes[eye]['gaze'][0]
                #     y_gaze = eyes[eye]['gaze'][1]
                #     pupil_width  = eyes[eye]['pupil'][0]
                #     pupil_height = eyes[eye]['pupil'][1]
                #     x_plot_index = plot_indices[e][0]
                #     y_plot_index = plot_indices[e][1]

                #     eye_blinks = run_data[f"blinks_{pupil_width}"]|run_data[f"blinks_{pupil_height}"]
                #     clean_run_data_eye = run_data[~eye_blinks].copy()
            
                #     for gaze, plot_index in list(zip([x_gaze, y_gaze], [x_plot_index, y_plot_index])):
                #         #plot gaze like before (from kdescott method)
                #         axs[x_plot_index].plot(clean_run_data_eye['TotalTime'], clean_run_data_eye[x_gaze], linewidth = 0.4, color=mygreen)
                #         axs[x_plot_index].set_ylim(0,1)
                #         axs[x_plot_index].set_title(f"eye_{eye}: X Gaze")
                    
                #         axs[y_plot_index].plot(clean_run_data_eye['TotalTime'], clean_run_data_eye[y_gaze], linewidth = 0.4, color=mygreen)
                #         axs[y_plot_index].set_ylim(0,1)
                #         axs[y_plot_index].set_title(f"eye_{eye}: Y Gaze") 
                
                # figname = f"gaze_timeseries_blinks_removed_{subject.name}_{run.name}"
                # fig.suptitle(figname)
                # fig.savefig(f"{run_preproc}/{figname}")

                # clean_run_data = run_data.copy()
                
                # clean_run_data.loc[
                #     (clean_run_data['blinks_PupilWidth']
                #     |clean_run_data['blinks_PupilHeight']
                #     |clean_run_data['blinks_PupilWidth.1']
                #     |clean_run_data['blinks_PupilHeight.1']), 
                #     [   'X_Gaze', 
                #         'Y_Gaze',
                #         'X_Gaze.1', 
                #         'Y_Gaze.1',
                #         'PupilWidth',
                #         'PupilHeight',
                #         'PupilWidth.1',
                #         'PupilHeight.1'
                #     ]] = np.nan
                

                
                
                # # clean_riu
                # output_file = f"fixation_recentered_kdescott_blink_removed_eyedata_{subject.name}_{run.name}.csv"
                # clean_run_data.to_csv(f"{run}/{output_file}")
                # # f"{run_preproc}/recentering_eye_run_data_based_on_central_fixation_{subject.name}_{run.name}"


