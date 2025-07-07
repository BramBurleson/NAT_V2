# Acquire and plot eyetracking data for every subject/sub
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

#integrate flag_indices for producing triggers out put file. or base on responses
#I dont think this: "FIGURE OUT WHY LOSING BLOCK DATA IN EYETRACK DF OUTPUT!" is relevant anymore
upsample = True #!!!!!!!!!!!!!!!!!!!!

#TEST THE TRIGGERS
#extract events file for eye and behav
#unity files are
    #1) one RECAP.csv with values referenced to the task start "5 press"
        #values comprises: 
            #i) 5 press
                #a)"PrepareBlock"
                #b)"StartDisplayInstruction"
                #c)"StartRecordingTrial"
                #d)"StartTrialKeyboard"
                #e)"Start"
                #f)"FinishBlock"
                #g)"EndTrialRecording"
            #ii) EndExperiment

    #2) many POSITION.csv one for each block with values referenced to file creation + 1 frame. The file is created
    #at the time c) "StartRecordingTrial" from the RECAP file.
    #these contain columns describing camera position in unity space and stimuli position in screenspace.
    #in screen space the camera is at the center of the display.

#eyetracking file is a continuous file filled with asynchronous triggers from unity.
        #a)"startBlock", 
        #b)"userStateTrigger", 
        #c)"flashTrigger", 
        #d)"userPressKeyTrigger" 
        #e)"finishBlock", 

#Preallocate relevant variables
#Two lists with corresponding triggers to loop through and match.
# position_triggers    =  ['start_block', 'Flash',        'KeyPress',         'finish_block']  # behav_triggers       =  ['Flash'] 
position_triggers    =  ['start_block', 'KeyPress',         'finish_block']
eyetrack_triggers    =  ['startBlock', 'userPressKeyTrigger', 'finishBlock'] # eyetrack_triggers    =  ['flashTrigger']

#Establish Paths
ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
behav_folder = Path(ROOT_DATASET, 'data', 'behav')

subjects_behav = list(behav_folder.glob("sub*"))
subjects_eyetrack = list(eyetrack_folder.glob("sub*"))

subjects = list(zip(subjects_behav, subjects_eyetrack))

#Main Loop
for subject in subjects:
        print(subject[0].name)
        # if subject[0].name == 'sub-07':
        behav_runs      = list(Path(subject[0], 'single_stream').glob("run*"))
        eyetrack_runs   = list(Path(subject[1], 'single_stream').glob("run*")) 
        runs = list(zip(behav_runs, eyetrack_runs))

        for r, run in enumerate(runs): 
                    print(run[0].name)
                    # if run[0].name == 'run_03':
                    # if r == 0:
                    position_file = list(run[0].glob("*position_file*"))[0]
                    response_file = list(run[0].glob("*flag_file*"))[0]
                    # eyetrack_file = list(run[1].glob("*eyetrack*_data*kde_scott*"))[0]
                    eyetrack_file = list(Path(run[1], 'preproc_steps').glob("eyetrack_*kde_scott*"))[0]
                
                    position = pd.read_csv(position_file)
                    response = pd.read_csv(response_file)
                    eyetrack = pd.read_csv(eyetrack_file)

                    eyetrack['Eyetrack_Triggers'] = '' #easy work around:
                    eyetrack.loc[eyetrack['5'] == 12, 'Eyetrack_Triggers'] = eyetrack.loc[eyetrack['5'] == 12, 'DeltaTime']
                
                    all_eye_and_pos_triggers= []
                    all_position_trigger_data = []
                    fig, axs = fig, axs = plt.subplots(2,2, figsize = (10,10), layout='constrained')
                    for tr, triggers in enumerate(list(zip(position_triggers, eyetrack_triggers))):
                        position_trigger_times = position.loc[position['Events'].str.contains(triggers[0]) & ~position['Events'].isnull(),'Time']
                        position_trigger_indices = position_trigger_times.index.to_numpy()
                        position_trigger_times   = position_trigger_times.to_numpy()
                        eyetrack_trigger_times = eyetrack.loc[eyetrack['Eyetrack_Triggers'].str.contains(triggers[1])& ~eyetrack['Eyetrack_Triggers'].isnull(), 'TotalTime']    
                        # if triggers[1] == 'flashTrigger':
                        #     every_other_eyetrack = np.arange(0,len(eyetrack_trigger_times),2)
                        #     eyetrack_trigger_times = eyetrack_trigger_times.iloc[every_other_eyetrack]
                        eyetrack_trigger_indices = eyetrack_trigger_times.index.to_numpy()
                        eyetrack_trigger_times   = eyetrack_trigger_times.to_numpy()

                        # position/eyetrack data checks
                        if len(eyetrack_trigger_times)==0:
                            print(f'no {triggers[0]} triggers in EYETRACK file for {subject[0].name} {run[0].name}')          
                        if len(position_trigger_times)==0:
                            print(f'n{triggers[0]} triggers in position file for {subject[0].name} {run[0].name}')   
                        if len(position_trigger_times)>len(eyetrack_trigger_times):
                            print(f'More {triggers[0]} triggers in position: {len(position_trigger_times)} than eyetrack : {len(eyetrack_trigger_times)} {subject[0].name} {run[0].name}')               
                        if len(position_trigger_times)<len(eyetrack_trigger_times):
                            print(f'More {triggers[0]} triggers in eyetrack : {len(eyetrack_trigger_times)} than position position: {len(position_trigger_times)} {subject[0].name} {run[0].name}')               
                        if len(position_trigger_times)==len(eyetrack_trigger_times):
                            print(f'Number of {triggers[0]} triggers in position {len(position_trigger_times)} == triggers in eyetrack {len(eyetrack_trigger_times)} {subject[0].name} {run[0].name}')
                        
                        #match triggers from each file.
                        #select the file with fewer triggers and match them to the closest triggers in the other file
                        if len(position_trigger_times)!=len(eyetrack_trigger_times):
                            if len(position_trigger_times)<len(eyetrack_trigger_times):
                                diff = np.abs(eyetrack_trigger_times[:, None] - position_trigger_times)
                                closest_in_b = eyetrack_trigger_times[np.argmin(diff, axis=0)]
                                idx = np.argmin(diff, axis=0)
                                eyetrack_trigger_times = closest_in_b
                                eyetrack_trigger_indices=eyetrack_trigger_indices[idx]

                            elif len(position_trigger_times)>len(eyetrack_trigger_times):
                                diff = np.abs(position_trigger_times[:, None] - eyetrack_trigger_times)
                                closest_in_b = position_trigger_times[np.argmin(diff, axis=0)]
                                idx = np.argmin(diff, axis=0)
                                position_trigger_times = closest_in_b
                                position_trigger_indices=position_trigger_indices[idx]
                        
                        #Store eyetracker and position indices (with reference to their respective data frames) in a shared dataframe:
                        eye_and_pos_triggers = pd.DataFrame({
                            'sub':subject[0].name,
                            'run':run[0].name,
                            # 'difference': abs(time_b_full - time_a_full),
                            'difference': eyetrack_trigger_times - position_trigger_times,
                            'position_trigger_times'    : position_trigger_times,
                            'eyetrack_trigger_times'    : eyetrack_trigger_times,      
                            'position_indices'          : position_trigger_indices,
                            'eyetrack_indices'          : eyetrack_trigger_indices, 
                            'position_name'             : triggers[0],
                            'eyetrack_name'             : triggers[1],
                        })

                        matched_eye_and_pos_triggers     = eye_and_pos_triggers[~eye_and_pos_triggers['difference'].isnull()] #the triggers that match
                        unmatched_eye_and_pos_triggers   = eye_and_pos_triggers[eye_and_pos_triggers['difference'].isnull()]  #the triggers that don't match

                        #Plot triggers (each trigger type gets its own subplot "ax")
                        ax = axs.flat[tr]
                        # axs.scatter(eye_and_pos_triggers.index, eye_and_pos_triggers['difference']*1000, c = 'k', s = 1)
                        ax.plot(matched_eye_and_pos_triggers.index, matched_eye_and_pos_triggers['difference']*1000)
                        ax.scatter(unmatched_eye_and_pos_triggers.index, np.tile(5, len(unmatched_eye_and_pos_triggers)), c = 'r', s=0.5)
                        ax.set_ylim(-50, 50)
                        ax.set_yticks([-50, -25, 0, 25, 50])
                        ax.set_ylabel("eyetrack_trigger_match - position_trigger [ms]")
                        ax.set_xlabel(f"{triggers[0]} index")

                        text_height = 40
                        # axs.text(0, text_height, f"Assigned values (same number for position and eyetrack because paired): {eye_and_pos_triggers['difference'].notna().sum()}")
                        ax.text(0, text_height, f"Total matched values: {eye_and_pos_triggers['difference'].notna().sum()}")
                        ax.text(0, text_height-5, f"Total unassigned values: {eye_and_pos_triggers['difference'].isna().sum()}")
                        ax.text(0, text_height-10, f"position unassigned values: {eye_and_pos_triggers['eyetrack_trigger_times'].isna().sum()}")
                        ax.text(0, text_height-15, f"Eyetrack unassigned values: {eye_and_pos_triggers['position_trigger_times'].isna().sum()}")
                        ax.text(0, text_height-20, f"Mean Difference: {np.round(np.mean(np.abs(matched_eye_and_pos_triggers['difference']))*1000,2)} [ms]")
                        ax.set_title(f"{triggers[0]}")
                        all_eye_and_pos_triggers.append(eye_and_pos_triggers)  
                    fig.suptitle(f'Trigger Timing differences eyetrack - behav \n upsampled \n {subject[0].name} {run[0].name}')
                    fig.savefig(run[1] / f"Trigger_eyetrack_behav_{subject[0].name}_{run[0].name}.png")

                    all_eye_and_pos_triggers= pd.concat(all_eye_and_pos_triggers)         
                
                ###############After visual inspection of trigger correspondence betweeen the two files
                ###############Carry only the eyetrack triggers to the new file.
                ###############The key will be to some how extract the responses from the other files

                    eyetrack['position_indices'] = np.nan
                    eyetrack['position_names'] = ''
                    eyetrack.loc[all_eye_and_pos_triggers['eyetrack_indices'], 'position_indices'] = all_eye_and_pos_triggers['position_indices'].values
                    eyetrack.loc[all_eye_and_pos_triggers['eyetrack_indices'], 'position_name']   = all_eye_and_pos_triggers['position_name'].values

                    eyetrack['Time'] = eyetrack['TotalTime']
                    eyetrack = eyetrack.sort_values('Time').reset_index(drop=True)
                    position = position.sort_values('Time').reset_index(drop=True)  
                    eye_and_pos = pd.merge_asof(eyetrack, position, on='Time', direction='nearest')
                    eye_and_pos.to_csv(f"{run[1]}/eye_and_pos_{subject[0].name}_{run[0].name}.csv")
                    # ###############GetResponse_i