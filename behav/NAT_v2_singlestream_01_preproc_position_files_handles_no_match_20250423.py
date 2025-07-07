#%%

import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt



# no access to distance, we do know when the flags disappear
# we could work backwards from there?
# like first from the turn than from passing the flags
# we can assume turn happens at 0 ?
ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
# print(ROOT_DATASET)
behav_folder = Path(ROOT_DATASET, 'data', 'behav')
subjects = list(behav_folder.glob("*sub*"))

#%%


# subjects = [sub for sub in behav_folder.iterdir() if sub.is_dir()]

def legend_without_duplicate_labels(ax): #https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # ax.legend(*zip(*unique), bbox_to_anchor=(1.4, 1), loc='upper center')
    ax.legend(*zip(*unique), bbox_to_anchor=(0.5, -0.5,), loc='lower center')

def create_flag_plots(ax, blocks, blockstart, events, conditions, markers, figname, legend):
    uniqueblocks = blocks.unique()
    for event_key, times in events.items():
        condition_names = []
        for b in uniqueblocks:         #for each block extract the events in that block based on whether their indices in the original data correspond to that block.
            block_true = blocks==b
            ax.plot(times[block_true] - blockstart[block_true], 
            np.full(times[block_true].shape, b), 
            linestyle='None', 
            marker=markers[event_key]['shape'], 
            ms=markers[event_key]['size'], 
            markeredgecolor=markers[event_key]['edge'], 
            markerfacecolor=markers[event_key]['face'], 
            markeredgewidth=1,
            label=markers[event_key]['label'])
            condition_names.append(conditions[block_true].iloc[0])
       
    if legend:
        legend_without_duplicate_labels(ax)
    ax.set_title(figname)
    ax.set_yticks(np.arange(len(uniqueblocks)))
    ax.set_yticklabels(uniqueblocks)
    ax.set_ylabel("Block ID n°")
    ax.set_xlabel("Time in Block [s]")
    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(len(condition_names)))
    ax2.set_yticklabels(condition_names)
    ax2.set_ylim(ax.get_ylim())

for subject in subjects:
    print(subject.name)
    for task_type in ['single_stream']:
        print(task_type)
        task_dir = Path(rf"{subject}/{task_type}")
        runs = task_dir.glob("run*")
        all_ego_sides = []
        for r, run in enumerate(runs):  
            # if r == 0:    
                print(run.name)
                raw_folder      = list(run.glob("raw*"))[0]
                condition_file  = list(raw_folder.glob("*instruction*.csv"))
                condition       = pd.read_csv(condition_file[0], delimiter=';', header=None) 

                position_files  = list(raw_folder.glob("*POSITION.csv"))
                pattern         = re.compile(r".*_(\d+)_POSITION\.csv$")
                position_files  = sorted(position_files, key=lambda x: int(pattern.match(x.name).group(1)))

                position_dfs    = []
                flag_dfs        = []
            
                #prep deltatimes plot
                fig1, axs1 = plt.subplots(1,1, figsize = (15,10))
                axs1.set_xlim(0,1000)
                axs1.set_xticks([0,50,100] + list(np.arange(100,1100,100)))
                axs1.set_ylim(0,10000)
                axs1.set_yticks(np.arange(0,1200,200))
                axs1.set_ylabel("counts")
                axs1.set_xlabel("sampling rate [Hz]")
                axs1.set_title(f"deltatimes")
                for p, position_file in enumerate(position_files):
                    # print(position_file)
                    position_df= pd.read_csv(position_file, skiprows=1, delimiter=';')
                    
                    #calc and plot deltatimes
                    deltatimes = np.diff(position_df['Time'])*1000
                    sampling_rate = 1000/deltatimes
                    dt_ms = position_df['Time'].diff() * 1000          
                    dt_ms = dt_ms.iloc[1:]                             
                    dt_ms = dt_ms.mask(dt_ms <= 0)
                    sampling_rate = (1000 / dt_ms).dropna()   
                    counts, bins = np.histogram(sampling_rate, 30)
                    axs1.stairs(counts, bins)

                    position_df['block'] = f'block {p}'
                    position_df['condition'] = condition.iloc[p, 0]
                    current_condition = np.unique(position_df['condition'])[0]
                    
                    if current_condition in [1,2,3]:
                        current_flag_item = 'key'
                    elif current_condition in [4,5,6]:
                        current_flag_item = 'eagle'
                    
                    condition_map1 = {'explore':[0], 'ego_key':[1], 'allo_key':[2], 'color_key':[3], 'ego_eagle':[4], 'allo_eagle':[5], 'color_eagle':[6]}
                    condition_map2 = {'explore':[0],'ego':[1,4], 'allo':[2,5], 'color':[3,6]}
                    for name, condition_map in {'condition_names1': condition_map1, 'condition_names2': condition_map2}.items():
                        for cond in condition_map.items():
                            position_df.loc[position_df['condition'].isin(cond[1]), name] = cond[0]

                    position_df['which_key'] = position_df['Press X'] + position_df['Press Y']*2

                    position_df['Events'] = ''                    
                    position_df.loc[position_df['which_key'].isin([1,2]), 'Events'] = 'KeyPress'
                    position_df.loc[2, 'Events'] = f'start_block_{p:02}'
                    position_df.loc[position_df.index[-1], 'Events'] = f'finish_block_{p:02}'
                    
                    stimuli_nbPressed_regex = r"Stimuli_\d+_nbPressed"
                    stimuli_nbPressed_columns =  position_df.columns[position_df.columns.str.contains(stimuli_nbPressed_regex, regex=True)]

                    stimuli_width_regex = r'^Stimuli_.*_LeftSizeX$'
                    stimuli_width_columns =  position_df.columns[position_df.columns.str.contains(stimuli_width_regex, regex=True)]
                    all_flag_offset_names = []
                    all_flag_offset_indices = [] 
                    for col in stimuli_width_columns:
                        flag_offset_name = f"{col.split('_')[0]}_{col.split('_')[1]}"
                        flag_width = position_df[col][position_df[col].notna()]
                        flag_present = (position_df[col]!=0).astype(int)
                        flag_onoff   = np.diff(flag_present)
                        flag_offset_index  = np.where((flag_onoff==-1).astype(int))[0][0]
                        all_flag_offset_indices.append(flag_offset_index) 
                        all_flag_offset_names.append(flag_offset_name)

                    all_flag_offset_times = list(position_df['TimeTrial'].iloc[all_flag_offset_indices])
                    all_flag_offsets = dict(list(zip(all_flag_offset_names, list(zip(all_flag_offset_indices, all_flag_offset_times)))))
                
                    if task_type == 'single_stream':
                        # response_indices = position_df.index[np.diff(position_df['which_key'])==1]         
                        response_indices = position_df.index[position_df['which_key'] != 0]
                        #match the responses to some flags
                        black_list = []
                        matched_responses_flags = []
                        no_match_counter = 0
                        for response_index in response_indices:
                            # get width of all of the flags when the response is made (as proxy for closeness)
                            stimuli_w_l = position_df.loc[response_index-10, position_df.columns.str.contains(r'^Stimuli_.*_LeftSizeX$')]
                            stimuli_w_l = pd.to_numeric(stimuli_w_l, errors='coerce')
                            stimuli_w_l = abs(stimuli_w_l)
                            widest_3 = stimuli_w_l.nlargest(3) #select 3 widest flags
                            closestflag_names = widest_3.index.tolist()
                            closestflag_names = [f"{col.split('_')[0]}_{col.split('_')[1]}" for col in closestflag_names]
                            closestflags_widths = widest_3.tolist()
                            for flag_name, flag_width in list(zip(closestflag_names, closestflags_widths)):
                                if flag_name not in black_list:
                                    flag_name
                                    black_list.append(flag_name)
                                    matched_responses_flags.append([flag_name, response_index])
                                    break #leave loop once first flag selected  
                            else: 
                                no_match_name = f"no_match_{no_match_counter}"
                                no_match_counter += 1
                                matched_responses_flags.append([no_match_name, response_index])
                                all_flag_offsets[no_match_name] = (np.nan, np.nan) 
                                # print("added null flag", no_match_name)
                        
                        #now that we have matched flags to the responses we need to extract some information about those flags 
                        #for each response:
                        matched_responses_flags = dict(matched_responses_flags)

                        flag_names                  = []
                        flag_configs                = []
                        flag_item_posXs             = []
                        flag_item_posYs             = []
                        flag_item_rots              = []
                        flag_item_sides             = []
                        flag_item_colors            = [] 
                        response_onsets             = []
                        response_indices2           = []
                        which_key                   = []
                        expected_responses          = []
                        flag_offset_indices         = []
                        flag_offset_times           = []
                        # flag_distances_to_player    = []

                        for flag_name in all_flag_offsets.keys():   
                                flag_names.append(flag_name)
                                if flag_name.startswith('no_match'): #if response has no flag
                                    # print(flag_name)
                                    flag_offset_indices.append(np.nan)
                                    flag_offset_times.append(np.nan)
                                    flag_configs.append(flag_name)
                                    flag_item_posXs.append(np.nan)
                                    flag_item_posYs.append(np.nan)
                                    flag_item_rots.append(np.nan)
                                    flag_item_sides.append(np.nan)
                                    flag_item_colors.append(np.nan)
                                    expected_responses.append(0)
                                    response_index = matched_responses_flags[flag_name] 
                                    response_indices2.append(response_index)
                                    response_onsets.append(position_df['TimeTrial'].iloc[response_index])
                                    which_key.append(position_df['which_key'].iloc[response_index])
                            
                                else:             
                                    flag_offset_index = all_flag_offsets[flag_name][0]
                                    flag_offset_indices.append(flag_offset_index)

                                    flag_offset_time = all_flag_offsets[flag_name][1]
                                    flag_offset_times.append(flag_offset_time)

                                    flag_config = position_df[f'{flag_name}_Name'].iloc[0]
                                    flag_config = f"{flag_config.split('_')[3]}_{flag_config.split('_')[4]}_{flag_config.split('_')[5]}"
                                    flag_configs.append(flag_config)

                                    #compute expected responses for all conditions (in eagle / key)
                                    if current_condition != 0:
        
                                        config_side = flag_config.split('_')[2]
                                        config_color_side = f"{flag_config.split('_')[1]}_{flag_config.split('_')[2]}"

                                        left_right = {
                                        #key                     #eagle	
                                        ('key', 'OKE'): 'Left',  ('eagle', 'OEK'): 'Left',
                                        ('key', 'OEK'): 'Right', ('eagle', 'OKE'): 'Right',
                                        }

                                        correct_allo = {
                                        #key                #eagle
                                        ('key', 'OKE'): 1, ('eagle', 'OEK'): 1, 
                                        ('key', 'OEK'): 2, ('eagle', 'OKE'): 2,
                                        }

                                        correct_color = {
                                        #key                    #eagle
                                        ('key', 'CYR_OKE'): 1, ('eagle', 'CYR_OKE'): 2,
                                        ('key', 'CYR_OEK'): 2, ('eagle', 'CYR_OEK'): 1, 
                                        ('key', 'CRY_OEK'): 1, ('eagle', 'CRY_OEK'): 2,   
                                        ('key', 'CRY_OKE'): 2, ('eagle', 'CRY_OKE'): 1,   
                                        }                            
                                                                                        
                                        #ego
                                        rotation_finish_row = position_df.iloc[np.where(position_df['UserState']==2)[0][0]]
                                        side = left_right.get((current_flag_item, config_side))
                                        flag_item_posXs.append(rotation_finish_row[f'{flag_name}_{side}PosX'])
                                        flag_item_posYs.append(rotation_finish_row[f'{flag_name}_{side}PosY'])
                                        flag_item_rots.append(rotation_finish_row[f'{flag_name}_{side}Rotation'])
                                        for flag_item_posX in flag_item_posXs:
                                            if flag_item_posX < 960:
                                                ego_side = 1
                                            elif flag_item_posX > 960:
                                                ego_side = 2
                                        #allo
                                        allo_side = correct_allo.get((current_flag_item, config_side))
                                        flag_item_sides.append(allo_side)    
                                
                                        #color
                                        color = correct_color.get((current_flag_item, config_color_side))
                                        flag_item_colors.append(color)

                                        #assign expected responses for actual condition
                                        if   current_condition in [1,4]: expected_responses.append(ego_side)
                                        elif current_condition in [2,5]: expected_responses.append(allo_side)
                                        elif current_condition in [3,6]: expected_responses.append(color)
                                    
                                    elif current_condition == 0:
                                        flag_item_posXs.append(np.nan)
                                        flag_item_posYs.append(np.nan)
                                        flag_item_rots.append(np.nan)
                                        flag_item_sides.append(np.nan)
                                        flag_item_colors.append(np.nan)
                                        # response_indices2.append(response_index)
                                        expected_responses.append(0)
                                
                                    if flag_name not in matched_responses_flags.keys(): #if flag has no response
                                        # response_indices2.append(np.nan)
                                        response_indices2.append(flag_offset_index)
                                        response_onsets.append(np.nan)
                                        which_key.append(0)
                                    else: #if flag has response and response has flag
                                        response_index = matched_responses_flags[flag_name] 
                                        response_indices2.append(response_index)
                                        response_onsets.append(position_df['TimeTrial'].iloc[response_index])
                                        which_key.append(position_df['which_key'].iloc[response_index])

                                    #Euclidean distance from flag to player at keypress (otherwise 0.2 seconds prior?)
                                # playerxz = rotation_finish_row['User Pos X', 'User Pos Z'].to_numpy()
                                # flagxz   = rotation_finish_row[f'{flag_name}_PosZ']
                                # # f'{flag_name}_{side}PosY''] 
                                # # a = np.array([a1, a2])
                                # # b = np.array([b1, b2])
                                # norm_diff = np.linalg.norm(a - b)
                                # print(response_index, flag_name, flag_config, allo_side, color)
                                    
                        flag_df =   pd.DataFrame({
                        'response_indices'          : response_indices2,
                        'response_onset'            : response_onsets,
                        'which_key'                 : which_key,
                        'expected_response'         : expected_responses,
                        'flag_name'                 : flag_names,
                        'flag_offset_indices'       : flag_offset_indices,
                        'flag_offset_times'         : flag_offset_times,
                        'flag_config'               : flag_configs,
                        'flag_item_posX'            : flag_item_posXs,
                        'flag_item_posY'            : flag_item_posYs,
                        'flag_item_rots'            : flag_item_rots,
                        'flag_item_side'            : flag_item_sides,
                        'flag_item_colors'          : flag_item_colors,       
                        })

                        flag_df.insert(4,'correct', flag_df['which_key'] == flag_df['expected_response']) # flag_df.insert(4,'correct', flag_df['which_key'] == (flag_df['expected_response']) & (flag_df['expected_response']!=0) & (flag_df['expected_response'].notna()))
                        flag_df.insert(5, 'error', flag_df['which_key'] != flag_df['expected_response']) # flag_df.insert(5,'error', flag_df['which_key']   != (flag_df['expected_response']) & (flag_df['expected_response']!=0) & (flag_df['expected_response'].notna()))
                       
                        #reindex to response_indices
                        flag_df = (flag_df.set_index('response_onset').reset_index())
                        flag_df['block_start_time']  = position_df['TimeTrial'].iloc[0]
                        flag_df['block_end_time']    = position_df['TimeTrial'].iloc[-1]
                        flag_df['subject']           = subject.name
                        flag_df['run']               = run.name
                        flag_df['block']             = position_df['block'].iloc[1]
                        flag_df['condition_name1']   = position_df['condition_names1'].iloc[1]
                        flag_df['condition_name2']   = position_df['condition_names2'].iloc[1]
                        flag_dfs.append(flag_df)      #add to run list

                        flag_df = flag_df.set_index("response_indices")
                        dupes = flag_df.columns.intersection(position_df.columns)
                        flag_df = flag_df.drop(columns=dupes)

                        position_df = position_df.join(flag_df, how="left")
                        position_dfs.append(position_df)
                
                position_df = pd.concat(position_dfs)
                position_df.to_csv(f'{run}/position_file_{subject.name}_{run.name}.csv')

                fig1.savefig(f'{run}/deltatimes_histogram_{subject.name}_{run.name}')
            
                flag_df = pd.concat(flag_dfs) #create dataframe with all data from run
                flag_df.to_csv(f'{run}/flag_file_{subject.name}_{run.name}.csv')

                # Quality Checks: plot whether response was error or correct
                #prep plots
                markers = {
                    'start'             : {'shape': '>', 'edge': 'k',   'face':     'k',    'label': 'Block_Start_Time',        'size': 7},
                    'end'               : {'shape': 's', 'edge': 'k',   'face':     'k',    'label': 'Block_End_Time',          'size': 7},
                    'flags'             : {'shape': 's', 'edge': '0.7', 'face':     '0.7',  'label': 'Flag_Offset_Time',        'size': 7},
                    'keypresses'        : {'shape': 'o', 'edge': 'b',   'face':     'b',    'label': 'Keypress_Time',           'size': 3},
                    'correct'           : {'shape': 'o', 'edge': 'g',   'face':     'g',    'label': 'Keypress_Time_Correct',   'size': 3}, 
                    'error'             : {'shape': 'o', 'edge': 'r',   'face':     'r',    'label': 'Keypress_Time_Error',     'size': 3},     
                    }

                events = {
                    'start'             : flag_df['block_start_time'], 
                    'end'               : flag_df['block_end_time'],
                    'flags'             : flag_df['flag_offset_times'],
                    'keypresses'        : flag_df['response_onset'],
                    'correct'           : flag_df['response_onset'].where(flag_df['which_key'].isin([1,2]) & flag_df['correct']),
                    'error'             : flag_df['response_onset'].where(flag_df['which_key'].isin([1,2]) & ~flag_df['correct'])
                    }   
        
                flashplottypes = {
                    'Events plot raw': (['start','end', 'flags','keypresses']),
                    # 'Events plot with response flag matching': (['start','end', 'flags','keypresses']), 
                    'Events plot with correct response and flag matching': (['start','end', 'flags','correct', 'error']), 
                }
                    
                #plot
                fig, axes = plt.subplots(1, len(flashplottypes), figsize=(7*len(flashplottypes), 5), layout="constrained")
                for i, fpt in enumerate(flashplottypes.items()):
                    name = f'{subject.name} {run.name} {fpt[0]}'
                    eventstoplot = {event: events[event] for event in fpt[1]}
                    create_flag_plots(axes[i], flag_df['block'], flag_df['block_start_time'], eventstoplot, flag_df['condition_name1'], markers, name, legend=True) #these reference the indices in eventstoplot
                    #add lines to illustrate matches
                    if i in [1,2]:
                        for b, block in enumerate(flag_df['block'].unique()):
                            for r, row in enumerate(flag_df[['response_onset', 'flag_offset_times']][flag_df['block'] == block].itertuples(index=False)):
                                response_onset_time, flag_offset_time = row
                                axes[i].plot([response_onset_time, flag_offset_time], [b+(r*0.025)-0.15, b+(r*0.025)-0.15], 'k', linewidth = 1)               
                fig.savefig(f"{run}/flag_plots_{subject.name}_{run.name}")