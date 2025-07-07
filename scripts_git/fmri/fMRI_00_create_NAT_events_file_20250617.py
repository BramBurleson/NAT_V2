import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import re

#replace every first other occurence of fixation with prep.

CORRECT_NUMBER_OF_EVENT_TYPES = 9 #removed user state

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
# print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
fmri_derivatives_folder = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
subjects = list(eyetrack_folder.glob('*sub*'))
# subjects = 

# event_names = ['fixation', 'startBlock', 'userStateTrigger', 'finishBlock']
event_names = ['fixation', 'startBlock', 'finishBlock']
for s, subject in enumerate(subjects): 
    print(subject.name)
    # if subject.name == "sub-07":

    subject_events_dir = rf'{fmri_derivatives_folder}/{subject.name}/events'  
    runs      = list(Path(subject, 'single_stream').glob("run*"))
    for r, run in enumerate(runs):
            # if run.name == "run_06":
                print(run.name)   

                eye_and_pos_file = list(run.glob(f'*eye_and_pos*'))[0]
                eye_and_pos = pd.read_csv(eye_and_pos_file)
                event_bool = eye_and_pos['DeltaTime'].isin(event_names)
                events = eye_and_pos['DeltaTime'][event_bool]
                event_times = eye_and_pos['TotalTime'][event_bool]
                # trial_type = eye_and_pos['condition_names1'][event_bool]

                event_durations = list(np.diff(event_times)) + [0]
                trial_types = eye_and_pos['DeltaTime'][event_bool]
                trial_types[trial_types == 'startBlock'] = eye_and_pos['condition_names1'][eye_and_pos['DeltaTime'] == 'startBlock']

                events_df = pd.DataFrame({
                "trial_type": trial_types,
                "onset": event_times,
                "duration": event_durations,
                })
                # events_file_bad = f'{subject.name}_run-{run.name[-2:]}_events_NEW.tsv'
                # events_file_bad_path = Path(subject_events_dir, events_file_bad)
                # if events_file_bad_path.exists():
                #         os.remove(f"{subject_events_dir}/{events_file_bad}")

                events_file = f'{subject.name}_run-{run.name[-2:]}_events.tsv'
                number_of_unique_events = len(events_df["trial_type"].unique())
                if number_of_unique_events != CORRECT_NUMBER_OF_EVENT_TYPES :
                        print(F"!!!!ERROR!!!! INCORRECT NUMBER OF EVENT TYPES: {number_of_unique_events} != {CORRECT_NUMBER_OF_EVENT_TYPES}")
                
        
                events_df.to_csv(f"{subject_events_dir}/{events_file}", sep="\t", index=False)
        # break


# position_events    =  ['start_block', 'instruction_block', 'prep_block',  'finish_block']

# #Establish Paths
# ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
# print(ROOT_DATASET)

# behav_folder = Path(ROOT_DATASET, 'data', 'behav')
# subjects = list(behav_folder.glob("sub*"))

# #Main Loop
# for subject in subjects:
#     print(subject.name)
#     runs      = list(Path(subject, 'single_stream').glob("run*"))
#     for r, run in enumerate(runs):
#         subject_events_dir = rf'{fmri_derivatives_folder}/{subject.name}/events'  
#         os.makedirs(subject_events_dir, exist_ok=True)

#         if r == 0:
#             position_file = list(run[0].glob("*position_file*"))[0]
#             data = pd.read_csv(position_file)

#             all_position_trigger_data = []
#             for tr, trigger in enumerate(position_events):
#                 position_trigger_times = data.loc[data['Events'].str.contains(trigger) & ~data['Events'].isnull(),'Time']
#                 position_trigger_indices = position_trigger_times.index.to_numpy()
#                 position_trigger_times   = position_trigger_times.to_numpy()

#                 tmp = pd.DataFrame()
#                 blockstart_times = np.unique(data['Time'][data['Events'].str.contains('start_block')])
#                 blockend_times   = np.unique(data['Time'][data['Events'].str.contains('finish_block')])
#                 instructionstart_times = np.unique(data['Time'][data['Events'].str.contains('start_block')])
#                 prep_times   = np.unique(data['Time'][data['Events'].str.contains('prep_block')])
#                 instructionstart_times = np.unique(data['Time'][data['Events'].str.contains('instruction_block')])

#                 durations = blockend_times - blockstart_times  
#                 start_of_first_pause = blockstart_times[0]-2
#                 blockstart_times = blockstart_times - start_of_first_pause
#                 instructionstart_times = instructionstart_times - start_of_first_pause
#                 blockend_times   = blockend_times-start_of_first_pause  
#                 onset_times = [val for pair in zip(blockstart_times, blockend_times) for val in pair]
                                        
#                 # # first_occurrence_indices = data[~data['blockstart_time'].duplicated()].index
#                 conditions = data['condition'][data['Events'].isin(blockstart_times)]
#                 instructions = np.tile('instructions', len(blockstart_times))
#                 trial_types = [val for pair in zip(instructions, conditions) for val in pair]
#                 instruction_durations = np.tile(2, len(blockstart_times))
#                 block_durations = blockend_times-blockstart_times
#                 durations = [val for pair in zip(blockstart_times, blockend_times) for val in pair]
                
#                 #create data frame?
#                 events = pd.DataFrame({
#                 "trial_type": trial_types,
#                 "onset": onset_times,
#                 "duration": durations,
#                 })

#                 events_file = f'{subject.name}_run-{run_numbers[r]}_events.tsv'
#                 # if f'run-{bad_runs[subjectname]}' not in events_file:
#                 #         events.to_csv(subject_events_dir / events_file, sep="\t", index=False)
#                 # else: 
#                 #         print(f'bad {events_file}')