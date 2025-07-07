import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

fraction_block = True
#important variables:
flash_side = {'left':0, 'right':1}

eyes = {'left':['X_Gaze', 'Y_Gaze'], 'right':['X_Gaze.1', 'Y_Gaze.1']}
corrected_eyes = {'left':['X_CorrectedGaze', 'Y_CorrectedGaze'], 'right':['X_CorrectedGaze.1', 'Y_CorrectedGaze.1']}
#to toggle if using raw or Viewpoint Eyetracker's corrected gaze points comment/uncomment below:
eyes = corrected_eyes

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
# mygreen = (0.35, 0.55, 0.7)
mygreen = (0.3, 0.7, 0.5)  # bright green
lightgray = (0.9,0.9,0.9)
darkergray = (0.7,0.7,0.7)
dotsize = 20
linewidth = 0.5

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = [sub for sub in eyetrack_folder.iterdir() if sub.is_dir()]

# Main loop
for subject in subjects:
    print(subject)
    runs = list(subject.glob("run_*"))
    for run in runs:  
        print(run.name)
        block_gaze_folder = f"{Path(run, f'{subject.name}_{run.name}_block_gaze_trajectory')}"
        if not os.path.exists(block_gaze_folder): os.makedirs(block_gaze_folder) 
     
        eyetrack_file = list(run.glob('eyetrack_with_position_behav*'))[0]
        eyetrack = pd.read_csv(eyetrack_file)

        block_starts = eyetrack[eyetrack['Events'].str.startswith('start_block', na=False)].index
        block_finishes = eyetrack[eyetrack['Events'].str.startswith('finish_block', na=False)].index
        conditions = eyetrack['condition_name2'][eyetrack['condition_name2'].notna()].index.to_numpy()
        
        diff = np.abs(conditions[:, None] - block_starts.to_numpy())
        closest_in_b = conditions[np.argmin(diff, axis=0)]     
        # condition_of_block = conditions  
        # condition_of_block = np.argmin(np.diff(condition.index, block_starts.to_numpy()))
        #get condition of block =>could go back to instruction file but safest option is to use condition column 

        blocks = list(zip(block_starts, block_finishes))
        for b, block in enumerate(blocks):
            if fraction_block==False:
                block_data = eyetrack.iloc[block[0]:block[1]].copy()
                fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                for e, eye in enumerate(eyes):
                    block_data[eyes[eye][0]] *= 1920 #transform to pixel coordinates
                    block_data[eyes[eye][1]] *= 1080 #transform to pixel coordinates

                    ax = axs[e]
                    ax.plot(block_data[eyes[eye][0]], block_data[eyes[eye][1]], color = darkergray, linewidth = linewidth)          
                    ax.scatter(block_data[eyes[eye][0]], block_data[eyes[eye][1]], dotsize, color = mygreen)             
                    ax.set_xlabel('x [pixels]')                  
                    ax.set_ylabel('y [pixels]')
                    ax.set_title(f'{subject.name}_{run.name}_block_{b"%02d"}_{eye} eye')
                    ax.set_xlim((0, 1920))
                    ax.set_ylim((0,1080))
                    ax.set_xticks(np.linspace(0, 1920, 7))
                    ax.set_yticks(np.linspace(0, 1080, 5))

                    x_mean, x_std = np.mean(block_data[eyes[eye][0]]), np.std(block_data[eyes[eye][0]])
                    y_mean, y_std = np.mean(block_data[eyes[eye][1]]), np.std(block_data[eyes[eye][1]])

                    ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color='black')
                    ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color='black')
            # unless
            else: 
                myfraction_number = 4
                myfractions = np.arange(myfraction_number)+1
                for myfraction in myfractions:
                    fraction_block_start_int    = (np.round((block[1]-block[0])/(myfraction_number)*(myfraction-1))+block[0]).astype(int)
                    fraction_block_end_int      = (np.round((block[1]-block[0])/(myfraction_number)*myfraction)+block[0]).astype(int)

                    block_data = eyetrack.iloc[fraction_block_start_int:fraction_block_end_int].copy()
        
                    #plot for each eye all gaze points (grey) and mean/std 
                    fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                    for e, eye in enumerate(eyes):
                        block_data[eyes[eye][0]] *= 1920 #transform to pixel coordinates
                        block_data[eyes[eye][1]] *= 1080 #transform to pixel coordinates
                        ax = axs[e]
                        ax.plot(block_data[eyes[eye][0]], block_data[eyes[eye][1]], color = darkergray, linewidth = linewidth)      
                        ax.scatter(block_data[eyes[eye][0]], block_data[eyes[eye][1]], dotsize, color = mygreen)          
                        ax.set_xlabel('x [pixels]')                  
                        ax.set_ylabel('y [pixels]')
                        ax.set_title(f'{eye}_eye')
                        ax.set_xlim((0, 1920))
                        ax.set_ylim((0,1080))
                        ax.set_xticks(np.linspace(0, 1920, 7))
                        ax.set_yticks(np.linspace(0, 1080, 5))

                        x_mean, x_std = np.mean(block_data[eyes[eye][0]]), np.std(block_data[eyes[eye][0]])
                        y_mean, y_std = np.mean(block_data[eyes[eye][1]]), np.std(block_data[eyes[eye][1]])

                        ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color='black')
                        ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color='black')
                    figname = f'Gaze_in_block_fractions_{subject.name}_{run.name}_block_{b:02d}_part_{myfraction}_of_{myfraction_number}'
                    fig.suptitle(figname)
                    fig.savefig(f'{block_gaze_folder}/{figname}.png')
                    plt.close()