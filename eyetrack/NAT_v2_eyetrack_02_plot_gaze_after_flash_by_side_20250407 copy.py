import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#important variables:
flash_side = {'left':0, 'right':1}

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
lightgray = (0.9,0.9,0.9)
darkergray = (0.7,0.7,0.7)
darkmagenta = (0.5, 0, 0.5)  # RGB for dark magenta
darkteal = (0, 0.5, 0.5)     # RGB for teal
colors = [darkmagenta, darkteal]
colors2 = [magenta, teal]
dotsize = 10 

#time window parameters
min_time = 200 #ms post flash
max_time = 500 #ms post flash

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = [sub for sub in eyetrack_folder.iterdir() if sub.is_dir()]

# Main loop
for subject in subjects:
    print(subject)
    # runs = [run for run in subject.iterdir() if run.is_dir()]
    runs = list(subject.glob("run_*"))
    for run in runs:  
        print(run)
        eyetrack_file = list(run.glob('eyetrack_with_position_behav*'))[0]
        eyetrack = pd.read_csv(eyetrack_file)

        #select data after flashes
        flash_indices = eyetrack[eyetrack['isflashing']==1].index
       
        mean_deltatime = np.mean(eyetrack['DeltaTime'][eyetrack['5']!=12].astype(float))  # was 16.68624364606602
        lowerbound = np.round(min_time / mean_deltatime).astype(int)
        upperbound = np.round(max_time / mean_deltatime).astype(int)     
        eyetrack_lowerbounds = flash_indices + lowerbound    
        eyetrack_upperbounds = flash_indices + upperbound
        bounds = list(zip(flash_indices, eyetrack_lowerbounds, eyetrack_upperbounds))
        
        flash_locked_data = []
        for bound in bounds:
            tmp = eyetrack.iloc[[bound[0]] + list(range(bound[1], bound[2]))].copy()
            tmp['flash_side'] = eyetrack['Stimulus_RightSide'].iloc[bound[0]]
            flash_locked_data.append(tmp)
        flash_locked_data = pd.concat(flash_locked_data)
        flash = flash_locked_data.copy()
        
        #plot for each eye all gaze points (grey) and mean/std 
        fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
        for e, eye in enumerate(eyes):
            flash[eyes[eye][0]] *= 1920 #transform to pixel coordinates
            flash[eyes[eye][1]] *= 1080 #transform to pixel coordinates

            ax = axs[e]
            ax.scatter(eyetrack[eyes[eye][0]]*1920, eyetrack[eyes[eye][1]]*1080, color = darkergray, s=dotsize)

            x_mean, x_std = np.mean(flash[eyes[eye][0]]), np.std(flash[eyes[eye][0]])
            y_mean, y_std = np.mean(flash[eyes[eye][1]]), np.std(flash[eyes[eye][1]])

            ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color='black')
            ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color='black')

            # for left and right flashes plot the gaze points within the time bounds after each flash and mean/std
            for s, (side_name, side_id) in enumerate(flash_side.items()):
                flash2         = flash[flash['flash_side'].isin([side_id])]

                x_mean, x_std = np.mean(flash2[eyes[eye][0]]), np.std(flash2[eyes[eye][0]])
                y_mean, y_std = np.mean(flash2[eyes[eye][1]]), np.std(flash2[eyes[eye][1]])

                ax.scatter(flash2[eyes[eye][0]], flash2[eyes[eye][1]], color=colors[s], s=dotsize)

                ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color=colors2[s])
                ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color=colors2[s])

                ax.text(1400,  100 - 50 * s, f'Gaze after {side_name} flash [{min_time}ms:{max_time}ms]', color=colors[s])
                 # ax.text(1500,  1000 - 50 * s, f'Flash {side_name} mean gaze x: {np.ceil(x_mean)}, mean gaze y: {np.ceil(y_mean)}', color=colors2[s])

            ax.text(1400,  150, f'All gaze points during recording')
            ax.set_xlabel('x [pixels]')
            ax.set_xlim((0, 1920))
            ax.set_ylim((0,1080))
            ax.set_xticks(np.linspace(0, 1920, 7))
            ax.set_yticks(np.linspace(0, 1080, 5))
            ax.set_ylabel('y [pixels]')
            ax.set_title(f'{eye} eye')

        fig.suptitle(f'Gaze after flash onset {subject.name} {run.name}')
        fig.savefig(f'{run}/Gaze_post_flash_{min_time}ms_{max_time}_{subject.name}_{run.name}.png')