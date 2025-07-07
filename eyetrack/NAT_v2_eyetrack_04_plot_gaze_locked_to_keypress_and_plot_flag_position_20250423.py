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
min_time = -500 #ms before/after press
max_time = 0    #ms before/after press

def rotate_rectangle(corners, center, angle):
    """Rotate each corner around the center by `angle` degrees"""
    angle = np.radians(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = center
    return [
        (
            (x - cx) * cos_a - (y - cy) * sin_a + cx,
            (x - cx) * sin_a + (y - cy) * cos_a + cy
        )
        for x, y in corners
    ]

ROOT_DATASET =  Path(__file__).resolve().parent.parent.parent
print(ROOT_DATASET)
data_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
subjects = data_folder.glob("sub*")

# Main loop
for subject in subjects:
    print(subject)
    # runs = [run for run in subject.iterdir() if run.is_dir()]
    runs = list(Path(subject, 'single_stream').glob("run_*"))
    for r, run in enumerate(runs):  
        if r==0:
            print(run)
            flag_gaze_press_plots_folder = Path(rf"{run}/flag_gaze_press_plots")
            if not os.path.exists(flag_gaze_press_plots_folder):
                os.mkdir(flag_gaze_press_plots_folder)

            data_file = list(run.glob("*eye_and_pos*"))[0]
            data = pd.read_csv(data_file)

            #select data after flashes
            # press_indices = data['response']#########extract indices of responses      
            press_indices   = data.index[data['Events'] == 'KeyPress']
            mean_deltatime  = np.mean(data['DeltaTime'][data['5']!=12].astype(float))  # was 16.68624364606602

            timewindow_start_index  = np.round(min_time / mean_deltatime).astype(int)
            timewindow_end_index    = np.round(max_time / mean_deltatime).astype(int)     

            data_timewindow_start_indices   = press_indices + timewindow_start_index    
            data_timewindow_end_indices     = press_indices + timewindow_end_index

            flag_names = data['flag_name'].iloc[press_indices].values

            #plot for each eye all gaze points (grey) and mean/std 

            for p, (press_index, flag_name, timewindow_start_ind, timewindow_end_ind) in enumerate(list(zip(press_indices, flag_names, data_timewindow_start_indices, data_timewindow_end_indices))):
                print(flag_name)
                press_df = data.iloc[timewindow_start_ind:timewindow_end_ind]
                press_df = press_df.reset_index(drop=True)
                
                fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
                fig.suptitle(
                        f"{flag_name} -- {data['flag_config'].iloc[press_index]}\n"
                        f"{data['condition_name1'].iloc[press_index]} -- {data['which_key'].iloc[press_index]}\n"
                        f"{data['correct'].iloc[press_index]}"
                    )
                for e, eye in enumerate(eyes):
                    press_df[eyes[eye][0]] *= 1920 #transform to pixel coordinates
                    press_df[eyes[eye][1]] *= 1080 #transform to pixel coordinates
                    ax = axs[e]
                    ax.set_xlim(0, 1920)
                    ax.set_ylim(0, 1080)
                    ax.scatter(press_df[eyes[eye][0]], press_df[eyes[eye][1]], color = darkergray, s=dotsize)

                    x_mean, x_std = np.mean(press_df[eyes[eye][0]]), np.std(press_df[eyes[eye][0]])
                    y_mean, y_std = np.mean(press_df[eyes[eye][1]]), np.std(press_df[eyes[eye][1]])

                    ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color='black')
                    ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color='black')

                    x_mean, x_std = np.mean(press_df[eyes[eye][0]]), np.std(press_df[eyes[eye][0]])
                    y_mean, y_std = np.mean(press_df[eyes[eye][1]]), np.std(press_df[eyes[eye][1]])

                    ax.scatter(press_df[eyes[eye][0]], press_df[eyes[eye][1]], color=gaze_color, s=dotsize)

                    ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color=gaze_color)
                    ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color=gaze_color)

                    if not flag_name.startswith('no_match'):        
                        def get_value(df, row, pattern):
                            val = df.loc[row, df.columns.str.contains(pattern)].values
                            return val[0] if len(val) else np.nan  # or raise an error

                        for r in range(len(press_df)):
                            x_l = get_value(press_df, r, f'{flag_name}_LeftPosX$')
                            y_l = get_value(press_df, r, f'{flag_name}_LeftPosY$')
                            x_r = get_value(press_df, r, f'{flag_name}_RightPosX$')
                            y_r = get_value(press_df, r, f'{flag_name}_RightPosY$')
                            w_l = get_value(press_df, r, f'{flag_name}_LeftSizeX$')
                            h_l = get_value(press_df, r, f'{flag_name}_LeftSizeY$')
                            w_r = get_value(press_df, r, f'{flag_name}_RightSizeX$')
                            h_r = get_value(press_df, r, f'{flag_name}_RightSizeY$')
                            rot = get_value(press_df, r, f'{flag_name}_RightRotation$')

                            rects_lr = []
                            ellipses = []
                            for rect in [[x_l, y_l,  w_l, h_l],[x_r, y_r, w_r, h_r]]:
                                x, y, w, h = rect            
                                max_x, min_x, max_y, min_y = x+w/2, x-w/2, y+h/2, y-h/2
                                rect_corners = [(min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y) ]
                                rotated_rect = rotate_rectangle(rect_corners, (x, y), rot)
                                rects_lr.append(rotated_rect)
                                ellipse = (x - 5, y - 5, x + 5, y + 5)
                                ellipses.append(ellipse)
        
                            if 960 <= x_r<= 1920 and 0 <= y_r <= 1080:                               
                                ax.add_patch(patches.Polygon(rects_lr[0], edgecolor=darkteal, fill=False))
                                ax.add_patch(patches.Polygon(rects_lr[1], edgecolor=darkteal, fill=False))
                                ax.add_patch(patches.Ellipse((x_l, y_l), 10, 10, edgecolor=darkteal, fill=False))
                                ax.add_patch(patches.Ellipse((x_r, y_r), 10, 10, edgecolor=darkteal, fill=False))


                            if 0 <= x_r< 960 and 0 <= y_r <= 1080: 
                                ax.add_patch(patches.Polygon(rects_lr[0], edgecolor=darkmagenta, fill=False))
                                ax.add_patch(patches.Polygon(rects_lr[1], edgecolor=darkmagenta, fill=False))
                                ax.add_patch(patches.Ellipse((x_l, y_l), 10, 10, edgecolor=darkmagenta, fill=False))
                                ax.add_patch(patches.Ellipse((x_r, y_r), 10, 10, edgecolor=darkmagenta, fill=False))
                    fig.savefig(rf"{flag_gaze_press_plots_folder}/flag_gaze_press_{subject.name}_{run.name}_{p:03d}.png")                  

#         video_writer.close()
#         print(f"Saved {output_video_path}")
# print(f"All videos processed.")









































        # response_locked_data = []
        # for timewindow in timewindows:
        #     tmp = data.iloc[timewindow[1] : timewindow[2]].copy()
        #     # tmp['flash_side'] = data['Stimulus_RightSide'].iloc[timewindow[0]]
        #     # tmp['flag_side'] = data['Stimulus_RightSide'].iloc[timewindow[0]] 
        #     #figure out 'Stimulus_RightSide' is!! It is probably a bool column in the flash file.
        #     response_locked_data.append(tmp)
        # response_locked_data = pd.concat(response_locked_data)
        # press_df = response_locked_data.copy()
        
        # #plot for each eye all gaze points (grey) and mean/std 
        # fig, axs = plt.subplots(1,2,figsize = [2*20, 20*1080/1920])
        # for e, eye in enumerate(eyes):
        #     press_df[eyes[eye][0]] *= 1920 #transform to pixel coordinates
        #     press_df[eyes[eye][1]] *= 1080 #transform to pixel coordinates
        #     ax = axs[e]
        #     ax.scatter(data[eyes[eye][0]]*1920, data[eyes[eye][1]]*1080, color = darkergray, s=dotsize)

        #     x_mean, x_std = np.mean(press_df[eyes[eye][0]]), np.std(press_df[eyes[eye][0]])
        #     y_mean, y_std = np.mean(press_df[eyes[eye][1]]), np.std(press_df[eyes[eye][1]])

        #     ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color='black')
        #     ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color='black')

        #     # # for left and right press_dfes plot the gaze points within the time windows after each press_df and mean/std
        #     # for s, (side_name, side_id) in enumerate(press_df_side.items()):
        #     #     press_df         = press_df[press_df['press_df_side'].isin([side_id])]

        #     x_mean, x_std = np.mean(press_df[eyes[eye][0]]), np.std(press_df[eyes[eye][0]])
        #     y_mean, y_std = np.mean(press_df[eyes[eye][1]]), np.std(press_df[eyes[eye][1]])

        #     ax.scatter(press_df[eyes[eye][0]], press_df[eyes[eye][1]], color=colors[0], s=dotsize)

        #     ax.errorbar([x_mean + x_std, x_mean - x_std], [y_mean, y_mean], color=colors2[0])
        #     ax.errorbar([x_mean, x_mean], [y_mean + y_std, y_mean - y_std], color=colors2[0])

        #         # ax.text(1400,  100 - 50 * s, f'Gaze after {side_name} press_df [{min_time}ms:{max_time}ms]', color=colors[s])
        #         # ax.text(1500,  1000 - 50 * s, f'press_df {side_name} mean gaze x: {np.ceil(x_mean)}, mean gaze y: {np.ceil(y_mean)}', color=colors2[s])

        #     ax.text(1400,  150, f'All gaze points during recording')
        #     ax.set_xlabel('x [pixels]')
        #     ax.set_xlim((0, 1920))
        #     ax.set_ylim((0,1080))
        #     ax.set_xticks(np.linspace(0, 1920, 7))
        #     ax.set_yticks(np.linspace(0, 1080, 5))
        #     ax.set_ylabel('y [pixels]')
        #     ax.set_title(f'{eye} eye')

        # fig.suptitle(f'Gaze after press_df onset {subject.name} {run.name}')
        # fig.savefig(f'{run}/Gaze_post_press_df_{min_time}ms_{max_time}_{subject.name}_{run.name}.png')