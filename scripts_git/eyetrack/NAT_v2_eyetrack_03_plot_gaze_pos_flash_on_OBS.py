import imageio
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_version())

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

# stimulus = 9
test = False

ROOT_DATASET = Path(__file__).resolve().parent.parent.parent
eyetrack_folder = Path(ROOT_DATASET, 'data', 'eyetrack')
OBS_folder = Path(ROOT_DATASET, 'data', 'obs_recordings')

subjects_eyetrack = [sub for sub in eyetrack_folder.iterdir() if sub.is_dir()]
subjects_OBS = [sub for sub in OBS_folder.iterdir() if sub.is_dir()]

subjects = list(zip(subjects_eyetrack, subjects_OBS))

#Main Loop
for subject in subjects:
    print(subject[0])
    eyetrack_runs =  [run for run in subject[0].iterdir() if run.is_dir()]
    OBS_runs = [run for run in subject[1].iterdir() if run.is_dir()]
    runs = list(zip(eyetrack_runs, OBS_runs))
    runs = runs[2:] #only runs for 2 and 3.

    for run in runs:
        print(run)
    
        #LOAD VIDEO
        video = list(run[1].glob('*cut.mp4'))[0]
        print(video)
        video_reader = imageio.get_reader(video, 'ffmpeg')
        first_frame = video_reader.get_data(0)
        # height, width, _ = first_frame.shape # shape of frame (height, width, channels)

        canvas_width, canvas_height = 1920, 1088 #maybe set back to 1080 for viewing or 1088
        target_fps = 60
        skip_beginning_frames = 0
        output_video_path = run[1] / f"overlay_stimuli_{target_fps}fps_full10_skipframes{skip_beginning_frames}_.mp4"
        video_writer = imageio.get_writer(output_video_path, fps=target_fps)

        ##LOAD INTEGRATED EYETRACK_BEHAV DATA
        eyetrack_file = list(run[0].glob("*UPSAMPLED_integrated_eyetrack_behav*"))[0]
        eyetrack = pd.read_csv(eyetrack_file)
        eyetrack_cols = list(eyetrack.columns)

        #video starts a little later than eyetrack : avidemux times: 'press 5': 15.333 'videostart' : 16.333
        #=>1.333 second difference * 60fps => remove first 80 frames from eyetrack
        eyetrack = eyetrack[eyetrack['5']==10]
        eyetrack = eyetrack[skip_beginning_frames:]

        # Eyetracker timestamps and data
        eye_timestamps = eyetrack['TotalTime'][eyetrack['5']==10].to_numpy()
        eye_x = (eyetrack['X_Gaze.1'][eyetrack['5']==10].to_numpy() * 1920).astype(int)
        eye_y = (eyetrack['Y_Gaze.1'][eyetrack['5']==10].to_numpy() * 1088).astype(int)

        #Stimuli
        stimuli_rightsided = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_isRightSide$')]
        
        stimuli_x_l = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_LeftPosX$')]
        stimuli_y_l = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_LeftPosY$')]    
        stimuli_x_r = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_RightPosX$')]
        stimuli_y_r = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_RightPosY$')]

        stimuli_w_l = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_LeftSizeX$')]
        stimuli_h_l = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_LeftSizeY$')]    
        stimuli_w_r = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_RightSizeX$')]
        stimuli_h_r = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_RightSizeY$')]

        stimuli_rot = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_RightRotation$')]

        #Flashes
        stimuli_flashing = eyetrack.loc[:, eyetrack.columns.str.contains(r'^Stimuli_.*_isFlashing$')]

        #Keypresses
        #CORRECT/INCORRECT /could be done by using relevant columns in 02_correct_response    

        num_eye_frames = len(eye_timestamps)
        num_video_frames = video_reader.count_frames()
        max_frames = min(num_eye_frames, num_video_frames)  # Ensure we stop when either runs out
        max_frames = num_eye_frames
        # max_frames = 1000

        lightmagenta   = (255, 0, 255)  
        darkmagenta    = (150, 0, 150)    
        lightteal      = (0, 255, 255)
        darkteal       = (0, 150, 150)

        print(max_frames)

        for i in range(max_frames):
            if i== 0 or i%500==0:
                print(f'frame {i}')
            frame = video_reader.get_data(i)  # Get the current video frame
            # Convert the frame to PIL Image format
            pil_frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_frame)
            # frame = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
            # draw = ImageDraw.Draw(frame)
            x, y = eye_x[i], eye_y[i]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 255, 0))  # Black dot
            
            for stim_coords in list(zip(stimuli_rightsided.iloc[i], 
                                        stimuli_x_l.iloc[i], stimuli_y_l.iloc[i], stimuli_x_r.iloc[i], stimuli_y_r.iloc[i], 
                                        stimuli_w_l.iloc[i], stimuli_h_l.iloc[i], stimuli_w_r.iloc[i], stimuli_h_r.iloc[i], 
                                        stimuli_rot.iloc[i],
                                        stimuli_flashing.iloc[i],
                                        )):
                # print(stim_coords)
                right_sided, x_l, y_l, x_r, y_r, w_l, h_l, w_r, h_r, rot, flashing = stim_coords
                # rect1 = [x_l-w_l/2, y_l-h_l/2, x_l+w_l/2, y_l+h_l/2]
                # rect2 = [x_r-w_r/2, y_r-h_r/2, x_r+w_r/2, y_r+h_r/2]
        
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

                if right_sided == 1:
                    if 960 <= x_r<= 1920 and 0 <= y_r <= 1080:                  
                        # if(w_l>=0 and h_l>=0 and w_r>=0 and h_r >= 0):
                        draw.polygon(rects_lr[0], outline=lightteal)# Teal right
                        draw.polygon(rects_lr[1], outline=lightteal)
                        draw.ellipse(ellipses[0], outline=darkteal)  
                        draw.ellipse(ellipses[1], outline=darkteal)  
                    if flashing == 1:
                        flash_x, flash_y = x_l + np.abs(w_l)/2, y_l - np.abs(h_l)
                        flash_ellipse = (flash_x-5, flash_y-5, flash_x+5, flash_y+5)
                        draw.ellipse(flash_ellipse, fill = (lightteal))
              
                elif right_sided == 0:
                    if 0 <= x_r< 960 and 0 <= y_r <= 1080:  # 
                        # if(w_l>=0 and h_l>=0 and w_r>=0 and h_r >= 0):
                        draw.polygon(rects_lr[0], outline=lightmagenta)# Magenta left
                        draw.polygon(rects_lr[1], outline=lightmagenta)
                        draw.ellipse(ellipses[0], outline=darkmagenta)  
                        draw.ellipse(ellipses[1], outline=darkmagenta)

                    if flashing == 1:
                            flash_x, flash_y = x_l + np.abs(w_l)/2, y_l - np.abs(h_l)
                            flash_ellipse = (flash_x-5, flash_y-5, flash_x+5, flash_y+5)
                            draw.ellipse(flash_ellipse, fill = (lightmagenta))        

        
            frame_resized = pil_frame.resize((canvas_width, canvas_height))
            video_writer.append_data(np.array(frame_resized))  # Resized frame writing

        video_writer.close()
        print(f"Saved {output_video_path}")

print(f"All videos processed.")



# # Loop through the video frames and the eye tracker data together, skipping frames to downsample
# for i in range(num_eye_data):
#     frame_index = i
#     # frame_index = i * skip_factor  # Skip frames to match 50 fps
#     # frame = video_reader.get_data(frame_index)  # Get the current video frame

#     # Convert the frame to PIL Image format
#     # pil_frame = Image.fromarray(frame)
#     draw = ImageDraw.Draw(frame_index)

#     # Get the eye tracker data for this frame
#     x, y = int(eye_x[i]), int(eye_y[i])

#     # Overlay eye tracker data as a circle on the frame
#     draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 0, 255))  # Draw a blue circle

#     # Convert back to numpy array for saving with imageio
#     frame_with_overlay = np.array(pil_frame)

#     # Write the frame to the output video
#     video_writer.append_data(frame_with_overlay)

# # Close the video writer
# video_writer.close()

# import os
# print(os.getcwd())