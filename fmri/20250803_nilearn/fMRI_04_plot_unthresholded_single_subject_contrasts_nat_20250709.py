from pathlib import Path
from nilearn.image import load_img
from nilearn import image, plotting, datasets


ROOT_DATASET    = Path(__file__).resolve().parent.parent.parent
derivatives_folder     = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')


subject_labels_nat = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07', 'sub-08', 'sub-09'] 
subject_labels_ff = ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07', 'sub-08', 'sub-09'] 

tasks = ["ff", "v2_nat"]

effect_ids_nat  = ["Allo-Ego", "Allo-Color", "Ego-Color"]
effect_ids_ff  = ["Allo-Ego_Flags", "Allo-Ego_Faces", "Allo-Color_Flags", "Allo-Color_Faces", "Ego-Color_Flags", "Ego-Color_Faces"]

glm_folder_nat = "glm"
glm_folder_ff = "glm_ff"

for task in tasks:
    print(task)
    if task == "v2_nat":
        subject_labels = subject_labels_nat
        effect_ids = effect_ids_nat
        glm_folder = glm_folder_nat
    elif task == "ff":
        subject_labels = subject_labels_ff
        effect_ids = effect_ids_ff
        glm_folder = glm_folder_ff

    for subject_label in subject_labels:
        print(subject_label)
        subjectfolder = Path(rf'{derivatives_folder}/{subject_label}') 
        subjectfolder_glm = Path(subjectfolder, glm_folder)

        for effect_id in effect_ids:
            print(effect_id)

            zmap_file = list(Path(subjectfolder, glm_folder).glob(f'*glm*{effect_id}*z_map.nii*'))[0]


            print(zmap_file)
            zmap_img = load_img(zmap_file)  #Append the loaded image to sub_img
                        # PLOT 3D SURFACES PIAL STATIC .PNG
            output_file     =  rf'{subjectfolder_glm}/{task}_{subject_label}_glm_{effect_id}_UNTHRESHOLDED_surface.png'
            plotting.plot_img_on_surf(
                zmap_img, 
                surf_mesh       = 'fsaverage',
                views           = ['lateral'], #'medial', 'ventral'],
                hemispheres     = ['left', 'right'],
                colorbar        = True,
                threshold       = None,
                darkness        = 1.0,
                title           = f'{task}_{subject_label}_glm_{effect_id}_unthresholded',
                output_file     = output_file,
            )