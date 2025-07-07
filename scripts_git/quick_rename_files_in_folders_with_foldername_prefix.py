# import os

# main_dir = r"K:\BramBurleson\000_datasets_and_scripts\NAT_V1_fMRI_pilots\results\fmri\20250307_1120 - Copy"
# folders = os.listdir(main_dir)
# for folder in folders:
#     prefix = os.path.basename(folder) + "_"  # Extract folder name as prefix

#     for filename in os.listdir(rf"{main_dir}\{folder}"):
#         if not filename.startswith(prefix):
#             old_path = os.path.join(folder, filename)
#             new_path = os.path.join(folder, prefix + filename)
#             os.rename(old_path, new_path)

import os

main_dir = r"K:\BramBurleson\000_datasets_and_scripts\NAT_V1_fMRI_pilots\results\fmri\20250307_1120 - Copy"
folders = os.listdir(main_dir)

for folder in folders:
    folder_path = os.path.join(main_dir, folder)
    if not os.path.isdir(folder_path):  # Skip if not a folder
        continue

    prefix = folder + "_"  # Extract folder name as prefix

    for filename in os.listdir(folder_path):
        if filename.lower() in ("thumbs.db", ".ds_store"):  # Ignore system files
            continue
        
        if not filename.startswith(prefix):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, prefix + filename)
            os.rename(old_path, new_path)
