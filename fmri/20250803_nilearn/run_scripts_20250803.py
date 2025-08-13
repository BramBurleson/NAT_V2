from pathlib import Path
import sys
import subprocess
current_dir = Path(__file__).resolve().parent
scripts = [
        # "FHO_fMRI_00B_smooth_20250627.py",
        # "FHO_fMRI_01_create_nuisance_regressors_file_20250724.py",
        "FHO_fMRI_01_GLM_firstlevel_compute_and_save_GLM_20250728.py",
        "FHO_fMRI_01_GLM_firstlevel_compute_contrasts_20250728.py",
]

    # Path to Conda env python exe
# conda_env_python = "C:/Users/bramb/anaconda3/envs/nilearn_env/python.exe"  # Replace with your environment's Python path
# conda_env_python = "/home/bramb/.conda/envs/Bram_nilearn/python.exe"

for script in scripts:
    script_path = current_dir/script
    # Run the script using the Python executable from the Conda environment
    print(f"running {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)