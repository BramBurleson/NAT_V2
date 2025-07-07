import os
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import image, plotting, datasets
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from scipy.stats import norm

# ───────────────────────── USER SETTINGS ─────────────────────────────
# output_root = "/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI"
ROOT_DATASET         = Path(__file__).resolve().parent.parent.parent
output_root          = Path(ROOT_DATASET, 'results', 'fmri')
derivatives_folder   = Path(ROOT_DATASET, 'data', 'fmri', 'derivatives')
motioncorrection_dir = Path(ROOT_DATASET, 'data', 'motioncorrection')

subject_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] 
task = "nat"
# subjects = [

#     {
#         "id": "Subj1",
#         "bold": r"/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI/Subj1_Thibaud/sub-03_task-nat_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
#         "events": r"/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI/Subj1_Thibaud/ThibaudLocalizer-Blocks.tsv",
#         "confounds": r"/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI/Subj1_Thibaud/sub-03_task-nat_run-02_desc-confounds_timeseries.tsv",
#     },

#simply build a subjects thing for each.



TR = 1.5
HRF_MODEL = "spm"
SMOOTH_FWHM = 6
P_FIG = 0.05
T_FIG = norm.isf(P_FIG)

POS_NEG = {
    "Flags_minus_Faces": (
        ["Allo Flags", "Ego Flags", "Color Flags"],
        ["Allo Faces", "Ego Faces", "Color Faces"],
    ),
    "AlloFlags_minus_EgoFlags": (["Allo Flags"], ["Ego Flags"]),
    "AlloFlags_minus_ColorFlags": (["Allo Flags"], ["Color Flags"]),
    "EgoFlags_minus_ColorFlags": (["Ego Flags"], ["Color Flags"]),
    "AlloFaces_minus_EgoFaces": (["Allo Faces"], ["Ego Faces"]),
    "AlloFaces_minus_ColorFaces": (["Allo Faces"], ["Color Faces"]),
    "EgoFaces_minus_ColorFaces": (["Ego Faces"], ["Color Faces"]),
}

# ────────────────────────── HELPERS ─────────────────────────────────

def stamp(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def fit_first_level(sub):
    stamp(f"── First‑level START  | {sub['id']}")
    subj_dir = os.path.join(output_root, sub["id"])
    os.makedirs(subj_dir, exist_ok=True)

    stamp(f"Loading BOLD: {sub['bold']}")
    img = image.load_img(sub["bold"])
    stamp(f"Original shape: {img.shape}")
    img = image.smooth_img(img, SMOOTH_FWHM)
    stamp(f"Applied smoothing FWHM={SMOOTH_FWHM}")

    events = pd.read_table(sub["events"])
    n_scans = img.shape[-1]
    frame_times = np.arange(n_scans) * TR
    stamp(f"Events rows: {len(events)} | n_scans={n_scans}")

    conf = load_confounds(
        sub["bold"],
        strategy=("high_pass", "motion", "compcor", "global_signal"),
        motion="basic",
        compcor="anat_combined",
        n_compcor=10,
        global_signal="basic",
    )[0]
    stamp(f"Confounds shape: {conf.shape}")

    design = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=HRF_MODEL,
        drift_model=None,
        high_pass=None,
        add_regs=conf,
    )
    plotting.plot_design_matrix(design).figure.savefig(
        os.path.join(subj_dir, f"{sub['id']}_design.png"), dpi=120
    )

    glm = FirstLevelModel(t_r=TR, slice_time_ref=0.0, hrf_model=HRF_MODEL)
    glm = glm.fit(img, design_matrices=design)
    stamp("Model fit complete")

    cols = list(design.columns)
    contrast_vecs = {}
    for name, (pos, neg) in POS_NEG.items():
        vec = np.zeros(design.shape[1])
        for p in pos:
            vec[cols.index(p)] += 1
        for n in neg:
            vec[cols.index(n)] -= 1
        contrast_vecs[name] = vec
        stamp(f"Contrast '{name}': non‑zeros {np.flatnonzero(vec).tolist()}")

    out_z = {}
    for name, vec in contrast_vecs.items():
        stamp(f"Computing {name}")
        zmap = glm.compute_contrast(vec, output_type="z_score")
        zfile = os.path.join(subj_dir, f"{sub['id']}_{name}_z.nii.gz")
        zmap.to_filename(zfile)
        out_z[name] = zfile
        stamp(f"Saved z‑map → {zfile}")

        surf_png = os.path.join(subj_dir, f"{sub['id']}_{name}_surf.png")
        fig, _ = plotting.plot_img_on_surf(
            zmap,
            hemispheres=["left", "right"],
            views=["lateral", "medial"],
            threshold=T_FIG,
            colorbar=True,
            title=f"{sub['id']} – {name}",
            darkness=0.7,
        )
        fig.savefig(surf_png, dpi=300, bbox_inches="tight")
        plt.close(fig)          # keep memory usage low
        stamp(f"Surface figure saved: {surf_png}")

    stamp(f"── First‑level DONE   | {sub['id']}\n")
    return out_z


def second_level(contrast, z_paths):
    if len(z_paths) < 2:
        stamp(f"Skipping {contrast} – {len(z_paths)} map(s)")
        return

    stamp(f"── Second‑level START | {contrast}")
    grp_dir = os.path.join(output_root, "group")
    os.makedirs(grp_dir, exist_ok=True)

    design = pd.DataFrame({"intercept": np.ones(len(z_paths))})
    slm = SecondLevelModel(smoothing_fwhm=6)
    slm = slm.fit(z_paths, design_matrix=design)
    zmap = slm.compute_contrast(output_type="z_score")

    zfile = os.path.join(grp_dir, f"group_{contrast}_z.nii.gz")
    zmap.to_filename(zfile)
    stamp(f"Group z‑map saved: {zfile}")

    fig_png = os.path.join(grp_dir, f"group_{contrast}_axial.png")
    plotting.plot_stat_map(
        zmap,
        bg_img=datasets.load_mni152_template(),
        threshold=T_FIG,
        display_mode="z",
        cut_coords=8,
        black_bg=True,
        title=f"Group – {contrast}",
    ).savefig(fig_png, dpi=180)
    stamp(f"Group figure saved: {fig_png}\n")

    # -----------------------------------------
    surf_png = os.path.join(grp_dir, f"group_{contrast}_surf.png")
    fig, _ = plotting.plot_img_on_surf(
        zmap,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        threshold=T_FIG,
        colorbar=True,
        title=f"Group – {contrast}",
        darkness=0.7,
    )
    fig.savefig(surf_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    stamp(f"Surface figure saved: {surf_png}")
    # -----------------------------------------

# ───────────────────────────── MAIN ──────────────────────────────────
if __name__ == "__main__":
    stamp("===== Pipeline started =====")
    all_z = {c: [] for c in POS_NEG}

    for sub in subject_labels:
        try:
            z_map_dict = fit_first_level(sub)
            for c, pth in z_map_dict.items():
                all_z[c].append(pth)
        except Exception as e:
            stamp(f"ERROR in first‑level {sub['id']}: {e}")
            sys.exit(1)

    for c, paths in all_z.items():
        try:
            second_level(c, paths)
        except Exception as e:
            stamp(f"ERROR in second‑level {c}: {e}")
            sys.exit(1)

    stamp("===== Pipeline finished. Results in " + output_root)
