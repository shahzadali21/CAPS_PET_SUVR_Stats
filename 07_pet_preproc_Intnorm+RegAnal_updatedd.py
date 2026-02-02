#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAPS PET regional stats using GTMSeg-on-MNI (affine), flexible for FDG and AV45.

- FDG: looks for trc-18FFDG + suvr-pons2_pet
       normalization mask: pons (label 174)
- AV45: looks for trc-18FAV45 + suvr-cerebellumPons2_pet
        normalization mask: cerebellum (labels 7,8,46,47)

Skips sessions without the requested PET file.

Outputs:
- per session: CAPS_DIR/subjects/<sub>/<ses>/pet_surface/<pet_norm_name>.nii.gz
- global Excel:
  PET_trc-<label>_suvr-<suvr>_statistics.xlsx
  sheets: Mean, MeanGMM, Std, Sample_Size
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import mahotas
from sklearn.mixture import GaussianMixture


# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_BASE_DIR = "/Users/shahzadali/Desktop/ADNI"
DEFAULT_CAPS_NAME = "CAPS_DIR"
DEFAULT_FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/")
# -----------------------------------------------


# ---------------- Utility ----------------
def log_message(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")


def robust_mode_GMM(values: np.ndarray) -> float:
    """
    Robust mode estimate using 2-component GMM.
    Safe for small n:
      - n == 0 -> NaN
      - n == 1 -> that value
      - 2 <= n < 10 -> median
      - n >= 10 -> 2-component GMM, return mean of higher-uptake Gaussian
    """
    v = np.asarray(values, dtype=np.float32).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(v[0])
    if n < 10:
        return float(np.median(v))

    X = v.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(X)
    means = gmm.means_.flatten()
    return float(means[np.argmax(means)])


def erode_mask(mask: np.ndarray) -> np.ndarray:
    """Remove border voxels (1-voxel erosion)."""
    return np.bitwise_xor(mask, mahotas.bwperim(mask))


def get_normalization_mask(seg_crop: np.ndarray, target: str) -> np.ndarray:
    """
    Returns a normalization mask given target:
    - 'pons' uses label 174
    - 'cb' uses labels 7,8,46,47 (cerebellum WM+cortex)
    - 'wm' uses labels 2,41
    """
    if target == "pons":
        pons = np.isin(seg_crop, [174])
        return erode_mask(pons)

    if target == "cb":
        cb = np.isin(seg_crop, [7, 8, 46, 47])
        return erode_mask(cb)

    if target == "wm":
        wm = np.isin(seg_crop, [2, 41])
        return erode_mask(wm)

    raise ValueError(f"Unsupported normalization target: {target}")


def crop_to_pet_space(seg_array, seg_affine, pet_shape, pet_affine):
    """
    Crop segmentation to PET space using affine origins.
    Works when differences are mainly origin shifts (same orientation).
    """
    seg_origin = np.round(seg_affine[:3, 3], 2)
    pet_origin = np.round(pet_affine[:3, 3], 2)
    delta = np.abs(seg_origin - pet_origin).astype(int)

    return seg_array[
        delta[0]:delta[0] + pet_shape[0],
        delta[1]:delta[1] + pet_shape[1],
        delta[2]:delta[2] + pet_shape[2],
    ]


def load_freesurfer_lut(lut_path: Path) -> dict:
    """Parse FreeSurferColorLUT.txt and return {label_id: label_name}."""
    lut = {}
    if not lut_path.exists():
        return lut

    with lut_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lab = int(parts[0])
            except ValueError:
                continue
            lut[lab] = parts[1]
    return lut


def label_id_to_name(label_id: int, lut: dict) -> str:
    """Excel-safe row label: '<id>_<name>'."""
    return f"{label_id}_{lut.get(label_id, 'unknown')}"


# ---------------- Tracer config ----------------
TRACER_CONFIG = {
    "FDG": {
        "trc": "18FFDG",
        "pet_glob": "*trc-18FFDG*space-MNI152NLin2009cSym*pons2_pet.nii.gz",
        "norm_target": "pons",
        "suvr_tag": "pons2",
    },
    "AV45": {
        "trc": "18FAV45",
        "pet_glob": "*trc-18FAV45*space-MNI152NLin2009cSym*cerebellumPons2_pet.nii.gz",
        "norm_target": "cb",
        "suvr_tag": "cerebellumPons2",
    },
}


def find_caps_sessions(caps_dir: Path, pet_glob: str):
    """
    Find sessions that have:
      - requested tracer PET in pet_linear matching pet_glob
      - GTMSeg on MNI (affine) in freesurfer_cross_sectional
    Returns entries list; missing PET => skip.
    """
    entries = []
    subjects_root = caps_dir / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"Missing CAPS subjects folder: {subjects_root}")

    for sub_dir in sorted(subjects_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue

            pet_linear_dir = ses_dir / "pet_linear"
            if not pet_linear_dir.exists():
                continue

            pet_candidates = sorted(pet_linear_dir.glob(pet_glob))
            if not pet_candidates:
                # no requested PET for this session
                continue
            pet_path = pet_candidates[0]

            fs_parent = ses_dir / "t1" / "freesurfer_cross_sectional"
            if not fs_parent.exists():
                continue

            fs_subject_dirs = sorted(fs_parent.glob("sub-*_ses-*"))
            if not fs_subject_dirs:
                continue
            fs_subject_dir = fs_subject_dirs[0]

            seg_path = fs_subject_dir / "mri" / "gtmseg_on_mni_affine.nii.gz"
            if not seg_path.exists():
                continue

            out_dir = ses_dir / "pet_surface"

            entries.append({
                "sub": sub_dir.name,
                "ses": ses_dir.name,
                "ses_dir": ses_dir,
                "pet_path": pet_path,
                "seg_path": seg_path,
                "out_dir": out_dir,
            })

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base folder containing CAPS_DIR")
    parser.add_argument("--caps_name", type=str, default=DEFAULT_CAPS_NAME, help="CAPS directory name")
    parser.add_argument("--tracer", type=str, required=True, choices=["FDG", "AV45"], help="Tracer type")
    parser.add_argument("--freesurfer_home", type=str, default=DEFAULT_FREESURFER_HOME, help="FreeSurfer home path")
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name

    cfg = TRACER_CONFIG[args.tracer]
    pet_glob = cfg["pet_glob"]
    norm_target = cfg["norm_target"]
    trc = cfg["trc"]
    suvr_tag = cfg["suvr_tag"]

    # Logs + outputs
    log_file = base_dir / f"logSubjects_petstats_{args.tracer}.txt"
    warning_log = base_dir / f"logSubjects_petstats_{args.tracer}_warning.txt"
    excel_out = base_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_statistics.xlsx"

    # LUT
    fs_lut_path = Path(args.freesurfer_home) / "FreeSurferColorLUT.txt"
    lut = load_freesurfer_lut(fs_lut_path)
    if lut:
        print(f"[INFO] Loaded FreeSurfer LUT: {fs_lut_path}")
    else:
        print(f"[WARN] Could not load LUT at: {fs_lut_path} (names will be 'unknown')")

    entries = find_caps_sessions(caps_dir, pet_glob=pet_glob)
    print(f"[INFO] Tracer={args.tracer} | Found {len(entries)} sessions with matching PET + GTMSeg-on-MNI.")

    df_meanGMM_all = pd.DataFrame()
    df_mean_all = pd.DataFrame()
    df_std_all = pd.DataFrame()
    df_size_all = pd.DataFrame()

    start_time = time.time()

    for e in entries:
        subj_id = f"{e['sub']}_{e['ses']}"
        print(f"\n#### Processing {subj_id} ####")

        pet_path = e["pet_path"]
        seg_path = e["seg_path"]
        out_dir = e["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if not seg_path.exists():
            log_message(warning_log, f"Segmentation missing for {subj_id}: {seg_path}")
            continue
        if not pet_path.exists():
            log_message(warning_log, f"PET missing for {subj_id}: {pet_path}")
            continue

        # Load PET + seg
        pet_img = nib.load(str(pet_path))
        pet_data = pet_img.get_fdata(dtype="float32")

        seg_img = nib.load(str(seg_path))
        seg_data = seg_img.get_fdata(dtype="float32")

        # Crop seg to PET if needed
        if seg_data.shape != pet_data.shape:
            seg_crop = crop_to_pet_space(seg_data, seg_img.affine, pet_data.shape, pet_img.affine)
        else:
            seg_crop = seg_data

        # Normalization
        norm_mask = get_normalization_mask(seg_crop, norm_target)
        if np.count_nonzero(norm_mask) == 0:
            log_message(warning_log, f"Empty norm mask for {subj_id} (target={norm_target})")
            continue

        # Your PET may already be SUVR-normalized; we still allow re-normalization.
        norm_value = robust_mode_GMM(pet_data[norm_mask])
        if not np.isfinite(norm_value) or norm_value == 0:
            log_message(warning_log, f"Bad norm_value for {subj_id}: {norm_value}")
            continue

        pet_norm = pet_data / norm_value

        # Save normalized PET under pet_surface
        # norm_pet_path = out_dir / f"PETonMNI_norm_{norm_target}.nii.gz"
        norm_pet_path = out_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_norm-{norm_target}.nii.gz"
        nib.save(nib.Nifti1Image(pet_norm, affine=pet_img.affine), str(norm_pet_path))

        # Regional stats: keep all regions, exclude only 0
        labels = np.unique(seg_crop).astype(int)
        VOIs = [lab for lab in labels if lab != 0]

        meanGMM_list, mean_list, std_list, size_list, idx_names = [], [], [], [], []

        for voi in VOIs:
            mask = (seg_crop == voi)
            values = pet_norm[mask]
            if values.size == 0:
                continue

            stat_gmm = robust_mode_GMM(values)
            if not np.isfinite(stat_gmm):
                continue

            idx_names.append(label_id_to_name(int(voi), lut))
            meanGMM_list.append(stat_gmm)
            mean_list.append(float(np.mean(values)))
            std_list.append(float(np.std(values)))
            size_list.append(int(np.count_nonzero(mask)))

        col = pd.MultiIndex.from_product([[subj_id]])
        df_meanGMM_all = pd.concat([df_meanGMM_all, pd.DataFrame(meanGMM_list, index=idx_names, columns=col)], axis=1)
        df_mean_all    = pd.concat([df_mean_all,    pd.DataFrame(mean_list,    index=idx_names, columns=col)], axis=1)
        df_std_all     = pd.concat([df_std_all,     pd.DataFrame(std_list,     index=idx_names, columns=col)], axis=1)
        df_size_all    = pd.concat([df_size_all,    pd.DataFrame(size_list,    index=idx_names, columns=col)], axis=1)

        log_message(log_file, f"OK {subj_id} | PET={pet_path.name} | OUT={norm_pet_path}")

    # Save Excel
    with pd.ExcelWriter(str(excel_out)) as writer:
        df_mean_all.transpose().to_excel(writer, sheet_name="Mean")
        df_meanGMM_all.transpose().to_excel(writer, sheet_name="MeanGMM")
        df_std_all.transpose().to_excel(writer, sheet_name="Std")
        df_size_all.transpose().to_excel(writer, sheet_name="Sample_Size")

    print(f"\n[INFO] Saved Excel: {excel_out}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
