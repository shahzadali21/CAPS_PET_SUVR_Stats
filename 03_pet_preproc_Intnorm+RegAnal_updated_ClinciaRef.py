#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAPS PET Regional Statistics (NO normalization; uses Clinica SUVR outputs)

Inputs (per session):
- PET FDG SUVR:  pet_linear/*trc-18FFDG*space-MNI152NLin2009cSym*suvr-pons2_pet.nii.gz
- PET AV45 SUVR: pet_linear/*trc-18FAV45*space-MNI152NLin2009cSym*suvr-cerebellumPons2_pet.nii.gz
- Seg: t1/freesurfer_cross_sectional/<sub>_<ses>/mri/gtmseg_on_mni_affine.nii.gz

Outputs (global, in BASE_DIR):
- PET_trc-<trc>_suvr-<suvr>_statistics.xlsx
  sheets: Mean, MeanGMM, Std, Sample_Size

Optional outputs (per session):
- Copies the Clinica SUVR PET into:
  CAPS_DIR/subjects/<sub>/<ses>/pet_surface/PET_trc-<trc>_suvr-<suvr>_pet.nii.gz
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.mixture import GaussianMixture


DEFAULT_BASE_DIR = "/Users/shahzadali/Desktop/ADNI"
DEFAULT_CAPS_NAME = "CAPS_DIR"
DEFAULT_FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/")


TRACER_CONFIG = {
    "FDG": {
        "trc": "18FFDG",
        "pet_glob": "*trc-18FFDG*space-MNI152NLin2009cSym*suvr-pons2_pet.nii.gz",
        "suvr_tag": "pons2",
    },
    "AV45": {
        "trc": "18FAV45",
        "pet_glob": "*trc-18FAV45*space-MNI152NLin2009cSym*suvr-cerebellumPons2_pet.nii.gz",
        "suvr_tag": "cerebellumPons2",
    },
}


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
    return f"{label_id}_{lut.get(label_id, 'unknown')}"


def find_caps_sessions(caps_dir: Path, pet_glob: str):
    """
    Find subject-sessions having:
      - matching PET SUVR file in pet_linear/
      - gtmseg_on_mni_affine.nii.gz in FS folder
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
                continue
            pet_path = pet_candidates[0]

            fs_parent = ses_dir / "t1" / "freesurfer_cross_sectional"
            if not fs_parent.exists():
                continue

            fs_subject_dirs = sorted(fs_parent.glob("sub-*_ses-*"))
            if not fs_subject_dirs:
                continue

            # Prefer the FS folder that matches this ses_dir if multiple exist
            matching = [d for d in fs_subject_dirs if d.name.endswith(f"_{ses_dir.name}")]
            fs_subject_dir = matching[0] if matching else fs_subject_dirs[0]

            seg_path = fs_subject_dir / "mri" / "gtmseg_on_mni_affine.nii.gz"
            if not seg_path.exists():
                continue

            entries.append({
                "sub": sub_dir.name,
                "ses": ses_dir.name,
                "ses_dir": ses_dir,
                "pet_path": pet_path,
                "seg_path": seg_path,
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description="Compute regional stats from Clinica SUVR PET (no normalization).")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--caps_name", type=str, default=DEFAULT_CAPS_NAME)
    parser.add_argument("--tracer", type=str, required=True, choices=["FDG", "AV45"])
    parser.add_argument("--freesurfer_home", type=str, default=DEFAULT_FREESURFER_HOME)

    # minor but useful options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-session pet_surface PET copy.")
    parser.add_argument("--no_pet_surface_copy", action="store_true", help="Do not create a PET copy in pet_surface/.")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name

    cfg = TRACER_CONFIG[args.tracer]
    trc = cfg["trc"]
    suvr_tag = cfg["suvr_tag"]
    pet_glob = cfg["pet_glob"]

    log_file = base_dir / f"logSubjects_petstats_{args.tracer}.txt"
    warning_log = base_dir / f"logSubjects_petstats_{args.tracer}_warning.txt"
    excel_out = base_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_statistics.xlsx"

    fs_lut_path = Path(args.freesurfer_home).expanduser().resolve() / "FreeSurferColorLUT.txt"
    lut = load_freesurfer_lut(fs_lut_path)
    if lut:
        print(f"[INFO] Loaded FreeSurfer LUT: {fs_lut_path}")
    else:
        print(f"[WARN] Could not load LUT at: {fs_lut_path} (names will be 'unknown')")

    entries = find_caps_sessions(caps_dir, pet_glob=pet_glob)
    print(f"[INFO] Tracer={args.tracer} | Found {len(entries)} sessions with matching Clinica SUVR PET + GTMSeg.")

    df_meanGMM_all, df_mean_all, df_std_all, df_size_all = (
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    start_time = time.time()

    for e in entries:
        subj_id = f"{e['sub']}_{e['ses']}"
        print(f"\n#### Processing {subj_id} ####")

        pet_path = Path(e["pet_path"])
        seg_path = Path(e["seg_path"])
        ses_dir = Path(e["ses_dir"])

        if not pet_path.exists():
            log_message(warning_log, f"Missing PET for {subj_id}: {pet_path}")
            continue
        if not seg_path.exists():
            log_message(warning_log, f"Missing seg for {subj_id}: {seg_path}")
            continue

        # Load PET (already SUVR) and segmentation
        pet_img = nib.load(str(pet_path))
        pet_data = pet_img.get_fdata(dtype="float32")

        seg_img = nib.load(str(seg_path))
        seg_data = seg_img.get_fdata(dtype="float32")

        # Align shapes if needed (crop)
        if seg_data.shape != pet_data.shape:
            seg_crop = crop_to_pet_space(seg_data, seg_img.affine, pet_data.shape, pet_img.affine)
        else:
            seg_crop = seg_data

        # Optional: create pet_surface copy (same data, just a canonical filename)
        if not args.no_pet_surface_copy:
            out_dir = ses_dir / "pet_surface"
            out_dir.mkdir(parents=True, exist_ok=True)

            pet_copy = out_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_pet.nii.gz"
            if args.overwrite or (not pet_copy.exists()):
                nib.save(nib.Nifti1Image(pet_data, affine=pet_img.affine), str(pet_copy))

        # all regions except background
        labels = np.unique(seg_crop).astype(int)
        VOIs = [lab for lab in labels if lab != 0]

        meanGMM_list, mean_list, std_list, size_list, idx_names = [], [], [], [], []

        for voi in VOIs:
            mask = (seg_crop == voi)
            values = pet_data[mask]
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

        log_message(log_file, f"OK {subj_id} | PET={pet_path.name}")

    with pd.ExcelWriter(str(excel_out)) as writer:
        df_mean_all.transpose().to_excel(writer, sheet_name="Mean")
        df_meanGMM_all.transpose().to_excel(writer, sheet_name="MeanGMM")
        df_std_all.transpose().to_excel(writer, sheet_name="Std")
        df_size_all.transpose().to_excel(writer, sheet_name="Sample_Size")

    print(f"\n[INFO] Saved Excel: {excel_out}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
