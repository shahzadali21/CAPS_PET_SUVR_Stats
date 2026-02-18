#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAPS PET Regional Statistics (NO normalization; uses Clinica SUVR outputs)
UPDATED — CAPS-safe + parallel + restart-safe + single TSV status logging
and robust seg->PET alignment via resampling (instead of origin-crop).

Inputs (per session):
- PET FDG SUVR:  <CAPS>/subjects/<sub>/<ses>/pet_linear/*trc-18FFDG*space-MNI152NLin2009cSym*suvr-pons2_pet.nii.gz
- PET AV45 SUVR: <CAPS>/subjects/<sub>/<ses>/pet_linear/*trc-18FAV45*space-MNI152NLin2009cSym*suvr-cerebellumPons2_pet.nii.gz
- Seg (MNI affine): <CAPS>/subjects/<sub>/<ses>/t1/freesurfer_cross_sectional/<sub>_<ses>/mri/gtmseg_on_mni_affine.nii.gz

Outputs (global, in BASE_DIR):
- PET_trc-<trc>_suvr-<suvr>_statistics.xlsx
  sheets: Mean, MeanGMM, Std, Sample_Size

Optional outputs (per session):
- Copies Clinica SUVR PET into:
  <CAPS>/subjects/<sub>/<ses>/pet_surface/PET_trc-<trc>_suvr-<suvr>_pet.nii.gz

Status TSV (in BASE_DIR):
- logSubjects_petstats_status_<TRACER>.tsv (parallel-safe with .lock)

Notes:
- Uses NearestNeighbor for segmentation resampling.
- Parallelizes per-session stats; Excel is written once at the end.
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

# robust resampling (part of nibabel)
from nibabel.processing import resample_from_to

# pip install filelock
from filelock import FileLock


DEFAULT_BASE_DIR = "/Volumes/Ali_X10Pro/ADNI"
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


# ----------------------------
# TSV status logging (parallel-safe)
# ----------------------------
def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def append_status_tsv(tsv_path: Path, row: dict):
    """
    Append one row to a TSV file safely (parallel-safe via file lock).
    """
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(tsv_path) + ".lock")

    columns = [
        "timestamp",
        "subject_id",
        "session_id",
        "status",
        "stage",
        "message",
        "pet_path",
        "seg_path",
        "pet_copy",
    ]
    for c in columns:
        row.setdefault(c, "")

    with lock:
        first_write = not tsv_path.exists()
        with tsv_path.open("a") as f:
            if first_write:
                f.write("\t".join(columns) + "\n")
            f.write("\t".join(str(row[c]) for c in columns) + "\n")


# ----------------------------
# PET stats utils
# ----------------------------
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


def resample_seg_to_pet(seg_img: nib.Nifti1Image, pet_img: nib.Nifti1Image) -> np.ndarray:
    """
    Robustly resample seg into PET grid using nearest neighbor.
    Returns seg data in PET space (same shape as pet).
    """
    seg_rs = resample_from_to(seg_img, pet_img, order=0)  # NN for labels
    seg_data = seg_rs.get_fdata(dtype="float32")
    # labels must be integers
    return np.rint(seg_data).astype(np.int32)


# ----------------------------
# Discovery
# ----------------------------
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

            entries.append(
                {
                    "sub": sub_dir.name,
                    "ses": ses_dir.name,
                    "ses_dir": ses_dir,
                    "pet_path": pet_path,
                    "seg_path": seg_path,
                }
            )

    return entries


# ----------------------------
# Per-session worker
# ----------------------------
def process_one_session(
    e: dict,
    lut: dict,
    trc: str,
    suvr_tag: str,
    pet_surface_copy: bool,
    overwrite_pet_copy: bool,
    status_tsv: Path,
):
    """
    Returns:
      subj_id, (Series meanGMM, Series mean, Series std, Series size)
    or:
      None if failed / skipped.
    """
    sub = e["sub"]
    ses = e["ses"]
    subj_id = f"{sub}_{ses}"

    pet_path = Path(e["pet_path"])
    seg_path = Path(e["seg_path"])
    ses_dir = Path(e["ses_dir"])

    pet_copy_path = ""
    def log(status, stage, msg):
        append_status_tsv(
            status_tsv,
            {
                "timestamp": _now_iso(),
                "subject_id": sub,
                "session_id": ses,
                "status": status,
                "stage": stage,
                "message": msg,
                "pet_path": str(pet_path),
                "seg_path": str(seg_path),
                "pet_copy": str(pet_copy_path) if pet_copy_path else "",
            },
        )

    # Basic checks
    if not pet_path.exists():
        log("FAILED", "precheck", f"Missing PET: {pet_path}")
        return None
    if not seg_path.exists():
        log("FAILED", "precheck", f"Missing seg: {seg_path}")
        return None

    try:
        pet_img = nib.load(str(pet_path))
        pet_data = pet_img.get_fdata(dtype="float32")
    except Exception as ex:
        log("FAILED", "load_pet", f"Failed to load PET: {repr(ex)}")
        return None

    try:
        seg_img = nib.load(str(seg_path))
    except Exception as ex:
        log("FAILED", "load_seg", f"Failed to load seg: {repr(ex)}")
        return None

    # Robust alignment: resample seg to PET grid
    try:
        seg_pet = resample_seg_to_pet(seg_img, pet_img)
    except Exception as ex:
        log("FAILED", "resample_seg", f"Failed seg->PET resample: {repr(ex)}")
        return None

    # Optional: create pet_surface copy
    if pet_surface_copy:
        out_dir = ses_dir / "pet_surface"
        out_dir.mkdir(parents=True, exist_ok=True)
        pet_copy = out_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_pet.nii.gz"
        pet_copy_path = pet_copy
        try:
            if overwrite_pet_copy or (not pet_copy.exists()):
                nib.save(nib.Nifti1Image(pet_data, affine=pet_img.affine), str(pet_copy))
        except Exception as ex:
            # Non-fatal
            log("WARNING", "pet_copy", f"Could not write pet_surface copy: {repr(ex)}")

    # VOI stats
    labels = np.unique(seg_pet).astype(np.int32)
    vois = [lab for lab in labels if lab != 0]

    meanGMM_list, mean_list, std_list, size_list, idx_names = [], [], [], [], []

    for voi in vois:
        mask = (seg_pet == voi)
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

    if len(idx_names) == 0:
        log("FAILED", "stats", "No valid VOIs after masking (empty stats).")
        return None

    s_meanGMM = pd.Series(meanGMM_list, index=idx_names, name=subj_id)
    s_mean    = pd.Series(mean_list,    index=idx_names, name=subj_id)
    s_std     = pd.Series(std_list,     index=idx_names, name=subj_id)
    s_size    = pd.Series(size_list,    index=idx_names, name=subj_id)

    log("COMPLETED", "done", f"OK | PET={pet_path.name} | n_vois={len(idx_names)}")
    return subj_id, s_meanGMM, s_mean, s_std, s_size


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute regional stats from Clinica SUVR PET (no normalization) using GTMSeg labels.")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--caps_name", type=str, default=DEFAULT_CAPS_NAME)
    parser.add_argument("--tracer", type=str, required=True, choices=["FDG", "AV45"])
    parser.add_argument("--freesurfer_home", type=str, default=DEFAULT_FREESURFER_HOME)

    # parallel
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs (default: cpu_count)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for progress printing (default: 50)")

    # outputs
    parser.add_argument("--overwrite_excel", action="store_true", help="Overwrite Excel if exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-session pet_surface PET copy.")
    parser.add_argument("--pet_surface_copy", action="store_true", default=False, help="Copy PET into pet_surface/ (default: False).")


    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name

    cfg = TRACER_CONFIG[args.tracer]
    trc = cfg["trc"]
    suvr_tag = cfg["suvr_tag"]
    pet_glob = cfg["pet_glob"]

    excel_out = base_dir / f"PET_trc-{trc}_suvr-{suvr_tag}_statistics.xlsx"
    status_tsv = base_dir / f"logSubjects_petstats_status_{args.tracer}.tsv"

    # LUT
    fs_lut_path = Path(args.freesurfer_home).expanduser().resolve() / "FreeSurferColorLUT.txt"
    lut = load_freesurfer_lut(fs_lut_path)
    if lut:
        print(f"[INFO] Loaded FreeSurfer LUT: {fs_lut_path}")
    else:
        print(f"[WARN] Could not load LUT at: {fs_lut_path} (names will be 'unknown')")

    # discovery
    entries = find_caps_sessions(caps_dir, pet_glob=pet_glob)
    print(f"[INFO] Tracer={args.tracer} | Found {len(entries)} sessions with matching Clinica SUVR PET + GTMSeg seg.")

    if len(entries) == 0:
        print("[WARN] No sessions found. Exiting.")
        return

    # excel overwrite policy
    if excel_out.exists() and (not args.overwrite_excel):
        print(f"[ERROR] Excel already exists: {excel_out}")
        print("        Use --overwrite_excel to overwrite.")
        return

    if args.n_jobs is None:
        args.n_jobs = cpu_count()

    start_time = time.time()

    # run in parallel
    results = []
    total = len(entries)

    # chunked progress: run Parallel on full list, but print progress per batch in main
    for i in range(0, total, args.batch_size):
        batch = entries[i:i + args.batch_size]
        batch_results = Parallel(n_jobs=int(args.n_jobs))(
            delayed(process_one_session)(
                e,
                lut,
                trc,
                suvr_tag,
                args.pet_surface_copy,
                args.overwrite,
                status_tsv,
            )
            for e in batch
        )
        results.extend([r for r in batch_results if r is not None])
        print(f"[INFO] Batch {i // args.batch_size + 1} done ({min(i + args.batch_size, total)}/{total}) | ok={len(results)}")

    if len(results) == 0:
        print("[ERROR] No successful sessions. Check TSV for failures:")
        print(f"        {status_tsv}")
        return

    # aggregate to DataFrames
    meanGMM_series = [r[1] for r in results]
    mean_series    = [r[2] for r in results]
    std_series     = [r[3] for r in results]
    size_series    = [r[4] for r in results]

    df_meanGMM = pd.concat(meanGMM_series, axis=1)
    df_mean    = pd.concat(mean_series, axis=1)
    df_std     = pd.concat(std_series, axis=1)
    df_size    = pd.concat(size_series, axis=1)

    # write excel (subjects as rows)
    with pd.ExcelWriter(str(excel_out)) as writer:
        df_mean.transpose().to_excel(writer, sheet_name="Mean")
        df_meanGMM.transpose().to_excel(writer, sheet_name="MeanGMM")
        df_std.transpose().to_excel(writer, sheet_name="Std")
        df_size.transpose().to_excel(writer, sheet_name="Sample_Size")

    print(f"\n[INFO] Saved Excel: {excel_out}")
    print(f"[INFO] Status TSV: {status_tsv}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
