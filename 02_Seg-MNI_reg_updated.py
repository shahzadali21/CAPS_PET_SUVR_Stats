#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module 2 — Convert GTMSeg MGZ to NIfTI and map to MNI using Clinica T1-linear affine.

Inputs (per session):
- CAPS_DIR/subjects/<sub>/<ses>/t1/freesurfer_cross_sectional/<sub>_<ses>/mri/gtmseg.mgz
- CAPS_DIR/subjects/<sub>/<ses>/t1_linear/*_space-MNI152NLin2009cSym_*_affine.mat
- CAPS_DIR/subjects/<sub>/<ses>/t1_linear/*_space-MNI152NLin2009cSym_*_T1w.nii.gz  (reference)

Outputs (per session, saved in the FS subject mri/ folder):
- .../mri/gtmseg.nii.gz
- .../mri/gtmseg_on_mni_affine.nii.gz

Logs (written to <base_dir>/):
- logSubjects_spanorm.txt
- logSubjects_spanorm_failed.txt
"""

import os
import sys
import time
import argparse
import contextlib
from pathlib import Path
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.ants import ApplyTransforms


# ----------------------------
# Utils
# ----------------------------
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Completely suppress stdout/stderr (file descriptor level)."""
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def append_log(log_path: Path, msg: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(msg + "\n")


# ----------------------------
# Discovery: find subject-sessions that have FS + T1-linear
# ----------------------------
def find_caps_entries(caps_dir: Path):
    """
    Returns list of entries with:
      - sub_id, ses_id
      - fs_subject_dir: .../t1/freesurfer_cross_sectional/sub-*_ses-*
      - gtmseg_mgz: .../mri/gtmseg.mgz
      - affine_mat: .../t1_linear/*_space-MNI152NLin2009cSym_*_affine.mat
      - mni_t1: .../t1_linear/*_space-MNI152NLin2009cSym_*_T1w.nii.gz
    """
    subjects_root = caps_dir / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"CAPS subjects folder not found: {subjects_root}")

    entries = []

    for sub_dir in sorted(subjects_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue

            fs_parent = ses_dir / "t1" / "freesurfer_cross_sectional"
            if not fs_parent.exists():
                continue

            fs_subject_dirs = sorted(fs_parent.glob("sub-*_ses-*"))
            if not fs_subject_dirs:
                continue

            t1_linear_dir = ses_dir / "t1_linear"
            if not t1_linear_dir.exists():
                continue

            affine_candidates = sorted(t1_linear_dir.glob("*_space-MNI152NLin2009cSym_*_affine.mat"))
            mni_t1_candidates = sorted(t1_linear_dir.glob("*_space-MNI152NLin2009cSym_*_T1w.nii.gz"))
            if not affine_candidates or not mni_t1_candidates:
                continue

            affine_mat = affine_candidates[0]
            mni_t1 = mni_t1_candidates[0]

            for fs_subject_dir in fs_subject_dirs:
                if not fs_subject_dir.is_dir():
                    continue

                gtmseg_mgz = fs_subject_dir / "mri" / "gtmseg.mgz"
                if not gtmseg_mgz.exists():
                    continue

                entries.append({
                    "sub_id": sub_dir.name,
                    "ses_id": ses_dir.name,
                    "fs_subject_dir": fs_subject_dir,
                    "gtmseg_mgz": gtmseg_mgz,
                    "affine_mat": affine_mat,
                    "mni_t1": mni_t1,
                })

    return entries


# ----------------------------
# Core processing: MGZ -> NIfTI -> Apply affine to MNI
# ----------------------------
def spanorm_caps(entry, log_dir: Path, quiet: bool = True):
    fs_subject_dir = entry["fs_subject_dir"]
    gtmseg_mgz = entry["gtmseg_mgz"]
    affine_mat = entry["affine_mat"]
    mni_t1 = entry["mni_t1"]

    out_dir = fs_subject_dir / "mri"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nii = out_dir / "gtmseg.nii.gz"
    out_on_mni = out_dir / "gtmseg_on_mni_affine.nii.gz"

    subject_tag = fs_subject_dir.name  # e.g., sub-XXX_ses-YYY

    # Convert gtmseg.mgz -> gtmseg.nii.gz
    try:
        segconv = MRIConvert()
        segconv.inputs.in_file = str(gtmseg_mgz)
        segconv.inputs.out_file = str(out_nii)
        segconv.inputs.out_type = "niigz"

        if quiet:
            with suppress_stdout_stderr():
                segconv.run()
        else:
            segconv.run()

    except Exception as e:
        msg = f"[FAIL][MRIConvert] {subject_tag} | {repr(e)}"
        print(msg)
        append_log(log_dir / "logSubjects_spanorm_failed.txt", msg)
        return False

    # Apply affine transform to move segmentation into MNI space
    try:
        at_seg = ApplyTransforms()
        at_seg.inputs.input_image = str(out_nii)
        at_seg.inputs.reference_image = str(mni_t1)
        at_seg.inputs.output_image = str(out_on_mni)

        # Clinica T1-linear provides affine only
        at_seg.inputs.transforms = [str(affine_mat)]
        at_seg.inputs.interpolation = "NearestNeighbor"

        if quiet:
            with suppress_stdout_stderr():
                at_seg.run()
        else:
            at_seg.run()

    except Exception as e:
        msg = f"[FAIL][ApplyTransforms] {subject_tag} | {repr(e)}"
        print(msg)
        append_log(log_dir / "logSubjects_spanorm_failed.txt", msg)
        return False

    msg = f"[OK] Spanorm affine complete: {subject_tag}"
    print(msg)
    append_log(log_dir / "logSubjects_spanorm.txt", msg)
    return True


# ----------------------------
# Batch processing
# ----------------------------
def process_in_batches(entries, log_dir: Path, batch_size: int = 5, n_jobs=None, quiet: bool = True):
    total = len(entries)
    if total == 0:
        print("[WARN] No entries found (need gtmseg.mgz + t1_linear affine + MNI T1).")
        return

    if n_jobs is None:
        n_jobs = cpu_count()

    for i in range(0, total, batch_size):
        batch = entries[i:i + batch_size]
        Parallel(n_jobs=int(n_jobs))(
            delayed(spanorm_caps)(entry, log_dir, quiet) for entry in batch
        )
        print(f"Batch {i // batch_size + 1} completed ({min(i + batch_size, total)}/{total})")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply Clinica T1-linear affine to GTMSeg outputs (CAPS)."
    )
    parser.add_argument(
        "--base_dir", type=str, default="/Users/shahzadali/Desktop/ADNI",
        help="Base folder containing CAPS_DIR (default: /Users/shahzadali/Desktop/ADNI)"
    )
    parser.add_argument(
        "--caps_name", type=str, default="CAPS_DIR",
        help="CAPS directory name inside base_dir (default: CAPS_DIR)"
    )
    parser.add_argument("--batch_size", type=int, default=7, help="Batch size (default: 5)")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs (default: cpu_count)")
    parser.add_argument("--no_quiet", action="store_true", help="Do not suppress tool outputs")
    parser.add_argument(
        "--freesurfer_home", type=str,
        default=os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/"),
        help="FreeSurfer home (default: $FREESURFER_HOME or /Applications/freesurfer/7.4.1/)"
    )
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name
    log_dir = base_dir  # logs in base_dir

    os.environ["FREESURFER_HOME"] = str(Path(args.freesurfer_home).expanduser().resolve())

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] CAPS dir: {caps_dir}")
    print(f"[INFO] FREESURFER_HOME: {os.environ['FREESURFER_HOME']}")

    entries = find_caps_entries(caps_dir)
    print(f"[INFO] Found {len(entries)} entries with gtmseg.mgz + t1_linear affine + MNI T1")

    start_time = time.time()
    process_in_batches(
        entries,
        log_dir=log_dir,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        quiet=(not args.no_quiet),
    )
    print(f"--- {time.time() - start_time:.2f} seconds (Spanorm affine total time) ---")


if __name__ == "__main__":
    main()
