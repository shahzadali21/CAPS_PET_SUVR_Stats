#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module 2 — Convert GTMSeg MGZ to NIfTI and map to MNI using Clinica T1-linear affine.

Inputs (per session):
- CAPS_DIR/subjects/<sub>/<ses>/t1/freesurfer_cross_sectional/<sub>_<ses>/mri/gtmseg.mgz
- CAPS_DIR/subjects/<sub>/<ses>/t1_linear/*_space-MNI152NLin2009cSym_*_affine.mat
- CAPS_DIR/subjects/<sub>/<ses>/t1_linear/*_space-MNI152NLin2009cSym_*_T1w.nii.gz (reference)

Outputs (per session, saved in the FS subject mri/ folder):
- .../mri/gtmseg.nii.gz
- .../mri/gtmseg_on_mni_affine.nii.gz

Status TSV (written to <base_dir>/):
- logSubjects_spanorm_status.tsv   (single file with status, stage, message, paths)

Notes:
- Restart-safe: if outputs exist, skip unless --overwrite.
- Uses NearestNeighbor interpolation for label maps.
- Parallel-safe TSV logging using file lock.
"""

import os
import sys
import time
import argparse
import contextlib
import datetime
import re
from pathlib import Path
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.ants import ApplyTransforms

# pip install filelock
from filelock import FileLock


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


def choose_first_or_none(paths):
    paths = list(paths)
    return paths[0] if paths else None


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
        "fs_subject_id",
        "status",
        "stage",
        "message",
        "gtmseg_mgz",
        "out_nii",
        "out_on_mni",
        "affine_mat",
        "mni_t1",
    ]

    for c in columns:
        row.setdefault(c, "")

    with lock:
        file_exists = tsv_path.exists()
        with tsv_path.open("a") as f:
            if not file_exists:
                f.write("\t".join(columns) + "\n")
            f.write("\t".join(str(row[c]) for c in columns) + "\n")


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


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

            # FreeSurfer CAPS
            fs_parent = ses_dir / "t1" / "freesurfer_cross_sectional"
            if not fs_parent.exists():
                continue

            fs_subject_dirs = sorted(fs_parent.glob("sub-*_ses-*"))
            if not fs_subject_dirs:
                continue

            # Clinica t1_linear CAPS
            t1_linear_dir = ses_dir / "t1_linear"
            if not t1_linear_dir.exists():
                continue

            affine_candidates = sorted(t1_linear_dir.glob("*_space-MNI152NLin2009cSym_*_affine.mat"))
            mni_t1_candidates = sorted(t1_linear_dir.glob("*_space-MNI152NLin2009cSym_*_T1w.nii.gz"))
            affine_mat = choose_first_or_none(affine_candidates)
            mni_t1 = choose_first_or_none(mni_t1_candidates)

            if affine_mat is None or mni_t1 is None:
                continue

            # Add one entry per FS subject dir (usually exactly one)
            for fs_subject_dir in fs_subject_dirs:
                if not fs_subject_dir.is_dir():
                    continue

                gtmseg_mgz = fs_subject_dir / "mri" / "gtmseg.mgz"
                if not gtmseg_mgz.exists():
                    continue

                entries.append(
                    {
                        "sub_id": sub_dir.name,
                        "ses_id": ses_dir.name,
                        "fs_subject_dir": fs_subject_dir,
                        "gtmseg_mgz": gtmseg_mgz,
                        "affine_mat": affine_mat,
                        "mni_t1": mni_t1,
                    }
                )

    return entries


# ----------------------------
# Core processing
# ----------------------------
def spanorm_caps(entry, log_dir: Path, quiet: bool = True, overwrite: bool = False):
    fs_subject_dir = entry["fs_subject_dir"]
    gtmseg_mgz = entry["gtmseg_mgz"]
    affine_mat = entry["affine_mat"]
    mni_t1 = entry["mni_t1"]

    sub_id = entry.get("sub_id", "")
    ses_id = entry.get("ses_id", "")
    fs_id = fs_subject_dir.name  # sub-XXX_ses-YYY

    out_dir = fs_subject_dir / "mri"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nii = out_dir / "gtmseg.nii.gz"
    out_on_mni = out_dir / "gtmseg_on_mni_affine.nii.gz"

    status_tsv = log_dir / "logSubjects_spanorm_status.tsv"

    def log_status(status: str, stage: str, message: str):
        append_status_tsv(
            status_tsv,
            {
                "timestamp": now_iso(),
                "subject_id": sub_id,
                "session_id": ses_id,
                "fs_subject_id": fs_id,
                "status": status,
                "stage": stage,
                "message": message,
                "gtmseg_mgz": str(gtmseg_mgz),
                "out_nii": str(out_nii),
                "out_on_mni": str(out_on_mni),
                "affine_mat": str(affine_mat),
                "mni_t1": str(mni_t1),
            },
        )

    # Restart-safe skip
    if (out_nii.exists() and out_on_mni.exists()) and (not overwrite):
        msg = "outputs exist"
        print(f"[SKIP] {fs_id} | {msg}")
        log_status("SKIPPED", "precheck", msg)
        return True

    # Basic input checks
    if not gtmseg_mgz.exists():
        msg = "missing gtmseg.mgz"
        print(f"[WARN] {fs_id} | {msg}")
        log_status("WARNING", "precheck", msg)
        return False

    if not affine_mat.exists() or not mni_t1.exists():
        msg = "missing t1_linear affine/T1"
        print(f"[WARN] {fs_id} | {msg} | affine={affine_mat} | ref={mni_t1}")
        log_status("WARNING", "precheck", msg)
        return False

    # 1) Convert MGZ -> NIfTI
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
        msg = f"MRIConvert failed: {repr(e)}"
        print(f"[FAIL] {fs_id} | {msg}")
        log_status("FAILED", "MRIConvert", msg)
        return False

    if not out_nii.exists():
        msg = f"MRIConvert output not created: {out_nii}"
        print(f"[FAIL] {fs_id} | {msg}")
        log_status("FAILED", "MRIConvert", msg)
        return False

    # 2) Apply affine to MNI
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
        msg = f"ApplyTransforms failed: {repr(e)}"
        print(f"[FAIL] {fs_id} | {msg}")
        log_status("FAILED", "ApplyTransforms", msg)
        return False

    if not out_on_mni.exists():
        msg = f"ApplyTransforms output not created: {out_on_mni}"
        print(f"[FAIL] {fs_id} | {msg}")
        log_status("FAILED", "ApplyTransforms", msg)
        return False

    msg = "Spanorm affine complete"
    print(f"[OK] {fs_id} | {msg}")
    log_status("COMPLETED", "done", msg)
    return True


# ----------------------------
# Batch processing
# ----------------------------
def process_in_batches(entries, log_dir: Path, batch_size: int = 20, n_jobs=None, quiet: bool = True, overwrite: bool = False):
    total = len(entries)
    if total == 0:
        print("[WARN] No entries found (need gtmseg.mgz + t1_linear affine + MNI T1).")
        return

    if n_jobs is None:
        n_jobs = cpu_count()

    for i in range(0, total, batch_size):
        batch = entries[i : i + batch_size]
        Parallel(n_jobs=int(n_jobs))(
            delayed(spanorm_caps)(entry, log_dir, quiet, overwrite) for entry in batch
        )
        print(f"Batch {i // batch_size + 1} completed ({min(i + batch_size, total)}/{total})")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert GTMSeg MGZ->NIfTI and apply Clinica T1-linear affine to MNI (CAPS), with TSV status logging."
    )
    parser.add_argument(
        "--base_dir", type=str, default="/Volumes/Ali_X10Pro/ADNI",
        help="Base folder containing CAPS_DIR"
    )
    parser.add_argument(
        "--caps_name", type=str, default="CAPS_DIR",
        help="CAPS directory name inside base_dir (default: CAPS_DIR)"
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size (default: 20)")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs (default: cpu_count)")
    parser.add_argument("--no_quiet", action="store_true", help="Do not suppress tool outputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs even if present")
    parser.add_argument(
        "--freesurfer_home", type=str,
        default=os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/"),
        help="FreeSurfer home (default: $FREESURFER_HOME or /Applications/freesurfer/7.4.1/)"
    )
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name
    log_dir = base_dir

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
        overwrite=args.overwrite,
    )
    print(f"--- {time.time() - start_time:.2f} seconds (Spanorm affine total time) ---")
    print(f"[INFO] Status TSV: {log_dir / 'logSubjects_spanorm_status.tsv'}")


if __name__ == "__main__":
    main()
