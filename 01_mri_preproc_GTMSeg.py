#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module 1 — Run FreeSurfer PETSurfer GTMSeg on CAPS FreeSurfer outputs.

CAPS discovery pattern:
  CAPS_DIR/subjects/sub-*/ses-*/t1/freesurfer_cross_sectional/sub-*_ses-*/mri/aseg.mgz

Inputs:
- CAPS directory with FreeSurfer cross-sectional outputs (must include aseg.mgz)

Outputs:
- GTMSeg outputs are written by FreeSurfer inside each FS subject folder:
  .../t1/freesurfer_cross_sectional/<sub>_<ses>/mri/gtmseg.mgz
- Logs are written to <base_dir>/:
  logSubjects_gtmseg.txt
  logSubjects_gtmseg_warning.txt
  logSubjects_gtmseg_failed.txt
  logSubjects_failed_once.txt
  logSubjects_failed_twice.txt

Notes:
- Retry logic uses recon-all-status.log (checks recon-all completion, not GTMSeg completion).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from nipype.interfaces.freesurfer import petsurfer


# ----------------------------
# Discovery: find FS folders in CAPS
# ----------------------------
def find_caps_freesurfer_subjects(caps_dir: Path):
    """
    Returns a list of dict entries, each describing one FS subject-session folder:
      - fs_subject_id: e.g., sub-ADNI002S2010_ses-M000
      - subjects_dir:  e.g., .../t1/freesurfer_cross_sectional
      - fs_subject_dir: e.g., .../t1/freesurfer_cross_sectional/sub-ADNI002S2010_ses-M000
    """
    subjects_root = caps_dir / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"CAPS subjects folder not found: {subjects_root}")

    results = []

    # CAPS pattern:
    # CAPS_DIR/subjects/sub-XXX/ses-YYY/t1/freesurfer_cross_sectional/sub-XXX_ses-YYY/mri/aseg.mgz
    for sub_dir in sorted(subjects_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            fs_parent = ses_dir / "t1" / "freesurfer_cross_sectional"
            if not fs_parent.exists():
                continue

            for fs_subject_dir in sorted(fs_parent.glob("sub-*_ses-*")):
                if not fs_subject_dir.is_dir():
                    continue

                aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
                if not aseg_path.exists():
                    continue

                results.append({
                    "fs_subject_id": fs_subject_dir.name,
                    "subjects_dir": fs_parent,
                    "fs_subject_dir": fs_subject_dir,
                })

    return results


# ----------------------------
# Logging helpers
# ----------------------------
def append_log(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


# ----------------------------
# GTMSeg runner
# ----------------------------
def run_gtmseg(entry, log_dir: Path):
    """
    Run GTMSeg for one FS subject-session entry.
    """
    fs_subject_id = entry["fs_subject_id"]
    subjects_dir = entry["subjects_dir"]
    fs_subject_dir = entry["fs_subject_dir"]

    aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
    if not aseg_path.exists():
        msg = f"Unable to run GTMSeg. aseg.mgz missing for {fs_subject_id}"
        print(msg)
        append_log(log_dir / "logSubjects_gtmseg_warning.txt", msg)
        return False

    try:
        gtmseg = petsurfer.GTMSeg()
        gtmseg.inputs.subject_id = fs_subject_id

        # Point to the folder containing all FS subject-session folders
        # (avoid relying on global SUBJECTS_DIR across parallel workers)
        gtmseg.inputs.subjects_dir = str(subjects_dir)

        gtmseg.run()

        msg = f"GTMSeg complete for subject {fs_subject_id}"
        print("\n########### " + msg + " ###########\n")
        append_log(log_dir / "logSubjects_gtmseg.txt", msg)
        return True

    except Exception as e:
        msg = f"GTMSeg FAILED for subject {fs_subject_id} | Error: {repr(e)}"
        print(msg)
        append_log(log_dir / "logSubjects_gtmseg_failed.txt", msg)
        return False


# ----------------------------
# Failure checking
# ----------------------------
def check_failed_subjects(entries):
    """
    Uses recon-all-status.log inside each FS subject folder to see if recon-all finished.
    NOTE: This checks recon-all completion, not GTMSeg completion.
    """
    failed = []
    for entry in entries:
        fs_subject_id = entry["fs_subject_id"]
        fs_subject_dir = entry["fs_subject_dir"]

        status_file = fs_subject_dir / "scripts" / "recon-all-status.log"
        if not status_file.exists():
            failed.append(entry)
            continue

        try:
            lines = status_file.read_text().splitlines()
            if not lines:
                failed.append(entry)
                continue

            last_line = lines[-1].strip()
            ok_prefix = f"recon-all -s {fs_subject_id} finished without error"
            if not last_line.startswith(ok_prefix):
                failed.append(entry)

        except Exception:
            failed.append(entry)

    return failed


# ----------------------------
# Batch processing
# ----------------------------
def process_in_batches(entries, log_dir: Path, batch_size=7, n_jobs=None):
    total = len(entries)
    if total == 0:
        print("No valid FS subject-session folders found (with aseg.mgz).")
        return

    if n_jobs is None:
        n_jobs = cpu_count()

    for i in range(0, total, batch_size):
        batch = entries[i:i + batch_size]
        Parallel(n_jobs=int(n_jobs))(
            delayed(run_gtmseg)(entry, log_dir) for entry in batch
        )
        print(f"Batch {i // batch_size + 1} completed ({min(i + batch_size, total)}/{total})")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer GTMSeg on CAPS freesurfer_cross_sectional outputs."
    )
    parser.add_argument(
        "--base_dir", type=str, default="/Volumes/Ali_X10Pro/ADNI",
        help="Base folder containing CAPS_DIR (default: /Users/shahzadali/Desktop/ADNI)"
    )
    parser.add_argument(
        "--caps_name", type=str, default="CAPS_DIR",
        help="CAPS directory name inside base_dir (default: CAPS_DIR)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
        help="How many subject-sessions per batch (default: 7)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None,
        help="Parallel jobs (default: cpu_count)"
    )
    parser.add_argument(
        "--freesurfer_home", type=str, default=os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/"),
        help="FreeSurfer home path (default: $FREESURFER_HOME or /Applications/freesurfer/7.4.1/)"
    )
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name
    log_dir = base_dir  # logs written here

    # Set FreeSurfer home (overridable)
    os.environ["FREESURFER_HOME"] = str(Path(args.freesurfer_home).expanduser().resolve())

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] CAPS dir: {caps_dir}")
    print(f"[INFO] FREESURFER_HOME: {os.environ['FREESURFER_HOME']}")

    entries = find_caps_freesurfer_subjects(caps_dir)
    print(f"[INFO] Found {len(entries)} FS subject-session folders with aseg.mgz")

    start_time = time.time()

    # First pass
    process_in_batches(entries, log_dir=log_dir, batch_size=args.batch_size, n_jobs=args.n_jobs)

    # Optional retry list based on recon-all status log
    first_failed = check_failed_subjects(entries)
    if first_failed:
        (log_dir / "logSubjects_failed_once.txt").write_text(
            "\n".join([e["fs_subject_id"] for e in first_failed]) + "\n"
        )
        print(f"[INFO] Retrying {len(first_failed)} subjects that look failed by recon-all-status.log ...")
        process_in_batches(first_failed, log_dir=log_dir, batch_size=args.batch_size, n_jobs=args.n_jobs)

        second_failed = check_failed_subjects(first_failed)
        if second_failed:
            (log_dir / "logSubjects_failed_twice.txt").write_text(
                "\n".join([e["fs_subject_id"] for e in second_failed]) + "\n"
            )
            print(f"[WARN] {len(second_failed)} subjects still look failed. See logSubjects_failed_twice.txt")

    print(f"--- {time.time() - start_time:.2f} seconds (GTMSeg total time) ---")


if __name__ == "__main__":
    main()
