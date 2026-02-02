#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

            # Typically exactly one FS subject dir here: sub-XXX_ses-YYY
            for fs_subject_dir in sorted(fs_parent.glob("sub-*_ses-*")):
                if not fs_subject_dir.is_dir():
                    continue

                aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
                if not aseg_path.exists():
                    # keep it in list anyway? usually better to skip
                    continue

                results.append({
                    "fs_subject_id": fs_subject_dir.name,
                    "subjects_dir": fs_parent,
                    "fs_subject_dir": fs_subject_dir,
                })

    return results


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

    # sanity check
    aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
    if not aseg_path.exists():
        msg = f"Unable to run GTMSeg. aseg.mgz missing for {fs_subject_id}"
        print(msg)
        (log_dir / "logSubjects_gtmseg_warning.txt").open("a").write(msg + "\n")
        return False

    try:
        gtmseg = petsurfer.GTMSeg()
        gtmseg.inputs.subject_id = fs_subject_id

        # IMPORTANT for CAPS: point to the parent that contains the FS folder
        # (avoids relying on global SUBJECTS_DIR env var across parallel jobs)
        gtmseg.inputs.subjects_dir = str(subjects_dir)

        gtmseg.run()

        msg = f"GTMSeg complete for subject {fs_subject_id}"
        print("\n########### " + msg + " ###########\n")
        (log_dir / "logSubjects_gtmseg.txt").open("a").write(msg + "\n")
        return True

    except Exception as e:
        msg = f"GTMSeg FAILED for subject {fs_subject_id} | Error: {repr(e)}"
        print(msg)
        (log_dir / "logSubjects_gtmseg_failed.txt").open("a").write(msg + "\n")
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
        print(f"Batch {i // batch_size + 1} completed ({min(i+batch_size, total)}/{total})")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run FreeSurfer GTMSeg on CAPS freesurfer_cross_sectional outputs.")
    parser.add_argument("--base_dir", type=str, default="/Users/shahzadali/Desktop/ADNI",
                        help="Base folder containing CAPS_DIR (default: /Users/shahzadali/Desktop/ADNI)")
    parser.add_argument("--caps_name", type=str, default="CAPS_DIR",
                        help="CAPS directory name inside base_dir (default: CAPS_DIR)")
    parser.add_argument("--batch_size", type=int, default=7, help="How many subjects per batch (default: 7)")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs (default: cpu_count)")
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name
    log_dir = base_dir  # logs will be written here (same as your original style)

    # If you want to enforce a FS home, keep it; otherwise remove
    os.environ["FREESURFER_HOME"] = "/Applications/freesurfer/7.4.1/"

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] CAPS dir: {caps_dir}")

    entries = find_caps_freesurfer_subjects(caps_dir)
    print(f"[INFO] Found {len(entries)} FS subject-session folders with aseg.mgz")

    start_time = time.time()

    # First pass
    process_in_batches(entries, log_dir=log_dir, batch_size=args.batch_size, n_jobs=args.n_jobs)

    # Optional: recon-all status based retry (same logic as your original script)
    first_failed = check_failed_subjects(entries)
    if first_failed:
        (log_dir / "logSubjects_failed_once.txt").write_text("\n".join([e["fs_subject_id"] for e in first_failed]) + "\n")
        print(f"[INFO] Retrying {len(first_failed)} subjects that look failed by recon-all-status.log ...")
        process_in_batches(first_failed, log_dir=log_dir, batch_size=args.batch_size, n_jobs=args.n_jobs)

        second_failed = check_failed_subjects(first_failed)
        if second_failed:
            (log_dir / "logSubjects_failed_twice.txt").write_text("\n".join([e["fs_subject_id"] for e in second_failed]) + "\n")
            print(f"[WARN] {len(second_failed)} subjects still look failed. See logSubjects_failed_twice.txt")

    print(f"--- {time.time() - start_time:.2f} seconds (GTMSeg total time) ---")


if __name__ == "__main__":
    main()
