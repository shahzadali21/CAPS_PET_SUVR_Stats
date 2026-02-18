#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module 1 — Run FreeSurfer PETSurfer GTMSeg on CAPS FreeSurfer outputs (restart-safe + TSV race-safe).

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
  logSubjects_gtmseg.tsv   (restart-safe tracker, UPDATED ONLY IN MAIN PROCESS)

Notes:
- Retry logic uses recon-all-status.log (checks recon-all completion, not GTMSeg completion).
- Restart-safe: reads logSubjects_gtmseg.tsv and skips already Completed entries.
- TSV race-safe: TSV updates happen only in the parent process (no parallel writes).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import csv

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
      - subject_id: e.g., sub-ADNI002S2010
      - session_id: e.g., ses-M000
    """
    subjects_root = caps_dir / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"CAPS subjects folder not found: {subjects_root}")

    results = []

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

                # Parse subject/session from FS folder name: sub-XXX_ses-YYY
                name = fs_subject_dir.name
                if "_ses-" in name:
                    subject_id = name.split("_ses-")[0]
                    session_id = "ses-" + name.split("_ses-")[1]
                else:
                    subject_id = name
                    session_id = "ses-UNKNOWN"

                results.append({
                    "fs_subject_id": name,
                    "subjects_dir": fs_parent,
                    "fs_subject_dir": fs_subject_dir,
                    "subject_id": subject_id,
                    "session_id": session_id,
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
# TSV tracker helpers (race-safe)
# ----------------------------
TSV_HEADER = ["Subject_ID", "session_id", "GTMSEG_status"]


def load_gtmseg_tsv(tsv_path: Path):
    """Load existing TSV into dict: status_map[(Subject_ID, session_id)] = status."""
    status_map = {}
    if not tsv_path.exists():
        return status_map

    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sub = (row.get("Subject_ID") or "").strip()
            ses = (row.get("session_id") or "").strip()
            st = (row.get("GTMSEG_status") or "").strip()
            if sub and ses:
                status_map[(sub, ses)] = st or "Unknown"
    return status_map


def write_gtmseg_tsv(tsv_path: Path, status_map: dict):
    """Rewrite TSV cleanly (no duplicates), sorted for readability."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(status_map.items(), key=lambda x: (x[0][0], x[0][1]))

    with tsv_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(TSV_HEADER)
        for (sub, ses), st in items:
            writer.writerow([sub, ses, st])


# ----------------------------
# GTMSeg runner (NO TSV writes here)
# ----------------------------
def run_gtmseg(entry, log_dir: Path):
    """
    Run GTMSeg for one FS subject-session entry.
    Returns: (subject_id, session_id, status_string)
      status_string in: Completed | Failed | Missing_aseg
    """
    fs_subject_id = entry["fs_subject_id"]
    subjects_dir = entry["subjects_dir"]
    fs_subject_dir = entry["fs_subject_dir"]
    subject_id = entry["subject_id"]
    session_id = entry["session_id"]

    aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
    if not aseg_path.exists():
        msg = f"Unable to run GTMSeg. aseg.mgz missing for {fs_subject_id}"
        print(msg)
        append_log(log_dir / "logSubjects_gtmseg_warning.txt", msg)
        return (subject_id, session_id, "Missing_aseg")

    try:
        gtmseg = petsurfer.GTMSeg()
        gtmseg.inputs.subject_id = fs_subject_id
        gtmseg.inputs.subjects_dir = str(subjects_dir)

        gtmseg.run()

        msg = f"GTMSeg complete for subject {fs_subject_id}"
        print("\n########### " + msg + " ###########\n")
        append_log(log_dir / "logSubjects_gtmseg.txt", msg)
        return (subject_id, session_id, "Completed")

    except Exception as e:
        msg = f"GTMSeg FAILED for subject {fs_subject_id} | Error: {repr(e)}"
        print(msg)
        append_log(log_dir / "logSubjects_gtmseg_failed.txt", msg)
        return (subject_id, session_id, "Failed")


# ----------------------------
# Failure checking (recon-all status)
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
# Batch processing (TSV updated ONLY here in parent)
# ----------------------------
def process_in_batches(entries, log_dir: Path, tsv_path: Path, status_map: dict,
                       batch_size=10, n_jobs=None):
    total = len(entries)
    if total == 0:
        print("No remaining subject-sessions to process.")
        return

    if n_jobs is None:
        n_jobs = cpu_count()

    for i in range(0, total, batch_size):
        batch = entries[i:i + batch_size]

        # Run in parallel: each worker returns (subject_id, session_id, status)
        results = Parallel(n_jobs=int(n_jobs))(
            delayed(run_gtmseg)(entry, log_dir) for entry in batch
        )

        # Update TSV map sequentially (no race)
        for subject_id, session_id, status in results:
            status_map[(subject_id, session_id)] = status

        # Write TSV ONCE per batch (safe)
        write_gtmseg_tsv(tsv_path, status_map)

        print(f"Batch {i // batch_size + 1} completed ({min(i + batch_size, total)}/{total})")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer GTMSeg on CAPS freesurfer_cross_sectional outputs (restart-safe + TSV race-safe)."
    )
    parser.add_argument("--base_dir", type=str, default="/Volumes/Ali_X10Pro/ADNI",
                        help="Base folder containing CAPS_DIR")
    parser.add_argument("--caps_name", type=str, default="CAPS_DIR",
                        help="CAPS directory name inside base_dir")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="How many subject-sessions per batch")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Parallel jobs (default: cpu_count)")
    parser.add_argument("--freesurfer_home", type=str,
                        default=os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1/"),
                        help="FreeSurfer home path")
    parser.add_argument("--tsv_name", type=str, default="logSubjects_gtmseg.tsv",
                        help="TSV tracker filename saved in base_dir")
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    base_dir = Path(args.base_dir).expanduser().resolve()
    caps_dir = base_dir / args.caps_name
    log_dir = base_dir

    os.environ["FREESURFER_HOME"] = str(Path(args.freesurfer_home).expanduser().resolve())

    tsv_path = log_dir / args.tsv_name
    status_map = load_gtmseg_tsv(tsv_path)

    # Ensure TSV exists with header (even if empty)
    if not tsv_path.exists():
        write_gtmseg_tsv(tsv_path, status_map)

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] CAPS dir: {caps_dir}")
    print(f"[INFO] FREESURFER_HOME: {os.environ['FREESURFER_HOME']}")
    print(f"[INFO] TSV tracker: {tsv_path}")

    entries = find_caps_freesurfer_subjects(caps_dir)
    print(f"[INFO] Found {len(entries)} FS subject-session folders with aseg.mgz")

    # Restart-safe: schedule only those NOT marked Completed in TSV
    remaining = []
    for e in entries:
        key = (e["subject_id"], e["session_id"])
        if status_map.get(key, "") == "Completed":
            continue
        remaining.append(e)

    print(f"[INFO] Remaining (not Completed in TSV): {len(remaining)}")

    start_time = time.time()

    # First pass only on remaining
    process_in_batches(
        remaining,
        log_dir=log_dir,
        tsv_path=tsv_path,
        status_map=status_map,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs
    )

    # Optional retry list based on recon-all status log (kept as-is)
    first_failed = check_failed_subjects(entries)
    if first_failed:
        (log_dir / "logSubjects_failed_once.txt").write_text(
            "\n".join([e["fs_subject_id"] for e in first_failed]) + "\n"
        )
        print(f"[INFO] Retrying {len(first_failed)} subjects that look failed by recon-all-status.log ...")
        process_in_batches(
            first_failed,
            log_dir=log_dir,
            tsv_path=tsv_path,
            status_map=status_map,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs
        )

        second_failed = check_failed_subjects(first_failed)
        if second_failed:
            (log_dir / "logSubjects_failed_twice.txt").write_text(
                "\n".join([e["fs_subject_id"] for e in second_failed]) + "\n"
            )
            print(f"[WARN] {len(second_failed)} subjects still look failed. See logSubjects_failed_twice.txt")

    print(f"--- {time.time() - start_time:.2f} seconds (GTMSeg total time) ---")


if __name__ == "__main__":
    main()
