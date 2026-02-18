#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time, contextlib, glob
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from nipype.interfaces.freesurfer import ReconAll

# -------------------------
# FreeSurfer environment
# -------------------------
os.environ['FREESURFER_HOME'] = '/Applications/freesurfer/7.4.1/'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Disable stdout and stderr at file descriptor level."""
    with open(os.devnull, 'w') as devnull:
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

# -------------------------
# Path helpers
# -------------------------
def find_bids_t1w(bids_root: str, sub_id: str, ses_id: str) -> str | None:
    """
    Find BIDS T1w file:
    BIDS/sub-XXX/ses-YYY/anat/sub-XXX_ses-YYY_*T1w.nii[.gz]
    """
    pattern = os.path.join(bids_root, sub_id, ses_id, "anat", f"{sub_id}_{ses_id}_*T1w.nii*")
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def caps_fs_subjects_dir(caps_root: str, sub_id: str, ses_id: str) -> str:
    """
    CAPS path where FreeSurfer SUBJECTS_DIR should point (parent folder):
    CAPS/subjects/sub-XXX/ses-YYY/t1/freesurfer_cross_sectional/
    """
    return os.path.join(
        caps_root, "subjects", sub_id, ses_id, "t1", "freesurfer_cross_sectional"
    )

def remove_isrunning(subjects_dir: str, fs_subject_id: str):
    """Remove FreeSurfer IsRunning locks if present."""
    scripts_dir = os.path.join(subjects_dir, fs_subject_id, "scripts")
    for fn in ["IsRunning.lh+rh", "IsRunning.lh", "IsRunning.rh"]:
        try:
            os.remove(os.path.join(scripts_dir, fn))
        except:
            pass

# -------------------------
# Core processing
# -------------------------
def ciclo_reconAll(job, path_project: str):
    """
    job = dict with keys:
      - sub_id, ses_id
      - t1_path
      - subjects_dir (CAPS freesurfer_cross_sectional)
      - fs_subject_id (sub-XXX_ses-YYY)
    """
    sub_id       = job["sub_id"]
    ses_id       = job["ses_id"]
    t1_path      = job["t1_path"]
    subjects_dir = job["subjects_dir"]
    fs_id        = job["fs_subject_id"]

    # Ensure CAPS FS parent exists
    os.makedirs(subjects_dir, exist_ok=True)

    # Remove locks if any
    remove_isrunning(subjects_dir, fs_id)

    if not (t1_path and os.path.exists(t1_path)):
        msg = f"Unable to run ReconAll. T1 is missing in BIDS for {fs_id} ({sub_id}/{ses_id})"
        print(msg)
        with open(os.path.join(path_project, "logSubjects_reconAll_warning.txt"), "a") as f:
            f.write(msg + "\n")
        return

    reconall = ReconAll()
    reconall.inputs.subjects_dir = subjects_dir          # <-- CAPS freesurfer_cross_sectional
    reconall.inputs.subject_id   = fs_id                 # <-- sub-XXX_ses-YYY
    reconall.inputs.T1_files     = t1_path               # <-- BIDS NIfTI input
    reconall.inputs.parallel     = True

    for step in ["autorecon1", "autorecon2", "autorecon3"]:
        print(f"Processing {fs_id}: {step} ...")
        reconall.inputs.directive = step
        with suppress_stdout_stderr():
            reconall.run()

    print(f"\n########### Subject {fs_id} complete ###########\n")
    with open(os.path.join(path_project, "logSubjects_reconAll.txt"), "a") as f:
        f.write(f"ReconAll complete for subject {fs_id}\n")

def status_ok(subjects_dir: str, fs_subject_id: str) -> bool:
    status_file = os.path.join(subjects_dir, fs_subject_id, "scripts", "recon-all-status.log")
    if not os.path.exists(status_file):
        return False
    try:
        with open(status_file, "r") as f:
            lines = f.readlines()
        if not lines:
            return False
        last_line = lines[-1].strip()
        return last_line.startswith(f"recon-all -s {fs_subject_id} finished without error")
    except:
        return False

def check_failed_jobs(jobs):
    failed = []
    for job in jobs:
        if not status_ok(job["subjects_dir"], job["fs_subject_id"]):
            failed.append(job)
    return failed

def process_in_batches(jobs, path_project: str, batch_size: int = 7):
    total = len(jobs)
    if total == 0:
        print("No jobs to run.")
        return

    # Safer parallelism: do not blindly launch cpu_count() recon-alls at once.
    # Keep your batching behavior, but cap parallel jobs per batch.
    n_jobs_parallel = min(cpu_count(), batch_size)

    for i in range(0, total, batch_size):
        batch = jobs[i:i+batch_size]
        Parallel(n_jobs=n_jobs_parallel)(
            delayed(ciclo_reconAll)(job, path_project) for job in batch
        )
        print(f"Batch {i//batch_size + 1} completed")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    sys.setrecursionlimit(5000)

    # ========== EDIT THESE 3 PATHS ==========
    path_project = "/Users/shahzadali/Desktop/"
    bids_root    = "/Users/shahzadali/Desktop/BIDS"      # <-- your BIDS root
    caps_root    = "/Users/shahzadali/Desktop/CAPS_DIR"      # <-- your CAPS root
    # =======================================

    # Collect jobs from BIDS
    subj_ids = sorted([d for d in os.listdir(bids_root)
                       if d.startswith("sub-") and os.path.isdir(os.path.join(bids_root, d))])

    jobs = []
    for sub_id in subj_ids:
        sub_path = os.path.join(bids_root, sub_id)
        ses_ids = sorted([d for d in os.listdir(sub_path)
                          if d.startswith("ses-") and os.path.isdir(os.path.join(sub_path, d))])

        for ses_id in ses_ids:
            t1 = find_bids_t1w(bids_root, sub_id, ses_id)

            # FreeSurfer subject ID in CAPS (matches your screenshot style):
            fs_id = f"{sub_id}_{ses_id}"

            # CAPS FreeSurfer parent directory:
            subjects_dir = caps_fs_subjects_dir(caps_root, sub_id, ses_id)

            jobs.append({
                "sub_id": sub_id,
                "ses_id": ses_id,
                "t1_path": t1,
                "subjects_dir": subjects_dir,
                "fs_subject_id": fs_id
            })

    start_time = time.time()

    # First pass
    process_in_batches(jobs, path_project, batch_size=7)

    # Retry pass
    first_failed = check_failed_jobs(jobs)
    if first_failed:
        with open(os.path.join(path_project, "logSubjects_failed_once.txt"), "w") as f:
            f.write("\n".join([j["fs_subject_id"] for j in first_failed]) + "\n")
        print(f"Retrying {len(first_failed)} subject-sessions that failed the first time...")
        process_in_batches(first_failed, path_project, batch_size=7)

        second_failed = check_failed_jobs(first_failed)
        if second_failed:
            with open(os.path.join(path_project, "logSubjects_failed_twice.txt"), "w") as f:
                f.write("\n".join([j["fs_subject_id"] for j in second_failed]) + "\n")
            print(f"Warning: {len(second_failed)} failed twice. Check logSubjects_failed_twice.txt")

    print(f"--- {time.time() - start_time} seconds (for recon-all) ---")
