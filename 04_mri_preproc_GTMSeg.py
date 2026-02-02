#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:42:00 2024

@author: alessio18
"""

import sys
import os
from nipype.interfaces.freesurfer import petsurfer
import time
from multiprocessing import cpu_count
from joblib import Parallel, delayed
os.environ['FREESURFER_HOME'] = '/Applications/freesurfer/7.4.1/' # '/usr/local/freesurfer/7.4.1/'

def ciclo_gtmseg(subj_id, path_data):
    if subj_id[0] != '.' and subj_id != 'fsaverage':
        check_reconall = path_data + subj_id + '/mri/aseg.mgz'
        if os.path.exists(check_reconall):
            gtmseg = petsurfer.GTMSeg()
            gtmseg.inputs.subject_id = subj_id
            gtmseg.run()
            print("\n" + "########### " + "Subject " + subj_id + " complete " + "###########" + "\n")
            f1 = open(path_project + 'logSubjects_gtmseg.txt', 'a')
            f1.write("GTMSeg complete for subject " + subj_id + "\n")
            f1.close()
        else:
            print("Unable to run GTMSeg. ReconAll is missing in Subject", subj_id)
            f2 = open(path_project + 'logSubjects_gtmseg_warning.txt', 'a')
            f2.write("Unable to run GTMSeg. ReconAll is missing in Subject " + subj_id + "\n")
            f2.close()
            
def check_failed_subjects(subj_List, path_data):
    failed_subjects = []
    for subj_id in subj_List:
        status_file = os.path.join(path_data, subj_id, 'scripts', 'recon-all-status.log')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                last_line = f.readlines()[-1].strip()
                if not last_line.startswith(f"recon-all -s {subj_id} finished without error"):
                    failed_subjects.append(subj_id)
        else:
            failed_subjects.append(subj_id)
    return failed_subjects
            
def process_in_batches(subj_List, path_data, batch_size=7):
    total_subjects = len(subj_List)
    for i in range(0, total_subjects, batch_size):
        batch = subj_List[i:i+batch_size]
        Parallel(n_jobs=int(cpu_count()))(delayed(ciclo_gtmseg)(subj_id, path_data) for subj_id in batch)
        print(f"Batch {i//batch_size + 1} completed")
            
if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    path_project = '/workspaces/testpippo/Data/DottoratoSimona_ADNI/dati_ADNI_nifti/'
    path_data = path_project + 'data/'
    os.environ['SUBJECTS_DIR'] = path_data
    subj_List = sorted(os.listdir(path_data))

    start_time = time.time()
    process_in_batches(subj_List, path_data)
    
    first_failed = check_failed_subjects(subj_List, path_data)
    if first_failed:
        with open(path_project + 'logSubjects_failed_once.txt', 'w') as f:
            f.write("\n".join(first_failed) + "\n")
        print(f"Retrying {len(first_failed)} subjects that failed the first time...")
        process_in_batches(first_failed, path_data)
        second_failed = check_failed_subjects(first_failed, path_data)
        if second_failed:
            with open(path_project + 'logSubjects_failed_twice.txt', 'w') as f:
                f.write("\n".join(second_failed) + "\n")
            print(f"Warning: {len(second_failed)} subjects failed twice. Check logSubjects_failed_twice.txt")
    
    print(f"--- {time.time() - start_time} seconds (for recon-all) ---")
