#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:10:27 2026

@author: alessio18
"""

import sys, os, time, contextlib
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.ants import RegistrationSynQuick, ApplyTransforms
from multiprocessing import cpu_count
from joblib import Parallel, delayed
os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer/7.4.1/'

def process_in_batches(subj_List, path_data, batch_size=5):
    total_subjects = len(subj_List)
    for i in range(0, total_subjects, batch_size):
        batch = subj_List[i:i+batch_size]
        Parallel(n_jobs=int(cpu_count()))(delayed(spanorm)(subj_id, path_data) for subj_id in batch)
        print(f"Batch {i//batch_size + 1} completed")
        
@contextlib.contextmanager
def suppress_stdout_stderr():
    """ Disabilita completamente stdout e stderr a livello di file descriptor. """
    with open(os.devnull, 'w') as devnull:
        # Salva i vecchi file descriptor
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        # Reindirizza stdout e stderr a /dev/null
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        try:
            yield  # Esegue il codice con output disabilitato
        finally:
            # Ripristina i file descriptor originali
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def spanorm(subj_id, path_data):
    if subj_id[0] != '.' and subj_id != 'fsaverage':
        
        MNI_T1 = '/mnt/d/Template_MNI/mni_icbm152_t1_tal_nlin_sym_55_ext.nii.gz'
        mri_registered = path_data + subj_id + '/mri/norm/'
        seg = path_data + subj_id + '/mri/gtmseg.mgz'
        if os.path.exists(seg):
            segconv = MRIConvert()
            segconv.inputs.in_file = seg
            segconv.inputs.out_file = path_data + subj_id + '/mri/gtmseg.nii.gz'
            segconv.inputs.out_type = 'niigz'
            with suppress_stdout_stderr():
                segconv.run()
            at_seg = ApplyTransforms()
            at_seg.inputs.input_image = path_data + subj_id + '/mri/gtmseg.nii.gz'
            at_seg.inputs.reference_image = MNI_T1
            at_seg.inputs.output_image = path_data + subj_id + '/mri/gtmseg_on_mni.nii.gz'
            at_seg.inputs.transforms = [mri_registered + 'MRIonMNI_1Warp.nii.gz', mri_registered + 'MRIonMNI_0GenericAffine.mat']
            at_seg.inputs.interpolation = 'NearestNeighbor'
            with suppress_stdout_stderr():
                at_seg.run()
        else:
            print("Unable to run Spanorm. Segmentation is missing in Subject", subj_id)
            f2 = open(path_project + 'logSubjects_spanorm_warning.txt', 'a')
            f2.write("Unable to run Spanorm. Segmentation is missing in Subject " + subj_id + "\n")
            f2.close()

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    path_project = '/mnt/d/Chiara_OIGE/'
    path_data = path_project #+ 'data/'
    os.environ['SUBJECTS_DIR'] = path_data
    subj_List = sorted(os.listdir(path_data))

    start_time = time.time()
    process_in_batches(subj_List, path_data)
    
    ris = Parallel(n_jobs=int(cpu_count()))(delayed(spanorm)(subj_id, path_data) for subj_id in subj_List)
    
    print("--- %s seconds ---" % (time.time() - start_time))