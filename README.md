# CAPS PET SUVR Regional Statistics (GTMseg-based)

To compute **regional PET SUVR statistics** from a **Clinica CAPS** dataset using **FreeSurfer/PETSurfer GTMseg** + **Clinica T1-linear** alignment.

---

## Repository structure

```
caps-pet-suvr-gtmseg/
├── scripts/
│   ├── 01_run_gtmseg_caps.py
│   ├── 02_gtmseg_to_mni_affine.py
│   └── 03_pet_suvr_stats.py
├── README.md
└── LICENSE
```

---

## Expected CAPS layout (example)

```
CAPS_DIR/subjects/sub-XXXX/ses-YYYY/
├── pet_linear/
│   ├── *trc-18FFDG*space-MNI152NLin2009cSym*suvr-pons2_pet.nii.gz
│   └── *trc-18FAV45*space-MNI152NLin2009cSym*suvr-cerebellumPons2_pet.nii.gz
└── t1/freesurfer_cross_sectional/sub-XXXX_ses-YYYY/
    └── mri/aseg.mgz
```

---

## Scripts (what they do)

### 1) `01_run_gtmseg_caps.py`
Runs PETSurfer GTMseg for each FreeSurfer subject-session in CAPS.  
**Input:** `.../mri/aseg.mgz`  
**Output:** `.../mri/gtmseg.mgz` (+ PETSurfer outputs)

### 2) `02_gtmseg_to_mni_affine.py`
Converts `gtmseg.mgz` to NIfTI and applies the **Clinica T1-linear affine** to create MNI-aligned GTMseg.  
**Inputs:** `.../mri/gtmseg.mgz` + `t1_linear/*_affine.mat` (+ MNI T1 reference)  
**Output:** `.../mri/gtmseg_on_mni_affine.nii.gz`

### 3) `03_pet_suvr_stats.py`
Computes regional PET statistics from Clinica SUVR PET + `gtmseg_on_mni_affine`.  
**Inputs:**
- FDG: `pet_linear/*trc-18FFDG*...*suvr-pons2_pet.nii.gz`
- AV45: `pet_linear/*trc-18FAV45*...*suvr-cerebellumPons2_pet.nii.gz`
- Seg: `.../mri/gtmseg_on_mni_affine.nii.gz`

**Outputs:**
- Per-session copy (consistent naming):  
  `.../pet_surface/PET_trc-<trc>_suvr-<suvr>_pet.nii.gz`
- Global Excel in `BASE_DIR`:  
  `PET_trc-<trc>_suvr-<suvr>_statistics.xlsx`  
  sheets: `Mean`, `MeanGMM`, `Std`, `Sample_Size`

Region names are taken from `FreeSurferColorLUT.txt`.

---

## Quick start

### Run GTMseg
```bash
python scripts/01_run_gtmseg_caps.py --base_dir /path/to/ADNI --caps_name CAPS_DIR
```

### Map GTMseg to MNI (affine)
```bash
python scripts/02_gtmseg_to_mni_affine.py --base_dir /path/to/ADNI --caps_name CAPS_DIR
```

### Compute PET stats (FDG)
```bash
python scripts/03_pet_suvr_stats.py --tracer FDG --base_dir /path/to/ADNI --caps_name CAPS_DIR
```

### Compute PET stats (AV45)
```bash
python scripts/03_pet_suvr_stats.py --tracer AV45 --base_dir /path/to/ADNI --caps_name CAPS_DIR
```

---

## Requirements
- FreeSurfer (with PETSurfer)
- Clinica outputs: `t1/freesurfer_cross_sectional`, `t1_linear`, `pet_linear`
- Python packages: `numpy`, `nibabel`, `pandas`, `scikit-learn`
- ANTs via `nipype` (for affine application)
