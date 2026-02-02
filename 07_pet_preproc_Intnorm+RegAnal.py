# -*- coding: utf-8 -*-
"""
Intensity Normalization and Regional PET Analysis
Author: Alessio Cirone
Email: alessio.cirone@hsanmartino.it
"""

# matplotlib qt5
import os, sys
import numpy as np
import nibabel as nib
import pandas as pd
import mahotas
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# ----------- CONFIGURATION -----------
path_project = '/mnt/d/Chiara_OIGE'
path_data = path_project #os.path.join(path_project, 'data/')
#gtmseg_path = os.path.join(path_data, subj_id, 'mri', 'gtmseg_on_mni.nii.gz')
#gtmseg = nib.load(gtmseg_path).get_fdata(dtype="float32")
region_to_normalize = 'cb'  # Can be: 'pons', 'cb', 'wm'
output_excel_name = 'tabella_FDG_norm.xlsx'
log_file = os.path.join(path_project, 'logSubjects.txt')
warning_log = os.path.join(path_project, 'logSubjects_warning.txt')
# -------------------------------------

def robust_mode_GMM(values, bins=100):
    """Estimate the mode using a 2-component Gaussian Mixture Model, taking the mean of the higher-uptake Gaussian."""
    values = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(values)
    means = gmm.means_.flatten()
    return means[np.argmax(means)]

def crop_to_pet_space(seg_array, seg_affine, pet_shape, pet_affine):
    """Crop segmentation to PET space using affine origins."""
    seg_origin = np.round(seg_affine[:3, 3], 2)
    pet_origin = np.round(pet_affine[:3, 3], 2)
    delta = np.abs(seg_origin - pet_origin).astype(int)
    return seg_array[
        delta[0]:delta[0]+pet_shape[0],
        delta[1]:delta[1]+pet_shape[1],
        delta[2]:delta[2]+pet_shape[2]
    ]

def erode_mask(mask):
    """Remove border voxels (1 voxel erosion)."""
    return np.bitwise_xor(mask, mahotas.bwperim(mask))

def get_normalization_mask(gtmseg, seg_crop, target):
    """Return the correct normalization mask."""
    if target == 'pons':
        pons = np.isin(gtmseg, [174])
        return erode_mask(pons)
    elif target == 'cb':
        cb = np.isin(seg_crop, [8, 47]) # cervelletto (materia bianca (7,46) e corteccia (8,47))
        return erode_mask(cb)
    elif target == 'wm':
        wm = np.isin(seg_crop, [2, 41]) # white matter destra e sinistra
        return erode_mask(wm)
    else:
        raise ValueError(f"Unsupported normalization target: {target}")

def log_message(filepath, message):
    with open(filepath, 'a') as f:
        f.write(message + "\n")

# --------- MAIN SCRIPT ----------
if __name__ == '__main__':
    sys.setrecursionlimit(5000)

    subj_list = sorted([s for s in os.listdir(path_data) if s[0] != '.' and s != 'fsaverage'])
    #subj_list = [subj_list[i] for i in [1,4,6,22,24,27,31,34,35,41,45,47,49,58,59,61,65,70,71,72]]

    # Prepare result containers
    df_meanGMM_all, df_mean_all, df_std_all, df_size_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for subj_id in subj_list:
        print(f"\n#### Processing subject {subj_id} ####")

        pet_path = os.path.join(path_data, subj_id, 'fdg', 'PETonMNI.nii.gz')
        #seg_path = '/mnt/d/Template_MNI/aparc+aseg_on_mni.nii.gz'
        #seg_path = '/mnt/d/Template_MNI/gtmseg_on_mni.nii.gz'
        seg_path = os.path.join(path_data, subj_id, 'mri', 'gtmseg_on_mni.nii.gz')

        if not os.path.exists(seg_path):
            log_message(warning_log, f"Segmentation missing for {subj_id}")
            continue
        if not os.path.exists(pet_path):
            log_message(warning_log, f"PET missing for {subj_id}")
            continue

        # Load images
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata(dtype="float32")
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata(dtype="float32")
        seg_crop = crop_to_pet_space(seg_data, seg_img.affine, pet_data.shape, pet_img.affine)

        # pet_img = nib.load(pet_path)
        # pet_img = nib.as_closest_canonical(pet_img)
        # pet_data = pet_img.get_fdata(dtype="float32")
        # seg_img = nib.load(seg_path)
        # seg_img = nib.as_closest_canonical(seg_img)
        # seg_data = seg_img.get_fdata(dtype="float32")
        # seg_crop = crop_to_pet_space(seg_data, seg_img.affine, pet_data.shape, pet_img.affine)

        # Normalization
        norm_mask = get_normalization_mask(seg_crop, seg_crop, region_to_normalize)  # plt.imshow(norm_mask[:,:,150])
        #norm_value = np.mean(pet_data[norm_mask])
        norm_value = robust_mode_GMM(pet_data[norm_mask])
        pet_norm = pet_data / norm_value

        # Save normalized PET
        out_path = os.path.join(path_data, subj_id, 'fdg', f'PETonMNI_norm_{region_to_normalize}.nii.gz')
        nib.save(nib.Nifti1Image(pet_norm, affine=pet_img.affine), out_path)

        # Regional statistics
        VOIs = [i for i in np.unique(seg_crop).astype(int)
                if i not in {0,2,4,5,7,8,14,15,16,24,28,29,30,31,41,43,44,46,47,60,62,63,72,77,80,85,130,165,172,174,251,252,253,254,255,257,258}]

        meanGMM_list, mean_list, std_list, size_list = [], [], [], []

        for voi in VOIs:
            mask = seg_crop == voi
            values = pet_norm[mask]
            meanGMM_list.append(robust_mode_GMM(values, bins=200))
            mean_list.append(np.mean(values))
            std_list.append(np.std(values))
            size_list.append(np.count_nonzero(mask))

        # Save to DataFrame
        col = pd.MultiIndex.from_product([[subj_id]])
        df_meanGMM_all = pd.concat([df_meanGMM_all, pd.DataFrame(meanGMM_list, index=VOIs, columns=col)], axis=1)
        df_mean_all = pd.concat([df_mean_all, pd.DataFrame(mean_list, index=VOIs, columns=col)], axis=1)
        df_std_all = pd.concat([df_std_all, pd.DataFrame(std_list, index=VOIs, columns=col)], axis=1)
        df_size_all = pd.concat([df_size_all, pd.DataFrame(size_list, index=VOIs, columns=col)], axis=1)

        log_message(log_file, f"Run complete for subject {subj_id}")

    # Save all tables
    with pd.ExcelWriter(os.path.join(path_project, output_excel_name)) as writer:
        df_mean_all.transpose().to_excel(writer, sheet_name='Mean')
        df_meanGMM_all.transpose().to_excel(writer, sheet_name='MeanGMM')
        df_std_all.transpose().to_excel(writer, sheet_name='Std')
        df_size_all.transpose().to_excel(writer, sheet_name='Sample_Size')
        
        
# Plot istogrammi
# voi=1003
# mask = seg_crop == voi
# values = pet_norm[mask]

# plt.figure(figsize=(8, 5))
# plt.hist(values, bins=100, color='skyblue', edgecolor='black')
# plt.title(f'Istogramma intensità per la regione {voi}')
# plt.xlabel('Intensità normalizzata')
# plt.ylabel('Frequenza')
# plt.grid(True)
# plt.show()



# rename_atlas_all_sheets

def rename_atlas_all_sheets(path_new_label, path_excel_orig, path_project):
    ext_ucsffsx7 = os.path.splitext(path_excel_orig)[1].lower()
    if ext_ucsffsx7 == '.xlsx':
        xlsx_path = path_excel_orig
    elif ext_ucsffsx7 == '.csv':  
        # Converte il CSV in Excel (unico foglio)
        df_csv = pd.read_csv(path_excel_orig, low_memory=False)
        xlsx_path = path_excel_orig.replace('.csv', '.xlsx')
        df_csv.to_excel(xlsx_path, index=False)
    else:
        raise ValueError("Formato file non supportato. Usa .xlsx o .csv")
    
    # Legge il dizionario di rinomina
    df_labels = pd.read_excel(path_new_label)
    label_mapping = dict(zip(df_labels['name'], df_labels['new_names']))
    label_keys = set(label_mapping.keys())

    # Legge tutti i fogli del file Excel
    sheets_dict = pd.read_excel(xlsx_path, sheet_name=None)
    renamed_sheets = {}
    missing_regions_global = set()

    # Applica la rinomina a ciascun foglio
    for sheet_name, df_orig in sheets_dict.items():
        df_renamed = df_orig.rename(columns=label_mapping)

        # Trova le colonne mancanti per questo foglio
        col_missing = [col for col in label_keys if col not in df_orig.columns]
        missing_regions_global.update(col_missing)

        renamed_sheets[sheet_name] = df_renamed

    # Salva tutte le versioni rinominate in un unico Excel
    output_path = os.path.join(path_project, os.path.splitext(os.path.basename(xlsx_path))[0] + '_renamed.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in renamed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Salva le regioni mancanti in un file di testo
    missing_path = os.path.join(path_project, 'Regioni_mancanti.txt')
    with open(missing_path, 'a') as f1:
        for reg in sorted(missing_regions_global):
            nome_regione = label_mapping.get(reg, '???')
            f1.write(f"{reg} : {nome_regione}\n")

    print(f"File rinominato salvato in: {output_path}")
    print(f"Lista delle regioni mancanti salvata in: {missing_path}")

    return renamed_sheets


# Esempio d’uso
renamed_sheets = rename_atlas_all_sheets(
    "/mnt/d/Chiara_OIGE/voi_names.xlsx",
    "/mnt/d/Chiara_OIGE/tabella_FDG_norm.xlsx",
    "/mnt/d/Chiara_OIGE/"
)