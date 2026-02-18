import sys
import os
import pandas as pd
import subprocess
from pathlib import Path

# Set FreeSurfer environment
os.environ['FREESURFER_HOME'] = '/Applications/freesurfer/7.4.1'

if __name__ == '__main__':
    sys.setrecursionlimit(5000)

    caps_dir = Path('/Volumes/X10Pro_Ali/ADNI/CAPS_DIR/subjects/')
    output_dir = Path('/Volumes/X10Pro_Ali/ADNI/freesurfer_outputs/')
    link_dir = output_dir / 'freesurfer_linked_subjects'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(link_dir, exist_ok=True)

    subj_list = []

    # Clear old links if any
    for f in link_dir.glob('sub-*'):
        if f.is_symlink():
            f.unlink()

    # Collect all FreeSurfer session folders and symlink them
    for subject in sorted(caps_dir.glob('sub-*')):
        ses_path = subject / 'ses-M000' / 't1' / 'freesurfer_cross_sectional'
        if ses_path.is_dir():
            for fs_folder in sorted(ses_path.glob('sub-*')):
                if (fs_folder / 'mri').is_dir():  # Confirm FS output
                    subj_name = fs_folder.name
                    subj_list.append(subj_name)
                    link_path = link_dir / subj_name
                    if not link_path.exists():
                        os.symlink(fs_folder, link_path)

    # Write subject list
    subj_list_path = output_dir / 'subj_List.txt'
    with open(subj_list_path, 'w') as f:
        for subj in subj_list:
            f.write(subj + '\n')

    # Set SUBJECTS_DIR to symlink folder
    os.environ['SUBJECTS_DIR'] = str(link_dir)

    print(f"SUBJECTS_DIR set to: {link_dir}")
    print(f"Found {len(subj_list)} valid FreeSurfer subjects.")

    os.chdir(output_dir)

    # Run aseg and aparc table generation
    subprocess.run([
        "asegstats2table", "--subjectsfile", "subj_List.txt",
        "--meas", "volume", "--tablefile", "aseg.txt", "--skip"
    ], check=True)

    subprocess.run([
        "aparcstats2table", "--subjectsfile", "subj_List.txt",
        "--hemi", "rh", "--meas", "thickness",
        "--tablefile", "rh_aparc.txt", "--skip"
    ], check=True)

    subprocess.run([
        "aparcstats2table", "--subjectsfile", "subj_List.txt",
        "--hemi", "lh", "--meas", "thickness",
        "--tablefile", "lh_aparc.txt", "--skip"
    ], check=True)

    # Load tables
    df1 = pd.read_csv('aseg.txt', sep='\t')
    df2 = pd.read_csv('rh_aparc.txt', sep='\t')
    df3 = pd.read_csv('lh_aparc.txt', sep='\t')

    # Save Excel file
    with pd.ExcelWriter('MRI_T1_Table.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Subcortical Volumes', index=False)
        df2.to_excel(writer, sheet_name='Cortical Thickness RH', index=False)
        df3.to_excel(writer, sheet_name='Cortical Thickness LH', index=False)

    print("MRI_T1_Table.xlsx created with", len(df1), "subjects.")
