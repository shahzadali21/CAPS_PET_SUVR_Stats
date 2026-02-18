#!/usr/bin/env python

from pathlib import Path
import re
import argparse

from clinica.utils.statistics import statistics_on_atlas
from clinica.utils.atlas import T1AndPetVolumeAtlasName


def recompute_pet_atlas_stats(caps_dir: Path, overwrite: bool = False, dry_run: bool = False):
    caps_dir = caps_dir.resolve()
    atlas_names = [a.value for a in T1AndPetVolumeAtlasName]

    print(f"Using CAPS directory: {caps_dir}")
    print(f"Atlases to compute: {atlas_names}")
    print(f"Overwrite existing TSVs: {overwrite}")
    print(f"Dry run: {dry_run}")
    print()

    # Pattern: .../subjects/sub-XX/ses-YY/pet/preprocessing/group-*/sub-XX_ses-YY_..._space-Ixi549Space_suvr-REF_pet.nii(.gz)
    suvr_pattern = "subjects/*/*/pet/preprocessing/group-*/sub-*_*_space-Ixi549Space_suvr-*_pet.nii*"

    suvr_files = sorted(caps_dir.glob(suvr_pattern))
    if not suvr_files:
        print("No PET SUVR files found with pattern:")
        print(f"  {suvr_pattern}")
        return

    for suvr in suvr_files:
        # Example filename:
        # sub-01_ses-M00_trc-AV45_space-Ixi549Space_suvr-PONS_pet.nii.gz
        m = re.match(
            r"(?P<base>sub-[^_]+.*)_space-[^_]+_suvr-(?P<ref>[^_]+)_pet\.nii(\.gz)?",
            suvr.name,
        )
        if not m:
            print(f"Skipping file with unexpected name: {suvr}")
            continue

        base = m.group("base")        # e.g. "sub-01_ses-M00_trc-AV45"
        ref = m.group("ref")          # e.g. "PONS"
        group_dir = suvr.parent       # .../pet/preprocessing/group-XYZ
        atlas_dir = group_dir / "atlas_statistics"
        atlas_dir.mkdir(exist_ok=True)

        print(f"\nSubject/session: {base}")
        print(f"  Group dir: {group_dir}")
        print(f"  SUVR reference: {ref}")

        for atlas in atlas_names:
            out_tsv = atlas_dir / f"{base}_space-{atlas}_suvr-{ref}_statistics.tsv"

            if out_tsv.exists() and not overwrite:
                print(f"  [SKIP] {out_tsv.name} (already exists)")
                continue

            if dry_run:
                print(f"  [DRY RUN] would compute {out_tsv.name}")
                continue

            print(f"  [RUN] Computing atlas {atlas} -> {out_tsv.name}")
            statistics_on_atlas(str(suvr), atlas, str(out_tsv))

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute PET-volume atlas statistics in an existing CAPS directory."
    )
    parser.add_argument(
        "--caps_dir",
        required=True,
        help="Path to the CAPS directory used by PET-volume.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TSVs if they already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be done, without writing any file.",
    )
    args = parser.parse_args()

    recompute_pet_atlas_stats(Path(args.caps_dir), overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
