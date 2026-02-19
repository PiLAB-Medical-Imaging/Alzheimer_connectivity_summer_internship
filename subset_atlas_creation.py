from os.path import join
import sys

import nibabel as nib
import numpy as np

from network_analyses import load_all_indices

def atlas_subsetting(
        network_definitions: str,
        save_path: str,
        subject: str,
        atlas_name = "AAL116"
):
    subject_split = subject.split("_")
    sub_num = subject_split[0] + subject_split[1].zfill(3)
    sub_session = subject_split[2]
    all_networks = load_all_indices(
        definitions_filepath=network_definitions
    )
    atlas_fp = join(
        save_path,
        sub_num,
        sub_session,
        "registered_atlas.nii.gz"
    )

    atlas_img = nib.load(atlas_fp)
    atlas_data = atlas_img.get_fdata()

    for network in all_networks.keys():
        atlas_subset = np.where(
            np.isin(
                atlas_data,
                all_networks[network][atlas_name]
            ), 
            atlas_data,
            0
        )
        atlas_filename = f"{network}_{atlas_name}_patient.nii.gz"
        atlas_path = join(
            save_path,
            sub_num,
            sub_session,
            atlas_filename
            
        )
        out = nib.Nifti1Image(
            dataobj=atlas_subset,
            affine = atlas_img
        )
        out.to_filename(
            atlas_path
        )

if __name__ == "__main__":
    network_definitions_path = sys.argv[1]
    save_path = sys.argv[2]
    subject = sys.argv[3]

    atlas_subsetting(
        network_definitions=network_definitions_path,
        save_path=save_path,
        subject=subject
    )