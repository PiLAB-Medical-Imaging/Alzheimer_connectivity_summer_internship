from os import system, path, makedirs
import os
import sys

if __name__ == "__main__":
    tractogram_folder = sys.argv[1]
    subject = sys.argv[2]
    study_folder = sys.argv[3]
    output_folder = path.join(
        tractogram_folder,
        "sift2"
    )
    makedirs(output_folder, exist_ok=True)
    output_path = path.join(
        output_folder,
        subject+"_sift_weights.txt"
    )
    mu_path = path.join(
        output_folder,
        subject+"_mu.txt"
    )
    subject_tractogram = path.join(
        tractogram_folder,
        subject + "_tractogram.tck"
    )

    fod_file = path.join(
        study_folder,
        subject,
        "dMRI",
        "ODF",
        "MSMT-CSD",
        subject + "_MSMT-CSD_WM_ODF.nii.gz"
    )

    system(
        command=f'tcksift2 {subject_tractogram} {fod_file} {output_path} -out_mu {mu_path}'
    )

