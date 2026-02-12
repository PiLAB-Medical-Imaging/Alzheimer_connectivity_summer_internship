import os
from os import path
import sys

from pipeline import register_atlases, find_anat_func_folder, anatamoy_crawler,dilate_atlases
from engagement_analysis import analyse_all_tracts, patient_registration

if __name__=="__main__":
    subject = sys.argv[1]
    atlas_folder = sys.argv[2]
    mni_path = sys.argv[3]
    outpath = sys.argv[4]
    save_csv_path = sys.argv[5]
    engagement_folder = sys.argv[6]
    t1w = ""

    patient_registration(
        atlas_folder=atlas_folder,
        original_space=mni_path,
        target_file=t1w,
        output_folder=outpath,
        subj=subject
    )

    analyse_all_tracts(
        tract_folder=outpath,
        output_folder=save_csv_path,
        engagement_file=
    )

