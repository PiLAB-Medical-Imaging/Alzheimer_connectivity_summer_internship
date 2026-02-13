import os
from os import path
import sys

from engagement_analysis import analyse_all_tracts, patient_registration

if __name__=="__main__":
    subject = sys.argv[1]
    atlas_folder = sys.argv[2]
    mni_path = sys.argv[3]
    outpath = sys.argv[4]
    save_csv_path = sys.argv[5]
    engagement_folder = sys.argv[6]
    derivatives_folder = sys.argv[7]
    outputs_folder = sys.argv[8]

    # Extract subject and session from input subject info
    subj_info = subject.split("_")
    id_val = subj_info[0]
    num = subj_info[1]
    num = num.zfill(3)
    session = subj_info[2]
    sub_id = id_val+num

    t1w_file = path.join(
        outputs_folder,
        sub_id,
        session,
        "brain_only_t1w.nii.gz"
    )
    patient_registration(
        atlas_folder=atlas_folder,
        original_space=mni_path,
        target_file=t1w_file,
        output_folder=outpath,
        subj=subject
    )
    eng_file = path.join(
        engagement_folder,
        subject + "_engagement.nii.gz"
    )
    analyse_all_tracts(
        tract_folder=atlas_folder, 
        output_folder=outputs_folder,
        engagement_file=eng_file,
        nodes=100, 
        subj=subject
    )

  
