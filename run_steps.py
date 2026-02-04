import sys
import os
from os import path
from pipeline import register_atlases, find_anat_func_folder, anatamoy_crawler,dilate_atlases

if __name__ == "__main__":
    fmri_prep_derivatives_folder = sys.argv[1]
    tractography_folder = sys.argv[2]
    subj_line = sys.argv[3]
    template_file = sys.argv[4]
    atlas_fp = sys.argv[5]
    output_folder = sys.argv[6]
    print(f"Sanity check\nLength of arguments {len(sys.argv)}")
    print(f"{fmri_prep_derivatives_folder}\n"
          f"{subj_line}\n"
          f"{template_file}\n"
          f"{output_folder}\n"
          f"")
    
    subj_info = subj_line.split("_")
    id_val = subj_info[0]
    num = subj_info[1]
    num = num.zfill(3)
    session = subj_info[2]
    destination_folder = path.join(
        output_folder,
        id_val+num,
        session
    )
    os.makedirs(
        destination_folder, 
        exist_ok=True
    )

    anatomy_folder, functional_folder = find_anat_func_folder(
        fmri_prep_derivatives=fmri_prep_derivatives_folder,
        subj_id=id_val+num,
        session_num=session
    )
    anatomy_fps = anatamoy_crawler(anatomy_folder)
    """register_atlases(
        template_file=template_file,
        atlas_fp=atlas_fp,
        output_folder=output_folder,
        anatomy_fps=anatomy_fps
    )"""
    dilate_atlases(
        brain_mask=anatomy_fps["brain_mask"],
        output_folder=destination_folder,
        dilation_width=2
    )


    

