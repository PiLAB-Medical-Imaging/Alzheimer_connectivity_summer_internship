import sys
import os
from os import path
from pipeline import register_atlases, find_anat_func_folder, anatamoy_crawler,dilate_atlases
import pipeline as ppl
if __name__ == "__main__":
    fmri_prep_derivatives_folder = sys.argv[1]
    tractography_folder = sys.argv[2]
    subj_line = sys.argv[3]
    template_file = sys.argv[4]
    atlas_fp = sys.argv[5]
    output_folder = sys.argv[6]
    dmri_folder = sys.argv[7]
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
    tractogram_file = path.join(
        tractography_folder,
        subj_line + "_tractogram.trk"
    )
    bold_filepath = ppl.find_bold_filepath(
        functional_folder=functional_folder
    )
    """
    register_atlases(
        template_file=template_file,
        atlas_fp=atlas_fp,
        output_folder=output_folder,
        anatomy_fps=anatomy_fps,
        subj_id=id_val+num,
        session=session
    )
    """
    dilate_atlases(
        output_folder=output_folder,
        subj_id=id_val+num,
        session=session,
        anatomy_fps=anatomy_fps,
        dilation_width=2
    )
    ppl.tractogram_registration(
        tractogram_file=tractogram_file,
        dmri_folder=dmri_folder,
        subj_line=subj_line,
        output_folder=output_folder,
        id_val=id_val+num,
        session=session
    )
    ppl.run_fc_matrix(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder,
        bold_fp=bold_filepath
    )
    ppl.run_sc_matrix(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder,
        tractogram_file=tractogram_file
    )
    ppl.run_simple_weighting(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder
    )
    ppl.run_EBC(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder
    )
    ppl.run_engagement(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder,
        anatamy_fps=anatomy_fps
    )
    """
    ppl.run_dynamic_engagement(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder,
        anatamy_fps=anatomy_fps,
        bold_fp=bold_filepath
    )
    ppl.run_functionnectome(
        subj_id=id_val+num,
        session=session,
        output_folder=output_folder,
        anatamy_fps=anatomy_fps,
        bold_fp=bold_filepath,
        tractogram_file=tractogram_file
    )"""




    
