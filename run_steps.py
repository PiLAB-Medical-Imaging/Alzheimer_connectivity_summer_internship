import sys
from pipeline import register_atlases

if __name__ == "__main__":
    fmri_prep_derivatives_folder = sys.argv[0]
    tractography_folder = sys.argv[1]
    subj_line = sys.argv[2]
    template_file = sys.argv[3]
    output_folder = sys.argv[4]
    atlas_fp = sys.argv[5]
    output_folder = sys.argv[6]


    register_atlases(
        fmri_prep_derivatives=fmri_prep_derivatives_folder,
        tractography_folder=tractography_folder,
        subject_line=subj_line,
        template_file=template_file,
        atlas_fp=atlas_fp,
        output_folder=output_folder
    )

