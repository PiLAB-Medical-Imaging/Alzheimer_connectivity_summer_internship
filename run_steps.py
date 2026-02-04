import sys
from pipeline import run_steps

if __name__ == "__main__":
    fmri_prep_derivatives_folder = sys.argv[0]
    tractography_folder = sys.argv[1]
    subj_line = sys.argv[2]

    run_steps(
        fmri_prep_derivatives=fmri_prep_derivatives_folder,
        tractography_folder=tractography_folder,
        subject_line=subj_line
    )
