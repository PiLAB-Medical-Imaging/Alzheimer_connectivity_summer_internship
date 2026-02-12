import sys

import engagement_analysis as ea

if __name__=="__main__":
    original_mni = sys.argv[1]
    atlas_original_space = sys.argv[2]
    outputs_folder = sys.argv[3]

    ea.correct_atlases(
        mni_path= original_mni,
        atlas_folder=atlas_original_space,
        outputs_folder=outputs_folder
    )