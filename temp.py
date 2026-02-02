import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from utilities import dilate_atlas_labels, atlas_masker, time_slicing
from utilities import split_nifti_to_visualise
from utilities import sl_to_roi_map
from time import time
trk = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk",
                      reference="same")
atlas = nib.load("/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz")
trk.to_vox()
trk.to_corner()

t1 = time()
sl_roi_map = sl_to_roi_map(
    trk.streamlines,
    atlas = atlas.get_fdata()
)
t2 = time()

print(f"Runtime: {t2-t1}")

