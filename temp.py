import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from utilities import dilate_atlas_labels, atlas_masker, time_slicing
from utilities import split_nifti_to_visualise, diffusion_to_t1space, mask_to_positions
from utilities import sl_to_roi_map, voxel_to_streamline_map_V2, conn_matrices, conn_matrices_V2
from engagement import generate_VWSC_matrices
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
print(f"sl_roi_map runtime: {t2-t1}")
t1 = time()
vx_sl_map = voxel_to_streamline_map_V2(
    trk.streamlines,
    vol_shape=trk.dimensions,
    subsegment=10
)
t2 = time()
print(f"vx_sl_map runtime: {t2-t1}")

""" t1 = time()
conn_mats = conn_matrices(
    sl_roi_map=sl_roi_map,
    vox_sl_map=vx_sl_map,
    atlas_data=atlas.get_fdata()
)
t2 = time()
print(f"Conn_mats 1 mapping runtime: {t2-t1}") """


mask = "/Users/sam/Desktop/sub-TAU001/test_mask_red.nii.gz"
mask_img = nib.load(mask)
mask_pos = mask_to_positions(mask_img)

t1 = time()
conn_mats = conn_matrices_V2(
    sl_roi_map=sl_roi_map,
    vox_sl_map=vx_sl_map,
    atlas_data=atlas.get_fdata(),
    mask_positions=mask_pos
)
t2 = time()

print(f"conn mats 2 runtime: {t2-t1}")

t1 = time()
cms_old_method = generate_VWSC_matrices(
    atlas_data=atlas.get_fdata(),
    trk =trk,
    v2f_mapping=vx_sl_map,
    white_matter_mask=mask
)
t2 = time()

print(f"Old method {t2-t1} s")

print(f"Old emthod\n"
      f"{cms_old_method}\n"
      f"New method\n"
      f"{conn_mats}")