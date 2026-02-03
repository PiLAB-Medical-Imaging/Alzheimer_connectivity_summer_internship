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
from engagement import generate_VWSC_matrices_entire_sl, generate_VWSC_matrices_ep_only
from time import time
import sparse
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


mask = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz"
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
cms_old_method, wm_positions = generate_VWSC_matrices_entire_sl(
    atlas_data=atlas.get_fdata(),
    trk =trk,
    v2f_mapping=vx_sl_map,
    white_matter_mask=mask
)
t2 = time()
print(f"Old method {t2-t1} s")

t1 = time()
cms_new_method, wm_positions = generate_VWSC_matrices_ep_only(
    atlas_data=atlas.get_fdata(),
    trk =trk,
    v2f_mapping=vx_sl_map,
    white_matter_mask=mask
)
t2 = time()

print(f"New method {t2-t1} s")

print(f"New non-zeros {cms_new_method.nnz}\n"
      f"Shape: {cms_new_method.shape}")

def sparse_equality(sparse_1, sparse_2):
    if sparse_1.shape != sparse_2.shape:
        return False
    if sparse_1.nnz != sparse_2.nnz:
        return False
    if (sparse_1-sparse_2).nnz !=0:
        return False
    else:
        return True

if  sparse_equality(cms_old_method, cms_new_method):
    print("Sweet as bruh")
else:
    print("Not the same.")
