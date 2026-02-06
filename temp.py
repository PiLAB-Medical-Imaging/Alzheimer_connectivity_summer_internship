import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from utilities import dilate_atlas_labels, atlas_masker, time_slicing
from utilities import split_nifti_to_visualise, diffusion_to_t1space, mask_to_positions
from utilities import sl_to_roi_map, voxel_to_streamline_map_V2, conn_matrices, conn_matrices_V2
from engagement import generate_VWSC_matrices_entire_sl, generate_VWSC_matrices_ep_only
from time import time
from nibabel import Nifti1Image
import sparse

""" engagement_old = nib.load("/Users/sam/Desktop/sub-TAU001/anat/02_threshold_engagement_10x.nii.gz")
engagement_new = nib.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng.nii.gz")

eng_old_data = engagement_old.get_fdata()
eng_new_data = engagement_new.get_fdata()

comparison = eng_old_data-eng_new_data

nnz = np.count_nonzero(comparison)
print(nnz)

plt.hist(eng_old_data.flatten(), color="g", stacked=True)
plt.hist(eng_new_data.flatten(), color="b", stacked=True)
plt.loglog()
plt.show()



def report(array):
    max_val = np.max(array)
    min_val = np.min(array)
    nnz = np.count_nonzero(array)
    nans = np.sum(np.isnan(array))
    print(f"Report\n"
          f"Max: {max_val}\n"
          f"Min: {min_val}\n"
          f"Nonzeros: {nnz}\n"
          f"Nans: {nans}")
    
report(eng_old_data)
report(eng_new_data) """


""" 
engagement = "/Users/sam/Desktop/tau100_eng_250_all_sl.nii.gz"
eng_img = nib.load(engagement)
eng_data = eng_img.get_fdata()
mask = np.where(eng_data > 0, 1, 0)
filtered = gaussian_filter(
    input=eng_data,
    sigma = 2.0
)
filtered = filtered*mask
filtered_img = Nifti1Image(filtered, affine=eng_img.affine)
filtered_img.to_filename(engagement[:-7]+"_filtered.nii.gz")



 """


t1w = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/sub-TAU056_ses-0_desc-preproc_T1w.nii.gz"
brain_mask = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/sub-TAU056_ses-0_desc-brain_mask.nii.gz"
trk_diffusion = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/TAU_56_ses-0_tractogram.trk"
atlas = "/Users/sam/Desktop/aal_mni_correct.nii.gz"
mni_template = "/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"
# Register the atlas: 
import utilities as utl

atlas_img = nib.load(atlas)
atlas_data = atlas_img.get_fdata()
print(np.unique(atlas_data.astype("int32")))
print(len(np.unique(atlas_data.astype("int32"))))
atlas_data = np.round(atlas_data)
print(np.unique(atlas_data))
print(len(np.unique(atlas_data)))
out = nib.Nifti1Image(
    atlas_data,
    affine = atlas_img.affine
)
out.to_filename(atlas)