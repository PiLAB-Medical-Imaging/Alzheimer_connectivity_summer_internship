import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from utilities import dilate_atlas_labels, atlas_masker


""" atlas = "/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz"
atlas_img = nib.load(atlas)
atlas = nib.load(atlas).get_fdata()

brain_mask = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-brain_mask.nii.gz"
brain_mask = nib.load(brain_mask).get_fdata()

dilated_atlas = dilate_atlas_labels(atlas=atlas, 
                    brain_mask=brain_mask, 
                    dilation_width=2)

out = nib.Nifti1Image(dilated_atlas, atlas_img.affine)
out.to_filename("/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz")


trk = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk",
                       reference="same") """


original_mask = nib.load("/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz")
new_maskname = "/Users/sam/Desktop/sub-TAU001/submask_TAU001.nii.gz"

target_indices = [1, 7, 46]

new_mask = atlas_masker(original_mask.get_fdata(), target_labels=target_indices)

print(np.count_nonzero(new_mask))

out = nib.Nifti1Image(new_mask, affine=original_mask.affine)
out.to_filename(new_maskname)