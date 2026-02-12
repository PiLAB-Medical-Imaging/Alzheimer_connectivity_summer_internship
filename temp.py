import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from utilities import dilate_atlas_labels, atlas_masker, time_slicing, trk2tck
from utilities import split_nifti_to_visualise, streamline_registration, mask_to_positions
from utilities import sl_to_roi_map, voxel_to_streamline_map_V2, conn_matrices, conn_matrices_V2
from engagement import generate_VWSC_matrices_entire_sl, generate_VWSC_matrices_ep_only
from time import time
from nibabel import Nifti1Image
from dipy.tracking.streamline import transform_streamlines
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
out.to_filename(atlas) """
""" 

OUR_MNI = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"
OUR_MNI_TRACT ="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/our_mni_bundles/mni_edited_AC.trk"
T1 = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz"
trk = load_tractogram(filename=OUR_MNI_TRACT, reference=OUR_MNI)
print("trk_space", trk.space)

tform = find_transform(
    moving_file=OUR_MNI,
    static_file=T1,
    diffeomorph=False
)

apply_transform(
    OUR_MNI,
    tform,
    static_file=T1,
    output_path="/Users/sam/Desktop/test_mni_patient_space.nii.gz"
)

data = trk.streamlines.get_data()
mins = data.min(axis=0)
maxs = data.max(axis=0)
print(trk.space)
print("Streamline bounds:")
print("X:", mins[0], "→", maxs[0])
print("Y:", mins[1], "→", maxs[1])
print("Z:", mins[2], "→", maxs[2])

new_sls = transform_streamlines(
    streamlines=trk.streamlines,
    mat = tform.affine
)

new_sls = transform_streamlines(
    streamlines=trk.streamlines,
    mat = np.linalg.inv(tform.affine)
)
new_trk = StatefulTractogram(
    streamlines=new_sls, 
    reference=T1, 
    space=Space.RASMM
)

data = new_trk.streamlines.get_data()
mins = data.min(axis=0)
maxs = data.max(axis=0)
print(new_trk.space)
print("Streamline bounds:")
print("X:", mins[0], "→", maxs[0])
print("Y:", mins[1], "→", maxs[1])
print("Z:", mins[2], "→", maxs[2])

save_tractogram(
    new_trk,
    filename="/Users/sam/Desktop/test_tracts.trk",
    bbox_valid_check=True
)
 
trk2tck("/Users/sam/Desktop/test_tracts.trk", False)
 """
trk = load_tractogram("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/wm_atlas_inverted/TAU001_mni_edited_IF0F_R.trk",
                reference="same")

print(trk)