from dipy.tracking.utils import target
from dipy.tracking.streamline import select_by_rois
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
import Functionnectome.functionnectome as funct
import numpy as np
from dipy.tracking.streamline import transform_streamlines
from unravel.utils import get_streamline_density
from regis.core import find_transform, apply_transform
from dipy.tracking.utils import density_map
from os import path
from tqdm import tqdm
import os
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn import image, masking
from numpy import tensordot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from nilearn.regions import signals_to_img_labels
from utilities import mask_generator
from collections import defaultdict
import json
from nibabel.nifti1 import Nifti1Image
import sparse
from utilities import voxel_to_streamline_map
from functionnectome import functionnectome_pipeline



# Make a simple image to use as a reference
# Simple 20×20×20 voxel grid
shape = (1, 3, 3)
affine = np.eye(4)

basic_img = nib.Nifti1Image(np.zeros(shape=shape), affine=affine)

# Save the test brain

basic_img.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/basic_brain.nii.gz")


""" 

test = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk", "same")
test.to_vox()
test.to_corner()
print(test.streamlines) """

streamline_1 = np.array([[0,0,0],[0,1,1], [0,2,2]]).astype(np.float64)
streamline_2 = np.array([[0,1,0],[0,1,1], [0,1,2]]).astype(np.float64)
#streamline_3 = np.array([[0,5,3],[0,5,4], [0,5,5], [0,5,6]]).astype(np.float64)
new_trk = StatefulTractogram([streamline_1, streamline_2],basic_img, space=Space.VOX, origin=Origin.TRACKVIS)
new_trk.to_corner()

save_tractogram(new_trk,filename="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/test_tracts.trk")

# Make a mask:
atlas_mask = np.zeros(shape=shape)
for i in range(shape[1]):
    for j in range(shape[2]):
        if i < 1 and j <1:
            atlas_mask[0][i][j] =1
        elif i > 1 and j < 1:
            atlas_mask[0][i][j] =3
        elif i < 1 and j > 1:
            atlas_mask[0][i][j] =2
        elif i > 1 and j > 1:
            atlas_mask[0][i][j] =4

atlas_img = nib.Nifti1Image(atlas_mask, affine=affine)
atlas_img.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/registered_atlas.nii.gz")

rois = np.unique(atlas_mask)
N = len(rois)

# Generate the fake FMRI data
T=10
fMRI = np.zeros((1, shape[1], shape[2], T))

for i in range(T):
    fMRI[0, 0:1, 0:1, i] = 5 * np.random.random(1)
    fMRI[0, 2, 2, i] = 2 * np.random.random(1)
    fMRI[0, 0, 2, i] = 10 * np.random.random(1)
# Save the fMRI file: 
fMRI_img = Nifti1Image(fMRI.astype(np.float64), 
                       affine)

fMRI_img.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/fake_fMRI.nii.gz")






atlas_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/registered_atlas.nii.gz"
fMRI_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/fake_fMRI.nii.gz"
reference_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/basic_brain.nii.gz"
save_registered_atlas = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/registered_atlas.nii.gz"
tractogram_filepath = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/test_tracts.trk"
moving_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/basic_brain.nii.gz"
density_map_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/"
functionnectome_savepath = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/functionnectome.nii.gz"

print("Testing the pipeline")
functionnectome_pipeline(atlas_path=atlas_path,
                            fMRI_path=fMRI_path,
                            t1w_file=reference_file,
                            tractogram=tractogram_filepath,
                            anatomical_scan_atlas_space=moving_file, 
                            save_registered_atlas=save_registered_atlas,
                            savepath_density_map = density_map_path,
                            functionnectome_savepath= functionnectome_savepath,
                            mode = "roi",
                            remap=True,
                            is_aligned=True,
                            brain_only_t1w_path=reference_file
                            )


# Inspect all the results. 

fMRI_data = nib.load(fMRI_path).get_fdata()




density_maps = nib.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/roi_density_map.nii.gz")

density_map_data = density_maps.get_fdata()

print(density_maps.shape)

for i in range(N-1):
    print(f"Density Map for region {i+1}")
    print(density_map_data[:,:,:,i])


#print(density_maps.get_fdata())

print("Checking the probability maps")

probability_maps = nib.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/probability_maps.nii.gz")
prob_map_data = probability_maps.get_fdata()

print(prob_map_data.shape)

for i in range(N-1):
    print(f"Probability Map for region {i+1}")
    print(prob_map_data[:,:,:,i])


""" 
print("Inspect the BOLD data")

for timepoint in range(T):
    print(f"Timepoint {timepoint}")
    print(fMRI_data[0, :, :, timepoint])

 """

print("Load and check the final functionnectome")
fct = nib.load(functionnectome_savepath)
fct_data = fct.get_fdata()

for i in range(T-5):
    print(fct_data[:, :, :, i])