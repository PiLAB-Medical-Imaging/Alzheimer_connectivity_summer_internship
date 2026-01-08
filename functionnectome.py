from dipy.tracking.utils import target
from dipy.tracking.streamline import select_by_rois
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
import timeit
import Functionnectome.functionnectome as funct
import numpy as np

from unravel.utils import get_streamline_density

MASK_DEBUG = False
RUN_LOOP = False

trk_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/NT1_track_msmt.trk"
voxel_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/voxel.nii"
out_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/output_tracks.tck"
trk = load_tractogram(trk_file, 'same')


def target_1(trk, mask, affine):  
    streamlines = trk.streamlines
    rel_streamlines = target(streamlines, trk.affine, mask)
    return rel_streamlines


mask = nib.load(voxel_file).get_fdata()

rel_streamlines = target_1(trk, mask, trk.affine)
trk_new = trk.from_sft(rel_streamlines, trk)
trk_new.to_vox()
trk_new.to_corner()
print(trk_new)
density = get_streamline_density(trk_new, resolution_increase=4)
print("successful to here")

#out = nib.Nifti1Image(density, trk.affine)
#out.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/density.nii.gz")
#save_tractogram(trk_new, out_file)

#print(timeit.timeit(lambda: target_1(trk, mask, trk.affine), number = 10)/10)

#print(trk.streamlines._offsets)

# try to calculate the different components: 


# Compute using the package. 

def compute_functionnectome():
    # Goal is to use the existing function first, and see how long this takes.
    pass

# Basic logic:
# Loop over the white matter voxels. 
#for 
        # Loop over the grey matter voxels. 
            # Calculate the density between the grey matter and all the relevant white matter voxels
            # Sum the values for all regions in the AAL116 atlas. Should be able to do this using the data from the nifti image file


# Convert to vectorised operations.

# Load the atlas:
atlas_path = "/Users/sam/Desktop/aal116-master/aal116MNI.nii.gz"
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
atlas_affine = atlas_img.affine

#trk = load_tractogram(trk_file, 'same')

# May need to remove these??
#trk.to_vox()
#trk.to_corner()


import numpy as np
method = 0


if method == 1:

    unique_values_in_atlas = np.unique(atlas_data)
    unique_values_in_atlas = unique_values_in_atlas[unique_values_in_atlas != 0]  # skip background

    for roi in unique_values_in_atlas:
        mask = [atlas_data == roi]       # boolean mask for this ROI
        roi_streamlines = list(select_by_rois(
            trk.streamlines,
            trk.affine,
            mask,
            include=[True],            # must be a list
            mode="either_end",
            tol=5
        ))
        print(f"ROI {roi}: {len(roi_streamlines)} streamlines")


# Alternative method
if method == 2:
    # Get all the streamlines that pass through a given voxel:
    # Need to first get the white matter mask
    wm_mask_img = nib.load("/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-MNI152NLin2009cAsym_label-WM_mask.nii.gz")
    wm_mask = wm_mask_img.get_fdata()
    # Need to generate a mask for each white matter voxel (iterate through the white matter mask)

    # First, take all the indices of locations in the scan where there is white matter.
    wm_positions = np.array(np.nonzero(wm_mask)).T
    n = len(wm_positions)

    # Make an array to store the masks. It should be n long, and then the same shape as the original white matter mask
    output = np.zeros((n,) + wm_mask.shape, dtype=wm_mask.dtype)

    # For each index in the wm_positions array, create a single 1.0 value in the mask.
    for i, idx in enumerate(wm_positions):
        output[i][tuple(idx)] = 1.0
    
    # See how long it takes to get the entire sequence of voxel densities
        # Iterate through the entire position and mask indices
    for mask in output:
        voxel_results = target_1(trk, mask, trk.affine)
        trk_new = trk.from_sft(voxel_results, trk)
        trk_new.to_vox()
        trk_new.to_corner()
        density = get_streamline_density(trk_new, resolution_increase=1)

    
def probability_maps(wm_mask, trk):
    wm_data = wm_mask.get_fdata()
    wm_positions = np.array(np.nonzero(wm_data)).T
    n = len(wm_positions)

    # Make an array to store the masks. It should be n long, and then the same shape as the original white matter mask
    output = np.zeros((n,) + wm_data.shape, dtype=wm_data.dtype)

    # For each index in the wm_positions array, create a single 1.0 value in the mask.
    for i, idx in enumerate(wm_positions):
        output[i][tuple(idx)] = 1.0

    # Check that the mask being produced is actually reasonable:
    if MASK_DEBUG == True:
        print(np.count_nonzero(output[0]))
        print("Index: ", wm_positions[0])
        out = nib.Nifti1Image(output[0].astype(np.uint8), wm_mask.affine, header=wm_mask.header.copy())
        out.to_filename("/Users/sam/Desktop/sub-TAU001/sub-TAU001_test_mask.nii.gz")

    
    mask = output[0]
    rel_streamlines = target_1(trk, mask, trk.affine)
    trk_new = trk.from_sft(rel_streamlines, trk)

    #trk_new.to_vox()
    #trk_new.to_corner()
    print(trk_new)
    density = get_streamline_density(trk_new, resolution_increase=4)



        
    # See how long it takes to get the entire sequence of voxel densities
        # Iterate through the entire position and mask indices
    if RUN_LOOP:
        for mask in output:
            voxel_results = target_1(trk, mask, trk.affine)
            trk_new = trk.from_sft(voxel_results, trk)
            trk_new.to_vox()
            trk_new.to_corner()


        density = get_streamline_density(trk_new, resolution_increase=1)

#print(timeit.timeit(lambda: target_1(trk, mask, trk.affine), number = 10)/10)


if __name__ == "__main__":
    print("Running main\n")
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()
    wm_mask_img = nib.load("/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-MNI152NLin2009cAsym_label-WM_mask.nii.gz")
    probability_maps(wm_mask_img, trk)