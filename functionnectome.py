from dipy.tracking.utils import target
from dipy.tracking.streamline import select_by_rois
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
import timeit
import Functionnectome.functionnectome as funct
import numpy as np
from dipy.tracking.streamline import transform_streamlines
from unravel.utils import get_streamline_density
from regis.core import find_transform, apply_transform
from dipy.tracking.utils import density_map
from os import path
from tqdm import tqdm
import os
from nilearn.input_data import NiftiLabelsMasker
from nilearn import image, masking
""" 
trk_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/NT1_track_msmt.trk"
voxel_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/voxel.nii"
out_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/output_tracks.tck"
trk = load_tractogram(trk_file, 'same')



mask = nib.load(voxel_file).get_fdata()
mask_img = nib.load(voxel_file)
rel_streamlines = target_1(trk, mask, trk.affine)
trk_new = trk.from_sft(rel_streamlines, trk)
trk_new.to_vox()
trk_new.to_corner()
print(f"Printing Nicolas' example:\n {trk.affine} \n mask affine \n {mask_img.affine}")
density = get_streamline_density(trk_new, resolution_increase=4)
print("successful to here")

out = nib.Nifti1Image(density, trk.affine)
out.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/density.nii.gz")
#save_tractogram(trk_new, out_file)

#print(timeit.timeit(lambda: target_1(trk, mask, trk.affine), number = 10)/10)

#print(trk.streamlines._offsets)
 """

""" 
# Load the atlas:
atlas_path = "/Users/sam/Desktop/aal116-master/aal116MNI.nii.gz"
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
atlas_affine = atlas_img.affine

trk = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram.trk", 'same')

# May need to remove these??
#trk.to_vox()
#trk.to_corner()


import numpy as np
method = 2


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
        try:
            voxel_results = target_1(trk, mask, trk.affine)
            trk_new = trk.from_sft(voxel_results, trk)
            trk_new.to_vox()
            trk_new.to_corner()
            density = get_streamline_density(trk_new, resolution_increase=1)
        except ValueError:
            continue

     """

def target_1(trk, mask, affine):  
    """
    Generates the subset of all streamlines that pass through a given voxel/masked value.
    
    :param trk: The trk file containing the tracts
    :param mask: A brain mask highlighting a region that we want to extract streamlines for
    :param affine: For now, unused. May be needed if the function is modified to take a tck and then the affine needs to be passed.

    :return rel_streamlines: A tractogram with the streamlines that pass through the masked location.
    """
    streamlines = trk.streamlines
    rel_streamlines = target(streamlines, affine, mask)
    return rel_streamlines


def generate_masks(wm_mask, test_masks = True):
    if test_masks:
        mask_1 = nib.load("/Users/sam/Desktop/sub-TAU001/one_white_matter_mask.nii.gz")
        mask_2 = nib.load("/Users/sam/Desktop/sub-TAU001/two_white_matter_mask.nii.gz")
        return np.stack([mask_1.get_fdata(), mask_2.get_fdata()], axis=0)
    else:
        wm_data = wm_mask.get_fdata()
        wm_positions = np.array(np.nonzero(wm_data)).T
        n = len(wm_positions)

        # Make an array to store the masks. It should be n long, and then the same shape as the original white matter mask
        output = np.zeros((n,) + wm_data.shape, dtype=wm_data.dtype)

        # For each index in the wm_positions array, create a single 1.0 value in the mask.
        for i, idx in enumerate(wm_positions):
            output[i][tuple(idx)] = 1.0
        return output



    

def probability_maps(trk, mask_array, mask_debug=False):
    """
    Compute the probability that a given voxel is connected to any other voxel. This method iterates through masks.

    This method will take an absurd amount of time to run. In the order of roughly 300 hours. May need to do groups of 2x2x2 voxels? This would bring down by a factor of 8, 
    
    :param wm_mask: Description
    :param trk: Description
    """
    # Initial attempt: Iterate through all and attempt the procedure. If they all come back empty, something is wrong.
    successes = 0
    failures = 0
    streamline_count = []
    for idx, mask in tqdm(enumerate(mask_array[0:999])):
        try:
            voxel_results = target_1(trk, mask, np.eye(4))
            trk_new = trk.from_sft(voxel_results, trk)
            streamline_count.append(len(trk_new.streamlines))
            density = get_streamline_density(trk_new, resolution_increase=1)
            successes += 1
        except (ValueError, IndexError) as e:
            failures += 1

    print(f"The number of successes: {successes}")
    print(f"Streamline Characteristics\n Mean: {np.mean(streamline_count)}\nMax: {np.max(streamline_count)}")
    print(f"Streamline counts:")
    print(streamline_count)


def vectorised_probability_maps(atlas_path, reference_file, trk, remap = False):
    """
    Docstring for vectorised_probability_maps
    
    :param atlas_path: str
        Path to the file containing the atlas.
    :param reference_file: str
        Path to a reference image for the patient. Should be a nifti file. Designed with a T1w file, could be a trk file with affine info.
    :param trk: Tractogram object.
        Tractogram for the subject
    :param remap: Boolean
        Determines if a new mapping is calculated, rather than using a saved one.

    :returns connection_probability
        An array - shape is r x 156 X 256 x 256, where r is the number of regions of interest defined.
    """
    
    # Move the tractogram to the corner of the voxel
    trk.to_vox()
    trk.to_corner()
    img = nib.load(reference_file)
    # First, match the atlas to the patient (this will be a slow step so try and cache it). Save it somewhere and then just check that filepath.
    sbj_atlas_path = "/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz"
    if  remap == False and path.exists(sbj_atlas_path):
        registered_atlas = nib.load(sbj_atlas_path)
    else:
        mapping = find_transform(moving_file="/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
                                static_file= reference_file,
                                level_iters=[1000, 100, 10],
                                diffeomorph=False)#, "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz", reference_file, level_iters=[1000, 100, 10], diffeomorph=False)
        
        registered_atlas = apply_transform(atlas_path, mapping, labels=True)

        # Save the label volume for validation

        out = nib.Nifti1Image(registered_atlas.astype(float), img.affine) 
        out.to_filename(sbj_atlas_path)
   
    # Extract the overall density map
    """     overall_density_map = density_map(trk.streamlines, 
                                       trk.affine,
                                       img.shape) """
    overall_density_map = np.ones(shape=(156, 256, 256))

    # Iterate through the AAL regions and extract only the streamlines that go through each region. (116 iterations, will give a NxN matrix for each ROI, where N is the number of white matter voxels) 
    atlas_matrix = registered_atlas.get_fdata()
    roi_ids = np.unique(atlas_matrix) # Bug currently: Only getting 37 indices, rather than 116. 

    # Stack the density maps up into a single array.
    all_density_maps = np.zeros(shape=(len(roi_ids),156, 256, 256))
    atlas = nib.load(atlas_path)
    
    for idx, roi in tqdm(enumerate(roi_ids),"Loading ROI streamlines"):
        if roi == 0:
            continue
        #print(f"Extracting Streamlines for Region {roi}")

        # Get all the streamlines that reach the grey matter ROI
        relevant_streamlines = target_1(trk = trk, 
                                        mask = (atlas_matrix == roi).astype(np.uint8),
                                        affine = np.eye(4))
        
        trk_new = trk.from_sft(relevant_streamlines, trk)
        #print("Tractogram Properties: ", trk_new._get_streamline_count())
        
        # Get a density map of the relevant streamlines
        roi_density_map = density_map(relevant_streamlines, trk.affine, trk.dimensions)

        # Output the density maps so that they can be visualised
        root_filepath = "/Users/sam/Desktop/sub-TAU001/DensityMaps"
        filepath = path.join(root_filepath, f"TAU001_{roi}_density_map.nii.gz")

        #out = nib.Nifti1Image(roi_density_map.astype(float), trk.affine)
        out = nib.Nifti1Image(roi_density_map.astype(float), trk.affine)
        out.to_filename(filename=filepath)

        all_density_maps[idx] = roi_density_map
    

    connection_probability = all_density_maps / overall_density_map

    return connection_probability



        
def functionnectome(probability_maps, fMRI_file, registered_atlas):
    """
    Computes the functionnectome based on a probability of connection map and the fmri data.
    
    :param probability_maps: array like
        Contains the probability of connection between a voxel and regions of interest.
    :param fMRI_file: str
        Contains the fMRI BOLD data - at this point this needs to be in the MNI space (which fMRI prep produces as well)
    :param registered_atlas: str
        Contains the atlas that has been adapted to the patient T1 space.
    """


    bold_img = image.load_img(fMRI_file)
    mask = nib.load("/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_desc-brain_mask.nii.gz")
    # Resample the mask to make sure they are compatible
    mask_resampled = image.resample_to_img(mask,
                                           bold_img,
                                           interpolation="nearest", 
                                            force_resample=True, 
                                            copy_header=True )

    masked = masking.apply_mask(bold_img, mask_resampled)


    bold_data = bold_img.get_fdata()
    print("Data shape", bold_data.shape)

    print("Probability Map shape", probability_maps.shape)

    # Try giving it the atlas in MNI space

    masker = NiftiLabelsMasker("/Users/sam/Desktop/aal116-master/aal116MNI.nii.gz", standardize=True)
    roi_time_series = masker.fit_transform(bold_img)
    print(roi_time_series.shape)





test_functionnectome = True
test_white_matter_iteration = False

if __name__ == "__main__":
    atlas_path = "/Users/sam/Desktop/aal116-master/aal116MNI.nii.gz"
    fMRI_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"

    print("Running main\n")
    wm_mask_img = nib.load("/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz")
    trk = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk", "same")
    trk.to_vox()
    trk.to_corner()
    save_tractogram(trk, "/Users/sam/Desktop/sub-TAU001/test_tract.trk")

    if test_white_matter_iteration:
        # Generate the white matter voxel masks.

        masks_array = generate_masks(wm_mask=wm_mask_img, 
                                    test_masks=False)
        
        print("The generated mask shapes: ", masks_array.shape)
        
        probability_maps(trk, masks_array) 


    # More efficient version? 
    reference_file = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w.nii.gz"
    probability_maps_computed  = vectorised_probability_maps(atlas_path,reference_file, trk)

    print(probability_maps_computed.shape)


    # Functionnectome Test
    if test_functionnectome:
        functionnectome(probability_maps = probability_maps_computed, 
                        fMRI_file= "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        registered_atlas="/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz")

    
