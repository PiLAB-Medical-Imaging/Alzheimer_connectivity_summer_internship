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
from numpy import tensordot
import matplotlib.pyplot as plt

NOISE_OFFSET = 5

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

    This method will take an absurd amount of time to run. In the order of roughly 300 hours. May need to do groups of 2x2x2 voxels? This would bring down by a factor of 8.
    
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


def vectorised_probability_maps(template_file, atlas_path, reference_file, trk, save_path, remap = False, save_output = None):
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
    labels = np.unique(nib.load(atlas_path).get_fdata())
    print("Number of labels: ", len(labels))
    # Move the tractogram to the corner of the voxel
    trk.to_vox()
    trk.to_corner()
    img = nib.load(reference_file)
    # First, match the atlas to the patient (this will be a slow step so try and cache it). Save it somewhere and then just check that filepath.
    sbj_atlas_path = save_path

    if  remap == False and path.exists(sbj_atlas_path):
        registered_atlas = nib.load(sbj_atlas_path)
    else:
        mapping = find_transform(moving_file= template_file,
                                static_file= reference_file,
                                level_iters=[1000, 100, 10],
                                diffeomorph=False)#, "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz", reference_file, level_iters=[1000, 100, 10], diffeomorph=False)
        
        registered_atlas = apply_transform(atlas_path, mapping, labels=True)

        # Save the label volume for validation

        out = nib.Nifti1Image(registered_atlas.astype(float), img.affine) 
        out.to_filename(sbj_atlas_path)
   
    # Extract the overall density map
    overall_density_map = density_map(trk.streamlines, 
                                       np.eye(4),
                                       trk.dimensions)
    
    #overall_density_map = np.ones(shape=(156, 256, 256))

    # Iterate through the AAL regions and extract only the streamlines that go through each region. (116 iterations, will give a NxN matrix for each ROI, where N is the number of white matter voxels) 
    atlas_matrix = registered_atlas.get_fdata()
    roi_ids = np.unique(atlas_matrix) 

    # Stack the density maps up into a single array.
    N = len(roi_ids)-1
    all_density_maps = np.zeros(shape=(N,156, 256, 256)) # todo: Make this reactive to the file dimensions that are input, rather than those that are hardcoded.

    root_filepath = "/Users/sam/Desktop/sub-TAU001/DensityMaps" # Replace this to make it more general @todo


    for idx, roi in tqdm(enumerate(roi_ids),"Computing ROI streamlines"):
        idx -= 1
        if roi == 0:
            continue
        filepath = path.join(root_filepath, f"TAU001_{roi}_density_map.nii.gz")

        if os.path.exists(filepath):
            roi_density_map = nib.load(filepath)
            all_density_maps[idx] = roi_density_map.get_fdata()
            continue
        
        # Get all the streamlines that reach the grey matter ROI
        mask =  (atlas_matrix == roi).astype(np.uint8)
        if mask.shape != atlas_matrix.shape:
            raise ValueError("The mask does not match the shape of the atlas")
        
        relevant_streamlines = target_1(trk = trk, 
                                        mask = mask,
                                        affine = np.eye(4))
        

        trk_new = trk.from_sft(relevant_streamlines, trk)


        #print("Tractogram Properties: ", trk_new._get_streamline_count())
        
        # Get a density map of the relevant streamlines
        roi_density_map = density_map(trk_new.streamlines, np.eye(4), trk_new.dimensions)

        # Output the density maps so that they can be visualised


        out = nib.Nifti1Image(roi_density_map.astype(float), trk.affine)
        out.to_filename(filename=filepath)

        #visual_inspection(out, roi)

        all_density_maps[idx] = roi_density_map
    

    connection_probability = all_density_maps / overall_density_map
    connection_probability = np.nan_to_num(connection_probability, True, nan=0)

    if save_output != None:
        if type(save_output) is not str:
           raise ValueError("Please ensure save_output is a string pointing to a location to save the probability maps")
        
        save_name = path.join(save_output, f"probability_maps.nii.gz")
        save_values = np.transpose(connection_probability, (1,2,3,0))
        out = nib.Nifti1Image(save_values.astype(float), trk.affine)
        out.to_filename(save_name)

    return connection_probability

def normalizer(funct_results, probability_maps, method = "basic", bold_min = None, bold_max = None):
    """
    Converts the raw functionnectome values to constrain them to within the range of the original BOLD signal. Some of these methods are experimental/require validation. 
    
    :param funct_results: Array-like
        Object containing the raw values of the functionnectome.
    :param probability_maps: Array
        Contains the probability maps for each voxel/ROI
    :param method: str
        String defining which method to use. Valid options are ('basic', 'self-sum', 'voxel_sum', 'hack')
    :param bold_min: OPTIONAL Numeric
        The minimum value in the bold data. Only required for the hack method
    :param bold_max: OPTIONAL Numeric
        The maximum value in the bold data. Only required for the hack method.
    """
    if method == "basic":
        summed_probs = np.sum(probability_maps) # Dud - makes values minute
    elif method == "self-sum": # Dud - makes values too large
        summed_probs = 0
        for probability_map in probability_maps:
            summed_probs += np.mean(probability_map)
    elif method == "voxel_sum": # Currently a dud - returns entirely NaNs.
        summed_probs = np.nansum(probability_maps, axis=0)
        print(f"Sum properties:\nNonzeros: {np.count_nonzero(summed_probs)}\nNansum: {np.nansum(summed_probs, axis = 0)}\nOriginal properties:\nNon-zeros: {np.count_nonzero(probability_maps)}")
    elif method == "hack": # I think this may sacrifice the physiological meaning of what we are doing.
        if bold_min == None or bold_max == None:
            raise ValueError("Please enter the parameters bold_min and bold_max (numeric) to use the hack method.")
        normalised = ((funct_results-funct_results.min())/(funct_results.max()-funct_results.min())) * (bold_max-bold_min)+bold_min
        return normalised
    else:
        raise ValueError("Please enter a valid method ('basic', 'self-sum', 'voxel_sum', 'hack')")
    
    normalised = funct_results/summed_probs   
    return normalised

        
def functionnectome(probability_maps, fMRI_file, brain_mask_path, registered_atlas):
    """
    Computes the functionnectome based on a probability of connection map and the fmri data.
    
    :param probability_maps: array like
        Contains the probability of connection between a voxel and regions of interest.
    :param fMRI_file: str
        Contains the fMRI BOLD data - at this point this needs to be in the MNI space (which fMRI prep produces as well) Maybe no longer as restrictive? 
    :param registered_atlas: str
        Contains the atlas that has been adapted to the patient T1 space.
    """


    bold_data = image.load_img(fMRI_file)

    masker = NiftiLabelsMasker(registered_atlas, standardize=True)

    # This is now a timepoints x ROI matrix.
    roi_time_series = masker.fit_transform(bold_data)

    plot_timeseries(roi_time_series=roi_time_series[NOISE_OFFSET:, :])

    bold_max = np.max(roi_time_series[NOISE_OFFSET:,:])
    bold_min = np.min(roi_time_series[NOISE_OFFSET:,:])
    print(f"Minimum BOLD value: {bold_min}\nMaximum BOLD value: {bold_max}") 
    
    funct_result = tensordot(roi_time_series[NOISE_OFFSET:, :], probability_maps,1) # need to double check the shapes of the roi_timeseries.

    funct_result = normalizer(funct_results=funct_result,
                                         probability_maps=probability_maps,
                                         method="hack",
                                         bold_min=bold_min,
                                         bold_max=bold_max)


    funct_result = np.transpose(funct_result, (1,2,3,0))

    print(f"Max F value: {funct_result.max()}\Min F value: {funct_result.min()}")
    out = nib.Nifti1Image(funct_result,bold_data.affine)
    out.to_filename("/Users/sam/Desktop/sub-TAU001/functionnectome.nii.gz")


def plot_timeseries(roi_time_series):
    plt.plot(roi_time_series)
    plt.show()


def visual_inspection(density_map_img, region):
    data = density_map_img.get_fdata()
    print(f"Report for {region}")
    print(f"Minimum Value: {np.min(data)}\nMaximum Value: {np.max(data)}\nNon-zeros    : {np.count_nonzero(data)}")


def create_masked_T1(t1_file, mask_file):
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    t1_data *= mask_data

    out = nib.Nifti1Image(t1_data, t1_img.affine)
    file_path = t1_file[:-7] + "_masked.nii.gz"
    out.to_filename(file_path)

test_functionnectome = True
test_white_matter_iteration = False

if __name__ == "__main__":
    atlas_path = "/Users/sam/Desktop/sub-TAU001/aal.nii.gz"
    fMRI_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    reference_file = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w.nii.gz"
    save_registered_atlas = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_desc-registered_atlas_space-T1w.nii.gz"
    tractogram_filepath = "/Users/sam/Desktop/sub-TAU001/test_tract.trk"
    wm_mask_filepath = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz"
    load_tractogram_path = "/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk"
    brain_mask_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-brain_mask.nii.gz"
    moving_file = "/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"

    print("Running main\n")
    wm_mask_img = nib.load(wm_mask_filepath)
    trk = load_tractogram(load_tractogram_path, "same")
    trk.to_vox()
    trk.to_corner()
    save_tractogram(trk, tractogram_filepath)

    if test_white_matter_iteration:
        # Generate the white matter voxel masks.

        masks_array = generate_masks(wm_mask=wm_mask_img, 
                                    test_masks=False)
        
        print("The generated mask shapes: ", masks_array.shape)
        
        probability_maps(trk, masks_array) 



    # More efficient version? 
    create_masked_T1(reference_file, brain_mask_path)
    reference_file = reference_file[:-7] + "_masked.nii.gz"
    probability_maps_computed  = vectorised_probability_maps(template_file= moving_file,
                                                             atlas_path=atlas_path,
                                                             reference_file=reference_file, 
                                                             trk=trk,
                                                             save_path=save_registered_atlas, 
                                                             remap=False,
                                                             save_output="/Users/sam/Desktop/sub-TAU001")
    

    # Functionnectome Test
    if test_functionnectome:
        functionnectome(probability_maps = probability_maps_computed, 
                        fMRI_file= fMRI_path,
                        registered_atlas=save_registered_atlas, 
                        brain_mask_path=brain_mask_path)
        


    
