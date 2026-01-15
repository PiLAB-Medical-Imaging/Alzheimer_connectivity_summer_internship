from dipy.tracking.utils import target
from dipy.tracking.streamline import select_by_rois
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
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
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn import image, masking
from numpy import tensordot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from nilearn.regions import signals_to_img_labels

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


def generate_masks(wm_mask, test_masks = False):
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


def vectorised_probability_maps(template_file, atlas_path, reference_file, trk, save_path, remap = False, save_output = None, smoothing = False, mode = "roi", save_density_map_path = None):
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
    # Move the tractogram to the corner of the voxel
    trk.to_vox()
    trk.to_corner()
    img = reference_file

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
    
    if smoothing:
        overall_density_map = gaussian_filter(overall_density_map, 1.0)
    

    # Iterate through the AAL regions and extract only the streamlines that go through each region. (116 iterations, will give a NxN matrix for each ROI, where N is the number of white matter voxels) 

    if save_density_map_path != None:
        filepath = path.join(save_density_map_path, f"density_map.nii.gz")
        if os.path.exists(filepath):
                print("\tLoading Pre-existing density maps")
                all_density_maps = nib.load(filepath)
                all_density_maps = all_density_maps.get_fdata()
                return all_density_maps, overall_density_map

    if mode == "roi":
        atlas_matrix = registered_atlas.get_fdata()
        roi_ids = np.unique(atlas_matrix) 

        # Stack the density maps up into a single array.
        N = len(roi_ids)-1
        all_density_maps = np.zeros(shape=(N,overall_density_map.shape[0], overall_density_map.shape[1], overall_density_map.shape[2]))

        for idx, roi in tqdm(enumerate(roi_ids),"\tComputing ROI streamlines"):
            idx -= 1
            if roi == 0:
                continue
            # Get all the streamlines that reach the grey matter ROI
            mask =  (atlas_matrix == roi).astype(np.uint8)


            if mask.shape != atlas_matrix.shape:
                raise ValueError("The mask does not match the shape of the atlas")
            
            relevant_streamlines = target_1(trk = trk, 
                                            mask = mask,
                                            affine = np.eye(4))
            

            trk_new = trk.from_sft(relevant_streamlines, trk)

            # Get a density map of the relevant streamlines
            # Use the white matter mask here? 
            roi_density_map = density_map(trk_new.streamlines, np.eye(4), trk_new.dimensions)


            if smoothing:
                roi_density_map = gaussian_filter(roi_density_map, 1.0)
            # Output the density maps so that they can be visualised


            all_density_maps[idx] = roi_density_map
        
        if save_density_map_path != None:
            complete_density_map_data = np.transpose(all_density_maps, (1,2,3,0))
            out = nib.Nifti1Image(complete_density_map_data.astype(float), trk.affine)
            out.to_filename(filename=filepath)

    elif mode == "vox":
        print("Voxel mode is not implemented... yet")
    else:
        raise ValueError("Please enter a valid mode. Valid modes are: 'roi', 'vox'")
        
    return all_density_maps, overall_density_map

def compute_connection_probability(overall_density_map, all_density_maps, save_output, trk):
    try:
        connection_probability = all_density_maps / overall_density_map
        connection_probability = np.nan_to_num(connection_probability, True, nan=0)
    except ValueError:
        all_density_maps = np.transpose(all_density_maps, (3,0,1,2))
        connection_probability = all_density_maps / overall_density_map
        connection_probability = np.nan_to_num(connection_probability, True, nan=0)


    if save_output != None:
        if type(save_output) is not str:
           raise ValueError("Please ensure save_output is a string filepath to save the probability maps")
        
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
    if np.sum(np.isnan(funct_results))>0:
        raise ValueError("Unscaled Functionnectome contains NAN values.")

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

        
def functionnectome(probability_maps, fMRI_file, registered_atlas, extensive_visualisation=None, debug_prints= False):
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
    roi_time_series = roi_time_series[NOISE_OFFSET:, :]

    if extensive_visualisation != None:
        reg_atlas_img = nib.load(registered_atlas)
        img_by_ROI = signals_to_img_labels(signals=roi_time_series,
                                        labels_img=reg_atlas_img)
        img_by_ROI.to_filename(extensive_visualisation)

    bold_max = np.max(roi_time_series)
    bold_min = np.min(roi_time_series)
    if debug_prints:
        print(f"Minimum BOLD value: {bold_min}\nMaximum BOLD value: {bold_max}") 
    
    funct_result = tensordot(roi_time_series, probability_maps,1) # need to double check the shapes of the roi_timeseries.

    funct_result = normalizer(funct_results=funct_result,
                                         probability_maps=probability_maps,
                                         method="hack",
                                         bold_min=bold_min,
                                         bold_max=bold_max)


    funct_result = np.transpose(funct_result, (1,2,3,0))
    if debug_prints:
        print(f"Max F value: {funct_result.max()}\nMin F value: {funct_result.min()}")

    return funct_result


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
    return out


def functionnectome_pipeline(atlas_path, fMRI_path, t1w_file, wm_mask_filepath, tractogram, anatomical_scan_atlas_space, save_registered_atlas, savepath_density_map, brain_mask_path = None, save_probability_maps = None, remap = False, functionnectome_savepath = None):
    task = 1

    # Load the tractogram and shift to voxel corner
    print(f"{task}. Load Tractogram")
    if type(tractogram) is str:
        trk = load_tractogram(tractogram, "same")    # Allows the user to pass either a string to the file, or the tractogram already loaded.
    elif type(tractogram) is StatefulTractogram:
        trk = tractogram
    trk.to_vox()
    trk.to_corner()
    task += 1


    # Create a masked T1w file. 
    if brain_mask_path != None:
        print(f"{task}. Generate brain-only T1w scan")
        brain_only_t1w = create_masked_T1(t1w_file, brain_mask_path) # Brain only t1w as a nifti image.
    else:
        print(f"{task}. Load brain-only T1w scan")
        brain_only_t1w = nib.load(t1w_file)
    task += 1

    # Generate the probability maps
    print(f"{task}. Generate density maps")
    all_density_maps, overall_density_map  = vectorised_probability_maps(template_file= anatomical_scan_atlas_space,
                                                            atlas_path=atlas_path,
                                                            reference_file=brain_only_t1w, 
                                                            trk=trk,
                                                            save_path=save_registered_atlas, 
                                                            remap=remap,
                                                            save_output=save_probability_maps, 
                                                            smoothing=False,
                                                            save_density_map_path=savepath_density_map)
    task += 1

    
    print(f"{task}. Generate connection probability")
    probability_maps_computed = compute_connection_probability(overall_density_map=overall_density_map, 
                                                               all_density_maps=all_density_maps,
                                                               save_output=savepath_density_map,
                                                               trk = trk)
    task += 1
    # Calculate the functionnectome
    print(f"{task}. Compute Functionnectome")
    funct_result = functionnectome(probability_maps = probability_maps_computed, 
                            fMRI_file= fMRI_path,
                            registered_atlas=save_registered_atlas)
    task += 1

    
    if functionnectome_savepath != None:
        print(f"{task}. Saving Functionnectome")
        out = nib.Nifti1Image(funct_result, brain_only_t1w.affine)
        out.to_filename(functionnectome_savepath)

    print("Complete!")





def plot_ROI_activity(registered_atlas, roi_timeseries):
    """
    Goal is to prepare a visualisation that contains the bold signal in each ROI 
    
    :param registered_atlas: Description
    :param roi_timeseries: Description
    """
    atlas_img = nib.load(registered_atlas)
    atlas_data = atlas_img.get_fdata()
    image_data = np.zeros(shape=(roi_timeseries.shape[0], roi_timeseries.shape[1], atlas_data.shape[0], atlas_data.shape[1], atlas_data.shape[2]))
    print("Shape is", image_data.shape)

    for idx in tqdm(range(roi_timeseries.shape[1]), "Preparing ROI Images"):
        for j in range(roi_timeseries.shape[0]):
            roi = idx + 1
            image_data[j, idx] = np.where(registered_atlas==roi, 
                                          roi_timeseries[j, idx], 
                                          np.nan)

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
    density_map_path = "/Users/sam/Desktop/sub-TAU001"
    functionnectome_savepath = "/Users/sam/Desktop/sub-TAU001/functionnectome.nii.gz"

    print("Testing the pipeline")
    functionnectome_pipeline(atlas_path=atlas_path,
                             fMRI_path=fMRI_path,
                             t1w_file=reference_file,
                             wm_mask_filepath=wm_mask_filepath,
                             tractogram=tractogram_filepath,
                             anatomical_scan_atlas_space=moving_file, 
                             brain_mask_path=brain_mask_path,
                             save_registered_atlas=save_registered_atlas,
                             savepath_density_map =    density_map_path,
                             functionnectome_savepath=functionnectome_savepath                         
                             )
