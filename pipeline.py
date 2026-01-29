from os import path
import os
import numpy as np
import sparse
import matplotlib.pyplot as plt
from string import ascii_letters
import pandas as pd
import seaborn as sns

import nibabel as nib
from nilearn.plotting import plot_matrix, show
from nilearn import image
from dipy.io.streamline import load_tractogram

from dipy.io.stateful_tractogram import StatefulTractogram

## My Imports (replace these with my package calls)
from utilities import connectivity_matrix_generation, visualise_square_mat, nifti_vs_img,mask_generator
from utilities import normalise, create_masked_T1, atlas_registration, mask_to_positions, voxel_to_streamline_map_V2
from utilities import create_ROI_time_series, create_VOX_time_series
from engagement import correlation_thresholding, ebc_computation
from engagement import generate_VWSC_matrices, engagement_calculation
from engagement import save_connectivity_matrices,save_engagement
from functionnectome import compute_connection_probability
from functionnectome import vectorised_probability_maps, functionnectome

CSFP_PATH = "CSF_probseg.nii.gz"
GMP_PATH = "GM_probseg.nii.gz"
WMP_PATH = "WM_probseg.nii.gz"

TEST_FUNCTIONNECTOME  = False
TEST_ENGAGEMENT = True

def functionnectome_pipeline(
        atlas_path, 
        fMRI_path,
        tractogram, 
        grey_matter_mask,
        t1w_file = None, 
        anatomical_scan_atlas_space = None,
        functionnectome_savepath = None, 
        mode = "roi", 
        register_atlas = False,
        v2f_mapping = None):
    
    task = 1

    # Load the tractogram and shift to voxel corner
    print(f"{task}. Load Tractogram")

    # Allow user to pass either the filepath or tractogram

    if type(tractogram) is str:
        trk = load_tractogram(tractogram, "same")
    elif type(tractogram) is StatefulTractogram:
        trk = tractogram

    # Shift to correct reference frame
    trk.to_vox()
    trk.to_corner()
    task += 1

 
    # Generate the probability maps
    print(f"{task}. Generate density maps")

    if register_atlas:
        
        if (anatomical_scan_atlas_space is None
                or t1w_file is None):
            raise ValueError("Please provide both an " \
                            "anatomical_scan_atlas_space "
                            "and t1w_file")
        

        registered_atlas = atlas_registration(
                            atlas_path=atlas_path,
                            template_file=anatomical_scan_atlas_space,
                            reference_file=t1w_file)
    else:
        registered_atlas = atlas_path

    grey_matter_mask_img = nifti_vs_img(grey_matter_mask)
    gm_positions = mask_to_positions(grey_matter_mask_img)
    registered_atlas = nifti_vs_img(registered_atlas)

    if v2f_mapping is None:
        v2f_mapping = voxel_to_streamline_map_V2(
            streamlines=trk.streamlines,
            vol_shape=trk.dimensions,
            subsegment=10)
    
    non_zeros = 0
    for key in v2f_mapping.keys():
        if len(v2f_mapping[key]) != 0:
            non_zeros += 1
    if non_zeros == 0:
        raise ValueError("All the voxels have no streamlines")

    all_density_maps, overall_density_map  = vectorised_probability_maps(
        registered_atlas=registered_atlas,
        trk=trk, 
        v2f_mapping=v2f_mapping,
        mask_positions=gm_positions,
        brain_template=grey_matter_mask_img,
        smoothing=False,
        mode=mode,
        verbose=False
    )

    if np.count_nonzero(all_density_maps) == 0:
        raise ValueError("All density maps are 0")
    if np.count_nonzero(overall_density_map) == 0:
        raise("All values in overall density are 0")
  
    task += 1

    print(f"{task}. Generate connection probability")

    probability_maps_computed = compute_connection_probability(
        overall_density_map=overall_density_map, 
        all_density_maps=all_density_maps)
    
    non_zeros = np.count_nonzero(probability_maps_computed)

    if non_zeros == 0:
        raise ValueError("Probability maps are entirely zeros")
    print(f"Nans: {np.sum(np.isnan(probability_maps_computed))}")
    task += 1

    # Calculate the functionnectome
    print(f"{task}. Compute Functionnectome")
    if mode == "roi" and grey_matter_mask != None:
        print("\tWarning, ignoring grey matter mask due to ROI mode selection!")
    bold_img = nib.load(fMRI_path)

    if mode == "roi":
        time_series = create_ROI_time_series(
            atlas=registered_atlas,
            bold_data = bold_img,
            bold_filepath=fMRI_path
        )
    elif mode == "vox":
        time_series = create_VOX_time_series(
            mask=gm_mask,
            bold_data=bold_img,
            bold_filepath=fMRI_path
        )
    else:
        raise ValueError("Please enter a valid mode")
    
    if np.count_nonzero(time_series) == 0:
        raise ValueError("All timeseries values are 0")
    
    funct_result = functionnectome(
                    probability_maps = probability_maps_computed,
                    timeseries=time_series,
                    registered_atlas=registered_atlas,
                    extensive_visualisation=False,
                    normalisation="none"
    )
    
    task += 1

    if functionnectome_savepath != None:
        print(f"{task}. Saving Functionnectome")
        out = nib.Nifti1Image(funct_result, grey_matter_mask_img.affine)
        out.to_filename(functionnectome_savepath)

    print("Complete!")

def engagement_pipeline(bold_data, atlas, 
                        tractogram_file, 
                        white_matter_prob, 
                        cache_pathway = None, 
                        plotting = False, 
                        save_engagement_filepath = None, 
                        verbose = False, 
                        confound_removal = False,
                        save_connectomes = False, 
                        white_matter_mask = None):
    """
    Pipeline that performs the entire engagement calculation - functions within 
    this will correspond to submodules that can be run with just the required 
    objects. This function works with the
    filepaths.
    
    :param bold_data: str
        Preprocessed bold data. Can be in any space (T1w, MNI), however this 
        must match the space of the atlas.
    :param atlas: str
        Filepath to the atlas. Space must match the bold data space
    :param tractogram_file: str
        Filepath for the tractogram
    :param white_matter_prob: str
        Filepath for the white matter probability map.
    :param plotting: boolean
        Default is False. If true, the correlation matrix will be plotted.
    :param save_engagement: str
        A filepath to save the engagement results.
    :param verbose: Boolean
        If true, more printing will occur.
    :param grey_matter_prob: Optional (unused as of right now)
    :param csf_prob: Optional (unused as of right now)
    :param save_connectomes: Boolean
        If true, the connectivity matrices for each voxel will be stored. 
        Be warned, that this generated at least 1-2 gb per patient. Save 
        location defaults to the same location as save_engagement
    """
    task = 1
    ################################ Step 1 ################################
    print(f"{task}. Preparing functional connectivity matrix")
    # Load the bold data and the atlas

    bold_img = image.load_img(bold_data)
    bold_img_data = bold_img.get_fdata()
    bold_img_data = bold_img_data[:, :, :, 3:]
    atlas_img = nib.load(atlas)
    atlas_data = atlas_img.get_fdata()

    # Check if the functional connectivity matrix already exists
    if cache_pathway is not None:
        fc_path = path.join(cache_pathway, "fc_matrix.npy")
        if path.exists(fc_path):
            fc_mat = np.load(fc_path)
        else:
            if confound_removal:
                fc_mat = connectivity_matrix_generation(
                            bold_img, 
                            atlas, 
                            False, 
                            bold_filepath=bold_data, 
                            method="nilearn")
            else:
                 fc_mat = connectivity_matrix_generation(
                            bold_img, 
                            atlas, 
                            False, 
                            method="nilearn")
    else:
        if confound_removal:
            fc_mat = connectivity_matrix_generation(
                        bold_img, atlas, False, 
                        bold_filepath=bold_data, 
                        method="nilearn")
        else:
            fc_mat = connectivity_matrix_generation(
                        bold_img, atlas, 
                        False, method="nilearn")
    
    if verbose:
        #print("The atlas looks like: ", atlas_data)
        #print("Atlas values are", atlas_values)
        print("The FC matrix looks like: ", fc_mat)

    # Plotting to see if the matrix makes sense (as of right now it does not!!!)
    if plotting:
        np.fill_diagonal(fc_mat, 0)
        plot_matrix(
        fc_mat,
        labels = (np.unique(atlas_img.get_fdata())
                  [np.unique(atlas_img.get_fdata())!=0]),
        figure=(10, 8),
        vmax=0.8,
        vmin=-0.8,
        title="Raw correlations",
        reorder=True,
        )
        plt.show()

    # Threshold the correlations to make it amenable to the EBC metric 
    # (may make sense to replace this with a metric that more accurately 
    # characterises the degree of "proximity" a node has to other nodes")
    fc_mat = correlation_thresholding(fc_mat, value_threshold= 0.2)

    if verbose:
        print("After thresholding: ", fc_mat)

    task+=1

    ################################ Step 2 ################################
    print(f"{task}. Computing Edge Between Connectedness Matrix") 

    ebc_mat = ebc_computation(fc_mat, inverted_values=False)

    if verbose:
        print("The ebc matrix")
        print(ebc_mat)
        print(ebc_mat.shape)

        print("EBC Report")
        print(f"NonZeros: {np.count_nonzero(ebc_mat)}\nMax: {ebc_mat.max()}")
        print(f"Min: {ebc_mat.min()}\nUNique values: {len(np.unique(ebc_mat))}")
        visualise_square_mat(ebc_mat, "EBC Visualisation")
    task += 1

    ################################ Step 3 ################################
    print(f"{task}. Computing all fibres that penetrate each voxel")   

    trk = load_tractogram(tractogram_file, "same")

    print(type(atlas_data))


    if white_matter_mask is not None:
         all_connectivity_matrices, wm_positions = generate_VWSC_matrices(
                                            white_matter_mask=white_matter_mask,
                                            trk=trk,
                                            atlas_data=atlas_data,
                                            verbose=verbose,
                                            segmentation=10
                                            )
    else:
        all_connectivity_matrices, wm_positions = generate_VWSC_matrices(
                                            white_matter_prob=white_matter_prob,
                                            trk=trk,
                                            atlas_data=atlas_data,
                                            verbose=verbose,
                                            segmentation=10
                                            )

    if verbose:
        print("The connectivity matrices:")
        print(all_connectivity_matrices)

    all_connectivity_matrices = sparse.asnumpy(all_connectivity_matrices)

    if save_connectomes and save_engagement_filepath != None:
        root_path = path.dirname(save_engagement_filepath)

        connectome_path = path.join(root_path, "connectivity_matrices.npy")
        save_connectivity_matrices(
            all_connectivity_mats=all_connectivity_matrices,
            save_path=connectome_path)

    task += 1
    ################################ Step 4 ################################
    print(f"{task}. Calculating Engagement")   

    engagement = engagement_calculation(EBC_matrix=ebc_mat,
                                        SC_matrices=all_connectivity_matrices,
                                        method = "custom")
    #engagement = normalise(engagement)
    print(f"Engagement Scorecard:\nMin:{engagement.min()}\nMax: {engagement.max()}")
    print(f"Unique Values: {len(np.unique(engagement))}")
    print(f"Number of non-zeros: {np.count_nonzero(engagement)}")


    task += 1

    ################################ Step 5 ################################
    if save_engagement_filepath!= None:
        print(f"{task}. Saving Result")
        print(engagement.shape)
        engagement = sparse.asnumpy(engagement)
        save_engagement(
            engagement, 
            wm_positions, 
            atlas_data.shape, 
            save_engagement_filepath, 
            atlas_img.affine)
    
    print("Finito!")

def anatamoy_crawler(anatomy_path):
    for file in os.listdir(anatomy_path):
        pass

def the_grand_central_pipeline(anatomy_path):
    pass

if __name__ == "__main__":

    if TEST_ENGAGEMENT:
        bold_filepath = ("/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-"
            "TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz")
        atlas_filepath = ("/Users/sam/Desktop/sub-TAU001/dilated_atlas_"
            "TAU001.nii.gz")
        tractogram_file = ("/Users/sam/Desktop/sub-TAU001/"
                            "TAU_1_ses-2_tractogram.trk")
        gm_prob = ("/Users/sam/Desktop/sub-TAU001/anat/"
            "sub-TAU001_label-GM_probseg.nii.gz"), 
        wm_mask = ("/Users/sam/Desktop/sub-TAU001/test_mask_red.nii.gz")
        wm_prob = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-"
            "WM_probseg.nii.gz")
        csf_prob = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-"
            "CSF_probseg.nii.gz")
        engagement_save_path = ("/Users/sam/Desktop/sub-TAU001/anat/02_"
            "threshold_engagement_10x.nii.gz")
        
        engagement_pipeline(
            bold_data=bold_filepath,
            atlas=atlas_filepath,
            white_matter_prob=wm_prob,
            tractogram_file=tractogram_file,
            save_engagement_filepath=engagement_save_path, 
            verbose=True, 
            plotting=False, 
            confound_removal=True, 
            save_connectomes=True)

        
    if TEST_FUNCTIONNECTOME:
        atlas_path = "/Users/sam/Desktop/sub-TAU001/aal.nii.gz"
        atlas_filepath = ("/Users/sam/Desktop/sub-TAU001/dilated_atlas_"
            "TAU001.nii.gz")
        fMRI_path = ("/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_"
            "ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz")
        reference_file = ("/Users/sam/Desktop/sub-TAU001/anat/sub-"
            "TAU001_desc-preproc_T1w.nii.gz")
        save_registered_atlas = ("/Users/sam/Desktop/sub-TAU001/sub-TAU001_"
            "desc-registered_atlas_space-T1w.nii.gz")
        tractogram_filepath = "/Users/sam/Desktop/sub-TAU001/test_tract.trk"
        wm_mask_filepath = ("/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-"
            "T1w_label-WM_mask.nii.gz")
        brain_mask_path = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_"
            "desc-brain_mask.nii.gz")
        moving_file = ("/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_"
            "brain.nii.gz")
        density_map_path = "/Users/sam/Desktop/sub-TAU001"
        functionnectome_savepath = ("/Users/sam/Desktop/sub-TAU001/"
            "functionnectome_streamlinecheck.nii.gz")
        gm_prob = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-"
            "GM_probseg.nii.gz")
        t1w_filepath = ("/Users/sam/Desktop/sub-TAU001/anat/sub-"
            "TAU001_desc-preproc_T1w_brain_only.nii.gz")
        wm_prob = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-"
            "WM_probseg.nii.gz")
        csf_prob = ("/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-"
            "CSF_probseg.nii.gz")


        print("Testing the pipeline")

        gm_mask = mask_generator(
            white_matter_probability=wm_prob,
            grey_matter_probability=gm_prob,
            csf_probability=csf_prob)


        functionnectome_pipeline(
            atlas_path=atlas_filepath,
            fMRI_path=fMRI_path,
            tractogram=tractogram_filepath,
            grey_matter_mask=gm_mask,
            functionnectome_savepath=functionnectome_savepath)
        
    else:
        the_grand_central_pipeline()



""" 
Detritus:

   # Create a masked T1w file. 

    if brain_only_t1w_path != None:
        print(f"{task}. Loading brain-only T1w scan")
        brain_only_t1w_path = brain_only_t1w_path
        brain_only_t1w = nib.load(brain_only_t1w_path)
    elif brain_mask_path != None:
        save_location = t1w_file[:-7]+"_brain_only.nii.gz"
        print(f"{task}. Generate brain-only T1w scan")
        brain_only_t1w = create_masked_T1(t1w_file, brain_mask_path, save_location) 
        brain_only_t1w_path = t1w_file[:-7] + "_masked.nii.gz"
    elif brain_mask_path == None and brain_only_t1w_path == None:
        raise ValueError("Please provide either a brain_only_t1w path, "
                        "or a brain_mask_t1w path and t1w_file path")
    task += 1


"""