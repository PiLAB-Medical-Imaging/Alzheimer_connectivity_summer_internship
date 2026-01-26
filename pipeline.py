from os import path
import logging
import numpy as np
import sparse
from matplotlib.pyplot import plt

import nibabel as nib
from nilearn.plotting import plot_matrix, show
from nilearn import image
from dipy.io.streamline import load_tractogram

from dipy.io.stateful_tractogram import StatefulTractogram

## My Imports (replace these with my package calls)
from utilities import connectivity_matrix_generation
from engagement import correlation_thresholding, ebc_computation, generate_VWSC_matrices
from engagement import engagement_calculation, save_connectivity_matrices,save_engagement
from functionnectome import create_masked_T1, compute_connection_probability, vectorised_probability_maps, functionnectome

TEST_FUNCTIONNECTOME  = False
TEST_ENGAGEMENT = True

def functionnectome_pipeline(atlas_path, fMRI_path, t1w_file, tractogram, anatomical_scan_atlas_space, 
                             save_registered_atlas, savepath_density_map, brain_mask_path = None, save_probability_maps = None, 
                             remap = False, functionnectome_savepath = None, grey_matter_path = None, white_matter_prob = None,
                             csf_prob = None, gm_mask = None, mode = "roi", is_aligned = False, brain_only_t1w_path = None):
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

    if brain_only_t1w_path != None:
        print(f"{task}. Loading brain-only T1w scan")
        brain_only_t1w_path = brain_only_t1w_path
        brain_only_t1w = nib.load(brain_only_t1w_path)
    elif brain_mask_path != None:
        print(f"{task}. Generate brain-only T1w scan")
        brain_only_t1w = create_masked_T1(t1w_file, brain_mask_path) 
        brain_only_t1w_path = t1w_file[:-7] + "_masked.nii.gz"
    elif brain_mask_path == None and brain_only_t1w_path == None:
        raise ValueError("Please provide either a brain_only_t1w path, or a brain_mask_t1w path and t1w_file path")
    task += 1

    # Generate the probability maps
    print(f"{task}. Generate density maps")
    all_density_maps, overall_density_map  = vectorised_probability_maps(template_file= anatomical_scan_atlas_space,
                                                            atlas_path=atlas_path,
                                                            reference_file= brain_only_t1w_path, 
                                                            trk=trk,
                                                            save_path=save_registered_atlas, 
                                                            remap=remap,
                                                            save_output=save_probability_maps, 
                                                            smoothing=False,
                                                            save_density_map_path=savepath_density_map,
                                                            mode=mode,
                                                            grey_matter_probs=grey_matter_path,
                                                            white_matter_probabilities=white_matter_prob,
                                                            csf_probability=csf_prob,
                                                            aligned=is_aligned
                                                            )
    task += 1

    print(f"{task}. Generate connection probability")
    probability_maps_computed = compute_connection_probability(overall_density_map=overall_density_map, 
                                                               all_density_maps=all_density_maps,
                                                               save_output=savepath_density_map,
                                                               trk = trk)
    task += 1
    # Calculate the functionnectome
    print(f"{task}. Compute Functionnectome")

    if mode == "roi" and grey_matter_path != None:
        print("\tWarning, ignoring grey matter mask due to ROI mode selection!")
        grey_matter_path = None

    funct_result = functionnectome(probability_maps = probability_maps_computed, 
                            fMRI_file= fMRI_path,
                            registered_atlas=save_registered_atlas,
                            grey_matter_mask=grey_matter_path)
    task += 1

    
    if functionnectome_savepath != None:
        print(f"{task}. Saving Functionnectome")
        out = nib.Nifti1Image(funct_result, brain_only_t1w.affine)
        out.to_filename(functionnectome_savepath)

    print("Complete!")

def engagement_pipeline(bold_data, atlas, 
                        tractogram_file, 
                        white_matter_prob, 
                        cache_pathway = None, 
                        plotting = False, 
                        save_engagement_filepath = None, 
                        verbose = False, 
                        grey_matter_prob = None,  
                        csf_prob = None,
                        confound_removal = False,
                        save_connectomes = False):
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

    atlas_values = np.unique(atlas_data)

    # Check if the functional connectivity matrix already exists
    if cache_pathway is not None:
        fc_path = path.join(cache_pathway, "fc_matrix.npy")
        if path.exists(fc_path):
            fc_mat = np.load(fc_path)
        else:
            if confound_removal:
                fc_mat = connectivity_matrix_generation(bold_img, 
                                                        atlas, 
                                                        False, 
                                                        bold_filepath=bold_data, 
                                                        method="nilearn")
            else:
                 fc_mat = connectivity_matrix_generation(bold_img, 
                                                         atlas, 
                                                         False, 
                                                         method="nilearn")
    else:
        if confound_removal:
            fc_mat = connectivity_matrix_generation(bold_img, atlas, False, 
                                                    bold_filepath=bold_data, 
                                                    method="nilearn")
        else:
            fc_mat = connectivity_matrix_generation(bold_img, atlas, 
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
        labels = np.unique(atlas_img.get_fdata())[np.unique(atlas_img.get_fdata())!=0],
        figure=(10, 8),
        vmax=0.8,
        vmin=-0.8,
        title="Raw correlations",
        reorder=True,
        )
        plt.show()

    # Threshold the correlations to make it amenable to the EBC metric (may make sense to replace 
    # this with a metric that more accurately characterises the degree of "proximity" a node has to other nodes")
    fc_mat = correlation_thresholding(fc_mat, value_threshold=0.2)

    if verbose:
        print("After thresholding: ", fc_mat)

    task+=1

    ################################ Step 2 ################################
    print(f"{task}. Computing Edge Between Connectedness Matrix") 
    ebc_mat = ebc_computation(fc_mat)
    if verbose:
        print("The ebc matrix")
        print(ebc_mat)
        print(ebc_mat.shape)

        print("EBC Report")
        print(f"NonZeros: {np.count_nonzero(ebc_mat)}\nMax: {ebc_mat.max()}")
        print(f"Min: {ebc_mat.min()}\nUNique values: {len(np.unique(ebc_mat))}")
    task += 1

    ################################ Step 3 ################################
    print(f"{task}. Computing all fibres that penetrate each voxel")   

    trk = load_tractogram(tractogram_file, "same")

    print(type(atlas_data))
    all_connectivity_matrices, wm_positions = generate_VWSC_matrices(
                                            white_matter_prob=white_matter_prob,
                                            trk=trk,
                                            atlas_data=atlas_data,
                                            verbose=verbose
                                            )

    if verbose:
        print("The connectivity matrices:")
        print(all_connectivity_matrices)

    all_connectivity_matrices = sparse.asnumpy(all_connectivity_matrices)

    if save_connectomes and save_engagement_filepath != None:
        root_path = path.dirname(save_engagement_filepath)

        connectome_path = path.join(root_path, "connectivity_matrices.npy")
        save_connectivity_matrices(all_connectivity_mats=all_connectivity_matrices,
                                   save_path=connectome_path)
       

    task += 1
    ################################ Step 4 ################################
    print(f"{task}. Calculating Engagement")   

    engagement = engagement_calculation(EBC_matrix=ebc_mat,
                                        SC_matrices=all_connectivity_matrices,
                                        method = "einsum")

    print(f"Engagment Scorecard:\nMin:{engagement.min()}\nMax: {engagement.max()}")
    print(f"Unique Values: {len(np.unique(engagement))}")
    task += 1

    ################################ Step 5 ################################
    if save_engagement_filepath!= None:
        print(f"{task}. Saving Result")
        print(engagement.shape)
        engagement = sparse.asnumpy(engagement)
        save_engagement(engagement, 
                        wm_positions, 
                        atlas_data.shape, 
                        save_engagement_filepath, 
                        atlas_img.affine)
    
    print("Finito!")




if __name__ == "__main__":

    if TEST_ENGAGEMENT:
        bold_filepath = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"
        atlas_filepath = "/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz"
        tractogram_file = "/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk"
        gm_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-GM_probseg.nii.gz"
        wm_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-WM_probseg.nii.gz"
        csf_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-CSF_probseg.nii.gz"
        engagement_save_path = "/Users/sam/Desktop/sub-TAU001/anat/engagement_test_covariance.nii.gz"
        engagement_pipeline(bold_data=bold_filepath,
                            atlas=atlas_filepath,
                            grey_matter_prob = gm_prob,
                            white_matter_prob=wm_prob,
                            csf_prob= csf_prob,
                            tractogram_file=tractogram_file,
                            save_engagement_filepath=engagement_save_path, 
                            verbose=True, 
                            plotting=False, 
                            confound_removal=True, 
                            save_connectomes=True)
        
        LOGFILENAME = "/Users/sam/Desktop/sub-TAU001/engagement.log"

        logging.basicConfig(filename= LOGFILENAME, level=logging.DEBUG)

        
    if TEST_FUNCTIONNECTOME:
        atlas_path = "/Users/sam/Desktop/sub-TAU001/aal.nii.gz"
        fMRI_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"
        reference_file = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w.nii.gz"
        save_registered_atlas = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_desc-registered_atlas_space-T1w.nii.gz"
        tractogram_filepath = "/Users/sam/Desktop/sub-TAU001/test_tract.trk"
        wm_mask_filepath = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz"
        brain_mask_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-brain_mask.nii.gz"
        moving_file = "/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"
        density_map_path = "/Users/sam/Desktop/sub-TAU001"
        functionnectome_savepath = "/Users/sam/Desktop/sub-TAU001/functionnectome.nii.gz"
        gm_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-GM_probseg.nii.gz"

        print("Testing the pipeline")
        functionnectome_pipeline(atlas_path=atlas_path,
                                fMRI_path=fMRI_path,
                                t1w_file=reference_file,
                                tractogram=tractogram_filepath,
                                anatomical_scan_atlas_space=moving_file, 
                                brain_mask_path=brain_mask_path,
                                save_registered_atlas=save_registered_atlas,
                                savepath_density_map =    density_map_path,
                                functionnectome_savepath=functionnectome_savepath,
                                white_matter_prob="/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-WM_probseg.nii.gz",
                                csf_prob="/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-CSF_probseg.nii.gz",
                                grey_matter_path=gm_prob,
                                mode="roi",
                                remap = True
                                )
