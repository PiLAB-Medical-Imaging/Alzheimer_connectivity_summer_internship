import numpy as np
import nibabel as nib
from fmri_processing import connectivity_matrix_generation
from nilearn import image
from networkx import edge_betweenness_centrality, Graph
import networkx as nx
import matplotlib.pyplot as plt
from nilearn.plotting import plot_matrix, show
from utilities import voxel_to_streamline_map
from dipy.io.streamline import load_tractogram
from utilities import mask_generator, generate_masks
from tqdm import tqdm
from unravel.analysis import connectivity_matrix
import sparse

def engagement_pipeline(bold_data, atlas, tractogram_file, grey_matter_prob, white_matter_prob, csf_prob,  plotting = False):
    """
    Pipeline that performs the entire engagement calculation - functions within this will correspond to submodules 
    that can be run with just the required objects. This function works with the filepaths.
    
    :param bold_data: str
        Preprocessed bold data. Can be in any space (T1w, MNI), however this must match the space of the atlas.
    :param atlas: str
        Filepath to the atlas. Space must match the bold data space
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
    ROIs = len(atlas_values)
    fc_mat = connectivity_matrix_generation(bold_img, atlas, False)

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
    fc_mat = correlation_thresholding(fc_mat, 0.2)

    task+=1

    ################################ Step 2 ################################
    print(f"{task}. Computing Edge Between Connectedness Matrix") 
    ebc_mat = ebc_computation(fc_mat)
    print(ebc_mat.shape)
    task += 1

    ################################ Step 3 ################################
    print(f"{task}. Computing all fibres that penetrate each voxel")   

    trk = load_tractogram(tractogram_file, "same")

    all_connectivity_matrices = generate_VWSC_matrices(white_matter_prob, atlas_data, ROIs, trk)

    task += 1
    ################################ Step 4 ################################
    print(f"{task}. Calculating Engagement")   
    engagement_calculation(EBC_matrix=ebc_mat,
                           SC_matrices=all_connectivity_matrices)

   

def engagement_calculation(EBC_matrix, SC_matrices):
    print(SC_matrices.shape)
    print(EBC_matrix.shape)
    result = sparse.einsum("ijk,jk->i", SC_matrices, EBC_matrix)
    print(result.shape)



def generate_VWSC_matrices(white_matter_prob, atlas_data, ROIs, trk):
    v2f_mapping = voxel_to_streamline_map(trk.streamlines, vol_shape=trk.dimensions)

    # Generate a white matter mask:
    wm_mask = mask_generator(white_matter_probability=white_matter_prob)
    
    # Generate all white matter positions
    wm_positions = generate_masks(wm_mask)

    # Naive method:
    all_connectivity_matrices = []

    proportion_non_zero = len(v2f_mapping.keys())/len(wm_positions)

    for idx, voxel in enumerate(tqdm(wm_positions, "VW SC matrices")):
        if tuple(voxel) not in v2f_mapping.keys():
            conn_mat = np.zeros(shape=(ROIs, ROIs))
        else:
            streamline_indices = v2f_mapping[tuple(voxel)]
            conn_mat = connectivity_matrix(trk.streamlines[streamline_indices], atlas_data,inclusive=False,)

        conn_mat = sparse.COO.from_numpy(conn_mat)
        all_connectivity_matrices.append(conn_mat)
    
    all_connectivity_matrices = sparse.stack(all_connectivity_matrices, axis = 0)
    return all_connectivity_matrices

def ebc_computation(numpy_matrix):
    """
    Simple wrapper to calculate the EBC matrix starting with a functional connectivity matrix.
    
    :param numpy_matrix: np array
        Functional connectivity array.
    """
    g = nx.from_numpy_array(numpy_matrix, 
                            edge_attr = "weight")
    
    ebc_dict= edge_betweenness_centrality(G=g, weight="weight")
    ebc_mat = np.zeros_like(numpy_matrix)
    for key in ebc_dict.keys():
        ebc_mat[key[0], key[1]] = ebc_dict[key]
        ebc_mat[key[1], key[0]] = ebc_dict[key]
    

    return ebc_mat

def correlation_thresholding(matrix, proportion=0.2):
    """
   Threshold a functional connectivity array, only retaining values that are in the top 0.x of the data.
    
    :param matrix: Array
        Numpy array
    :param proportion: numeric
        Decimal proportion of data to keep. Range between 0 and 1
    """
    if proportion < 0 or proportion > 1.0:
        raise ValueError("Ensure proportion is between 0 and 1")
    cutoff = np.quantile(matrix, 1-proportion)
    filtered_mat = np.where(matrix >= cutoff, matrix, 0)
    return filtered_mat

if __name__ == "__main__":
    bold_filepath = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    atlas_filepath = "/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz"
    tractogram_file = "/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk"
    gm_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-GM_probseg.nii.gz"
    wm_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-WM_probseg.nii.gz"
    csf_prob = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-CSF_probseg.nii.gz"
    engagement_pipeline(bold_data=bold_filepath,
                        atlas=atlas_filepath,
                        grey_matter_prob = gm_prob,
                        white_matter_prob=wm_prob,
                        csf_prob= csf_prob,
                        tractogram_file=tractogram_file)