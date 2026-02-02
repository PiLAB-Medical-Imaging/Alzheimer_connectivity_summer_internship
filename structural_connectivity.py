"""
This script takes the preprocessed dMRI data and returns a connectivity matrix. 
"""

############## Imports ###############
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from regis.core import find_transform, apply_transform
from unravel.analysis import connectivity_matrix

from dipy.io.streamline import load_tractogram
import os
import pickle
#from network_definitions import intra_network_analysis, inter_network_analyses
import sys
import stat


############## Helper FUnctions ###############
os.umask(0o022)

def load_or_compute_mapping(atlas_path, subj_file, saved_map_path):
    """
    Load a saved mapping if it exists, otherwise compute it and save.
    """
    if os.path.exists(saved_map_path):
        print(f"Using cached mapping at {saved_map_path}")
        with open(saved_map_path, "rb") as f:
            mapping = pickle.load(f)
    else:
        print("Computing new mapping...")
        mapping = find_transform(atlas_path, subj_file, level_iters=[1000, 100, 10], diffeomorph=False)
        os.makedirs(os.path.dirname(saved_map_path), exist_ok=True)
        os.chmod(os.path.dirname(saved_map_path),  0o755)
        with open(saved_map_path, "wb") as f:
            pickle.dump(mapping, f)

        os.chmod(saved_map_path, 0o644)     
    return mapping


def compute_connectivity_matrix(trk_file, label_volume):
    """
    Generate the connectivity matrix from tractography and labels.
    """

    #print(f"Attempting to load trk file: {trk_file} with label volume: {label_volume}")
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    streamlines = trk.streamlines

    """     print("Label volume shape:", label_volume.shape)
    print("Tractogram reference shape:", trk._data_per_streamline[0].shape if hasattr(trk, '_data_per_streamline') else "unknown")
    print("Streamline bounds:", np.min(np.vstack(trk.streamlines)), np.max(np.vstack(trk.streamlines)))
    """

    matrix = connectivity_matrix(streamlines, label_volume, inclusive=False)

    # Remove background row/column (index 0)
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    return matrix



def generate_connectivity_matrix(root_in, out_path, subj_id, atlas_path, label_path):
    """ Main function that generates the connectivity matrix from preprocessed dMRI data. It saves t at out_path. Also returns the matrix as a variable. """
    matrix_path = os.path.join(out_path, "connec_mat.npy")
    saved_map_path = os.path.join(out_path, "map.pkl")
    subject_load_path = os.path.join(root_in, subj_id)
    subj_file = os.path.join(subject_load_path, f"dMRI/microstructure/dti/{subj_id}_FA.nii.gz")

    # Return cached matrix if it exists
    if os.path.exists(matrix_path):
        print("Using cached connectivity matrix")
        matrix =  np.load(matrix_path, allow_pickle=True)

        if matrix.size == 0:
            print("Matrix is empty, attempting to recompute")
        else:
            return matrix

    # Load or compute the mapping
    mapping = load_or_compute_mapping(atlas_path, subj_file, saved_map_path)

    # Apply the mapping to the label volume
    label_volume = apply_transform(label_path, mapping, labels=True)
    img = nib.load(subj_file)
    out = nib.Nifti1Image(label_volume.astype(float), img.affine)
    sbj_atlas_path = os.path.join(out_path, f"{subj_id}_atlas.nii.gz") # maybe add the artlas name to this.
    out.to_filename(sbj_atlas_path)

    # Compute the connectivity matrix
    trk_file = os.path.join(subject_load_path, "dMRI", "tractography", f"{subj_id}_tractogram.trk")

    # Insert the sift2 shenanigans here.

    matrix = compute_connectivity_matrix(trk_file, label_volume)

    # Save the matrix
    # os.makedirs(subject_specific_location, exist_ok=True)
    # np.save(matrix_path, matrix)

    print(f"The structural matrix for {subj_id} is:")
    print(matrix)

    return matrix
