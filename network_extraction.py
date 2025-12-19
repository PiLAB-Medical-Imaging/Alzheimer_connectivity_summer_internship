""" This script extracts specified networks from a precomputed matrix"""

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_aal
import pandas as pd
import matplotlib.pyplot as plt
from regis.core import find_transform, apply_transform
from unravel.analysis import connectivity_matrix
from dipy.io.streamline import load_tractogram
import networkx as nx
import matplotlib.pyplot as plt
import os


NETWORKS = ["dmn-basic", "dmn-ext", "salience", "ecn"]


def load_all_indices(definitions_filepath):
    """
    Load ALL network → atlas index mappings from a definitions file.

    Returns:
        dict: {
            "dmn_basic": {
                "AAL116": [...],
                "OtherAtlas": [...],
                ...
            },
            "dmn_ext": {
                "AAL116": [...],
                ...
            }
        }
    """

    # Auto-detect delimiter (handles tabs, semicolons, commas)
    df = pd.read_csv(definitions_filepath, sep="\\t", engine="python")
    print(df.columns)
    df.columns = df.columns.str.strip('"')

    if "Network" not in df.columns:
        raise ValueError("Definitions file must contain a 'Network' column.")

    # Identify atlas columns = everything except 'Network'
    atlas_columns = [col for col in df.columns if col.lower() != "network"]

    # Clean network names (strip whitespace)
    df["Network"] = df["Network"].str.strip()

    # Build output structure
    all_networks = {}

    for _, row in df.iterrows():
        network_name = row["Network"]
        print(network_name)
        all_networks[network_name] = {}

        for atlas in atlas_columns:
            raw = str(row[atlas]).strip()

            if raw == "" or raw.lower() == "nan":
                indices = []
            else:
                # Parse comma-separated integers safely
                indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

            all_networks[network_name][atlas] = indices

    return all_networks


def define_network(connectivity_matrix_filepath, subject_atlas_path, definitions_filepath, atlas_name, network_name):

    # Load a graph (either a functional or structural connectivity matrix. It is essential that the atlas used matches)
    conn_matrix = np.load(connectivity_matrix_filepath)

    ## Load the atlas (a subject specific one - already need to have performed registration)
    atlas_img=nib.load(subject_atlas_path)
    atlas = atlas_img.get_fdata().astype(float)

    # Load all network to brain network mappings
    network_mappings = load_all_indices(definitions_filepath)

    # Define network indices
    target_indices = network_mappings[network_name][atlas_name]

    # Mask for required indices
    target_mask = np.zeros_like(atlas)
    target_mask[np.isin(atlas, target_indices)] = atlas[np.isin(atlas, target_indices)]

    #out=nib.Nifti1Image(target_mask, atlas_img.affine , atlas_img.header)
    #out.to_filename(network_path) # This creates amd saves a visualisation of the subjects brain network.

    # Next - look at the subset of the connectivity matrix and work with that.
    subset_matrix = conn_matrix[np.ix_(target_indices, target_indices)]

    return subset_matrix


###### Main logic #####
def network_extraction(subj_connectivity_folder, matrix_filepath, subj_id, definitions_filepath, atlas):
    atlas_filepath = os.path.join(subj_connectivity_folder, f"{subj_id}_atlas.nii.gz")

    return_networks = {}

    for network in NETWORKS:
        selected_network = define_network(matrix_filepath,atlas_filepath,definitions_filepath, atlas, network)
        return_networks[f"{atlas}_{network}_{subj_id}"] = selected_network

    return return_networks









####### Out dated - these are functions that only work with the AAL atlas #######
def reorder_matrix(matrix, labels, indices=[]):
    """
    Docstring for reorder_matrix
    
    :param matrix: Numpy array that will be reordered
    :param labels: Label list that corresponds to the file that contains the index labels
    :param indices: the indices that you want to reorder. By default, reorders the entire matrix.
    """

    node_labels = load_labels(labels)

    if len(indices) > 0:
        node_labels = [node_labels[i] for i in indices]

    left_idx = [i for i, label in enumerate(node_labels) if label.endswith("_L")]
    right_idx = [i for i, label in enumerate(node_labels) if label.endswith("_R")]

    # Combined new order: all left, then all right
    new_order = left_idx + right_idx[::-1]

    reordered_matrix = matrix[np.ix_(new_order, new_order)]
    ## Need to reorder the labels (they are not right)
    return reordered_matrix, new_order
        

def load_labels(labels):
    with open(labels, "r") as f: 
        label_set   = [line.strip() for line in f]

    return label_set