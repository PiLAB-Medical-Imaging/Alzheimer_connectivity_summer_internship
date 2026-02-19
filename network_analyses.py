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




# Specify which network
network_name = "dmn_basic"




def load_indices(network_name, atlas_name, definitions_filepath):
    """
    Retrieve the atlas indices that correspond to the specified network. Need to modify this so it actually works
    
    :param network_name: (string) Which network you want to build a mask for. Only valid answers are the Networks defined.
    :param atlas_name: (string) Description
    :param definitions_filepath: Description
    
    """
    # @todo: Add some robustness. Check for the network names. Compare cases. Probably makes more sense for this to retrieve all the indices, and then to select which one you want a bit later.
    df = pd.read_csv(definitions_filepath, delimiter="\t")
    print(df)
    df[atlas_name] = df[atlas_name].str.split(',').apply(lambda x: [int(i) for i in x])
    network_dict = dict(zip(df["Network"], df[atlas_name]))
    return network_dict



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
    df = pd.read_csv(definitions_filepath, sep="\t", engine="python")
    df.columns = df.columns.str.strip()

    if "Network" not in df.columns:
        raise ValueError(f"Definitions file must contain a 'Network' column."
                         f"Columns: {df.columns}")

    # Identify atlas columns = everything except 'Network'
    atlas_columns = [col for col in df.columns if col.lower() != "network"]

    # Clean network names (strip whitespace)
    df["Network"] = df["Network"].str.strip()

    # Build output structure
    all_networks = {}

    for _, row in df.iterrows():
        network_name = row["Network"]
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



def intra_network_analysis(matrix, subject_atlas_path, definitions_filepath, atlas_name ):
    matrix = np.load(matrix)
    ## Load the atlas
    atlas_img=nib.load(subject_atlas_path)
    atlas = atlas_img.get_fdata().astype(float)

    # Load all network to brain network mappings
    network_mappings = load_all_indices(definitions_filepath)

    # Define network indices
    target_indices = network_mappings[network_name][atlas_name]
    # Mask for required indices
    target_mask = np.zeros_like(atlas)
    target_mask[np.isin(atlas, target_indices)] = atlas[np.isin(atlas, target_indices)]

    out=nib.Nifti1Image(target_mask, atlas_img.affine , atlas_img.header)
    out.to_filename(network_path) # This creates amd saves a visualisation of the subjects brain network.

    # Next - look at the subset of the connectivity matrix and work with that.
    subset_matrix = matrix[np.ix_(target_indices, target_indices)]
    # Reorder the matrix
    subset_matrix, new_order = reorder_matrix(subset_matrix, labels_path, target_indices)
    # Work with just the network
    network_isolated = nx.from_numpy_array(subset_matrix)
    network_isolated = attach_labels(network_isolated, labels_path, new_order)
    plot_graph(network_isolated)



def inter_network_analyses(matrix):
    pass

def attach_labels(G, labels, indices = []):
    label_set = load_labels(labels)
    
    if len(indices)>0:
        label_set = [label_set[i] for i in indices]
    
    node_labels = {i: label_set[i] for i in range(len(label_set))}

    nx.set_node_attributes(G, node_labels, "label")
    
    return G



def load_labels(labels):
    with open(labels, "r") as f: 
        label_set   = [line.strip() for line in f]

    return label_set

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
        


def plot_graph(G):
    pos = nx.circular_layout(G)  
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Normalize weights for nicer plotting
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos,
        with_labels=True,
        #width=weights,       # edge thickness by weight
        edge_color=weights,  # edge color by weight
        edge_cmap=plt.cm.viridis,
        node_size=300,
        font_size=8)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"))
    plt.show()




##############################################
#           Different Analyses
##############################################
def node_basic_metrics(graph, save_location):
    """
    Compute and store node level metrics for a graph
    
    :param graph: Description
    :param save_location: Description
    """
    degree_dict = graph.degree()
    deg_centrality = nx.degree_centrality(graph)
    node_betweenness = nx.betweenness_centrality(graph)
    edge_betweenness = nx.edge_betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

def graph_level_metrics(graph, save_location):
    """
    Calculates and saves graph level metrics relevant to a brain network
    
    :param graph: Description
    :param save_location: Description
    """
    clustering_coefficient_avg = nx.cluster.average_clustering(graph)
    transitivity = nx.transitivity(graph)

