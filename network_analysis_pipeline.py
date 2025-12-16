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


#--------------------------------------------
#           OPTIONS
#--------------------------------------------
NETWORK_ANALYSIS = True


#--------------------------------------------
#           Functions
#--------------------------------------------


import os
import pickle
import stat
import nibabel as nib
import numpy as np


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
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()
    streamlines = trk.streamlines
    matrix = connectivity_matrix(streamlines, label_volume, inclusive=False)

    # Remove background row/column (index 0)
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    return matrix


def generate_connectivity_matrix(root_in, out_path, subj_id, atlas_path, label_path):
    """
    Main function to generate or load a connectivity matrix for a subject.
    """
    subject_specific_location = os.path.join(out_path, subj_id)
    matrix_path = os.path.join(subject_specific_location, "connec_mat.npy")
    saved_map_path = os.path.join(subject_specific_location, "map.pkl")
    subject_load_path = os.path.join(root_in, subj_id)
    subj_file = os.path.join(subject_load_path, f"dMRI/microstructure/dti/{subj_id}_FA.nii.gz")

    # Return cached matrix if it exists
    if os.path.exists(matrix_path):
        print("Using cached connectivity matrix")
        return np.load(matrix_path, allow_pickle=True)

    # Load or compute the mapping
    mapping = load_or_compute_mapping(atlas_path, subj_file, saved_map_path)

    # Apply the mapping to the label volume
    label_volume = apply_transform(label_path, mapping, labels=True)
    img = nib.load(subj_file)
    out = nib.Nifti1Image(label_volume.astype(float), img.affine)
    sbj_atlas_path = os.path.join(subject_specific_location, f"{subj_id}_atlas.nii.gz")
    out.to_filename(sbj_atlas_path)

    # Compute the connectivity matrix
    trk_file = os.path.join(subject_load_path, "dMRI", "tractography", f"{subj_id}_tractogram.trk")
    matrix = compute_connectivity_matrix(trk_file, label_volume)

    # Save the matrix
    os.makedirs(subject_specific_location, exist_ok=True)
    np.save(matrix_path, matrix)

    return matrix



def run_pipeline(atlas_path, labels_path, root_in, root_out, subj_id):
    # Generate and save the connectivity matrix
    matrix  = generate_connectivity_matrix(root_in, root_out, subj_id, atlas_path, labels_path)
    

    # Run various analyses (ultimately wrap this into a single script in the network_definitions file and set up a 
    # config file that allows the selection of options)
    subject = "TAU10000"
    out_path = os.path.join(root_out, f"{subject}_atlas.nii.gz")

    # Run a series of analyses:
    #intra_network_analysis(matrix, out_path, definition_path)
    #inter_network_analyses(matrix)


if __name__ == '__main__':
    atlas_path = sys.argv[1]
    labels_path = sys.argv[2]
    root_in = sys.argv[3]
    root_out = sys.argv[4]
    subj_id = sys.argv[5]
    #definition_path = sys.argv[6]
    run_pipeline(atlas_path, labels_path, root_in, root_out, subj_id)
    