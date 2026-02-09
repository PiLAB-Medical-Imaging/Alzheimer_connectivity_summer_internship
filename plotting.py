from nilearn.plotting import plot_glass_brain, view_connectome
from nilearn import plotting 
import nibabel as nib
import numpy as np
from regis.core import find_transform, apply_transform


""" eng_file = ("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/"
            "data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng.nii.gz")
mni_template = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"
patient_t1w = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz"

transform = find_transform(
    moving_file=patient_t1w,
    static_file=mni_template,
)
transformed_eng = apply_transform(
    moving_file=eng_file,
    mapping=transform,
    static_file=mni_template,
    output_path=eng_file[:-7] + "_mni_space.nii.gz"
)
"""

transformed_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng_mni_space.nii.gz"
transformed_img = nib.load(transformed_file)
display = plot_glass_brain(
    stat_map_img=transformed_img
)
plotting.show()
display.close()
"""
# Load the weighted matrix. 
sw_file = ("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/"
        "data_temp/TestFileStructure/Outputs/TAU001/ses-2/"
        "TAU001_2sw_matrix.npy")

atlas_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/aal.nii.gz"
atlas_img = nib.load(atlas_file)
data = atlas_img.get_fdata()
affine = atlas_img.affine
labels = np.unique(data)
labels = labels[labels != 0]
centers = {}
for label in labels:
    coords = np.column_stack(np.where(data == label))
    center_voxel = coords.mean(axis=0)
    center_mni = nib.affines.apply_affine(affine, center_voxel)
    centers[int(label)] = center_mni
weight_matrix = np.load(
    file=sw_file
)

center_list = []
for key in centers.keys():
    center_list.append(centers[key])

coords_array = np.array(
    center_list
)
view = view_connectome(
    adjacency_matrix=weight_matrix,
    node_coords=coords_array,
    edge_threshold=0.1
)
view.open_in_browser()
"""