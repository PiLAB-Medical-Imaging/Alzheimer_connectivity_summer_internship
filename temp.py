import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from utilities import dilate_atlas_labels, atlas_masker, time_slicing, trk2tck
from utilities import split_nifti_to_visualise, streamline_registration, mask_to_positions
from utilities import sl_to_roi_map, voxel_to_streamline_map_V2, conn_matrices, conn_matrices_V2
from engagement import generate_VWSC_matrices_entire_sl, generate_VWSC_matrices_ep_only
from time import time
from nibabel import Nifti1Image
from dipy.tracking.streamline import transform_streamlines
import sparse

""" engagement_old = nib.load("/Users/sam/Desktop/sub-TAU001/anat/02_threshold_engagement_10x.nii.gz")
engagement_new = nib.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng.nii.gz")

eng_old_data = engagement_old.get_fdata()
eng_new_data = engagement_new.get_fdata()

comparison = eng_old_data-eng_new_data

nnz = np.count_nonzero(comparison)
print(nnz)

plt.hist(eng_old_data.flatten(), color="g", stacked=True)
plt.hist(eng_new_data.flatten(), color="b", stacked=True)
plt.loglog()
plt.show()



def report(array):
    max_val = np.max(array)
    min_val = np.min(array)
    nnz = np.count_nonzero(array)
    nans = np.sum(np.isnan(array))
    print(f"Report\n"
          f"Max: {max_val}\n"
          f"Min: {min_val}\n"
          f"Nonzeros: {nnz}\n"
          f"Nans: {nans}")
    
report(eng_old_data)
report(eng_new_data) """


""" 
engagement = "/Users/sam/Desktop/tau100_eng_250_all_sl.nii.gz"
eng_img = nib.load(engagement)
eng_data = eng_img.get_fdata()
mask = np.where(eng_data > 0, 1, 0)
filtered = gaussian_filter(
    input=eng_data,
    sigma = 2.0
)
filtered = filtered*mask
filtered_img = Nifti1Image(filtered, affine=eng_img.affine)
filtered_img.to_filename(engagement[:-7]+"_filtered.nii.gz")



 """

""" 
t1w = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/sub-TAU056_ses-0_desc-preproc_T1w.nii.gz"
brain_mask = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/sub-TAU056_ses-0_desc-brain_mask.nii.gz"
trk_diffusion = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/TAU_56_ses-0_tractogram.trk"
atlas = "/Users/sam/Desktop/aal_mni_correct.nii.gz"
mni_template = "/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"
# Register the atlas: 
import utilities as utl

atlas_img = nib.load(atlas)
atlas_data = atlas_img.get_fdata()
print(np.unique(atlas_data.astype("int32")))
print(len(np.unique(atlas_data.astype("int32"))))
atlas_data = np.round(atlas_data)
print(np.unique(atlas_data))
print(len(np.unique(atlas_data)))
out = nib.Nifti1Image(
    atlas_data,
    affine = atlas_img.affine
)
out.to_filename(atlas) """

"""
OUR_MNI = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"
OUR_MNI_TRACT ="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/our_mni_bundles/mni_edited_AC.trk"
T1 = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz"
trk = load_tractogram(filename=OUR_MNI_TRACT, reference=OUR_MNI)
print("trk_space", trk.space)

tform = find_transform(
    moving_file=OUR_MNI,
    static_file=T1,
    diffeomorph=False
)

apply_transform(
    OUR_MNI,
    tform,
    static_file=T1,
    output_path="/Users/sam/Desktop/test_mni_patient_space.nii.gz"
)

data = trk.streamlines.get_data()
mins = data.min(axis=0)
maxs = data.max(axis=0)
print(trk.space)
print("Streamline bounds:")
print("X:", mins[0], "→", maxs[0])
print("Y:", mins[1], "→", maxs[1])
print("Z:", mins[2], "→", maxs[2])

new_sls = transform_streamlines(
    streamlines=trk.streamlines,
    mat = tform.affine
)

new_sls = transform_streamlines(
    streamlines=trk.streamlines,
    mat = np.linalg.inv(tform.affine)
)
new_trk = StatefulTractogram(
    streamlines=new_sls, 
    reference=T1, 
    space=Space.RASMM
)

data = new_trk.streamlines.get_data()
mins = data.min(axis=0)
maxs = data.max(axis=0)
print(new_trk.space)
print("Streamline bounds:")
print("X:", mins[0], "→", maxs[0])
print("Y:", mins[1], "→", maxs[1])
print("Z:", mins[2], "→", maxs[2])

save_tractogram(
    new_trk,
    filename="/Users/sam/Desktop/test_tracts.trk",
    bbox_valid_check=True
)
 
trk2tck("/Users/sam/Desktop/test_tracts.trk", False)
"""

# Install nilearn if needed:
# pip install nilearn matplotlib

from nilearn import plotting, image
from nilearn.datasets import load_mni152_template
import matplotlib.pyplot as plt

""" # ------------------------------------------------------------------
# Option 1: Plot your own BOLD fMRI NIfTI file
# ------------------------------------------------------------------
# Replace with the path to your BOLD fMRI NIfTI file (.nii or .nii.gz)
bold_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

# Load image
bold_img = image.load_img(bold_path)

# If 4D (time series), select one volume (e.g., first time point)
if bold_img.ndim == 4:
    bold_img = image.index_img(bold_img, 30)

# Plot
plotting.plot_stat_map(
    bold_img,
    bg_img=load_mni152_template(),
    threshold=None,
    black_bg=False,
    display_mode="ortho",
    title="BOLD fMRI Volume"
)

plt.show()

 """
# ------------------------------------------------------------------
# Option 2: Example using nilearn sample dataset
""" # ------------------------------------------------------------------
from nilearn.datasets import fetch_development_fmri
data = fetch_development_fmri(n_subjects=1)
example_bold = data.func[0]
bold_img = image.load_img(example_bold)
bold_img = image.index_img(bold_img, 0)
plotting.plot_stat_map(bold_img, display_mode="ortho", title="Sample BOLD fMRI")
plt.show()

 """


""" import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib
import os
from nibabel.nifti1 import Nifti1Image
import re
from nilearn.plotting import plot_matrix, show

from nilearn.interfaces.fmriprep import load_confounds_strategy

def connectivity_matrix_generation(bold, atlas, normalise=True, method="nilearn",
                                   kind="covariance", bold_filepath=None):
    # Load atlas
    if isinstance(atlas, str):
        atlas_img = nib.load(atlas)
    elif isinstance(atlas, nib.Nifti1Image):
        atlas_img = atlas
    else:
        raise TypeError("The atlas should be a path or a Nifti1Image object.")
    
    # Masker to extract time series
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=normalise)
    
    # Extract time series with or without confounds
    if bold_filepath is not None:
        confounds_df, _ = load_confounds_strategy(bold_filepath, denoise_strategy="simple")
        time_series = masker.fit_transform(bold, confounds=confounds_df)
    else:
        time_series = masker.fit_transform(bold)
    
    # --- Plot timeseries ---
    plt.figure(figsize=(12, 4))
    plt.plot(time_series[:,0:4 ])
    plt.xlabel('Time points')
    plt.ylabel('Signal')
    plt.title('BOLD Time Series per Region')
    plt.show()
    
    # Compute connectivity matrix
    if method == "nilearn":
        conn_measure = ConnectivityMeasure(kind=kind)
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    else:
        raise ValueError("Enter a valid method: 'nilearn' or 'custom'")
    
    # --- Plot connectivity matrix ---
    plotting.plot_matrix(conn_matrix, figure=(10, 8), labels=None, colorbar=True, vmax=1.0)
    plt.show()
    
    # --- Plot connectome on brain ---
    # Get region coordinates
    coords = plotting.find_parcellation_cut_coords(labels_img=atlas_img)
    
    # Plot connectome
    plotting.plot_connectome(conn_matrix, coords, edge_threshold="80%", node_size=50, title="Connectome")
    plt.show()
    
    return conn_matrix

connectivity_matrix_generation(
    bold="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_desc-preproc_bold.nii.gz",
    atlas="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/registered_atlas.nii.gz",
    normalise=True,
    bold_filepath="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_desc-preproc_bold.nii.gz",
) """

struct_mat = np.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/TAU001_ses-2_simple_weighting.npy")
from nilearn.plotting import plot_matrix
#struct_mat = np.clip(struct_mat, 0, 100)
plot_matrix(struct_mat)
plt.show()



import pyvista as pv
from unravel.viz import plot_trk

def create_gif(plotter, file_path:str):
        """Create a 360° rotation GIF of the current 3D view."""

        # Ensure the file has a .gif extension
        if not file_path.lower().endswith(".gif"):
            file_path += ".gif"

        # Create the 360° rotation GIF
        plotter.open_gif(file_path, fps=20)
        n_frames = 360
        for i in range(n_frames):
            plotter.camera.azimuth += 360 / n_frames
            plotter.render()
            plotter.write_frame()
        plotter.close()
        
trk_file="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/wm_atlas_inverted/TAU001_mni_edited_CC.trk"
#trk_file="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk"
#gif_file="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/temp.gif"
gif_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Presentations/Images/whole_trk.gif"

plotter=pv.Plotter()
plot_trk(trk_file=trk_file,
         plotter=plotter,
         background="white")
create_gif(plotter, gif_file)
plotter.show()


#eng_img = nib.load("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng_mni_space.nii.gz")
#eng = eng_img.get_fdata()
#eng=np.clip(eng, 0,5)

mni_atlas = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"
mni_atlas_img = nib.load(mni_atlas)
mni = mni_atlas_img.get_fdata()
grid = pv.ImageData()
 
grid.dimensions = np.array(mni.shape) + 1
grid.cell_data['values'] = mni.flatten(order='F')
plotter = pv.Plotter()
plotter.add_volume(grid, cmap='gray', opacity=[0.0, 0.045], show_scalar_bar=False)
#plot_trk(trk_file,plotter=plotter, background='white')
#plot_trk(trk_file,plotter=plotter, background='white', scalar=eng, color_map="turbo")

# plotter.show()

from unravel.stream import get_roi_sections_from_nodes, extract_nodes

#point_array = extract_nodes(trk_file=trk_file, nodes=30)
#rois_arrays = get_roi_sections_from_nodes(trk_file, point_array)
#plot_trk(trk_file,plotter=plotter, background='white', scalar=rois_arrays, color_map="Set3")
#plotter.show()