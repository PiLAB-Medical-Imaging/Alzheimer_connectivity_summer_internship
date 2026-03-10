import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.colors import ListedColormap

from data_compiler import load_all_indices

# -------------------------------
# Load patient atlas & network definitions
# -------------------------------
net_defs = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/definitions.csv"
atlas = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/aal.nii.gz"
indices = load_all_indices(net_defs)
atlas_img = nib.load(atlas)
atlas_data = atlas_img.get_fdata().astype(int)

# -------------------------------
# Function to create binary network mask
# -------------------------------
def create_network_mask(atlas_data, region_indices):
    print(region_indices)
    print(np.unique(atlas_data))
    mask = np.isin(atlas_data, region_indices).astype(int)

    print(np.unique(mask))
    return nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header)

masked_atlas = create_network_mask(atlas_data, indices["salience"]["AAL116"])



plotting.plot_glass_brain(
    stat_map_img=masked_atlas,
    display_mode='ortho',  # 'ortho', 'x', 'y', 'z', 'lzry' etc.
    cmap='cool',
    black_bg=False,
    threshold= 0,
    colorbar=False,
)

plt.show()