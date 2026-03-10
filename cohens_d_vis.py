# Filename: generate_cohens_d_map.py

import nibabel as nib
from dipy.io.streamline import load_trk
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn import plotting
# -------------------------------
# USER SETTINGS
# -------------------------------
tract_folder = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/our_mni_bundles"  # folder containing all .trk files
template_img_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"  # reference image in MNI space
output_file = "/Users/sam/Desktop/cohens_d_map.nii.gz"
output_png = "/Users/sam/Desktop/cohens_d_map.png"
# Cohen's d values for each tract (example, add all 44)
cohens_d_dict = {
    "mni_edited_FPT_L": -0.378795513232893,
    "mni_edited_FPT_R": -0.368520218329306,
    "mni_edited_PPT_R": -0.367640714766227,
    "mni_edited_CST_R": -0.365497124915572,
    "mni_edited_PPT_L": -0.363931911911354,
    "mni_edited_AR_R": -0.348126783427613,
    "mni_edited_CST_L": -0.346444414371649,
    "mni_edited_AR_L": -0.330180272266562,
    "mni_edited_AC": -0.322675512128760,
    "mni_edited_MdLF_R": -0.313932628578833,
    "mni_edited_SLF_L": -0.311635351231125,
    "mni_edited_AF_R": -0.305329380649312,
    "mni_edited_OPT_R": -0.303231504832667,
    "mni_edited_IFOF_L": -0.303023808629581,
    "mni_edited_CT_R": -0.299905812224032,
    "mni_edited_SLF_R": -0.297019817079801,
    "mni_edited_AF_L": -0.292134895185485,
    "mni_edited_CS_L": -0.284980522835601,
    "mni_edited_CT_L": -0.283311432129888,
    "mni_edited_OPT_L": -0.279996833256593,
    "mni_edited_CS_R": -0.279890163703921,
    "mni_edited_TPT_R": -0.275928192690016,
    "mni_edited_TPT_L": -0.272188038558469,
    "mni_edited_UF_R": -0.269146744965161,
    "mni_edited_MdLF_L": -0.260624632422356,
    "mni_edited_IFOF_R": -0.256011460702132,
    "mni_edited_EMC_R": -0.251518978836845,
    "mni_edited_EMC_L": -0.248582756218653,
    "mni_edited_ILF_R": -0.242995314331736,
    "mni_edited_CC": -0.239913487665192,
    "mni_edited_OR_L": -0.236879678513298,
    "mni_edited_C_L": -0.232960854955777,
    "mni_edited_OR_R": -0.226391435846188,
    "mni_edited_UF_L": -0.224994110713827,
    "mni_edited_ILF_L": -0.222572509194659,
    "mni_edited_VOF_R": -0.196561136993319,
    "mni_edited_PC": -0.191394140317899,
    "mni_edited_CC_ForcepsMinor": -0.185491085277711,
    "mni_edited_CC_ForcepsMajor": -0.177355471771859,
    "mni_edited_VOF_L": -0.125685528264680,
    "mni_edited_C_R": -0.0925035027444102,
    "mni_edited_F_L_R": 0.134222485043702,
}

# -------------------------------
# LOAD TEMPLATE
# -------------------------------
template_img = nib.load(template_img_file)
shape = template_img.shape
affine = template_img.affine

# Initialize empty map
merged_map = np.zeros(shape)

# -------------------------------
# PROCESS EACH TRACT
# -------------------------------
for tract_file in os.listdir(tract_folder):
    if tract_file.endswith(".trk"):
        tract_name = os.path.splitext(tract_file)[0]
        print(f"Processing {tract_name}...")

        if tract_name not in cohens_d_dict:
            print(f"  Warning: Cohen's d value not found for {tract_name}, skipping.")
            continue

        # Load tract streamlines
        trk_path = os.path.join(tract_folder, tract_file)
        tractogram = load_trk(trk_path, reference=template_img_file)
        streamlines = tractogram.streamlines

        # Create empty volume for this tract
        tract_vol = np.zeros(shape)

        # Rasterize streamlines to voxel space
        for sl in streamlines:
            coords = np.round(nib.affines.apply_affine(np.linalg.inv(affine), sl)).astype(int)
            # Remove out-of-bounds indices
            coords = coords[
                (coords[:,0]>=0)&(coords[:,0]<shape[0]) &
                (coords[:,1]>=0)&(coords[:,1]<shape[1]) &
                (coords[:,2]>=0)&(coords[:,2]<shape[2])
            ]
            tract_vol[coords[:,0], coords[:,1], coords[:,2]] = 1

        # Apply Cohen's d
        tract_vol *= cohens_d_dict[tract_name]

        # Add to merged map
        merged_map += tract_vol

# -------------------------------
# SAVE MERGED COHEN'S D MAP
# -------------------------------
merged_img = nib.Nifti1Image(merged_map, affine)
nib.save(merged_img, output_file)
print(f"\nSaved merged Cohen's d map to {output_file}")

# -------------------------------
# PLOT PUBLICATION-READY FIGURE
# -------------------------------
fig = plt.figure(figsize=(10,5))
display = plotting.plot_glass_brain(
    merged_img,
    display_mode='ortho',
    colorbar=True,
    cmap='coolwarm',  # diverging colormap
    threshold=0.05,   # show only meaningful effect sizes
    plot_abs=False
)
plt.title("Tract-wise Cohen's d Effect Sizes")
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved to {output_png}")