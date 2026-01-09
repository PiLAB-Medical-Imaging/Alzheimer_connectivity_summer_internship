import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from nilearn.image import resample_to_img
from nibabel.processing import resample_from_to
import os
import os.path as path

gm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"
wm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz"
csf_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz"
save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz"
wm_save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-MNI152NLin2009cAsym_label-WM_mask.nii.gz"
bold_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


# Get the probabilities for each tissue type
gm_img = nib.load(gm_path)
gm_data = gm_img.get_fdata()

wm_img = nib.load(wm_path)
wm_data = wm_img.get_fdata()

csf_img = nib.load(csf_path)
csf_data = csf_img.get_fdata()

# Smooth the probabilities
gm_smooth = gaussian_filter(gm_data, sigma = 1.0)
wm_smooth = gaussian_filter(wm_data, sigma = 1.0)


# Make the mask (Note we are requiring a minimum value, then that it is also the dominant tissue type.)
gm_mask = (
    (gm_smooth > 0.2) &
    (gm_smooth > wm_data) & 
    (gm_smooth > csf_data)         
           ).astype("uint8")

wm_mask = (
    (wm_smooth >0.2) &
    (wm_smooth > gm_data) & 
    (wm_smooth > csf_data)
        ).astype("uint8")


# Clean mask

gm_mask = binary_fill_holes(gm_mask)
labels, _ = label(gm_mask)
counts = np.bincount(labels.ravel())

wm_mask = binary_fill_holes(wm_mask)
label_wm, _ = label(wm_mask)
wm_counts = np.bincount(label_wm.ravel())

# remove clusters smaller than 200 voxels
gm_mask[counts[labels]<200] = False
wm_mask[wm_counts[label_wm] < 200] = False

print(f"The affines before saving the masks:\ngrey matter\n{gm_img.affine}\nwhite matter\n{wm_img.affine}")
# Save the T1w mask.
nib.save(nib.Nifti1Image(gm_mask, gm_img.affine, gm_img.header), save_name)
nib.save(nib.Nifti1Image(wm_mask, wm_img.affine, wm_img.header), wm_save_name)


# Resample the mask to the space of the bold scan.
bold_ref = nib.load(bold_path)
gm_resampled = resample_to_img(gm_img, 
                               bold_ref,
                               interpolation="nearest"
                               )


print("GM Voxels: ", gm_resampled.get_fdata().sum())

aff = gm_resampled.affine
bold_aff = bold_ref.affine

# To make this run, you need the exact MNI space that they used in the functionnectome package. I will conform for now, but it may be worth doing our own priors to avoid this. 

print(aff)
print(bold_aff)

from templateflow.api import get

# This will download the template if needed
template_path = get('MNI152NLin2009cAsym', resolution=2, desc=None, suffix='T1w', extension='nii.gz')
template = nib.load(template_path)
print(template.affine)



def complete_data_compiler(dfmri_fp, bold_fp, data_filepath=None):
    """
    Docstring for complete_data_compiler
    
    :param dfmri_fp: Description
    :param bold_fp: str
        Filepath to the derivative folder where preprocessed fMRI data lives
    :param data_filepath: Description
    """

    # First crawl through the dMRI folder and get every subject and session pair for whoch there is data
    dict_for_results = {}
    for folder in os.listdir(dfmri_fp):
        split_name = folder.split(sep = "_")
        subj_number = split_name[1]
        session = split_name[-1]
        identifier = f"TAU{subj_number.zfill(3)}"
        available_data = [session, True, False, False]

        dict_for_results[identifier] = available_data
    
       
    # Iterate through all the patients that we have dmri data for

    for participant in dict_for_results:
        session = dict_for_results[participant][0]
        bold_path = path.join(bold_fp, f"sub-{participant}", session, "func", f"sub-{participant}_{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        if os.path.exists(bold_path):
            dict_for_results[participant][2]= True


    
