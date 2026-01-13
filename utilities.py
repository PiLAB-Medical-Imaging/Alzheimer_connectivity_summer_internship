import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from nilearn.image import resample_to_img
from nibabel.processing import resample_from_to
import os
import os.path as path
from unravel.stream import smooth_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
from regis.core import find_transform
from unravel.utils import get_streamline_density

gm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-GM_probseg.nii.gz"
wm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-WM_probseg.nii.gz"
csf_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-CSF_probseg.nii.gz"
save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-GM_mask.nii.gz"
wm_save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz"
bold_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

""" 
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
wm_mask=wm_mask*1.0

# Save the T1w mask.
nib.save(nib.Nifti1Image(gm_mask, gm_img.affine, gm_img.header), save_name)
nib.save(nib.Nifti1Image(wm_mask, wm_img.affine, wm_img.header), wm_save_name)
out=nib.Nifti1Image(wm_mask,wm_img.affine)
out.to_filename(wm_save_name)

print("white matter image")
print(wm_img.affine, wm_img.header)

 """

from templateflow.api import get

# This will download the template if needed
template_path = get('MNI152NLin2009cAsym', resolution=2, desc=None, suffix='T1w', extension='nii.gz')
template = nib.load(template_path)
print(template.affine)

def mask_generator(white_matter_probability, grey_matter_probability=None, csf_probability = None, mask_type = "white"):
    """
    Generates a white or grey matter mask in the T1 space of the patient. Functionality starts with just a white matter mask generator

    :param white_matter_probability: str
        Path to a file containing the white matter probability map in T1w space (essential that it is T1 space!! Do not use a mask in MNI space)
    
    """

    if mask_type == "white":
        white_matter_img = nib.load(white_matter_probability)
        wm_data = white_matter_img.get_fdata()
        wm_smooth = gaussian_filter(wm_data, sigma = 1.0)
        wm_mask = (wm_smooth > 0.1) # This is commonly used apparently.
        out = nib.Nifti1Image(wm_mask, white_matter_img.affine, white_matter_img.header) 
        return out
    elif mask_type == "grey": # Eventually this will feature different behaviour that allows the mask to be computed a little more advanced (as in the commented code above.)
        gm_img = nib.load(gm_path)
        gm_data = gm_img.get_fdata()  
        gm_smooth = gaussian_filter(gm_data, sigma=1.0)
        gm_mask = (gm_smooth > 0.2)
        out =  nib.Nifti1Image(gm_mask, gm_img.affine, gm_img.header) 
        return out
    else: 
        print(f"Invalid mask type specified: {mask_type}. Valid values are \"white\" and \"grey\"")





def complete_data_compiler(dfmri_fp, bold_fp, data_filepath=None):
    """
    Docstring for complete_data_compiler
    
    :param dfmri_fp: Description
    :param bold_fp: str
        Filepath to the derivative folder where preprocessed fMRI data lives
    :param data_filepath: Description
    """

    # First crawl through the dMRI folder and get every subject and session pair for which there is data
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


    



def diffusion_to_t1space(moving_file, static_file, mni= False, smooth = False):

    # Diffusion space file
    moving_file = '/Users/sam/Desktop/sub-TAU001/TAU_1_ses-2_FA.nii.gz'

    if mni:
    # Ignore
        static_file = 'C:/Users/nicol/Documents/Doctorat/Data/Atlas_Maps/FSL_HCP1065_FA_1mm.nii.gz'
        mapping = find_transform(static_file, moving_file, diffeomorph=False)
    else:
    # T1 file
        static_file = '/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w.nii.gz'
        mapping = find_transform(static_file, moving_file, only_affine=True)


    # For every tract you want to register
    for r in ["1"]:
    # Or hardcode the filename
        trk_file = '/Users/sam/Desktop/TAU_1_ses-2_tractogram.trk'
        #trk_file = 'C:/Users/nicol/Desktop/temp_anais/10_'+r+'.trk'

        trk = load_tractogram(trk_file, 'same')

        stream_reg = transform_streamlines(trk.streamlines,
                                       # np.linalg.inv(mapping.affine))
                                       mapping.affine)

        sft_reg = StatefulTractogram(
        stream_reg, nib.load(static_file), Space.RASMM)

    # trk_new = StatefulTractogram(streams, trk, Space.VOX,
    #                                  origin=Origin.TRACKVIS)

        if mni:
            out_file = trk_file[:-4]+'_mni.trk'
        else:
            out_file = trk_file[:-4]+'_T1.trk'

        save_tractogram(sft_reg, out_file, bbox_valid_check=False)

        if smooth:
        # For visualization, not computing
            smooth_streamlines(out_file, out_file=out_file[:-4]+'_smoothed.trk',
                           iterations=50)



wm_mask = mask_generator(white_matter_probability=wm_path,
                         grey_matter_probability= gm_path,
                         csf_probability= csf_path)

print(wm_mask)