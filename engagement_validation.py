from engagement import engagement_pipeline
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from nibabel import Nifti1Image
import matplotlib.pyplot as plt

""" atlas_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/registered_atlas.nii.gz"
fMRI_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/fake_fMRI.nii.gz"
reference_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/basic_brain.nii.gz"
save_registered_atlas = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/registered_atlas.nii.gz"
tractogram_filepath = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/test_tracts.trk"
moving_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/basic_brain.nii.gz"
density_map_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/"
engagement_savepath = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/engagement.nii.gz"
wm_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/wm_probs.nii.gz"
gm_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/gm_probs.nii.gz"
csf_path = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/FunctValidation/csf_probs.nii.gz"

# Generate the fake FMRI data
T=10
shape = (1, 3, 3)
affine = np.eye(4)
fMRI = np.zeros((1, shape[1], shape[2], T))


for i in range(T):
    fMRI[0, 0, 0, i] = 5 * i
    fMRI[0, 2, 2, i] = 2 * np.random.random(1)
    fMRI[0, 0, 2, i] = 5.1 * np.random.random(1)
    fMRI[0, 2, 0, i] = -5 * np.random.normal(loc= 0, scale = 2, size=1)
# Save the fMRI file: 
fMRI_img = Nifti1Image(fMRI.astype(np.float64), 
                       affine)

fMRI_img.to_filename(fMRI_path)

basic_img = nib.Nifti1Image(np.zeros(shape=shape), affine=affine)

# Make some more white matter connections in the trk.
streamline_1 = np.array([[0,0,0],[0,1,1], [0,2,2]]).astype(np.float64)
streamline_2 = np.array([[0,1,0],[0,1,1], [0,1,2]]).astype(np.float64)
streamline_3 = np.array([[0,2,0],[0,1,1], [0,0,2]]).astype(np.float64)
streamline_4 = np.array([[0,0,0], [0, 0, 1], [0, 0, 2]]).astype(np.float64)
streamline_5 = np.array([[0,0,0], [0, 1, 0], [0, 2, 0], [0, 2, 1], [0, 2, 2]]).astype(np.float64)
streamline_6 = np.array([[0,0,0], [0, 1, 0], [0, 2, 0]]).astype(np.float64)
streamline_7 = np.array([[0,0,0], [0, 0, 1], [0, 2, 2 ]]).astype(np.float64)
new_trk = StatefulTractogram([streamline_1, streamline_2, streamline_3, streamline_4, streamline_5, streamline_6, streamline_7],basic_img, space=Space.VOX, origin=Origin.TRACKVIS)
print("before save",new_trk.origin)
print(new_trk.streamlines)
save_tractogram(new_trk,filename=tractogram_filepath)


# Generate some probability maps
white_matter_probability = np.zeros(shape=shape)
white_matter_probability[0, 1, 0] = 0.2
white_matter_probability[0, 1, 1] = 0.2
white_matter_probability[0, 1, 2] = 0.3
white_matter_probability[0, 0, 1] = 0.2
white_matter_probability[0, 2, 1] = 0.2
gm_probs = np.zeros(shape)
csf_probs = np.zeros(shape)

# Save the maps
wm = nib.Nifti1Image(white_matter_probability, np.eye(4))
wm.to_filename(wm_path)
gm =  nib.Nifti1Image(gm_probs, np.eye(4))
gm.to_filename(gm_path)

csf =  nib.Nifti1Image(csf_probs, np.eye(4))
csf.to_filename(csf_path)

engagement_pipeline(bold_data=fMRI_path,
                    atlas =save_registered_atlas,
                    tractogram_file=tractogram_filepath,
                    white_matter_prob=wm_path,
                    grey_matter_prob=gm_path,
                    csf_prob=csf_path,
                    save_engagement_filepath=engagement_savepath,
                    verbose=True,
                    save_connectomes=True
                    )

 """
# Look at the results of the real deal:
engagement = nib.load("/Users/sam/Desktop/sub-TAU001/anat/engagement_test_covariance.nii.gz")
engagement_data = engagement.get_fdata()
uniques = np.unique(engagement_data)
print("The unique values", uniques)
print("Non zeros:", np.count_nonzero(engagement_data))
plt.hist(engagement.get_fdata().flatten())
plt.semilogy()
plt.show()
