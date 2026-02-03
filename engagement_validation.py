import time
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from nibabel import Nifti1Image
import matplotlib.pyplot as plt
from nilearn import image
from engagement import generate_VWSC_matrices_entire_sl, correlation_thresholding, ebc_computation
from engagement import engagement_calculation, reshape_engagement_slices, generate_VWSC_matrices_V2
from engagement import fc_mat_gen, create_ROI_time_series, dynamic_engagement
from utilities import connectivity_matrix_generation, visualise_square_mat, time_slicing, diffusion_to_t1space
import sparse
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
""" # Look at the results of the real deal:
engagement = nib.load("/Users/sam/Desktop/sub-TAU001/anat/engagement_test_covariance.nii.gz")
engagement_data = engagement.get_fdata()
uniques = np.unique(engagement_data)
print("The unique values", uniques)
print("Non zeros:", np.count_nonzero(engagement_data))
plt.hist(engagement.get_fdata().flatten())
plt.semilogy()
plt.show()
 """

 # Run a simple test case for the white matter
dilated_atlas = "/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz"
dilated_atlas = nib.load(dilated_atlas)
#trk_file = "/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1_10x.trk"
trk_file = "/Users/sam/Desktop/sub-TAU001/TAU_1_ses-2_tractogram.trk"
trk = load_tractogram(trk_file, reference="same")

""" cms, wm_pos = generate_VWSC_matrices(atlas_data=dilated_atlas.get_fdata(),
                            trk=trk,
                            white_matter_mask="/Users/sam/Desktop/sub-TAU001/test_mask_red.nii.gz",
                            verbose=True)

 """
# Test the functional connectivity matrix and thresholding

# Load the actual bold data and see the difference
bold_data_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-T1w_desc-preproc_bold.nii.gz"
atlas = "/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz"
bold_img = image.load_img(bold_data_path)
bold_img_data = bold_img.get_fdata()
bold_img_data = bold_img_data[:, :, :, 3:]
atlas_img = nib.load(atlas)
atlas_data = atlas_img.get_fdata()
wmm_path = "/Users/sam/Desktop/sub-TAU001/test_mask_red.nii.gz"
#thresh_mat = correlation_thresholding(fc, value_threshold= 0.2)

# Time series

t1 = time.time()
total_ts = create_ROI_time_series(atlas=atlas_img,
                              bold_data=bold_img,
                              bold_filepath=bold_data_path)
t2 = time.time()

print(f"Time series generation: {t2-t1} s")
t1 = time.time()
sliced_timeseries = time_slicing(total_ts, 10, True, axis=0)
t2 = time.time()
print(f"Time slicing generation: {t2-t1} s")

t1 = time.time()
for slice in sliced_timeseries:
    fc_mat_gen(slice)
t2 = time.time()
print(f"Time to generate all FC mats: {t2-t1} s")

t1 = time.time()

trk = diffusion_to_t1space(
    moving_file="/Users/sam/Desktop/sub-TAU001/TAU_1_ses-2_FA.nii.gz",
    static_file="/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz",
    trk_file=trk_file,
    save=True
)

all_cms, wm_pos = generate_VWSC_matrices_V2(
        atlas_data=atlas_data,
        trk = trk, 
        white_matter_mask=wmm_path,
        segmentation=10)
t2 = time.time()

print(all_cms.nnz)

fc_mat = fc_mat_gen(
    timeseries=total_ts
)
fc_mat = correlation_thresholding(
    fc_mat,
    value_threshold=0.2
)
ebc_mat = ebc_computation(
    fc_mat,
    False
)
print(all_cms, all_cms.shape)
eng = engagement_calculation(
    EBC_matrix= ebc_mat,
    SC_matrices=all_cms,
)

plt.hist(eng)
plt.show()


print(f"Time to generate mapping: {t2-t1} s")
""" 
t1 = time.time()
all_eng = dynamic_engagement(sliced_timeseries,
                             all_cms)
t2 = time.time()


print(f"Time to get all engagements: {t2-t1} s")



t1 = time.time()
reshaped_engagement = reshape_engagement_slices(all_eng,
                                                atlas_data,
                                                wm_pos)
t2 = time.time()


print(f"Time to reshape {t2-t1} s")

print(reshaped_engagement.shape)



 """





"""cms = sparse.asnumpy(cms2
 for idx, matrix in enumerate(cms):
    title = f"Voxel {wm_pos[idx]} Connectivity"
    #print(np.unique(matrix))
    #visualise_square_mat(matrix, title)


numerators = []

for conn_mat in cms:
    numerator = np.sum(np.multiply(conn_mat, ebc))
    numerators.append(numerator)
    print(numerator)

numerator_np = np.array(numerators)
 """