import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from utilities import dilate_atlas_labels


atlas = "/Users/sam/Desktop/sub-TAU001/check_atlas_TAU001.nii.gz"
atlas_img = nib.load(atlas)
atlas = nib.load(atlas).get_fdata()

brain_mask = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-brain_mask.nii.gz"
brain_mask = nib.load(brain_mask).get_fdata()

dilated_atlas = dilate_atlas_labels(atlas=atlas, 
                    brain_mask=brain_mask, 
                    dilation_width=2)

out = nib.Nifti1Image(dilated_atlas, atlas_img.affine)
out.to_filename("/Users/sam/Desktop/sub-TAU001/dilated_atlas_TAU001.nii.gz")


trk = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk",
                       reference="same")





subsegment = 10
streams = trk.streamlines
print(streams._offsets)
print(streams._offsets.shape)

streams = trk.streamlines
point = streams.get_data()

# Creating subpoints
subpoint = np.linspace(point, np.roll(point, -1, axis=0),
                        subsegment+1, axis=1)
point = subpoint[:, :-1, :].reshape(point.shape[0]*subsegment, 3)




new_trk = StatefulTractogram.from_sft(sft=trk, 
             streamlines=expanded_streamlines)

save_tractogram(sft=new_trk,
                filename="/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1_10x.trk")
