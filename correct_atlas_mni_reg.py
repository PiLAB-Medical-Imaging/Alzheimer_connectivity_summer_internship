import nibabel as nib
from regis.core import find_transform, apply_transform
atlas_file = "/Users/sam/Desktop/sub-TAU001/aal.nii.gz"
mni_file = "/Users/sam/Desktop/sub-TAU001/MNI152_T1_1mm_brain.nii.gz"
tform = find_transform(
    moving_file=atlas_file,
    static_file=mni_file,
    only_affine=True
)
apply_transform(
    moving_file=atlas_file,
    mapping=tform,
    static_file=mni_file,
    output_path="/Users/sam/Desktop/aal_mni_correct.nii.gz"
)
