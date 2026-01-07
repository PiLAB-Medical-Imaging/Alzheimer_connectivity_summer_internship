from dipy.tracking.utils import target
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
import timeit
import Functionnectome.functionnectome as funct

from unravel.utils import get_streamline_density

trk_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/NT1_track_msmt.trk"
voxel_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/voxel.nii"
out_file = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/output_tracks.tck"
trk = load_tractogram(trk_file, 'same')


def target_1(trk, mask, affine):  
    streamlines = trk.streamlines
    rel_streamlines = target(streamlines, trk.affine, mask)
    return rel_streamlines


mask = nib.load(voxel_file).get_fdata()

rel_streamlines = target_1(trk, mask, trk.affine)
trk_new = trk.from_sft(rel_streamlines, trk)
trk_new.to_vox()
trk_new.to_corner()
density = get_streamline_density(trk_new, resolution_increase=4)

out = nib.Nifti1Image(density, trk.affine)
out.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/density.nii.gz")
save_tractogram(trk_new, out_file)

print(timeit.timeit(lambda: target_1(trk, mask, trk.affine), number = 10)/10)

print(trk.streamlines._offsets)

# try to calculate the different components: 
