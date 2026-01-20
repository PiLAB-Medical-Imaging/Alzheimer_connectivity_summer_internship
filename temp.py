from unravel.stream import smooth_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
from unravel.utils import get_streamline_density

from regis.core import find_transform
import nibabel as nib

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


        


trk=load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram.trk", "same")
trk.to_vox()
trk.to_corner()
density = get_streamline_density(trk, resolution_increase=1, color=True)
print(trk._dimensions)

trk_new = load_tractogram("/Users/sam/Desktop/TAU_1_ses-2_tractogram_T1.trk", 
                          'same')

trk_new.to_vox()
trk_new.to_corner()
density_new = get_streamline_density(trk_new, resolution_increase=1, color=True)
print(trk_new._dimensions, density_new.shape)


out = nib.Nifti1Image(density, trk.affine)
out.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/density.nii.gz")

out_new= nib.Nifti1Image(density_new, trk_new.affine)
out_new.to_filename("/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/density_new.nii.gz")