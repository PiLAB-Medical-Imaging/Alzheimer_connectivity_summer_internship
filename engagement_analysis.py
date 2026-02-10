import time
import os
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from unravel.stream import get_streamline_density, align_streamline, _smooth_streamline
from tqdm import tqdm

from utilities import diffusion_to_t1space

MNI_PATH = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"
ATLAS_FOLDER = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/bundles"
T1_ANAT_FILE = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz"
OUT_FOLDER = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/Engagement_analysis"
def extract_nodes(trk_file: str, nodes: int = 32, smooth_iter: int = 10):
    '''
    Return the streamline with the most density, subsampled into a defined
    number of nodes.

    Parameters
    ----------
    trk_file : str
        Path to tractogram file
    nodes : int, optional
        Number of points in the average pathway. The default is 32.
    smooth_iter : int, optional
        Number of iterations for smoothing the average pathway. Not recommended
        when there are few nodes. The default is 10.

    Returns
    -------
    centroid : 2D array of size (n, 3)
        Coordinates (x,y,z) of the n mean trajectory points.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    dens = get_streamline_density(trk, subsegment=5)

    streams = trk.streamlines
    s_dens = np.empty(len(streams))

    for i in tqdm(range(len(streams)), desc="Computing centroid streamline"):

        s_dens[i] = np.sum(dens[np.floor(streams[i]).astype(np.int32)],
                           dtype=np.float32)

    s_i_max = np.argmax(s_dens)

    deltas = np.diff(streams[s_i_max], axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)),
                                     dtype=np.float32)
    new_distances = np.linspace(0, cumulative_dist[-1], nodes, dtype=np.float32)
    indices = np.searchsorted(cumulative_dist, new_distances)
    centroid = streams[s_i_max][indices]

    centroid = align_streamline(centroid)
    if smooth_iter > 0:
        centroid = _smooth_streamline(centroid, iterations=smooth_iter)

    return centroid
 
def atlas_to_T1(
        mni_path: str,
        t1_anat_file: str,
        atlas_folder: str,
        save_destination: str
):
    for atlas_file in os.listdir(atlas_folder):
        save_name = os.path.join(
            save_destination,
            "sbj_" + atlas_file
        )
        atlas_path = os.path.join(
            atlas_folder,
            atlas_file
        )
        diffusion_to_t1space(
            moving_file=mni_path,
            static_file=t1_anat_file,
            trk_file=atlas_path,
            save=save_name
        )

if __name__ == "__main__":
    test = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/bundles/AR_R.trk"
    mni_image = nib.load(MNI_PATH)
    trk = load_tractogram(test, "same", bbox_valid_check=False)
    trk.to_vox()
    trk.to_corner()
    dimensions = trk.dimensions
    sls = trk.streamlines.get_data()
    sls = sls + dimensions/2
    trk = StatefulTractogram(
        streamlines=sls,
        reference=mni_image,
        space= Space.VOX, 
        origin=Origin.TRACKVIS
    )
    print(dimensions)
    print(mni_image.get_fdata().shape)
    print(trk.affine)
    print(np.min(trk.streamlines.get_data()))

    """     atlas_to_T1(
        mni_path=MNI_PATH,
        t1_anat_file=T1_ANAT_FILE,
        atlas_folder=ATLAS_FOLDER,
        save_destination=OUT_FOLDER
    ) """