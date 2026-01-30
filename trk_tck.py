from dipy.io.streamline import load_tractogram, save_tractogram


def trk2tck(input_file: str):

    tract = load_tractogram(input_file, 'same')
    save_tractogram(tract, input_file[:-3]+'tck')


tck = trk2tck("/Users/sam/Desktop/sub-TAU001/TAU_1_ses-2_tractogram.trk")

