from os import listdir, makedirs
from os.path import join, exists


def generate_complete_data(dMRI_root_file, fMRI_root_file):
    """
    Generates a list of all the patients and sessions for which there is complete data available
    
    :param dMRI_root_file: str
        Path to the root directory containing diffusion study data. 
    :param fMRI_root_file: str
        Path to the fMRI path containing fMRIprep outputs. Must be in BIDS format (really, only use this if you used fMRI prep to do your preprocessing)
    """

    data = {}

    for subject_id in listdir(fMRI_root_file):
        
        subject_folder = join(fMRI_root_file, subject_id)

        for session in listdir(subject_folder):

            session_folder = join(subject_folder, session)

            if "func" in listdir(session_folder):

                
    