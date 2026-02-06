from os import makedirs, listdir
from os.path import join, exists

import nibabel as nib
import numpy as np
import networkx as nx

from utilities import nifti_vs_img
from graph_metrics import graph_level_metrics

ENGAGEMENT_NAME = "engagement.nii.gz"
SIMPLE_WEIGHTING_NAME = "simple_weighting.npy"
ENG_MEAN_NAME = "mean_eng"
SW_METRIC_NAME = "sw_metrics"

def avg_engagement(engagement):
    eng_img = nifti_vs_img(engagement)
    eng_data = eng_img.get_fdata()
    mean_eng = np.mean(eng_data)
    return mean_eng

def sw_analysis(sw_matrix):
    """
    Runs the graph level metrics and returns them as a dictionary
    
    :param sw_matrix: Description
    :return: Description
    :rtype: Callable[[Iterable[object]], bool]
    """
    g = nx.from_numpy_array(
        A=sw_matrix,
        parallel_edges=False
    )
    metrics = graph_level_metrics(
        graph=g
    )
    return metrics

def data_crawler(
        outputs_folder: str
):
    """
    A crawler that navigates through the outputs folder and performs
    any calculations that you choose. Will return all values to a 
    dictionary. 
    
    :param outputs_folder: str 
        Filepath to the outputs folder. Assumes a structure of 
        outputs_folder/subject/session/ and that all relevant files have
        a generic name in the final folder.
    """
    data = []
    for subject in listdir(outputs_folder):
        subj_data = {}
        subject_folder = join(
            outputs_folder,
            subject_folder
        )
        for session in listdir(subject_folder):
            session_folder = join(
                subject_folder,
                session
            )
            subj_data[session] = {}
            eng_path = join(
                session_folder,
                ENGAGEMENT_NAME
            )
            mean_eng = avg_engagement(
                engagement=eng_path
            )
            subj_data[session][ENG_MEAN_NAME] = mean_eng
            
            sw_fp = join(
                session_folder,
                SIMPLE_WEIGHTING_NAME
            )
            sw_mat = np.load(sw_fp)
            sw_metrics = sw_analysis(
                sw_matrix=sw_mat
            )
            subj_data[session][SW_METRIC_NAME] = sw_metrics



