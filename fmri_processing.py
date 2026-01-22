from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib
import os
from nibabel.nifti1 import Nifti1Image
import re
from nilearn.plotting import plot_matrix, show
from nilearn.interfaces.fmriprep import load_confounds_strategy


bold_filepath = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
atlas_filepath = "/Users/sam/Desktop/aal116-master/aal116MNI.nii.gz"


def fmri_process(atlas_location, label_location):
    atlas_image = image.load_img(atlas_location)
    tree = ET.parse(label_location)
    root = tree.getroot()
    labels = {}

    region_code = []
    region_name = []
    region_index = []
    # Loop over all 'label' elements anywhere in the tree
    i = 0
    for label in root.iter("label"):
        # Find sub-elements 'index' and 'name'
        idx_elem = label.find("index")
        name_elem = label.find("name")
        if idx_elem is not None and name_elem is not None:
            idx = int(idx_elem.text)
            name = name_elem.text.strip()
            labels[idx] = name
            region_code.append(idx)
            region_name.append(name)
            region_index.append(i)
            i = i+1

    lut = pd.DataFrame(np.column_stack([region_index, region_name]), columns=["index", "name"])

    print(lut)

def connectivity_matrix_generation(bold, atlas, normalise, method= "custom", bold_filepath=None):
    if type(atlas) is str:
        aal_img = nib.load(atlas)
    elif type(atlas) is Nifti1Image:
        aal_img = atlas
    else:
        raise TypeError("The Atlas should be provided as either a path to an image, or the Nifti image object.")
    
    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=normalise)

    if bold_filepath is not None:
        counfounds_df= load_confounds_strategy(bold_filepath,
                                            denoise_strategy="simple")
        time_series = masker.fit_transform(bold, 
                                           counfounds=counfounds_df)
    else:
        time_series = masker.fit_transform(bold)

    
    # Correlation Matrix
    if method == "nilearn":
        conn_measure = ConnectivityMeasure(kind="correlation")
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif method == "custom":
        conn_matrix = matrix_computation(time_series)
    else:
        raise ValueError("Enter a valid method: nilearn or custom")

    return conn_matrix


def matrix_computation(time_series):
    matrix = np.corrcoef(time_series,rowvar=False )
    return matrix

def process_fMRI_dataset(root_path, atlas_path):
    # First, crawl through the derivates folder and find successful fMRI data
    for directory in os.listdir(root_path):
        if os.path.isdir(directory):
            for subfolder in os.listdir(os.path.join(root_path, directory)):
                if subfolder.__contains__("ses"):
                    filename = directory + "_" + subfolder + "_" + "task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                    target_file = os.path.join(root_path, directory, subfolder, "func", filename)
                    if os.path.exists(filename):
                        fconn_matrix = connectivity_matrix_generation(filename, atlas_path)
                        np.save()


def process_fMRI(rootpath, subj_id, session_number, atlas_path):
    # First make sure the subj_id is in the right format:
    number = (re.findall(r'-?\d*\.?\d+', subj_id))
    number = int(number[0]) 
    converted_num = "TAU-{:03d}".format(number)

    

    filepath = os.path.join(rootpath, subj_id, session_number, "task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

    if os.path.exists(filepath):
        fconn_matrix = connectivity_matrix_generation(filepath, atlas_path)
    else:
        raise Exception(f"There is no fMRI data for this patient and session combination:\nSubj ID: {subj_id} \n Session Number: {subj_id}")
    return fconn_matrix

    
            



    
if __name__ == "__main__":
    #atlas_location = sys.argv[1]
    #label_location =sys.argv[2]
    #fmri_process(atlas_location, label_location)
    #matrix = connectivity_matrix_generation(bold_filepath, atlas_filepath)
    #print(matrix)

    process_fMRI("/Users/sam/Desktop", "TAU_1")
