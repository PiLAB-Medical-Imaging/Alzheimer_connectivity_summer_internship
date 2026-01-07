from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib


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

def connectivity_matrix_generation(bold_filepath, atlas_filepath):
    aal_img = nib.load(atlas_filepath)
    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=True)
    time_series = masker.fit_transform(bold_filepath)

    # Correlation Matrix
    conn_measure = ConnectivityMeasure(kind="correlation")
    conn_matrix = conn_measure.fit_transform([time_series])[0]


    return conn_matrix


    
if __name__ == "__main__":
    #atlas_location = sys.argv[1]
    #label_location =sys.argv[2]
    #fmri_process(atlas_location, label_location)
    matrix = connectivity_matrix_generation(bold_filepath, atlas_filepath)
    print(matrix)
