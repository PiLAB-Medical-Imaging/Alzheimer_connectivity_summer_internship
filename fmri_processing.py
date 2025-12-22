from nilearn import datasets
from nilearn import image
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np



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

    
if __name__ == "__main__":
    atlas_location = sys.argv[1]
    label_location =sys.argv[2]
    fmri_process(atlas_location, label_location)
