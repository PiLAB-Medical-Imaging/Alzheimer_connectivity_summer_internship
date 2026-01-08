import pandas as pd
import numpy as np
import os
import structural_connectivity as sc
import joint_representations as jr
import network_extraction as ne
import graph_metrics as gm
import json
import sys
from fmri_processing import process_fMRI

def load_dataset(filepath):
    dataset = pd.read_excel(filepath)
    return dataset

def dataset_pipeline(dMRI_data_path, fMRI_data_path, subj_id, out_path, atlas_path, label_path, definitions_filepath):
    # Create the out_path if it doesnt already exist
    out_path = os.path.join(out_path, "analyses")
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"{subj_id}")
    os.makedirs(out_path, exist_ok=True)

    # Generate connectivity matrices and the combined representations (remaining is the fMRI computation)
    ####################################################################################################
    # Define a save path for the three matrices and metrics
    SC_save_path = os.path.join(out_path, "structural")
    FC_save_path = os.path.join(out_path, "functional")
    JR_save_path = os.path.join(out_path, "combined")
    os.makedirs(SC_save_path, exist_ok=True)
    os.makedirs(FC_save_path, exist_ok=True)
    os.makedirs(JR_save_path, exist_ok=True)


    # Generate and save the structural matrix
    SC = sc.generate_connectivity_matrix(dMRI_data_path, out_path, subj_id, atlas_path, label_path)
    SC_filepath = os.path.join(SC_save_path, "SC_matrix.npy")
    np.save(SC_filepath, SC)

    
    # Generate and save the functional matrix
    #FC = np.random.rand(len(SC),len(SC)) # This is just until I have the data for fMRI processed
    FC = process_fMRI(fMRI_data_path, subj_id, session_number)
    FC_filepath = os.path.join(FC_save_path, "FC_matrix.npy")
    np.save(FC_filepath, FC)


    # Generate and save the joint matrices

    # Safety check for dimensions
    if SC.shape != FC.shape:
        print(f"The size of the structral and functional connectivity matrices are not compatible for {subj_id}")
        raise ValueError

    JR =  jr.create_combined_matrices(SC_filepath, FC_filepath, subj_id, out_path)
    JR_filepath = os.path.join(JR_save_path, "JR_matrix.npy")
    np.save(JR_filepath, JR)

    # for each type of representation, compute networks. Output each to a different subfolder
    dMRI_network_path = os.path.join(SC_save_path, "networks")
    os.makedirs(dMRI_network_path, exist_ok=True)
    structural_networks = ne.network_extraction(out_path, 
                                                SC_filepath, 
                                                subj_id, 
                                                definitions_filepath, 
                                                "AAL116")
    process_dictionary2save(structural_networks, dMRI_network_path)

    fMRI_network_path = os.path.join(FC_save_path, "networks")
    os.makedirs(fMRI_network_path, exist_ok=True)
    functional_networks =  ne.network_extraction(out_path, FC_filepath,subj_id,definitions_filepath, "AAL116" )
    process_dictionary2save(functional_networks, fMRI_network_path)

    combined_network_path = os.path.join(JR_save_path, "networks")
    os.makedirs(combined_network_path, exist_ok=True)
    combined_networks =  ne.network_extraction(out_path, JR_filepath, subj_id, definitions_filepath, "AAL116" )
    process_dictionary2save(combined_networks, combined_network_path)

    # Graph metrics on each graph that is designated
    network_path_names = [dMRI_network_path, fMRI_network_path, combined_network_path]

    for network_path in network_path_names:
        for network_file in os.listdir(network_path):

            if network_file.__contains__("json"):
                continue

            adj_matrix = np.load(os.path.join(network_path, network_file), allow_pickle=True)
            graph_metrics = gm.analyse_graph(adj_matrix)
            identifier = network_file.split(".")[0]

            saving_location = os.path.join(network_path, f"graph-metrics_{identifier}_values.json")
            with open(saving_location, "w") as f:
                json.dump(graph_metrics, f, indent=2)    


############ Utility ###############
def process_dictionary2save(network_object, parent_folder):
    for key in network_object:
        
        key_split = key.split("_")
        identifier = f"{key_split[0]}_{key_split[1]}"

        save_destination = os.path.join(parent_folder, f"network_{key}")
        np.save(save_destination, network_object[key])



########### Main #####################
if __name__ == "__main__":

    print("The values that are input are: ")
    for element in sys.argv:
        print(element)
    dMRI_data_path = sys.argv[1]
    fMRI_data_path = sys.argv[2]
    subj_id = sys.argv[3]
    atlas_path = sys.argv[4]
    label_path = sys.argv[5]
    definitions_filepath = sys.argv[6]
    out_path = sys.argv[7]



    dataset_pipeline(dMRI_data_path, fMRI_data_path, subj_id, out_path, atlas_path, label_path, definitions_filepath)