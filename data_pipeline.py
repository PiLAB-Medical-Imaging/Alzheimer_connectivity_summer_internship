import pandas as pd
import numpy as np
import os
import structural_connectivity as sc
import joint_representations as jr
import network_extraction as ne
import graph_metrics as gm

def load_dataset(filepath):
    dataset = pd.read_excel(filepath)
    return dataset

def dataset_pipeline(dMRI_data_path, fMRI_data_path, subj_id, out_path, atlas_path, label_path, definitions_filepath):
    # Create the out_path if it doesnt already exist
    os.makedirs(out_path, exist_ok=True)
    os.path.join(out_path, f"{subj_id}")

    # Generate connectivity matrices and the combined representations (remaining is the fMRI computation)
    ####################################################################################################
    
    

    # Define a save path for the three matrices and metrics
    SC_save_path = os.path.join(out_path, "structural")
    FC_save_path = os.path.join(out_path, "functional")
    JR_save_path = os.path.join(out_path, "combined")
    os.makedirs(SC_save_path, exist_ok=True)
    os.makedirs(FC_save_path, exist_ok=True)
    os.makedirs(JR_save_path, exist_ok=True)


    # Generate and save the connectivity matrix
    SC = sc.generate_connectivity_matrix(dMRI_data_path, out_path, subj_id, atlas_path, label_path)
    SC_filepath = os.path.join(SC_save_path, "SC_matrix.npy")
    np.save(SC_filepath, SC)

    # Generate and save the functional matrix
    FC = np.random.rand(116,116)
    FC_filepath = os.path.join(FC_save_path, "FC_matrix.npy")
    np.save(FC_filepath, FC)

    # Generate and save the joint metrics
    JR =  jr.create_combined_matrices(dMRI_data_path, fMRI_data_path,subj_id, out_path)
    JR_filepath = os.path.join(JR_save_path, "JR_matrix.npy")
    np.save(JR_filepath, JR)

    # for each type of representation, compute networks. Output each to a different subfolder
    dMRI_network_path = os.path.join(SC_filepath, "networks")
    os.makedirs(dMRI_network_path, exist_ok=True)
    structural_networks = ne.network_extraction(SC_save_path, SC_filepath, subj_id, definitions_filepath "AAL116")
    process_dictionary2save(structural_networks, dMRI_network_path)

    fMRI_network_path = os.path.join(FC_filepath, "networks")
    os.makedirs(fMRI_network_path, exist_ok=True)
    functional_networks =  ne.network_extraction(FC_save_path, FC_filepath,subj_id,definitions_filepath, "AAL116" )
    process_dictionary2save(functional_networks, fMRI_network_path)

    
    # Graph metrics on each graph that is designated
    #gm.analyse_graph()



def process_dictionary2save(network_object, parent_folder):
    for key in network_object:
        save_destination = os.path.join(parent_folder, f"network_{key}")
        np.save(save_destination, network_object[key])