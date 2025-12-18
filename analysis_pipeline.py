"""Script to collect network metrics and build a dataset"""

# Imports
import numpy as np
import pandas as pd
import os
import json
import sys


def retrieve_participant_data(root_direc, subj_id):
    """
    Retrieves a single participants data from the file location specified as root_direc
    
    :param root_direc: The file path to where the subjects data is stored. Data should be in a dictionary form. Create a 
    docs that has an example.
    :param subj_id: The subject identifier. This will be used to pair the data with behavioural measures.

    :return participant_data: Dictionary containing network measures for the participant.
    """
    network_types = ["structural", "functional", "combined"]

    participant_data = {}

    # Iterate through and capture participant data. 
    for network in network_types:
        directory = os.path.join(root_direc, subj_id, network, "networks")
        try:
            for file_name in os.listdir(directory):
                if file_name.__contains__("json"):

                    data_path = os.path.join(directory, file_name)
                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    identifier = file_name.split(".")[0]
                    participant_data[f"{network}_{identifier}"] = data
        except:
            print(f"Warning: {subj_id} did not find {directory}! Skipping. ")

    return participant_data

def build_dataset(root_directory, patient_list_json_path):
    """
    Generates a heirarchical dictionary containing all participant data. 
    
    :param root_directory: Location of the root directory in which all participant data is stored in a bids like format (add a check for bids format)
    :param patient_list_json_path: A list of all participants.

    :return all_data: Dictionary of all participant data. 
    """
    with open(patient_list_json_path, "r") as f:
        patient_list = json.load(f)
    
    all_data = {}

    for patient in patient_list:
        print(patient)
        patient_data = retrieve_participant_data(root_directory, patient)
        all_data[f"{patient}"] = patient_data
    
    return all_data

def dict2DF(dictionary_version):
    """
    Converts from the dictionary representation to a pandas dataframe. 
    
    :param dictionary_version: Description
    """

    # There is probably a way to refactor this nicely.
    reordered_dict = {}
    for idx, top_level_key in enumerate(dictionary_version.keys()):
        split_key = top_level_key.split("_")
        subj_num = split_key[1]
        session_num = split_key[2]
        reordered_dict[idx] = {"subject_num":subj_num,
                               "session":session_num}

        for secondary_key in dictionary_version[top_level_key].keys():
            split_secondary_key = secondary_key.split("_")
            net_type = split_secondary_key[0]
            atlas = split_secondary_key[3]
            network = split_secondary_key[4]

            for tertiary_key in dictionary_version[top_level_key][secondary_key].keys():
                    new_label = f"{network}_{atlas}_{net_type}_{tertiary_key}"
                    reordered_dict[idx][new_label] = dictionary_version[top_level_key][secondary_key][tertiary_key]

    df_version = pd.DataFrame.from_dict(reordered_dict, orient="index")
    return df_version



if __name__=="__main__":
    root_directory = sys.argv[1]
    patient_list_path = sys.argv[2]
    bulk_data = build_dataset(root_directory, patient_list_path)

    # Convert to a nicely arranged data frame
    reordered_version = dict2DF(bulk_data)

    excel_filename = os.path.join(root_directory, "network_data.csv")

    reordered_version.to_csv(excel_filename)

