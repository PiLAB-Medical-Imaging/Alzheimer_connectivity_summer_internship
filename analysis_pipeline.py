"""Script to collect network metrics and build a dataset"""

# Imports
import numpy as np
import pandas as pd
import os
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sbs

NETWORK_TYPES = ["structural", "functional", "combined"]
CURRENT_METRICS = ["MMSE", "MEMORY_Composite", "LANGUAGE_Composite", "EXECUTIVE_Composite", "VISUOSPATIAL_Composite", "GLOBAL_COGNITIVE_Composite"]

def retrieve_participant_data(root_direc, subj_id):
    """
    Retrieves a single participants data from the file location specified as root_direc
    
    :param root_direc: The file path to where the subjects data is stored. Data should be in a dictionary form. Create a 
    docs that has an example.
    :param subj_id: The subject identifier. This will be used to pair the data with behavioural measures.

    :return participant_data: Dictionary containing network measures for the participant.
    """
    network_types = NETWORK_TYPES

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

    data_list = []

    for top_level_key in dictionary_version.keys():
        split_key = top_level_key.split("_")
        subj_num = split_key[1]
        session_num = split_key[2].split("-")[1]

        
        for secondary_key in dictionary_version[top_level_key].keys():
            split_secondary_key = secondary_key.split("_")
            net_type = split_secondary_key[0]
            atlas = split_secondary_key[3]
            network = split_secondary_key[4]

            row = {"subj_id": subj_num,
                   "session_num":session_num,
                   "type": net_type,
                   "atlas": atlas,
                   "network":network}
            
            for tertiary_key in dictionary_version[top_level_key][secondary_key]:
                row[tertiary_key] = dictionary_version[top_level_key][secondary_key][tertiary_key]
        
            data_list.append(row)

    df_version = pd.DataFrame.from_dict(data_list)
    print(df_version.filter(like="density").dtypes)

    return df_version


def compare_to_behavioural_data(behavioural_data_fp, network_data_fp):
    behavioural_data = pd.read_excel(behavioural_data_fp, decimal=",")
    network_data = pd.read_csv(network_data_fp, sep=";", decimal=",")

    print("behavioural keys: ")
    print(behavioural_data.keys())
    print("Keys for network data")
    print(network_data.keys())

    # Merge them into a long format. 

    long_combined = behavioural_data.merge(network_data, 
                                           how="outer",
                                           left_on=["ID", "Visit_number"], 
                                           right_on=["subj_id", "session_num"])

    return long_combined

import seaborn as sns
import matplotlib.pyplot as plt

def simple_plotting(long_data, network, network_metric):
    # Seaborn styling
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(
        len(NETWORK_TYPES),
        len(CURRENT_METRICS),
        figsize=(30, 16),
        sharex=True
    )
    for i, network_type in enumerate(NETWORK_TYPES):

        data_subset = long_data.loc[
        (long_data["type"] == network_type) &
        (long_data["network"] == network)
        ]
        print("Debugging")
        print(data_subset[network_metric])

        for idx, metric in enumerate(CURRENT_METRICS):
            print(data_subset[metric])

            sns.scatterplot(
                data=data_subset,
                x=network_metric,
                y=metric,
                ax=axes[i, idx],
                s=60,
                alpha=0.7,
                hue = "Demented",
                legend = False, 
                
            )

            axes[i, idx].set_title(f"{metric} vs {network_metric}")
            axes[i, idx].set_xlabel(network_metric)
            axes[i,idx].set_title(metric)

        y_pos = (len(NETWORK_TYPES) - i - 0.5) / len(NETWORK_TYPES)  # center on row
        fig.text(0.04, y_pos, network_type, va='center', rotation='vertical', fontsize=14)
    
    sns.despine()
    plt.tight_layout()

    name = f"scatter_{network}_{network_metric}.png"

    return fig, name




if __name__=="__main__":
    root_directory = sys.argv[1]
    patient_list_path = sys.argv[2]
    patient_behavioural_data = sys.argv[3]

    bulk_data = build_dataset(root_directory, patient_list_path)

    # Convert to a nicely arranged data frame
    reordered_version = dict2DF(bulk_data)

    excel_filename = os.path.join(root_directory, "network_data.csv")

    reordered_version.to_csv(excel_filename, sep=";", decimal = ",")

    long_version = compare_to_behavioural_data(patient_behavioural_data, excel_filename)

    long_save_name = os.path.join(root_directory, "long_form_combined.csv")
    long_version.to_csv(long_save_name, sep=";", decimal = ",")

    fig, fig_name = simple_plotting(long_version, "ecn", "m_connectivity")
    figure_savepath = os.path.join(root_directory, "figures")
    os.makedirs(figure_savepath, exist_ok=True)
    figure_savepath = os.path.join(figure_savepath, fig_name)
    fig.savefig(figure_savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

