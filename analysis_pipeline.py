"""Script to collect network metrics and build a dataset"""

# Imports
import numpy as np
import pandas as pd
import os
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sbs
import chardet
import itertools

NETWORK_TYPES = ["structural", "functional", "combined"]
CURRENT_METRICS = ["MMSE", "MEMORY_Composite", "LANGUAGE_Composite", "EXECUTIVE_Composite", "VISUOSPATIAL_Composite", "GLOBAL_COGNITIVE_Composite"]
NETWORK_METRICS = ["m_connectivity", "density", "diameter", "global_clustering", "isolates"]
DEFINED_NETWORKS = ["ecn", "salience", "dmn-basic", "dmn-ext"]


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
    """
    Builds a dataset that combines the behavioural metrics with the network metrics.
    
    :param behavioural_data_fp: str
        filepath containing the behavioural data. 

    :param network_data_fp: str
        filepath containing the network data.
    """

    with open(behavioural_data_fp, "rb") as f:
        result = chardet.detect(f.read())

    behavioural_data = pd.read_csv(behavioural_data_fp,sep=";", decimal=",", encoding="latin-1")
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
    """
    This fucker sucks. Needs lots of work.
    
    :param long_data: Description
    :param network: Description
    :param network_metric: Description
    """
    # Seaborn styling
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(
        len(NETWORK_TYPES),
        len(CURRENT_METRICS),
        figsize=(30, 16),
        sharex=False
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

            
            axes[i, idx].set_xlabel(network_metric)
            axes[0,idx].set_title(metric)

        y_pos = (len(NETWORK_TYPES) - i - 0.5) / len(NETWORK_TYPES)  # center on row
        fig.text(0.01, y_pos, network_type, va='center', rotation='vertical', fontsize=20)
    
    for i in range(len(NETWORK_TYPES)):
        for j in range(len(CURRENT_METRICS)):
            axes[i, j].set_ylabel("")
            axes[i, j].set_yticklabels([])
            axes[i, j].tick_params(left=False)

    sns.despine()
    plt.tight_layout()

    name = f"scatter_{network}_{network_metric}.png"

    return fig, name




def plot_diagnosis_network_characteristics(data_long, metric_of_interest, network_type):
    """
    Generates Distribution plots for each diagnosis a disorder. Does not compare demented vs non-demented. This works. Could be refined visually. 
    
    :param data_long: Long data. 
    """

    # Note these lines are just for testing on my data - the real thing will need to handle the exceptions. 
    print(data_long.duplicated(subset=["subj_id", "session_num", "type", "network"]))
    data_minus_empty = data_long.dropna(subset=["subj_id", "session_num", "type", "network"])
    print(data_minus_empty.duplicated(subset=["subj_id", "session_num", "type", "network"]))
    data_minus_duplicates = data_long.drop_duplicates(subset=["subj_id", "session_num", "type", "network"])
    name = metric_of_interest+network_type
    df_long = data_minus_duplicates.melt(
        id_vars=["subj_id", "network", "type", metric_of_interest],
        value_vars=["MEM_DISORDER","LANG_DISORDER","EXE_DISORDER","VS_DISORDER"],
        var_name="diagnosis",
        value_name="has_diagnosis"
        )

    g = sns.FacetGrid(
        df_long[df_long["type"] == network_type],
        col="diagnosis",
        row="network",
        hue="has_diagnosis",
        height=3,
        aspect=1.2
        )

    g.map_dataframe(
        sns.kdeplot,
        x=metric_of_interest,
        common_norm=False
    )

    g.add_legend(title="Diagnosis present")
    #plt.show()
    return g, name


def group_comparisons(network_name, data):
    #relevant_data = data[data["network"]==network_name]
    relevant_data = data[data["type"]=="structural"]
    print(relevant_data.columns)
    longer_data = relevant_data.melt(id_vars = ["subj_id", "session_num", "Diagnostic cognitif détaillé_CLASSIF_1", "network", "Demented"],
                                    value_vars = ["m_connectivity", "diameter", "density", "global_clustering", "isolates"],
                                    var_name = "metric",
                                    value_name = "score")
    
    # First plot all patients across different time points
    sns.relplot(data    = longer_data,
                x           = "session_num",
                y           =  "score",
                hue         = "Diagnostic cognitif détaillé_CLASSIF_1",
                col         = "metric",
                row         = "network",
                facet_kws   ={"sharey": False})

    plt.show()

    # Next plot the average for each group over time points.
    sns.relplot(data = longer_data,
            x           = "session_num",
            y           = "score",
            hue         = "Diagnostic cognitif détaillé_CLASSIF_1",
            col         = "metric", 
            kind        = "line",
            estimator   = "mean",
            row         = "network", 
            facet_kws   ={"sharey": False})

    plt.show()

    sns.catplot(data    = longer_data,
            x           = "Diagnostic cognitif détaillé_CLASSIF_1",
            y           =  "score",
            hue         = "Diagnostic cognitif détaillé_CLASSIF_1",
            col         = "metric",
            row         = "network",
            kind        = "violin",
            sharey      = False)

    plt.show()

    sns.catplot(data    = longer_data,
        x           = "Demented",
        y           =  "score",
        hue         = "Demented",
        col         = "metric",
        row         = "network",
        kind        = "violin",
        sharey      = False)

    plt.show()


def rel_composite_to_network(data, composite, metric):
    data = data[data["type"]=="structural"]
    sns.relplot(data=data,
                x = metric,
                y = composite,
                col = "network",
                facet_kws   ={"sharey": False},
                )
    plt.show()

    sns.lmplot(data=data,
                x   = metric,
                y   = composite,
                hue = "Demented",
                col = "network",
                facet_kws   ={"sharey": False})
    plt.show()

def run_tests():
    patient_data = "/Users/sam/Desktop/long_form_combined.csv"
    data_long = pd.read_csv(patient_data, sep=";", decimal=",")
    group_comparisons("ecn",data=data_long)
    #rel_composite_to_network(data_long, "MEMORY_Composite", "density")
    #plot_diagnosis_network_characteristics(data_long, "density", "structural")
    



TESTING = True

if __name__=="__main__":
    if not TESTING:
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

        ########## Plotting Basic Relationships ################
        for network in DEFINED_NETWORKS:
            for metric in NETWORK_METRICS:
                fig, fig_name = simple_plotting(long_version, network, metric)
                figure_savepath = os.path.join(root_directory, "figures")
                os.makedirs(figure_savepath, exist_ok=True)
                figure_savepath = os.path.join(figure_savepath, fig_name)
                fig.savefig(figure_savepath, dpi=300, bbox_inches="tight")
                plt.close(fig)

        for metric in CURRENT_METRICS:
            for network_type in NETWORK_TYPES:
                g, g_name = plot_diagnosis_network_characteristics(long_version, metric, )
                figure_savepath = os.path.join(root_directory, "figures", f"netw-properties_{g_name}")
                g.savefig(figure_savepath, dpi = 300, bbox_inces = "tight")
    else:
        run_tests()
