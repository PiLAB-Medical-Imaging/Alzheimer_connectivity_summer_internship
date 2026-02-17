from os import makedirs, listdir
from os.path import join, exists
import sys

import nibabel as nib
import numpy as np
import networkx as nx
import pandas as pd

from utilities import nifti_vs_img
from graph_metrics import graph_level_metrics

ENGAGEMENT_NAME = "engagement.nii.gz"
ATLAS_NAME = "registered_atlas.nii.gz"
SIMPLE_WEIGHTING_NAME = "simple_weighting"
STRUCTURAL = "sc_matrix"
FUNCTIONAL = "fc_matrix"
ENG_MEAN_NAME = "mean_eng"
SW_METRIC_NAME = "sw_metrics"
NETWORKS = ["dmn-basic", "dmn-ext", "salience", "ecn", "emot"]
NETWORK_TYPES = [SIMPLE_WEIGHTING_NAME,STRUCTURAL, FUNCTIONAL]
NETWORK_NAMES = ["sw", "struct", "funct"]

def avg_engagement(engagement):
    """
    Simple function to get the average engagement score
    
    :param engagement: Engagement array
    :return: Float, mean of the engagement array. 
    """
    eng_img = nifti_vs_img(engagement)
    eng_data = eng_img.get_fdata()
    mean_eng = np.mean(eng_data)
    return mean_eng

def load_all_indices(definitions_filepath):
    """
    Load ALL network → atlas index mappings from a definitions file.

    Returns:
        dict: {
            "dmn_basic": {
                "AAL116": [...],
                "OtherAtlas": [...],
                ...
            },
            "dmn_ext": {
                "AAL116": [...],
                ...
            }
        }
    """
    # Auto-detect delimiter (handles tabs, semicolons, commas)
    df = pd.read_csv(definitions_filepath, sep="\\t", engine="python")
    print(df.columns)
    df.columns = df.columns.str.strip('"')

    if "Network" not in df.columns:
        raise ValueError("Definitions file must contain a 'Network' column.")

    # Identify atlas columns = everything except 'Network'
    atlas_columns = [col for col in df.columns if col.lower() != "network"]

    # Clean network names (strip whitespace)
    df["Network"] = df["Network"].str.strip('"')

    # Build output structure
    all_networks = {}

    for _, row in df.iterrows():
        network_name = row["Network"]
        print(network_name)
        all_networks[network_name] = {}

        for atlas in atlas_columns:
            raw = str(row[atlas]).strip()

            if raw == "" or raw.lower() == "nan":
                indices = []
            else:
                # Parse comma-separated integers safely
                indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

            all_networks[network_name][atlas] = indices

    return all_networks


def matrix_subsetting(
        connectivity_matrix_filepath, 
        subject_atlas_path, 
        definitions_filepath,
        atlas_name, 
        network_name
    ):
    """
    Subsets a matrix based on a network definition. 
    
    :param connectivity_matrix_filepath: Description
    :param subject_atlas_path: Description
    :param definitions_filepath: Description
    :param atlas_name: Description
    :param network_name: Description
    :return: Description
    :rtype: Callable[[Iterable[object]], bool]
    """
    # Load a graph (either a functional or structural connectivity matrix. 
    # It is essential that the atlas used matches)
    conn_matrix = np.load(
        connectivity_matrix_filepath
    )
    ## Load the atlas (already need to have performed registration)
    atlas_img=nib.load(subject_atlas_path)
    atlas = atlas_img.get_fdata().astype(float)
    # Load all network to brain network mappings
    network_mappings = load_all_indices(definitions_filepath)
    # Define network indices
    try:
        target_indices = network_mappings[network_name][atlas_name]
    except KeyError:
        print(f"Failure on {network_name}:"
              f"The dictionary has {network_mappings.keys()} available.")
        raise KeyError

    # Next - look at the subset of the connectivity matrix and work with that.
    subset_matrix = conn_matrix[np.ix_(target_indices, target_indices)]

    return subset_matrix


def network_extraction_V2(
        atlas_filepath, 
        matrix_filepath, 
        definitions_filepath, 
        atlas
    ):
    """
    Gives a dictionary containing metrics calculated per network. 
    
    :param atlas_filepath: Description
    :param matrix_filepath: Description
    :param definitions_filepath: Description
    :param atlas: Description
    :return: Description
    :rtype: Callable[[Iterable[object]], bool]
    """
    all_values = {}
    for network in NETWORKS:
        selected_network = matrix_subsetting(
            matrix_filepath,
            atlas_filepath,
            definitions_filepath,
            atlas,
            network
        )
        metrics = graph_level_metrics(graph=selected_network)
        for key in metrics.keys():
            new_key = network + "_" + key
            all_values[new_key] = metrics[key]
    return all_values

def subnet_analysis(
        subject_folder,
        matrix_path,
        network_definitions
):

    if exists(matrix_path):
        atlas_fp = join(
            subject_folder,
            ATLAS_NAME
        )
        values = network_extraction_V2(
            atlas_filepath=atlas_fp,
            matrix_filepath=matrix_path,
            definitions_filepath=network_definitions,
            atlas="AAL116"
        )
        return values
    else:
        return None


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
        outputs_folder: str,
        save_path: str,
        network_definitions: str
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
    print("Commencing Compilation!")
    data = []
    total = len(listdir(outputs_folder))
    for i, subject in enumerate(listdir(outputs_folder)):
        print(f'Analysing {i} of total: {subject}')
        subject_folder = join(
            outputs_folder,
            subject
        )
        for session in listdir(subject_folder):
            print(f"\t{session}")
            subj_data = {"subj": subject}
            session_folder = join(
                subject_folder,
                session
            )
            subj_data[session] = session
            
            #Engagement analysis
            eng_path = join(
                session_folder,
                ENGAGEMENT_NAME
            )
            if exists(eng_path):
                mean_eng = avg_engagement(
                    engagement=eng_path
                )
            else:
                print(f"\tEngagement {eng_path} not found")
                mean_eng = None
            subj_data[ENG_MEAN_NAME] = mean_eng

            # Overall simple weighting metrics
            sw_fp = join(
                session_folder,
                subject + "_" + session + "_" + SIMPLE_WEIGHTING_NAME
            )
            if exists(sw_fp):
                sw_mat = np.load(sw_fp)
                sw_metrics = sw_analysis(
                    sw_matrix=sw_mat
                )
                for key in sw_metrics:
                    subj_data[key] = sw_metrics[key]
            else:
                print(f"Warning: Simple weighting matrix was not found at"
                      f"{sw_fp}" )

            # Subnetwork Analysis for the simple weighting data
            matrix_name = subject + "_" + session+ "_" + "simple_weighting.npy"
            subj_results = subnet_analysis(
                subject_folder=session_folder,
                matrix_path=matrix_name,
                network_definitions=network_definitions
            )
            if subj_results is None:
                continue
            for key in subj_results:
                subj_data[key] = subj_results[key]

            data.append(subj_data)

    df = pd.DataFrame(data)
    save_location = join(
        save_path, 
        "compiled_data.csv"
    )
    df.to_csv(
        path_or_buf=save_location
    )


def subject_data_crawler(
        outputs_folder: str,
        subj_number,
        save_path : str,
        network_definitions: str,
):
    """
    A crawler that navigates through the outputs folder for a single patient
    and performs any calculations that you choose. 
    Saves values as a csv
    
    :param outputs_folder: str 
        Filepath to the outputs folder. Assumes a structure of 
        outputs_folder/subject/session/ and that all relevant files have
        a generic name in the final folder.
    """
    print("Commencing Compilation!")
    makedirs(
        name=save_path,
        exist_ok=True
    )
    subj_components = subj_number.split("_")
    subj_id = subj_components[0] + subj_components[1].zfill(3)
    subject_folder = join(
        outputs_folder,
        subj_id
    )
    all_data = []
    for session in listdir(subject_folder):
        print(f"\t{session}")
        if session == "wm_atlas":
            continue
        subject = subj_number[4:7]
        
        session_folder = join(
            subject_folder,
            session
        )
        #Engagement analysis
        eng_path = join(
            session_folder,
            ENGAGEMENT_NAME
        )
        if exists(eng_path):
            mean_eng = avg_engagement(
                engagement=eng_path
            )
        else:
            print(f"\tEngagement {eng_path} not found")
            mean_eng = None

        for i, net_type in enumerate(NETWORK_TYPES):
            subj_data = {"subj": subject}
            subj_data["session"] = session[-1]
            subj_data[ENG_MEAN_NAME] = mean_eng
            subj_data["net_type"] = NETWORK_NAMES[i]
            matrix_fp = join(
                session_folder,
                subj_id + "_" + session + "_" + net_type + ".npy"
            )
            # This is bizarre
            print(matrix_fp)
            if exists(matrix_fp):
                mat = np.load(matrix_fp)
                print(mat)
                print(np.sum(mat))
                mat_metrics = sw_analysis(
                    sw_matrix=mat
                )
                print(mat_metrics)
                for key in mat_metrics:
                    subj_data[key] = mat_metrics[key]
               
                subj_results = subnet_analysis(
                    subject_folder=session_folder,
                    matrix_path=matrix_fp,
                    network_definitions=network_definitions
                )
                if subj_results is None:
                    continue
                for key in subj_results:
                    subj_data[key] = subj_results[key]
            else:
                print(f"Warning: Matrix was not found at"
                        f"{matrix_fp}" )

            all_data.append(subj_data)

    df = pd.DataFrame(
        data = all_data
    )
    save_location = join(
        save_path, 
        subj_id + "_compiled_data.csv"
    )
    df.to_csv(
        path_or_buf=save_location
    )


def subject_wise_compilation(directory):
    all_dfs = []
    for file in listdir(directory):
        file_path = join(
            directory,
            file
        )
        df = pd.read_csv(
            filepath_or_buffer=file_path,
            index_col=0
        )
        all_dfs.append(df)

    final_df = pd.concat(all_dfs)
    return final_df


if __name__ == "__main__":
    output_folder = sys.argv[1]
    save_folder = sys.argv[2]
    network_definitions_fp = sys.argv[3]
    subj_num = sys.argv[4]
    print(f"Input files: \n"
          f"{output_folder}\n"
          f"{save_folder}\n"
          f"{network_definitions_fp}")
    subject_data_crawler(
        outputs_folder=output_folder,
        subj_number=subj_num,
        save_path=save_folder,
        network_definitions=network_definitions_fp
    )
    """
    final_df = subject_wise_compilation(save_folder)
    final_result_path = join(
       save_folder,
        "all_subjects.csv"
    )
    final_df.to_csv(
       final_result_path
    )
    """

